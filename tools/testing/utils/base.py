"""
Oasis Crypto Trade - Testing Framework
======================================

Comprehensive testing infrastructure with:
- Base test classes for different test types
- Database and cache test fixtures
- Mock trading data generators
- Performance testing utilities
- Integration test helpers
- Test data factories
- Async test support

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from ...libs.infrastructure.cache.redis_client import OasisRedisManager
from ...libs.infrastructure.database.connection import OasisDatabaseManager, OasisBase
from ...libs.shared.config.base import OasisBaseSettings
from ...libs.shared.logging.config import setup_logging

# Initialize Faker for generating test data
fake = Faker()

T = TypeVar('T')


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestSettings(OasisBaseSettings):
    """Test-specific configuration settings."""
    
    # Override for testing
    ENVIRONMENT: str = "testing"
    DEBUG: bool = True
    TESTING: bool = True
    
    # Use in-memory databases for testing
    POSTGRES_HOST: str = "localhost"
    POSTGRES_DB: str = "oasis_test_db"
    REDIS_DB: int = 1  # Use different Redis DB for tests
    
    # Faster settings for tests
    CACHE_TTL: int = 60
    REQUEST_TIMEOUT: int = 5
    
    # Disable some features for testing
    TRACING_ENABLED: bool = False
    METRICS_ENABLED: bool = False
    
    class Config:
        env_file = ".env.test"


def get_test_settings() -> TestSettings:
    """Get test-specific settings."""
    return TestSettings()


# =============================================================================
# BASE TEST CLASSES
# =============================================================================

class BaseTest:
    """
    Base test class with common utilities.
    
    Provides:
    - Test data generation
    - Mock helpers
    - Assertion utilities
    - Setup/teardown hooks
    """
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup for each test method."""
        self.test_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Setup test logging
        self.setup_logging()
        
        yield
        
        # Cleanup after test
        self.cleanup_test()
    
    def setup_logging(self):
        """Setup test-specific logging."""
        settings = get_test_settings()
        setup_logging(settings)
    
    def cleanup_test(self):
        """Cleanup after test completion."""
        execution_time = time.time() - self.start_time
        print(f"Test {self.test_id} completed in {execution_time:.3f}s")
    
    def generate_test_id(self) -> str:
        """Generate unique test identifier."""
        return f"test_{uuid.uuid4().hex[:8]}"
    
    def assert_almost_equal(
        self,
        actual: float,
        expected: float,
        tolerance: float = 1e-6,
        message: str = None
    ):
        """Assert two floats are almost equal within tolerance."""
        diff = abs(actual - expected)
        assert diff <= tolerance, (
            message or f"Values not within tolerance: {actual} != {expected} "
            f"(diff: {diff}, tolerance: {tolerance})"
        )
    
    def assert_timestamp_recent(
        self,
        timestamp: datetime,
        max_age_seconds: float = 60.0,
        message: str = None
    ):
        """Assert timestamp is recent."""
        age = (datetime.utcnow() - timestamp).total_seconds()
        assert age <= max_age_seconds, (
            message or f"Timestamp too old: {age}s > {max_age_seconds}s"
        )
    
    def create_mock_async_context_manager(self, return_value: Any = None):
        """Create mock async context manager."""
        mock = AsyncMock()
        mock.__aenter__.return_value = return_value or AsyncMock()
        mock.__aexit__.return_value = None
        return mock


class AsyncBaseTest(BaseTest):
    """Base test class for async tests."""
    
    @pytest.fixture(autouse=True)
    async def async_setup_test(self):
        """Async setup for each test method."""
        await self.async_setup()
        yield
        await self.async_cleanup()
    
    async def async_setup(self):
        """Override in subclasses for async setup."""
        pass
    
    async def async_cleanup(self):
        """Override in subclasses for async cleanup."""
        pass
    
    async def wait_for_condition(
        self,
        condition_func,
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met within timeout"
    ):
        """Wait for a condition to become true."""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        
        raise TimeoutError(error_message)


# =============================================================================
# DATABASE TEST UTILITIES
# =============================================================================

class DatabaseTest(AsyncBaseTest):
    """Base class for database tests."""
    
    @pytest.fixture(autouse=True)
    async def setup_database(self):
        """Setup test database."""
        self.test_settings = get_test_settings()
        
        # Create test database manager
        self.db_manager = OasisDatabaseManager(self.test_settings)
        await self.db_manager.initialize()
        
        # Create all tables
        await self.create_test_tables()
        
        yield
        
        # Cleanup database
        await self.cleanup_database()
    
    async def create_test_tables(self):
        """Create test database tables."""
        # Create tables using SQLAlchemy metadata
        engine = self.db_manager.get_engine("primary")
        async with engine.begin() as conn:
            # Drop all tables first
            await conn.run_sync(OasisBase.metadata.drop_all)
            # Create all tables
            await conn.run_sync(OasisBase.metadata.create_all)
    
    async def cleanup_database(self):
        """Cleanup test database."""
        if self.db_manager:
            await self.db_manager.close()
    
    async def get_test_session(self) -> AsyncSession:
        """Get database session for testing."""
        async with self.db_manager.session() as session:
            yield session
    
    async def clear_all_tables(self):
        """Clear all data from test tables."""
        async with self.db_manager.session() as session:
            # Get all table names
            tables = OasisBase.metadata.tables.keys()
            
            # Disable foreign key constraints
            await session.execute("SET foreign_key_checks = 0")
            
            # Truncate all tables
            for table_name in tables:
                await session.execute(f"TRUNCATE TABLE {table_name}")
            
            # Re-enable foreign key constraints
            await session.execute("SET foreign_key_checks = 1")
            
            await session.commit()


# =============================================================================
# CACHE TEST UTILITIES
# =============================================================================

class CacheTest(AsyncBaseTest):
    """Base class for cache tests."""
    
    @pytest.fixture(autouse=True)
    async def setup_cache(self):
        """Setup test cache."""
        self.test_settings = get_test_settings()
        
        # Create test cache manager
        self.cache_manager = OasisRedisManager(self.test_settings)
        await self.cache_manager.initialize()
        
        # Clear test cache
        await self.clear_test_cache()
        
        yield
        
        # Cleanup cache
        await self.cleanup_cache()
    
    async def cleanup_cache(self):
        """Cleanup test cache."""
        if self.cache_manager:
            await self.clear_test_cache()
            await self.cache_manager.close()
    
    async def clear_test_cache(self):
        """Clear all test cache data."""
        # Delete all keys with test prefix
        await self.cache_manager.redis.flushdb()
    
    async def set_test_data(self, key: str, value: Any, ttl: int = 60):
        """Set test data in cache."""
        return await self.cache_manager.set(f"test:{key}", value, ttl=ttl)
    
    async def get_test_data(self, key: str, default: Any = None):
        """Get test data from cache."""
        return await self.cache_manager.get(f"test:{key}", default=default)


# =============================================================================
# TRADING TEST DATA FACTORIES
# =============================================================================

class TradingDataFactory:
    """Factory for generating trading test data."""
    
    @staticmethod
    def generate_symbol() -> str:
        """Generate random trading symbol."""
        base_currencies = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE']
        quote_currencies = ['USD', 'USDT', 'EUR', 'BTC', 'ETH']
        
        base = fake.random_element(base_currencies)
        quote = fake.random_element(quote_currencies)
        
        # Avoid same currency pairs
        while base == quote:
            quote = fake.random_element(quote_currencies)
        
        return f"{base}/{quote}"
    
    @staticmethod
    def generate_price(min_price: float = 0.01, max_price: float = 100000.0) -> Decimal:
        """Generate random price."""
        price = fake.pydecimal(
            left_digits=6,
            right_digits=8,
            positive=True,
            min_value=Decimal(str(min_price)),
            max_value=Decimal(str(max_price))
        )
        return price
    
    @staticmethod
    def generate_quantity(min_qty: float = 0.001, max_qty: float = 1000.0) -> Decimal:
        """Generate random quantity."""
        qty = fake.pydecimal(
            left_digits=4,
            right_digits=8,
            positive=True,
            min_value=Decimal(str(min_qty)),
            max_value=Decimal(str(max_qty))
        )
        return qty
    
    @staticmethod
    def generate_order_data(**overrides) -> Dict[str, Any]:
        """Generate order test data."""
        order_sides = ['BUY', 'SELL']
        order_types = ['MARKET', 'LIMIT', 'STOP_LOSS', 'STOP_LIMIT']
        
        data = {
            'order_id': str(uuid.uuid4()),
            'symbol': TradingDataFactory.generate_symbol(),
            'side': fake.random_element(order_sides),
            'type': fake.random_element(order_types),
            'quantity': TradingDataFactory.generate_quantity(),
            'price': TradingDataFactory.generate_price(),
            'status': 'PENDING',
            'created_at': fake.date_time_between(start_date='-1h', end_date='now'),
            'updated_at': fake.date_time_between(start_date='-1h', end_date='now'),
        }
        
        data.update(overrides)
        return data
    
    @staticmethod
    def generate_trade_data(**overrides) -> Dict[str, Any]:
        """Generate trade test data."""
        data = {
            'trade_id': str(uuid.uuid4()),
            'order_id': str(uuid.uuid4()),
            'symbol': TradingDataFactory.generate_symbol(),
            'side': fake.random_element(['BUY', 'SELL']),
            'quantity': TradingDataFactory.generate_quantity(),
            'price': TradingDataFactory.generate_price(),
            'fee': TradingDataFactory.generate_price(max_price=10.0),
            'fee_currency': fake.random_element(['USD', 'BTC', 'ETH']),
            'timestamp': fake.date_time_between(start_date='-1h', end_date='now'),
        }
        
        data.update(overrides)
        return data
    
    @staticmethod
    def generate_market_data(**overrides) -> Dict[str, Any]:
        """Generate market data."""
        base_price = float(TradingDataFactory.generate_price())
        
        data = {
            'symbol': TradingDataFactory.generate_symbol(),
            'timestamp': fake.date_time_between(start_date='-1m', end_date='now'),
            'open': base_price,
            'high': base_price * fake.pyfloat(min_value=1.0, max_value=1.1),
            'low': base_price * fake.pyfloat(min_value=0.9, max_value=1.0),
            'close': base_price * fake.pyfloat(min_value=0.95, max_value=1.05),
            'volume': float(TradingDataFactory.generate_quantity(max_qty=10000)),
            'quote_volume': base_price * float(TradingDataFactory.generate_quantity(max_qty=10000)),
            'trades_count': fake.random_int(min=1, max=1000),
        }
        
        data.update(overrides)
        return data
    
    @staticmethod
    def generate_orderbook_data(symbol: str = None, levels: int = 10) -> Dict[str, Any]:
        """Generate orderbook test data."""
        symbol = symbol or TradingDataFactory.generate_symbol()
        base_price = float(TradingDataFactory.generate_price())
        
        # Generate bids (buy orders) - prices below base price
        bids = []
        for i in range(levels):
            price = base_price * (1 - (i + 1) * 0.001)  # Descending prices
            quantity = float(TradingDataFactory.generate_quantity())
            bids.append([price, quantity])
        
        # Generate asks (sell orders) - prices above base price
        asks = []
        for i in range(levels):
            price = base_price * (1 + (i + 1) * 0.001)  # Ascending prices
            quantity = float(TradingDataFactory.generate_quantity())
            asks.append([price, quantity])
        
        return {
            'symbol': symbol,
            'timestamp': fake.date_time_between(start_date='-1m', end_date='now'),
            'bids': bids,
            'asks': asks,
        }


# =============================================================================
# PERFORMANCE TEST UTILITIES
# =============================================================================

class PerformanceTest(BaseTest):
    """Base class for performance testing."""
    
    def __init__(self):
        super().__init__()
        self.performance_metrics = {}
    
    def benchmark_function(self, func, *args, **kwargs):
        """Benchmark function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        func_name = func.__name__
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = []
        
        self.performance_metrics[func_name].append(execution_time)
        return result, execution_time
    
    async def benchmark_async_function(self, func, *args, **kwargs):
        """Benchmark async function execution time."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        func_name = func.__name__
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = []
        
        self.performance_metrics[func_name].append(execution_time)
        return result, execution_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for func_name, times in self.performance_metrics.items():
            if times:
                summary[func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'times': times,
                }
        
        return summary
    
    def assert_performance_within_limit(
        self,
        func_name: str,
        max_time_seconds: float,
        percentile: float = 95.0
    ):
        """Assert function performance is within limits."""
        times = self.performance_metrics.get(func_name, [])
        assert times, f"No performance data for function '{func_name}'"
        
        # Calculate percentile
        sorted_times = sorted(times)
        percentile_index = int((percentile / 100.0) * len(sorted_times))
        percentile_time = sorted_times[min(percentile_index, len(sorted_times) - 1)]
        
        assert percentile_time <= max_time_seconds, (
            f"Function '{func_name}' {percentile}th percentile time "
            f"{percentile_time:.3f}s exceeds limit {max_time_seconds:.3f}s"
        )


# =============================================================================
# MOCK HELPERS
# =============================================================================

class MockExchangeAPI:
    """Mock exchange API for testing."""
    
    def __init__(self):
        self.call_history = []
        self.responses = {}
        self.errors = {}
        self.delays = {}
    
    def set_response(self, method: str, response: Any):
        """Set mock response for method."""
        self.responses[method] = response
    
    def set_error(self, method: str, error: Exception):
        """Set mock error for method."""
        self.errors[method] = error
    
    def set_delay(self, method: str, delay_seconds: float):
        """Set mock delay for method."""
        self.delays[method] = delay_seconds
    
    async def call_api(self, method: str, *args, **kwargs):
        """Mock API call."""
        self.call_history.append({
            'method': method,
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.utcnow(),
        })
        
        # Simulate delay if configured
        if method in self.delays:
            await asyncio.sleep(self.delays[method])
        
        # Raise error if configured
        if method in self.errors:
            raise self.errors[method]
        
        # Return response if configured
        if method in self.responses:
            return self.responses[method]
        
        # Default response
        return {'success': True, 'data': None}
    
    def get_call_count(self, method: str = None) -> int:
        """Get number of API calls made."""
        if method is None:
            return len(self.call_history)
        
        return len([call for call in self.call_history if call['method'] == method])
    
    def get_last_call(self, method: str = None) -> Optional[Dict[str, Any]]:
        """Get last API call made."""
        calls = self.call_history
        if method:
            calls = [call for call in calls if call['method'] == method]
        
        return calls[-1] if calls else None
    
    def reset(self):
        """Reset mock state."""
        self.call_history.clear()
        self.responses.clear()
        self.errors.clear()
        self.delays.clear()


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_settings():
    """Test settings fixture."""
    return get_test_settings()


@pytest.fixture
async def test_db_manager(test_settings):
    """Test database manager fixture."""
    db_manager = OasisDatabaseManager(test_settings)
    await db_manager.initialize()
    
    yield db_manager
    
    await db_manager.close()


@pytest.fixture
async def test_cache_manager(test_settings):
    """Test cache manager fixture."""
    cache_manager = OasisRedisManager(test_settings)
    await cache_manager.initialize()
    await cache_manager.redis.flushdb()  # Clear test data
    
    yield cache_manager
    
    await cache_manager.redis.flushdb()  # Clean up
    await cache_manager.close()


@pytest.fixture
def trading_data_factory():
    """Trading data factory fixture."""
    return TradingDataFactory()


@pytest.fixture
def mock_exchange_api():
    """Mock exchange API fixture."""
    return MockExchangeAPI()


@pytest.fixture
def temp_directory():
    """Temporary directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_test_symbols(count: int = 10) -> List[str]:
    """Generate list of test trading symbols."""
    return [TradingDataFactory.generate_symbol() for _ in range(count)]


def generate_test_orders(count: int = 10) -> List[Dict[str, Any]]:
    """Generate list of test orders."""
    return [TradingDataFactory.generate_order_data() for _ in range(count)]


def generate_test_trades(count: int = 10) -> List[Dict[str, Any]]:
    """Generate list of test trades."""
    return [TradingDataFactory.generate_trade_data() for _ in range(count)]


def generate_price_history(
    symbol: str,
    days: int = 30,
    interval_minutes: int = 60
) -> List[Dict[str, Any]]:
    """Generate historical price data."""
    history = []
    current_time = datetime.utcnow() - timedelta(days=days)
    current_price = float(TradingDataFactory.generate_price())
    
    while current_time < datetime.utcnow():
        # Price random walk
        price_change = fake.pyfloat(min_value=-0.05, max_value=0.05)
        current_price = current_price * (1 + price_change)
        
        high = current_price * fake.pyfloat(min_value=1.0, max_value=1.02)
        low = current_price * fake.pyfloat(min_value=0.98, max_value=1.0)
        volume = float(TradingDataFactory.generate_quantity(max_qty=1000))
        
        history.append({
            'symbol': symbol,
            'timestamp': current_time,
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume,
        })
        
        current_time += timedelta(minutes=interval_minutes)
    
    return history


# =============================================================================
# TEST UTILITIES
# =============================================================================

def skip_if_no_database():
    """Skip test if database is not available."""
    return pytest.mark.skipif(
        os.getenv('SKIP_DB_TESTS', 'false').lower() == 'true',
        reason="Database tests disabled"
    )


def skip_if_no_cache():
    """Skip test if cache is not available."""
    return pytest.mark.skipif(
        os.getenv('SKIP_CACHE_TESTS', 'false').lower() == 'true',
        reason="Cache tests disabled"
    )


def skip_if_no_external_services():
    """Skip test if external services are not available."""
    return pytest.mark.skipif(
        os.getenv('SKIP_EXTERNAL_TESTS', 'false').lower() == 'true',
        reason="External service tests disabled"
    )


def parametrize_symbols(symbols: List[str] = None):
    """Parametrize test with multiple trading symbols."""
    if symbols is None:
        symbols = generate_test_symbols(5)
    return pytest.mark.parametrize('symbol', symbols)


# =============================================================================
# TEST MARKERS
# =============================================================================

# Custom pytest markers for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.slow = pytest.mark.slow
pytest.mark.external = pytest.mark.external
pytest.mark.trading = pytest.mark.trading
pytest.mark.ml = pytest.mark.ml


if __name__ == "__main__":
    """Test framework validation."""
    
    # Test data generation
    print("🧪 Testing Oasis Test Framework")
    
    # Test trading data factory
    factory = TradingDataFactory()
    
    symbol = factory.generate_symbol()
    print(f"✅ Generated symbol: {symbol}")
    
    order = factory.generate_order_data(symbol=symbol)
    print(f"✅ Generated order: {order['order_id']} - {order['side']} {order['quantity']} {order['symbol']}")
    
    trade = factory.generate_trade_data(symbol=symbol)
    print(f"✅ Generated trade: {trade['trade_id']} - {trade['quantity']}@{trade['price']}")
    
    market_data = factory.generate_market_data(symbol=symbol)
    print(f"✅ Generated market data: {market_data['symbol']} OHLC: {market_data['open']:.4f}")
    
    orderbook = factory.generate_orderbook_data(symbol=symbol)
    print(f"✅ Generated orderbook: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
    
    # Test price history generation
    history = generate_price_history(symbol, days=1, interval_minutes=15)
    print(f"✅ Generated price history: {len(history)} data points")
    
    print("\n✅ Test framework validation completed")