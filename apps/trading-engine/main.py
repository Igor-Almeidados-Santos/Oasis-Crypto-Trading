#!/usr/bin/env python3
"""
Oasis Crypto Trade - Trading Engine
===================================

High-performance algorithmic trading engine for cryptocurrency markets.

Core Features:
- Sub-millisecond order execution
- Multi-strategy orchestration
- Real-time risk management
- Advanced order types
- Exchange abstraction layer
- Performance monitoring
- Graceful shutdown handling

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from libs.infrastructure.database.connection import (
    initialize_database, 
    close_database, 
    get_db_session
)
from libs.infrastructure.cache.redis_client import (
    initialize_cache, 
    close_cache, 
    get_cache
)
from libs.infrastructure.messaging.kafka_producer import (
    initialize_kafka_producer, 
    close_kafka_producer,
    send_trading_event
)
from libs.shared.config.base import get_settings
from libs.shared.logging.config import setup_logging, get_logger
from libs.shared.exceptions.base import (
    OasisException, 
    TradingException,
    format_exception_for_api
)

# Import trading engine components (will be created in future sprints)
# from .domain.trading_engine import TradingEngine
# from .domain.strategy_manager import StrategyManager
# from .domain.order_manager import OrderManager
# from .domain.risk_manager import RiskManager
# from .api.routes import trading_router, health_router

# Configuration
settings = get_settings()
logger = get_logger("oasis.trading_engine")

# Global state
trading_engine_state = {
    'engine': None,
    'strategy_manager': None,
    'order_manager': None,
    'risk_manager': None,
    'startup_time': None,
    'shutdown_requested': False,
}


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting Oasis Trading Engine")
    
    try:
        # Initialize infrastructure
        await initialize_infrastructure()
        
        # Initialize trading components
        await initialize_trading_components()
        
        # Start trading engine
        await start_trading_engine()
        
        trading_engine_state['startup_time'] = datetime.utcnow()
        
        logger.info("✅ Trading Engine started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start Trading Engine: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Shutdown sequence
        logger.info("🔄 Shutting down Trading Engine")
        trading_engine_state['shutdown_requested'] = True
        
        try:
            await shutdown_trading_engine()
            await shutdown_infrastructure()
            logger.info("✅ Trading Engine shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}", exc_info=True)


async def initialize_infrastructure():
    """Initialize infrastructure services."""
    logger.info("📦 Initializing infrastructure services")
    
    # Database
    await initialize_database(settings)
    logger.info("✅ Database initialized")
    
    # Cache
    await initialize_cache(settings)
    logger.info("✅ Cache initialized")
    
    # Message queue
    await initialize_kafka_producer(settings)
    logger.info("✅ Kafka producer initialized")


async def initialize_trading_components():
    """Initialize trading engine components."""
    logger.info("⚙️ Initializing trading components")
    
    # TODO: Initialize actual trading components
    # trading_engine_state['risk_manager'] = RiskManager(settings)
    # trading_engine_state['order_manager'] = OrderManager(settings)
    # trading_engine_state['strategy_manager'] = StrategyManager(settings)
    # trading_engine_state['engine'] = TradingEngine(settings)
    
    # For now, create placeholder objects
    trading_engine_state['risk_manager'] = MockRiskManager()
    trading_engine_state['order_manager'] = MockOrderManager()
    trading_engine_state['strategy_manager'] = MockStrategyManager()
    trading_engine_state['engine'] = MockTradingEngine()
    
    logger.info("✅ Trading components initialized")


async def start_trading_engine():
    """Start the trading engine."""
    logger.info("🎯 Starting trading engine")
    
    # Start all components
    engine = trading_engine_state['engine']
    if engine:
        await engine.start()
    
    logger.info("✅ Trading engine started")


async def shutdown_trading_engine():
    """Shutdown trading engine gracefully."""
    logger.info("⏹️ Stopping trading engine")
    
    # Stop trading engine
    engine = trading_engine_state['engine']
    if engine:
        await engine.stop()
    
    logger.info("✅ Trading engine stopped")


async def shutdown_infrastructure():
    """Shutdown infrastructure services."""
    logger.info("📦 Shutting down infrastructure")
    
    # Close message queue
    await close_kafka_producer()
    logger.info("✅ Kafka producer closed")
    
    # Close cache
    await close_cache()
    logger.info("✅ Cache closed")
    
    # Close database
    await close_database()
    logger.info("✅ Database closed")


# =============================================================================
# MOCK TRADING COMPONENTS (Sprint 1-2 Placeholders)
# =============================================================================

class MockTradingEngine:
    """Mock trading engine for Sprint 1-2."""
    
    def __init__(self):
        self.running = False
        self.strategies = []
        self.orders = []
        self.positions = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    async def start(self):
        """Start the trading engine."""
        logger.info("🚀 Mock Trading Engine starting")
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_task())
        asyncio.create_task(self._performance_monitoring_task())
    
    async def stop(self):
        """Stop the trading engine."""
        logger.info("⏹️ Mock Trading Engine stopping")
        self.running = False
    
    async def _heartbeat_task(self):
        """Heartbeat monitoring task."""
        while self.running:
            logger.debug("💓 Trading Engine heartbeat")
            await asyncio.sleep(30)
    
    async def _performance_monitoring_task(self):
        """Performance monitoring task."""
        while self.running:
            await self._update_performance_metrics()
            await asyncio.sleep(60)
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        # Mock performance calculation
        import random
        self.performance_metrics.update({
            'total_trades': self.performance_metrics['total_trades'] + random.randint(0, 5),
            'winning_trades': self.performance_metrics['winning_trades'] + random.randint(0, 3),
            'total_pnl': self.performance_metrics['total_pnl'] + random.uniform(-100, 200),
            'sharpe_ratio': random.uniform(0.5, 3.0),
            'max_drawdown': random.uniform(0.01, 0.1)
        })
    
    def get_status(self) -> Dict:
        """Get engine status."""
        return {
            'running': self.running,
            'strategies_count': len(self.strategies),
            'active_orders': len(self.orders),
            'positions_count': len(self.positions),
            'performance': self.performance_metrics
        }


class MockStrategyManager:
    """Mock strategy manager for Sprint 1-2."""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategies = set()
    
    async def load_strategy(self, strategy_config: Dict) -> str:
        """Load a trading strategy."""
        strategy_id = f"strategy_{len(self.strategies) + 1}"
        self.strategies[strategy_id] = strategy_config
        logger.info(f"📈 Strategy loaded: {strategy_id}")
        return strategy_id
    
    async def start_strategy(self, strategy_id: str):
        """Start a trading strategy."""
        if strategy_id in self.strategies:
            self.active_strategies.add(strategy_id)
            logger.info(f"▶️ Strategy started: {strategy_id}")
        else:
            raise TradingException(f"Strategy not found: {strategy_id}")
    
    async def stop_strategy(self, strategy_id: str):
        """Stop a trading strategy."""
        self.active_strategies.discard(strategy_id)
        logger.info(f"⏹️ Strategy stopped: {strategy_id}")
    
    def get_strategies(self) -> Dict:
        """Get all strategies."""
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'strategies': list(self.strategies.keys())
        }


class MockOrderManager:
    """Mock order manager for Sprint 1-2."""
    
    def __init__(self):
        self.orders = {}
        self.order_counter = 0
    
    async def place_order(self, order_data: Dict) -> str:
        """Place a trading order."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        
        order = {
            'id': order_id,
            'symbol': order_data.get('symbol'),
            'side': order_data.get('side'),
            'quantity': order_data.get('quantity'),
            'price': order_data.get('price'),
            'status': 'PENDING',
            'created_at': datetime.utcnow().isoformat(),
            **order_data
        }
        
        self.orders[order_id] = order
        
        # Send trading event
        await send_trading_event(
            event_type="order_placed",
            symbol=order['symbol'],
            data=order
        )
        
        logger.info(f"📋 Order placed: {order_id} - {order['side']} {order['quantity']} {order['symbol']}")
        return order_id
    
    async def cancel_order(self, order_id: str):
        """Cancel an order."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            logger.info(f"❌ Order cancelled: {order_id}")
        else:
            raise TradingException(f"Order not found: {order_id}")
    
    def get_orders(self) -> Dict:
        """Get all orders."""
        return {
            'total_orders': len(self.orders),
            'orders': list(self.orders.values())
        }


class MockRiskManager:
    """Mock risk manager for Sprint 1-2."""
    
    def __init__(self):
        self.risk_limits = {
            'max_position_size': 0.1,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.05
        }
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
    
    async def check_risk_limits(self, order_data: Dict) -> bool:
        """Check if order passes risk limits."""
        # Mock risk checking logic
        position_size = float(order_data.get('quantity', 0))
        
        if position_size > self.risk_limits['max_position_size']:
            logger.warning(f"⚠️ Position size {position_size} exceeds limit {self.risk_limits['max_position_size']}")
            return False
        
        logger.debug(f"✅ Risk check passed for order")
        return True
    
    def get_risk_status(self) -> Dict:
        """Get current risk status."""
        return {
            'risk_limits': self.risk_limits,
            'current_exposure': self.current_exposure,
            'daily_pnl': self.daily_pnl,
            'risk_level': 'LOW'  # Mock risk level
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI application
app = FastAPI(
    title="Oasis Trading Engine",
    description="High-performance algorithmic trading engine for cryptocurrency markets",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine = trading_engine_state.get('engine')
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'environment': settings.ENVIRONMENT,
        'startup_time': trading_engine_state.get('startup_time', '').isoformat() if trading_engine_state.get('startup_time') else None,
        'components': {
            'database': 'connected',
            'cache': 'connected',
            'message_queue': 'connected',
            'trading_engine': 'running' if engine and engine.running else 'stopped'
        }
    }
    
    return health_status


@app.get("/status")
async def get_status():
    """Get trading engine status."""
    engine = trading_engine_state.get('engine')
    strategy_manager = trading_engine_state.get('strategy_manager')
    order_manager = trading_engine_state.get('order_manager')
    risk_manager = trading_engine_state.get('risk_manager')
    
    return {
        'engine': engine.get_status() if engine else None,
        'strategies': strategy_manager.get_strategies() if strategy_manager else None,
        'orders': order_manager.get_orders() if order_manager else None,
        'risk': risk_manager.get_risk_status() if risk_manager else None,
    }


@app.post("/orders")
async def place_order(order_data: Dict, background_tasks: BackgroundTasks):
    """Place a trading order."""
    try:
        # Validate order data
        required_fields = ['symbol', 'side', 'quantity']
        for field in required_fields:
            if field not in order_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Risk check
        risk_manager = trading_engine_state.get('risk_manager')
        if risk_manager:
            risk_passed = await risk_manager.check_risk_limits(order_data)
            if not risk_passed:
                raise HTTPException(
                    status_code=400,
                    detail="Order rejected by risk management"
                )
        
        # Place order
        order_manager = trading_engine_state.get('order_manager')
        if not order_manager:
            raise HTTPException(
                status_code=503,
                detail="Order manager not available"
            )
        
        order_id = await order_manager.place_order(order_data)
        
        return {
            'success': True,
            'order_id': order_id,
            'message': 'Order placed successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel a trading order."""
    try:
        order_manager = trading_engine_state.get('order_manager')
        if not order_manager:
            raise HTTPException(
                status_code=503,
                detail="Order manager not available"
            )
        
        await order_manager.cancel_order(order_id)
        
        return {
            'success': True,
            'message': f'Order {order_id} cancelled successfully'
        }
        
    except TradingException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error cancelling order: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/strategies")
async def create_strategy(strategy_config: Dict):
    """Create and load a trading strategy."""
    try:
        strategy_manager = trading_engine_state.get('strategy_manager')
        if not strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy manager not available"
            )
        
        strategy_id = await strategy_manager.load_strategy(strategy_config)
        
        return {
            'success': True,
            'strategy_id': strategy_id,
            'message': 'Strategy created successfully'
        }
        
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """Start a trading strategy."""
    try:
        strategy_manager = trading_engine_state.get('strategy_manager')
        if not strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy manager not available"
            )
        
        await strategy_manager.start_strategy(strategy_id)
        
        return {
            'success': True,
            'message': f'Strategy {strategy_id} started successfully'
        }
        
    except TradingException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """Stop a trading strategy."""
    try:
        strategy_manager = trading_engine_state.get('strategy_manager')
        if not strategy_manager:
            raise HTTPException(
                status_code=503,
                detail="Strategy manager not available"
            )
        
        await strategy_manager.stop_strategy(strategy_id)
        
        return {
            'success': True,
            'message': f'Strategy {strategy_id} stopped successfully'
        }
        
    except Exception as e:
        logger.error(f"Error stopping strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(OasisException)
async def oasis_exception_handler(request, exc: OasisException):
    """Handle Oasis exceptions."""
    return JSONResponse(
        status_code=500,
        content=format_exception_for_api(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            'error': True,
            'message': 'An unexpected error occurred',
            'type': 'internal_server_error'
        }
    )


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        trading_engine_state['shutdown_requested'] = True
        # The actual shutdown is handled by the lifespan context manager
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to start the trading engine."""
    
    # Setup logging
    setup_logging(settings)
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info("🏛️ Oasis Trading Engine Starting")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"API Host: {settings.API_HOST}:{settings.API_PORT}")
    
    # Run the application
    uvicorn.run(
        "apps.trading_engine.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        reload_dirs=["apps", "libs"] if settings.DEBUG else None,
        log_config=None,  # We handle logging ourselves
        access_log=False,  # Disable uvicorn access logs
        workers=1,  # Single worker for now (trading engine state)
    )


if __name__ == "__main__":
    main()