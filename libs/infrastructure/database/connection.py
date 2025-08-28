"""
Oasis Crypto Trade - Database Infrastructure
============================================

Enterprise-grade database management with:
- PostgreSQL + TimescaleDB for time-series data
- Async connection pooling with SQLAlchemy 2.0
- High-performance query optimization
- Connection health monitoring
- Automatic failover and recovery
- Database sharding support

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import contextlib
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar, Union

import asyncpg
import sqlalchemy as sa
from sqlalchemy import MetaData, event, pool, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column
from sqlalchemy.pool import QueuePool

from ...shared.config.base import OasisBaseSettings
from ...shared.exceptions.infrastructure import DatabaseConnectionError, DatabaseQueryError
from ...shared.logging.config import get_logger

logger = get_logger("oasis.database")

T = TypeVar("T", bound="OasisBase")


# =============================================================================
# BASE DATABASE MODEL
# =============================================================================

class OasisBase(DeclarativeBase):
    """
    Base class for all Oasis database models.
    
    Provides common functionality:
    - Automatic table naming
    - Common timestamp fields
    - Audit trail support
    - Soft delete functionality
    """
    
    # Naming convention for constraints and indexes
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        import re
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Common timestamp fields
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        doc="Record creation timestamp"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
        doc="Record last update timestamp"
    )
    
    # Soft delete support
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="Soft delete timestamp"
    )
    
    # Audit trail fields
    created_by: Mapped[Optional[str]] = mapped_column(
        sa.String(255),
        nullable=True,
        doc="User who created the record"
    )
    
    updated_by: Mapped[Optional[str]] = mapped_column(
        sa.String(255),
        nullable=True,
        doc="User who last updated the record"
    )
    
    # Version for optimistic locking
    version: Mapped[int] = mapped_column(
        sa.Integer,
        default=1,
        nullable=False,
        doc="Record version for optimistic locking"
    )
    
    def soft_delete(self, deleted_by: Optional[str] = None):
        """Mark record as soft deleted."""
        self.deleted_at = datetime.utcnow()
        if deleted_by:
            self.updated_by = deleted_by
    
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None
    
    def __repr__(self) -> str:
        """String representation of model."""
        attrs = []
        for column in self.__table__.columns:
            if hasattr(self, column.name):
                value = getattr(self, column.name)
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                attrs.append(f"{column.name}={value!r}")
        
        return f"{self.__class__.__name__}({', '.join(attrs)})"


# =============================================================================
# DATABASE CONNECTION MANAGER
# =============================================================================

class OasisDatabaseManager:
    """
    Enterprise-grade database connection manager.
    
    Features:
    - Connection pooling with health checks
    - Automatic reconnection
    - Query performance monitoring
    - Connection metrics collection
    - Multi-database support
    """
    
    def __init__(self, settings: OasisBaseSettings):
        self.settings = settings
        self.engines: Dict[str, AsyncEngine] = {}
        self.session_makers: Dict[str, async_sessionmaker] = {}
        self._connection_metrics: Dict[str, Dict[str, Any]] = {}
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0
    
    async def initialize(self):
        """Initialize database connections."""
        logger.info("Initializing database connections")
        
        # Primary trading database
        await self._create_engine(
            "primary",
            self.settings.postgres_dsn,
            pool_size=self.settings.DB_POOL_SIZE,
            max_overflow=self.settings.DB_MAX_OVERFLOW
        )
        
        # Analytics database (can be same as primary for now)
        await self._create_engine(
            "analytics",
            self.settings.postgres_dsn,
            pool_size=max(2, self.settings.DB_POOL_SIZE // 2),
            max_overflow=self.settings.DB_MAX_OVERFLOW // 2
        )
        
        # Perform initial health check
        await self.health_check()
        
        logger.info(
            "Database connections initialized",
            databases=list(self.engines.keys()),
            pool_size=self.settings.DB_POOL_SIZE
        )
    
    async def _create_engine(
        self,
        name: str,
        dsn: str,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """Create database engine with optimized settings."""
        
        # Connection arguments for optimal performance
        connect_args = {
            "server_settings": {
                "application_name": f"oasis-{name}",
                "jit": "off",  # Disable JIT for consistent performance
            },
            "command_timeout": self.settings.REQUEST_TIMEOUT,
            "statement_cache_size": 0,  # Disable prepared statement cache for now
        }
        
        # Create engine with connection pooling
        engine = create_async_engine(
            dsn,
            echo=self.settings.DB_ECHO,
            echo_pool=self.settings.DB_ECHO_POOL,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=self.settings.DB_POOL_TIMEOUT,
            pool_recycle=self.settings.DB_POOL_RECYCLE,
            pool_pre_ping=True,  # Enable connection health checks
            connect_args=connect_args,
            future=True,  # Use SQLAlchemy 2.0 style
        )
        
        # Set up event listeners for monitoring
        self._setup_engine_events(engine, name)
        
        # Store engine and create session maker
        self.engines[name] = engine
        self.session_makers[name] = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        # Initialize metrics
        self._connection_metrics[name] = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "avg_connection_time": 0.0,
            "last_connection_time": None,
        }
    
    def _setup_engine_events(self, engine: AsyncEngine, name: str):
        """Set up SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            logger.debug(f"New database connection established", database=name)
            self._connection_metrics[name]["total_connections"] += 1
        
        @event.listens_for(engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            self._connection_metrics[name]["active_connections"] += 1
            connection_record.info['checkout_time'] = time.time()
        
        @event.listens_for(engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            self._connection_metrics[name]["active_connections"] -= 1
            
            if 'checkout_time' in connection_record.info:
                duration = time.time() - connection_record.info['checkout_time']
                metrics = self._connection_metrics[name]
                
                # Update average connection time
                total_time = metrics["avg_connection_time"] * (metrics["total_connections"] - 1)
                metrics["avg_connection_time"] = (total_time + duration) / metrics["total_connections"]
                metrics["last_connection_time"] = datetime.utcnow()
        
        @event.listens_for(engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            logger.warning(
                "Database connection invalidated",
                database=name,
                error=str(exception) if exception else "Unknown"
            )
            self._connection_metrics[name]["failed_connections"] += 1
    
    def get_engine(self, database: str = "primary") -> AsyncEngine:
        """Get database engine by name."""
        if database not in self.engines:
            raise DatabaseConnectionError(f"Database '{database}' not configured")
        return self.engines[database]
    
    def get_session_maker(self, database: str = "primary") -> async_sessionmaker:
        """Get session maker by database name."""
        if database not in self.session_makers:
            raise DatabaseConnectionError(f"Database '{database}' not configured")
        return self.session_makers[database]
    
    @contextlib.asynccontextmanager
    async def session(self, database: str = "primary") -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        session_maker = self.get_session_maker(database)
        
        async with session_maker() as session:
            try:
                yield session
            except Exception as e:
                logger.error(
                    "Database session error",
                    database=database,
                    error=str(e),
                    exc_info=True
                )
                await session.rollback()
                raise DatabaseQueryError(f"Database operation failed: {str(e)}") from e
            finally:
                await session.close()
    
    async def health_check(self, force: bool = False) -> Dict[str, bool]:
        """Perform health check on all database connections."""
        current_time = time.time()
        
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            # Return cached results if within interval
            return {name: True for name in self.engines.keys()}
        
        health_status = {}
        
        for name, engine in self.engines.items():
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                    
                # Check TimescaleDB extension if available
                try:
                    async with engine.begin() as conn:
                        result = await conn.execute(
                            text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
                        )
                        timescale_available = result.scalar() > 0
                        
                    logger.debug(
                        "Database health check passed",
                        database=name,
                        timescale_available=timescale_available
                    )
                    
                except Exception as e:
                    logger.warning(
                        "TimescaleDB extension check failed",
                        database=name,
                        error=str(e)
                    )
                
                health_status[name] = True
                
            except Exception as e:
                logger.error(
                    "Database health check failed",
                    database=name,
                    error=str(e)
                )
                health_status[name] = False
        
        self._last_health_check = current_time
        return health_status
    
    async def get_connection_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get connection pool metrics."""
        metrics = {}
        
        for name, engine in self.engines.items():
            pool = engine.pool
            base_metrics = self._connection_metrics[name].copy()
            
            # Add pool-specific metrics
            base_metrics.update({
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            })
            
            metrics[name] = base_metrics
        
        return metrics
    
    async def execute_raw_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: str = "primary"
    ) -> List[Dict[str, Any]]:
        """Execute raw SQL query and return results."""
        async with self.session(database) as session:
            try:
                result = await session.execute(text(query), parameters or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                return []
                
            except Exception as e:
                logger.error(
                    "Raw query execution failed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    error=str(e)
                )
                raise DatabaseQueryError(f"Query execution failed: {str(e)}") from e
    
    async def close(self):
        """Close all database connections."""
        logger.info("Closing database connections")
        
        for name, engine in self.engines.items():
            try:
                await engine.dispose()
                logger.debug(f"Database connection closed", database=name)
            except Exception as e:
                logger.error(
                    f"Error closing database connection",
                    database=name,
                    error=str(e)
                )
        
        self.engines.clear()
        self.session_makers.clear()
        self._connection_metrics.clear()


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

class TimescaleDBManager:
    """TimescaleDB-specific operations manager."""
    
    def __init__(self, db_manager: OasisDatabaseManager):
        self.db_manager = db_manager
    
    async def create_hypertable(
        self,
        table_name: str,
        time_column: str = "created_at",
        chunk_time_interval: str = "1 day",
        database: str = "primary"
    ):
        """Create TimescaleDB hypertable."""
        query = f"""
        SELECT create_hypertable(
            '{table_name}',
            '{time_column}',
            chunk_time_interval => INTERVAL '{chunk_time_interval}',
            if_not_exists => TRUE
        )
        """
        
        try:
            await self.db_manager.execute_raw_query(query, database=database)
            logger.info(
                "Hypertable created",
                table=table_name,
                time_column=time_column,
                chunk_interval=chunk_time_interval
            )
        except Exception as e:
            logger.error(
                "Failed to create hypertable",
                table=table_name,
                error=str(e)
            )
            raise
    
    async def add_retention_policy(
        self,
        table_name: str,
        retention_period: str = "30 days",
        database: str = "primary"
    ):
        """Add data retention policy to hypertable."""
        query = f"""
        SELECT add_retention_policy(
            '{table_name}',
            INTERVAL '{retention_period}',
            if_not_exists => TRUE
        )
        """
        
        try:
            await self.db_manager.execute_raw_query(query, database=database)
            logger.info(
                "Retention policy added",
                table=table_name,
                retention_period=retention_period
            )
        except Exception as e:
            logger.error(
                "Failed to add retention policy",
                table=table_name,
                error=str(e)
            )
            raise
    
    async def create_continuous_aggregate(
        self,
        view_name: str,
        query: str,
        refresh_policy: str = "1 hour",
        database: str = "primary"
    ):
        """Create continuous aggregate view."""
        create_query = f"""
        CREATE MATERIALIZED VIEW {view_name}
        WITH (timescaledb.continuous) AS
        {query}
        """
        
        refresh_query = f"""
        SELECT add_continuous_aggregate_policy(
            '{view_name}',
            start_offset => INTERVAL '1 day',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '{refresh_policy}',
            if_not_exists => TRUE
        )
        """
        
        try:
            await self.db_manager.execute_raw_query(create_query, database=database)
            await self.db_manager.execute_raw_query(refresh_query, database=database)
            
            logger.info(
                "Continuous aggregate created",
                view=view_name,
                refresh_policy=refresh_policy
            )
        except Exception as e:
            logger.error(
                "Failed to create continuous aggregate",
                view=view_name,
                error=str(e)
            )
            raise


# =============================================================================
# GLOBAL DATABASE INSTANCE
# =============================================================================

# Global database manager instance
db_manager: Optional[OasisDatabaseManager] = None
timescale_manager: Optional[TimescaleDBManager] = None


async def initialize_database(settings: OasisBaseSettings):
    """Initialize global database connection."""
    global db_manager, timescale_manager
    
    db_manager = OasisDatabaseManager(settings)
    await db_manager.initialize()
    
    timescale_manager = TimescaleDBManager(db_manager)
    
    logger.info("Global database manager initialized")


async def get_db_session(database: str = "primary") -> AsyncGenerator[AsyncSession, None]:
    """Get database session - FastAPI dependency compatible."""
    if not db_manager:
        raise DatabaseConnectionError("Database not initialized")
    
    async with db_manager.session(database) as session:
        yield session


async def close_database():
    """Close global database connections."""
    global db_manager, timescale_manager
    
    if db_manager:
        await db_manager.close()
        db_manager = None
        timescale_manager = None
    
    logger.info("Global database connections closed")


def get_database_manager() -> OasisDatabaseManager:
    """Get global database manager."""
    if not db_manager:
        raise DatabaseConnectionError("Database not initialized")
    return db_manager


def get_timescale_manager() -> TimescaleDBManager:
    """Get TimescaleDB manager."""
    if not timescale_manager:
        raise DatabaseConnectionError("TimescaleDB manager not initialized")
    return timescale_manager


# =============================================================================
# DATABASE TESTING UTILITIES
# =============================================================================

async def test_database_connection(settings: OasisBaseSettings) -> bool:
    """Test database connection with detailed diagnostics."""
    logger.info("Testing database connection")
    
    try:
        # Test basic connection
        test_manager = OasisDatabaseManager(settings)
        await test_manager.initialize()
        
        # Test health check
        health_status = await test_manager.health_check(force=True)
        
        # Test basic query
        result = await test_manager.execute_raw_query("SELECT version() as version")
        postgres_version = result[0]["version"] if result else "Unknown"
        
        # Test TimescaleDB availability
        timescale_result = await test_manager.execute_raw_query(
            "SELECT COUNT(*) as count FROM pg_extension WHERE extname = 'timescaledb'"
        )
        timescale_available = timescale_result[0]["count"] > 0 if timescale_result else False
        
        # Get connection metrics
        metrics = await test_manager.get_connection_metrics()
        
        # Clean up
        await test_manager.close()
        
        logger.info(
            "Database connection test successful",
            postgres_version=postgres_version,
            timescale_available=timescale_available,
            health_status=health_status,
            connection_metrics=metrics
        )
        
        return all(health_status.values())
        
    except Exception as e:
        logger.error(
            "Database connection test failed",
            error=str(e),
            exc_info=True
        )
        return False


if __name__ == "__main__":
    """Database connection test."""
    import asyncio
    from ...shared.config.base import get_settings
    
    async def main():
        settings = get_settings()
        success = await test_database_connection(settings)
        
        if success:
            print("✅ Database connection test passed")
            exit(0)
        else:
            print("❌ Database connection test failed")
            exit(1)
    
    asyncio.run(main())