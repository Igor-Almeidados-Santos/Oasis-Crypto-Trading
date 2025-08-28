"""
Oasis Crypto Trade - Base Configuration System
===============================================

Enterprise-grade configuration management with environment-based settings,
security-first design, and comprehensive validation.

Author: Oasis Trading Systems
License: Proprietary
"""

import os
import secrets
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseSettings,
    Field,
    SecretStr,
    validator,
    root_validator,
)
from pydantic.networks import PostgresDsn, RedisDsn


class Environment(str, Enum):
    """Application environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class OasisBaseSettings(BaseSettings):
    """
    Base configuration class for all Oasis services.
    
    Implements enterprise-grade configuration patterns with:
    - Environment-based configuration
    - Secure secret management
    - Comprehensive validation
    - Auto-discovery of configuration files
    """
    
    # ==========================================================================
    # APPLICATION CORE SETTINGS
    # ==========================================================================
    
    # Application Identity
    APP_NAME: str = Field("Oasis Crypto Trade", description="Application name")
    APP_VERSION: str = Field("1.0.0", description="Application version")
    APP_DESCRIPTION: str = Field(
        "Enterprise-grade algorithmic trading system",
        description="Application description"
    )
    
    # Environment Configuration
    ENVIRONMENT: Environment = Field(
        Environment.DEVELOPMENT,
        description="Runtime environment"
    )
    
    DEBUG: bool = Field(False, description="Debug mode flag")
    TESTING: bool = Field(False, description="Testing mode flag")
    
    # API Configuration
    API_PREFIX: str = Field("/api/v1", description="API URL prefix")
    API_HOST: str = Field("0.0.0.0", description="API bind host")
    API_PORT: int = Field(8000, description="API bind port")
    API_WORKERS: int = Field(1, description="API worker processes")
    
    # Security
    SECRET_KEY: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="Application secret key"
    )
    
    # Request/Response Configuration
    REQUEST_TIMEOUT: int = Field(30, description="Request timeout in seconds")
    MAX_REQUEST_SIZE: int = Field(16 * 1024 * 1024, description="Max request size in bytes")
    CORS_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins")
    
    # ==========================================================================
    # DATABASE CONFIGURATION
    # ==========================================================================
    
    # PostgreSQL Primary Database
    POSTGRES_HOST: str = Field("oasis-postgres", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field("oasis_trading_db", description="PostgreSQL database name")
    POSTGRES_USER: str = Field("oasis_admin", description="PostgreSQL username")
    POSTGRES_PASSWORD: SecretStr = Field("oasis_secure_2024", description="PostgreSQL password")
    POSTGRES_SCHEMA: str = Field("public", description="Default schema")
    
    # Database Pool Configuration
    DB_POOL_SIZE: int = Field(10, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(20, description="Database connection pool overflow")
    DB_POOL_TIMEOUT: int = Field(30, description="Database pool timeout")
    DB_POOL_RECYCLE: int = Field(3600, description="Database connection recycle time")
    
    # Database Features
    DB_ECHO: bool = Field(False, description="Enable SQL query echoing")
    DB_ECHO_POOL: bool = Field(False, description="Enable connection pool echoing")
    
    # Analytics Database (Future)
    ANALYTICS_DB_HOST: str = Field("oasis-postgres", description="Analytics DB host")
    ANALYTICS_DB_PORT: int = Field(5432, description="Analytics DB port")
    ANALYTICS_DB_NAME: str = Field("oasis_analytics_db", description="Analytics database")
    
    @property
    def postgres_dsn(self) -> str:
        """Build PostgreSQL connection string."""
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD.get_secret_value(),
            host=self.POSTGRES_HOST,
            port=str(self.POSTGRES_PORT),
            path=f"/{self.POSTGRES_DB}",
        )
    
    # ==========================================================================
    # CACHE CONFIGURATION (Redis)
    # ==========================================================================
    
    # Redis Primary Configuration
    REDIS_HOST: str = Field("oasis-redis", description="Redis host")
    REDIS_PORT: int = Field(6379, description="Redis port")
    REDIS_PASSWORD: SecretStr = Field("oasis_redis_2024", description="Redis password")
    REDIS_DB: int = Field(0, description="Redis database number")
    REDIS_SSL: bool = Field(False, description="Redis SSL enabled")
    
    # Redis Connection Pool
    REDIS_POOL_SIZE: int = Field(10, description="Redis connection pool size")
    REDIS_POOL_TIMEOUT: int = Field(20, description="Redis connection timeout")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(True, description="Retry on Redis timeout")
    
    # Cache Configuration
    CACHE_TTL: int = Field(3600, description="Default cache TTL in seconds")
    CACHE_PREFIX: str = Field("oasis:", description="Cache key prefix")
    
    @property
    def redis_dsn(self) -> str:
        """Build Redis connection string."""
        return RedisDsn.build(
            scheme="redis",
            password=self.REDIS_PASSWORD.get_secret_value(),
            host=self.REDIS_HOST,
            port=str(self.REDIS_PORT),
            path=f"/{self.REDIS_DB}",
        )
    
    # ==========================================================================
    # MESSAGE QUEUE CONFIGURATION (Kafka)
    # ==========================================================================
    
    # Kafka Broker Configuration
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(
        ["oasis-kafka:29092"],
        description="Kafka bootstrap servers"
    )
    KAFKA_SECURITY_PROTOCOL: str = Field("PLAINTEXT", description="Kafka security protocol")
    KAFKA_SASL_MECHANISM: Optional[str] = Field(None, description="SASL mechanism")
    KAFKA_SASL_USERNAME: Optional[str] = Field(None, description="SASL username")
    KAFKA_SASL_PASSWORD: Optional[SecretStr] = Field(None, description="SASL password")
    
    # Kafka Producer Configuration
    KAFKA_PRODUCER_ACKS: str = Field("all", description="Producer acknowledgment level")
    KAFKA_PRODUCER_RETRIES: int = Field(3, description="Producer retry attempts")
    KAFKA_PRODUCER_BATCH_SIZE: int = Field(16384, description="Producer batch size")
    KAFKA_PRODUCER_LINGER_MS: int = Field(10, description="Producer linger time")
    KAFKA_PRODUCER_MAX_REQUEST_SIZE: int = Field(1048576, description="Max request size")
    
    # Kafka Consumer Configuration
    KAFKA_CONSUMER_GROUP_ID: str = Field("oasis-consumer", description="Consumer group ID")
    KAFKA_CONSUMER_AUTO_OFFSET_RESET: str = Field("latest", description="Auto offset reset")
    KAFKA_CONSUMER_ENABLE_AUTO_COMMIT: bool = Field(False, description="Auto commit enabled")
    KAFKA_CONSUMER_MAX_POLL_RECORDS: int = Field(500, description="Max poll records")
    
    # Schema Registry
    SCHEMA_REGISTRY_URL: str = Field(
        "http://oasis-schema-registry:8081",
        description="Schema Registry URL"
    )
    
    # ==========================================================================
    # LOGGING CONFIGURATION
    # ==========================================================================
    
    LOG_LEVEL: LogLevel = Field(LogLevel.INFO, description="Logging level")
    LOG_FORMAT: str = Field("json", description="Log format: json or text")
    LOG_FILE: Optional[Path] = Field(None, description="Log file path")
    LOG_MAX_SIZE: str = Field("100MB", description="Max log file size")
    LOG_BACKUP_COUNT: int = Field(5, description="Number of log backups")
    
    # Structured Logging
    LOG_CORRELATION_ID: bool = Field(True, description="Include correlation ID in logs")
    LOG_REQUEST_ID: bool = Field(True, description="Include request ID in logs")
    LOG_USER_ID: bool = Field(False, description="Include user ID in logs")
    
    # ==========================================================================
    # MONITORING & OBSERVABILITY
    # ==========================================================================
    
    # Metrics Configuration
    METRICS_ENABLED: bool = Field(True, description="Enable metrics collection")
    METRICS_PREFIX: str = Field("oasis_", description="Metrics name prefix")
    METRICS_PORT: int = Field(9090, description="Metrics server port")
    
    # Health Check Configuration
    HEALTH_CHECK_ENABLED: bool = Field(True, description="Enable health checks")
    HEALTH_CHECK_PATH: str = Field("/health", description="Health check endpoint")
    
    # Distributed Tracing
    TRACING_ENABLED: bool = Field(True, description="Enable distributed tracing")
    TRACING_SAMPLE_RATE: float = Field(0.1, description="Trace sampling rate")
    JAEGER_ENDPOINT: Optional[str] = Field(None, description="Jaeger endpoint")
    
    # ==========================================================================
    # SECURITY CONFIGURATION
    # ==========================================================================
    
    # JWT Configuration
    JWT_SECRET_KEY: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="JWT signing key"
    )
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRE_MINUTES: int = Field(30, description="JWT token expiry in minutes")
    JWT_REFRESH_EXPIRE_DAYS: int = Field(7, description="Refresh token expiry in days")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")
    
    # CORS Configuration
    CORS_ALLOW_CREDENTIALS: bool = Field(True, description="Allow credentials in CORS")
    CORS_ALLOW_METHODS: List[str] = Field(
        ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        ["*"],
        description="Allowed CORS headers"
    )
    
    # ==========================================================================
    # TRADING SPECIFIC CONFIGURATION
    # ==========================================================================
    
    # Market Data Configuration
    MARKET_DATA_WEBSOCKET_TIMEOUT: int = Field(
        30,
        description="WebSocket timeout for market data"
    )
    MARKET_DATA_RECONNECT_INTERVAL: int = Field(
        5,
        description="Reconnection interval in seconds"
    )
    MARKET_DATA_MAX_RECONNECT_ATTEMPTS: int = Field(
        10,
        description="Maximum reconnection attempts"
    )
    
    # Risk Management
    MAX_POSITION_SIZE: float = Field(0.1, description="Maximum position size (% of portfolio)")
    MAX_DAILY_LOSS: float = Field(0.02, description="Maximum daily loss (% of portfolio)")
    MAX_DRAWDOWN: float = Field(0.05, description="Maximum drawdown (% of portfolio)")
    
    # Performance Monitoring
    PERFORMANCE_CALCULATION_INTERVAL: int = Field(
        60,
        description="Performance calculation interval in seconds"
    )
    
    # ==========================================================================
    # VALIDATORS
    # ==========================================================================
    
    @validator("ENVIRONMENT", pre=True)
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("LOG_LEVEL", pre=True)
    def validate_log_level(cls, v):
        """Validate log level setting."""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator("API_PORT")
    def validate_api_port(cls, v):
        """Validate API port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError("API port must be between 1024 and 65535")
        return v
    
    @validator("DB_POOL_SIZE")
    def validate_pool_size(cls, v):
        """Validate database pool size."""
        if v < 1:
            raise ValueError("Database pool size must be at least 1")
        return v
    
    @validator("TRACING_SAMPLE_RATE")
    def validate_sample_rate(cls, v):
        """Validate tracing sample rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Tracing sample rate must be between 0.0 and 1.0")
        return v
    
    @root_validator
    def validate_environment_settings(cls, values):
        """Validate environment-specific settings."""
        env = values.get("ENVIRONMENT")
        debug = values.get("DEBUG")
        
        # Production environment validations
        if env == Environment.PRODUCTION:
            if debug:
                raise ValueError("Debug mode cannot be enabled in production")
            
            # Ensure secure secrets in production
            secret_key = values.get("SECRET_KEY")
            if secret_key and len(secret_key.get_secret_value()) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
        
        return values
    
    # ==========================================================================
    # DERIVED PROPERTIES
    # ==========================================================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration dictionary."""
        config = {
            "bootstrap_servers": self.KAFKA_BOOTSTRAP_SERVERS,
            "security_protocol": self.KAFKA_SECURITY_PROTOCOL,
        }
        
        if self.KAFKA_SASL_USERNAME:
            config.update({
                "sasl_mechanism": self.KAFKA_SASL_MECHANISM,
                "sasl_plain_username": self.KAFKA_SASL_USERNAME,
                "sasl_plain_password": self.KAFKA_SASL_PASSWORD.get_secret_value(),
            })
        
        return config
    
    @property
    def jwt_expire_delta(self) -> timedelta:
        """Get JWT expiration timedelta."""
        return timedelta(minutes=self.JWT_EXPIRE_MINUTES)
    
    @property
    def jwt_refresh_expire_delta(self) -> timedelta:
        """Get JWT refresh expiration timedelta."""
        return timedelta(days=self.JWT_REFRESH_EXPIRE_DAYS)
    
    # ==========================================================================
    # CONFIGURATION LOADING
    # ==========================================================================
    
    class Config:
        """Pydantic configuration."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        use_enum_values = True
        validate_assignment = True
        
        # Environment variable prefixes
        env_nested_delimiter = "__"
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            """Customize configuration sources priority."""
            return (
                init_settings,          # 1. Explicit initialization
                env_settings,           # 2. Environment variables
                file_secret_settings,   # 3. Secret files (Docker secrets)
            )


# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================

def get_settings() -> OasisBaseSettings:
    """
    Factory function to create settings instance.
    
    Implements singleton pattern for configuration management.
    """
    return OasisBaseSettings()


# Global settings instance
settings = get_settings()


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def get_database_url(settings: OasisBaseSettings) -> str:
    """Get formatted database URL."""
    return settings.postgres_dsn


def get_redis_url(settings: OasisBaseSettings) -> str:
    """Get formatted Redis URL."""  
    return settings.redis_dsn


def get_kafka_config(settings: OasisBaseSettings) -> Dict[str, Any]:
    """Get Kafka configuration dictionary."""
    return settings.kafka_config


def is_development_mode(settings: OasisBaseSettings) -> bool:
    """Check if application is running in development mode."""
    return settings.is_development


def is_production_mode(settings: OasisBaseSettings) -> bool:
    """Check if application is running in production mode."""
    return settings.is_production


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration(settings: OasisBaseSettings) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        settings: Configuration settings instance
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Database validation
    if not settings.POSTGRES_HOST:
        issues.append("PostgreSQL host is required")
    
    # Redis validation  
    if not settings.REDIS_HOST:
        issues.append("Redis host is required")
    
    # Kafka validation
    if not settings.KAFKA_BOOTSTRAP_SERVERS:
        issues.append("Kafka bootstrap servers are required")
    
    # Production-specific validations
    if settings.is_production:
        if len(settings.SECRET_KEY.get_secret_value()) < 32:
            issues.append("SECRET_KEY must be at least 32 characters in production")
        
        if settings.DEBUG:
            issues.append("Debug mode must be disabled in production")
    
    return issues


if __name__ == "__main__":
    """Configuration validation script."""
    config = get_settings()
    issues = validate_configuration(config)
    
    if issues:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        exit(1)
    else:
        print("✅ Configuration validation passed")
        print(f"Environment: {config.ENVIRONMENT}")
        print(f"Debug: {config.DEBUG}")
        print(f"Database: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
        print(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
        print(f"Kafka: {', '.join(config.KAFKA_BOOTSTRAP_SERVERS)}")