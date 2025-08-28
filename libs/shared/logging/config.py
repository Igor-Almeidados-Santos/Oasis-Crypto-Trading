"""
Oasis Crypto Trade - Enterprise Logging System
==============================================

Comprehensive logging infrastructure with:
- Structured JSON logging
- OpenTelemetry integration
- Correlation ID tracking
- Performance monitoring
- Security audit logging
- Distributed tracing support

Author: Oasis Trading Systems
License: Proprietary
"""

import contextvars
import logging
import logging.config
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pythonjsonlogger import jsonlogger

from ..config.base import OasisBaseSettings


# =============================================================================
# CONTEXT VARIABLES FOR REQUEST TRACKING
# =============================================================================

# Request context variables
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)
correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)
user_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)
session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'session_id', default=None
)
trace_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'trace_id', default=None
)


# =============================================================================
# STRUCTURED LOGGING PROCESSORS
# =============================================================================

def add_correlation_context(logger, method_name, event_dict):
    """Add correlation context to log entries."""
    # Request tracking
    request_id = request_id_ctx.get()
    if request_id:
        event_dict['request_id'] = request_id
    
    correlation_id = correlation_id_ctx.get()
    if correlation_id:
        event_dict['correlation_id'] = correlation_id
    
    # User context
    user_id = user_id_ctx.get()
    if user_id:
        event_dict['user_id'] = user_id
    
    session_id = session_id_ctx.get()
    if session_id:
        event_dict['session_id'] = session_id
    
    # Distributed tracing
    trace_id = trace_id_ctx.get()
    if trace_id:
        event_dict['trace_id'] = trace_id
    
    # OpenTelemetry span context
    span = trace.get_current_span()
    if span.is_recording():
        span_context = span.get_span_context()
        event_dict['span_id'] = f"{span_context.span_id:016x}"
        event_dict['trace_id'] = f"{span_context.trace_id:032x}"
    
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add precise timestamp to log entries."""
    event_dict['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    event_dict['@timestamp'] = event_dict['timestamp']  # For ELK compatibility
    return event_dict


def add_service_context(logger, method_name, event_dict):
    """Add service context information."""
    event_dict['service'] = 'oasis-crypto-trade'
    event_dict['component'] = logger.name
    event_dict['environment'] = getattr(add_service_context, '_environment', 'unknown')
    event_dict['version'] = getattr(add_service_context, '_version', '1.0.0')
    return event_dict


def add_performance_context(logger, method_name, event_dict):
    """Add performance metrics to log entries."""
    # Add execution time if available
    if hasattr(logger, '_start_time'):
        execution_time = time.time() - logger._start_time
        event_dict['execution_time'] = round(execution_time * 1000, 2)  # milliseconds
    
    return event_dict


def format_exception(logger, method_name, event_dict):
    """Format exception information."""
    if 'exception' in event_dict:
        exc_info = event_dict['exception']
        if exc_info:
            # Add detailed exception information
            exc_type, exc_value, exc_traceback = exc_info
            event_dict['exception_type'] = exc_type.__name__
            event_dict['exception_message'] = str(exc_value)
            event_dict['exception_traceback'] = traceback.format_exception(
                exc_type, exc_value, exc_traceback
            )
            # Remove the raw exception to avoid serialization issues
            del event_dict['exception']
    
    return event_dict


def censor_sensitive_data(logger, method_name, event_dict):
    """Censor sensitive data from logs."""
    sensitive_keys = {
        'password', 'secret', 'token', 'key', 'private_key', 
        'api_key', 'authorization', 'auth_token', 'session_token',
        'credit_card', 'ssn', 'social_security'
    }
    
    def _censor_dict(d: dict) -> dict:
        """Recursively censor dictionary values."""
        censored = {}
        for key, value in d.items():
            lower_key = key.lower()
            if any(sensitive in lower_key for sensitive in sensitive_keys):
                censored[key] = '***CENSORED***'
            elif isinstance(value, dict):
                censored[key] = _censor_dict(value)
            elif isinstance(value, list):
                censored[key] = [_censor_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                censored[key] = value
        return censored
    
    # Censor the entire event_dict
    return _censor_dict(event_dict)


# =============================================================================
# CUSTOM JSON FORMATTER
# =============================================================================

class OasisJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for Oasis logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_name = "oasis-crypto-trade"
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add service information
        log_record['service'] = self.service_name
        log_record['logger'] = record.name
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add process information
        log_record['process'] = record.process
        log_record['thread'] = record.thread
        log_record['thread_name'] = record.threadName
        
        # Add correlation context
        request_id = request_id_ctx.get()
        if request_id:
            log_record['request_id'] = request_id
        
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            log_record['correlation_id'] = correlation_id


# =============================================================================
# CUSTOM LOG HANDLERS
# =============================================================================

class TradingAuditHandler(logging.Handler):
    """Custom handler for trading-specific audit logs."""
    
    def __init__(self, audit_file: Optional[Path] = None):
        super().__init__()
        self.audit_file = audit_file or Path("logs/trading_audit.log")
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        """Emit audit log record."""
        if hasattr(record, 'audit') and record.audit:
            try:
                with open(self.audit_file, 'a', encoding='utf-8') as f:
                    log_entry = self.format(record)
                    f.write(log_entry + '\n')
                    f.flush()
            except Exception:
                self.handleError(record)


class PerformanceHandler(logging.Handler):
    """Custom handler for performance monitoring logs."""
    
    def __init__(self):
        super().__init__()
        self.performance_data = []
    
    def emit(self, record):
        """Emit performance log record."""
        if hasattr(record, 'performance') and record.performance:
            try:
                # Store performance data for analysis
                perf_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': getattr(record, 'operation', 'unknown'),
                    'duration': getattr(record, 'duration', 0),
                    'success': getattr(record, 'success', True),
                    'details': getattr(record, 'details', {})
                }
                self.performance_data.append(perf_data)
                
                # Keep only recent performance data (last 1000 entries)
                if len(self.performance_data) > 1000:
                    self.performance_data = self.performance_data[-1000:]
                    
            except Exception:
                self.handleError(record)


# =============================================================================
# LOGGING CONFIGURATION BUILDER
# =============================================================================

class OasisLoggingConfig:
    """Oasis logging configuration builder."""
    
    def __init__(self, settings: OasisBaseSettings):
        self.settings = settings
        self._configure_service_context()
    
    def _configure_service_context(self):
        """Configure global service context."""
        add_service_context._environment = self.settings.ENVIRONMENT.value
        add_service_context._version = self.settings.APP_VERSION
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Build logging configuration dictionary."""
        
        log_level = self.settings.LOG_LEVEL.value
        log_format = self.settings.LOG_FORMAT
        
        # Base formatters
        formatters = {
            'json': {
                '()': OasisJSONFormatter,
                'format': '%(timestamp)s %(level)s %(logger)s %(message)s'
            },
            'detailed': {
                'format': (
                    '%(asctime)s | %(levelname)-8s | %(name)s | '
                    '%(funcName)s:%(lineno)d | %(message)s'
                ),
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s | %(name)s | %(message)s'
            }
        }
        
        # Base handlers
        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'formatter': 'json' if log_format == 'json' else 'detailed',
                'level': log_level,
            },
            'audit': {
                '()': TradingAuditHandler,
                'formatter': 'json',
                'level': 'INFO',
            },
            'performance': {
                '()': PerformanceHandler,
                'formatter': 'json',
                'level': 'DEBUG',
            }
        }
        
        # Add file handler if log file is specified
        if self.settings.LOG_FILE:
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(self.settings.LOG_FILE),
                'maxBytes': self._parse_size(self.settings.LOG_MAX_SIZE),
                'backupCount': self.settings.LOG_BACKUP_COUNT,
                'formatter': 'json',
                'level': log_level,
            }
        
        # Logger configuration
        loggers = {
            '': {  # Root logger
                'level': log_level,
                'handlers': ['console'],
                'propagate': False,
            },
            'oasis_crypto_trade': {
                'level': log_level,
                'handlers': ['console', 'audit'] + (['file'] if self.settings.LOG_FILE else []),
                'propagate': False,
            },
            'trading': {
                'level': 'INFO',
                'handlers': ['audit', 'performance'],
                'propagate': True,
            },
            'performance': {
                'level': 'DEBUG',
                'handlers': ['performance'],
                'propagate': True,
            },
            # Third-party library loggers
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False,
            },
            'kafka': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False,
            },
        }
        
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': loggers,
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)


# =============================================================================
# STRUCTLOG CONFIGURATION
# =============================================================================

def configure_structlog(settings: OasisBaseSettings):
    """Configure structlog for structured logging."""
    
    # Choose renderer based on format preference
    if settings.LOG_FORMAT == 'json':
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=not settings.is_production)
    
    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        add_correlation_context,
        add_timestamp,
        add_service_context,
        censor_sensitive_data,
        format_exception,
    ]
    
    # Add performance processor if enabled
    if settings.DEBUG:
        processors.append(add_performance_context)
    
    # Add final renderer
    processors.append(renderer)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# =============================================================================
# OPENTELEMETRY INTEGRATION
# =============================================================================

def setup_tracing(settings: OasisBaseSettings):
    """Setup OpenTelemetry distributed tracing."""
    
    if not settings.TRACING_ENABLED:
        return
    
    # Create resource
    resource = Resource(attributes={
        SERVICE_NAME: settings.APP_NAME,
        SERVICE_VERSION: settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add Jaeger exporter if endpoint is configured
    if settings.JAEGER_ENDPOINT:
        jaeger_exporter = JaegerExporter(
            agent_host_name=settings.JAEGER_ENDPOINT.split('://')[1].split(':')[0],
            agent_port=int(settings.JAEGER_ENDPOINT.split(':')[-1]),
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context."""
    correlation_id_ctx.set(correlation_id)


def set_request_id(request_id: str):
    """Set request ID for current context."""
    request_id_ctx.set(request_id)


def set_user_context(user_id: str, session_id: Optional[str] = None):
    """Set user context for current request."""
    user_id_ctx.set(user_id)
    if session_id:
        session_id_ctx.set(session_id)


def clear_context():
    """Clear all context variables."""
    for ctx_var in [request_id_ctx, correlation_id_ctx, user_id_ctx, session_id_ctx, trace_id_ctx]:
        try:
            ctx_var.set(None)
        except LookupError:
            pass


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


# =============================================================================
# AUDIT LOGGING UTILITIES
# =============================================================================

def audit_log(message: str, **kwargs):
    """Log an audit message."""
    logger = get_logger("audit")
    logger.info(message, audit=True, **kwargs)


def trading_audit_log(action: str, symbol: str, **kwargs):
    """Log a trading-specific audit message."""
    logger = get_logger("trading.audit")
    logger.info(
        f"Trading action: {action}",
        audit=True,
        action=action,
        symbol=symbol,
        **kwargs
    )


def performance_log(operation: str, duration: float, success: bool = True, **kwargs):
    """Log performance metrics."""
    logger = get_logger("performance")
    logger.debug(
        f"Performance: {operation}",
        performance=True,
        operation=operation,
        duration=duration,
        success=success,
        **kwargs
    )


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_logging(settings: OasisBaseSettings):
    """
    Setup complete logging infrastructure.
    
    Args:
        settings: Application settings
    """
    
    # Create logs directory
    if settings.LOG_FILE:
        settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard logging
    config = OasisLoggingConfig(settings)
    logging_config = config.get_logging_config()
    logging.config.dictConfig(logging_config)
    
    # Configure structlog
    configure_structlog(settings)
    
    # Setup distributed tracing
    setup_tracing(settings)
    
    # Log successful setup
    logger = get_logger("oasis.logging")
    logger.info(
        "Logging system initialized",
        environment=settings.ENVIRONMENT.value,
        log_level=settings.LOG_LEVEL.value,
        log_format=settings.LOG_FORMAT,
        tracing_enabled=settings.TRACING_ENABLED,
    )


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

class LoggingContext:
    """Context manager for request-scoped logging context."""
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id or generate_request_id()
        self.correlation_id = correlation_id or generate_correlation_id()
        self.user_id = user_id
        self.session_id = session_id
        self.original_values = {}
    
    def __enter__(self):
        """Enter logging context."""
        # Store original values
        self.original_values = {
            'request_id': request_id_ctx.get(None),
            'correlation_id': correlation_id_ctx.get(None),
            'user_id': user_id_ctx.get(None),
            'session_id': session_id_ctx.get(None),
        }
        
        # Set new values
        request_id_ctx.set(self.request_id)
        correlation_id_ctx.set(self.correlation_id)
        if self.user_id:
            user_id_ctx.set(self.user_id)
        if self.session_id:
            session_id_ctx.set(self.session_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        # Restore original values
        for key, value in self.original_values.items():
            ctx_var = globals()[f"{key}_ctx"]
            ctx_var.set(value)


if __name__ == "__main__":
    """Logging system test."""
    from ..config.base import get_settings
    
    # Setup logging
    settings = get_settings()
    setup_logging(settings)
    
    # Test logging
    logger = get_logger("test")
    
    with LoggingContext() as ctx:
        logger.info("Test log message", test_data={"key": "value"})
        logger.warning("Test warning", extra_field="test")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Test exception logging")
    
    # Test audit logging
    audit_log("System startup", component="test")
    trading_audit_log("BUY", "BTC/USD", quantity=1.0, price=50000.0)
    performance_log("test_operation", 125.5, success=True)
    
    print("✅ Logging system test completed")