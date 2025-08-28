"""
Oasis Crypto Trade - Exception System
=====================================

Comprehensive exception hierarchy with:
- Structured error handling
- Error codes and categories
- Context preservation
- Logging integration
- Error recovery strategies
- User-friendly error messages
- Security-aware error disclosure

Author: Oasis Trading Systems
License: Proprietary
"""

import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..logging.config import get_logger

logger = get_logger("oasis.exceptions")


# =============================================================================
# ERROR CATEGORIES AND CODES
# =============================================================================

class ErrorCategory(str, Enum):
    """High-level error categories."""
    
    # System and Infrastructure
    SYSTEM = "system"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    
    # Trading and Market
    TRADING = "trading"
    MARKET_DATA = "market_data"
    RISK_MANAGEMENT = "risk_management"
    
    # Data and Validation
    VALIDATION = "validation"
    DATA_ACCESS = "data_access"
    SERIALIZATION = "serialization"
    
    # Authentication and Authorization
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    
    # External Services
    EXCHANGE_API = "exchange_api"
    EXTERNAL_SERVICE = "external_service"
    
    # Business Logic
    BUSINESS_RULE = "business_rule"
    STRATEGY = "strategy"
    
    # Client and User Interface
    CLIENT = "client"
    USER_INPUT = "user_input"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"           # Minor issues, system can continue
    MEDIUM = "medium"     # Notable issues, some functionality affected
    HIGH = "high"         # Serious issues, major functionality affected
    CRITICAL = "critical" # System-threatening issues


class ErrorCode(str, Enum):
    """Specific error codes for different scenarios."""
    
    # System Errors (SYS-xxx)
    SYSTEM_ERROR = "SYS-001"
    MEMORY_ERROR = "SYS-002"
    TIMEOUT_ERROR = "SYS-003"
    RESOURCE_EXHAUSTED = "SYS-004"
    
    # Infrastructure Errors (INF-xxx)
    DATABASE_CONNECTION = "INF-001"
    CACHE_CONNECTION = "INF-002"
    MESSAGE_QUEUE_ERROR = "INF-003"
    NETWORK_ERROR = "INF-004"
    
    # Configuration Errors (CFG-xxx)
    INVALID_CONFIGURATION = "CFG-001"
    MISSING_CONFIGURATION = "CFG-002"
    CONFIGURATION_CONFLICT = "CFG-003"
    
    # Trading Errors (TRD-xxx)
    INVALID_ORDER = "TRD-001"
    INSUFFICIENT_BALANCE = "TRD-002"
    MARKET_CLOSED = "TRD-003"
    POSITION_NOT_FOUND = "TRD-004"
    EXECUTION_FAILED = "TRD-005"
    ORDER_REJECTED = "TRD-006"
    
    # Market Data Errors (MKT-xxx)
    DATA_UNAVAILABLE = "MKT-001"
    STALE_DATA = "MKT-002"
    DATA_CORRUPTION = "MKT-003"
    FEED_DISCONNECTED = "MKT-004"
    
    # Risk Management Errors (RSK-xxx)
    RISK_LIMIT_EXCEEDED = "RSK-001"
    INVALID_POSITION_SIZE = "RSK-002"
    DRAWDOWN_LIMIT = "RSK-003"
    EXPOSURE_LIMIT = "RSK-004"
    
    # Validation Errors (VAL-xxx)
    INVALID_INPUT = "VAL-001"
    MISSING_REQUIRED_FIELD = "VAL-002"
    INVALID_FORMAT = "VAL-003"
    OUT_OF_RANGE = "VAL-004"
    
    # Authentication Errors (AUT-xxx)
    INVALID_CREDENTIALS = "AUT-001"
    TOKEN_EXPIRED = "AUT-002"
    INSUFFICIENT_PERMISSIONS = "AUT-003"
    
    # Exchange API Errors (EXC-xxx)
    API_KEY_INVALID = "EXC-001"
    RATE_LIMIT_EXCEEDED = "EXC-002"
    API_ERROR = "EXC-003"
    MAINTENANCE_MODE = "EXC-004"


# =============================================================================
# BASE EXCEPTION CLASSES
# =============================================================================

class OasisException(Exception):
    """
    Base exception class for all Oasis trading system errors.
    
    Provides structured error information with:
    - Error categorization and codes
    - Context preservation
    - User-friendly messages
    - Developer debugging information
    - Integration with logging system
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[ErrorCode] = None,
        category: Optional[ErrorCategory] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None,
        safe_for_client: bool = False,
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or ErrorCode.SYSTEM_ERROR
        self.category = category or ErrorCategory.SYSTEM
        self.severity = severity
        self.context = context or {}
        self.user_message = user_message or self._generate_user_message()
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        self.safe_for_client = safe_for_client
        
        # Metadata
        self.timestamp = datetime.utcnow()
        self.exception_id = self._generate_exception_id()
        self.stack_trace = self._capture_stack_trace()
        
        # Log the exception
        self._log_exception()
    
    def _generate_exception_id(self) -> str:
        """Generate unique exception identifier."""
        import uuid
        return f"OAS-{str(uuid.uuid4())[:8].upper()}"
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message."""
        # Default user-friendly messages based on category
        user_messages = {
            ErrorCategory.TRADING: "An error occurred while processing your trading request.",
            ErrorCategory.MARKET_DATA: "Market data is currently unavailable.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.VALIDATION: "The provided information is invalid.",
            ErrorCategory.INFRASTRUCTURE: "A system service is currently unavailable.",
            ErrorCategory.CONFIGURATION: "System configuration error.",
        }
        
        return user_messages.get(
            self.category,
            "An unexpected error occurred. Please try again."
        )
    
    def _capture_stack_trace(self) -> List[str]:
        """Capture formatted stack trace."""
        return traceback.format_tb(self.__traceback__)
    
    def _log_exception(self):
        """Log the exception with appropriate level based on severity."""
        log_data = {
            'exception_id': self.exception_id,
            'error_code': self.error_code.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'user_message': self.user_message,
        }
        
        if self.original_exception:
            log_data['original_exception'] = str(self.original_exception)
        
        # Log based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(self.message, **log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(self.message, **log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(self.message, **log_data)
        else:
            logger.info(self.message, **log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            'exception_id': self.exception_id,
            'timestamp': self.timestamp.isoformat(),
            'error_code': self.error_code.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'user_message': self.user_message,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'safe_for_client': self.safe_for_client,
        }
    
    def add_context(self, **kwargs) -> 'OasisException':
        """Add context information to the exception."""
        self.context.update(kwargs)
        return self
    
    def add_recovery_suggestion(self, suggestion: str) -> 'OasisException':
        """Add recovery suggestion."""
        self.recovery_suggestions.append(suggestion)
        return self
    
    def __str__(self) -> str:
        """String representation with exception ID."""
        return f"[{self.exception_id}] {self.message}"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code.value}', "
            f"category='{self.category.value}', "
            f"severity='{self.severity.value}', "
            f"exception_id='{self.exception_id}'"
            f")"
        )


# =============================================================================
# SYSTEM AND INFRASTRUCTURE EXCEPTIONS
# =============================================================================

class SystemException(OasisException):
    """Base exception for system-level errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class InfrastructureException(OasisException):
    """Base exception for infrastructure-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.INFRASTRUCTURE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ConfigurationException(OasisException):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('error_code', ErrorCode.INVALID_CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


# =============================================================================
# TRADING EXCEPTIONS
# =============================================================================

class TradingException(OasisException):
    """Base exception for trading-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.TRADING)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class InvalidOrderException(TradingException):
    """Exception for invalid trading orders."""
    
    def __init__(self, message: str, order_details: Optional[Dict] = None, **kwargs):
        kwargs.setdefault('error_code', ErrorCode.INVALID_ORDER)
        if order_details:
            kwargs.setdefault('context', {}).update({'order_details': order_details})
        super().__init__(message, **kwargs)


class InsufficientBalanceException(TradingException):
    """Exception for insufficient account balance."""
    
    def __init__(
        self,
        required_amount: float,
        available_balance: float,
        currency: str = "USD",
        **kwargs
    ):
        message = f"Insufficient balance: required {required_amount} {currency}, available {available_balance} {currency}"
        kwargs.setdefault('error_code', ErrorCode.INSUFFICIENT_BALANCE)
        kwargs.setdefault('context', {}).update({
            'required_amount': required_amount,
            'available_balance': available_balance,
            'currency': currency,
            'shortfall': required_amount - available_balance
        })
        super().__init__(message, **kwargs)


class OrderExecutionException(TradingException):
    """Exception for order execution failures."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', ErrorCode.EXECUTION_FAILED)
        if order_id:
            kwargs.setdefault('context', {}).update({'order_id': order_id})
        super().__init__(message, **kwargs)


class PositionNotFoundException(TradingException):
    """Exception when requested position is not found."""
    
    def __init__(self, position_id: str, **kwargs):
        message = f"Position not found: {position_id}"
        kwargs.setdefault('error_code', ErrorCode.POSITION_NOT_FOUND)
        kwargs.setdefault('context', {}).update({'position_id': position_id})
        super().__init__(message, **kwargs)


# =============================================================================
# MARKET DATA EXCEPTIONS
# =============================================================================

class MarketDataException(OasisException):
    """Base exception for market data errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MARKET_DATA)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class DataUnavailableException(MarketDataException):
    """Exception when market data is unavailable."""
    
    def __init__(self, symbol: str, data_type: str = "unknown", **kwargs):
        message = f"Market data unavailable for {symbol} ({data_type})"
        kwargs.setdefault('error_code', ErrorCode.DATA_UNAVAILABLE)
        kwargs.setdefault('context', {}).update({
            'symbol': symbol,
            'data_type': data_type
        })
        super().__init__(message, **kwargs)


class StaleDataException(MarketDataException):
    """Exception when market data is stale."""
    
    def __init__(
        self,
        symbol: str,
        age_seconds: float,
        max_age_seconds: float = 60,
        **kwargs
    ):
        message = f"Stale data for {symbol}: {age_seconds}s old (max: {max_age_seconds}s)"
        kwargs.setdefault('error_code', ErrorCode.STALE_DATA)
        kwargs.setdefault('context', {}).update({
            'symbol': symbol,
            'age_seconds': age_seconds,
            'max_age_seconds': max_age_seconds
        })
        super().__init__(message, **kwargs)


class FeedDisconnectedException(MarketDataException):
    """Exception when market data feed is disconnected."""
    
    def __init__(self, feed_name: str, **kwargs):
        message = f"Market data feed disconnected: {feed_name}"
        kwargs.setdefault('error_code', ErrorCode.FEED_DISCONNECTED)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('context', {}).update({'feed_name': feed_name})
        super().__init__(message, **kwargs)


# =============================================================================
# RISK MANAGEMENT EXCEPTIONS
# =============================================================================

class RiskManagementException(OasisException):
    """Base exception for risk management errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RISK_MANAGEMENT)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class RiskLimitExceededException(RiskManagementException):
    """Exception when risk limits are exceeded."""
    
    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        unit: str = "",
        **kwargs
    ):
        message = f"{limit_type} limit exceeded: {current_value}{unit} > {limit_value}{unit}"
        kwargs.setdefault('error_code', ErrorCode.RISK_LIMIT_EXCEEDED)
        kwargs.setdefault('context', {}).update({
            'limit_type': limit_type,
            'current_value': current_value,
            'limit_value': limit_value,
            'unit': unit,
            'excess_amount': current_value - limit_value
        })
        super().__init__(message, **kwargs)


class InvalidPositionSizeException(RiskManagementException):
    """Exception for invalid position sizes."""
    
    def __init__(
        self,
        requested_size: float,
        max_allowed_size: float,
        symbol: str,
        **kwargs
    ):
        message = f"Invalid position size for {symbol}: {requested_size} (max: {max_allowed_size})"
        kwargs.setdefault('error_code', ErrorCode.INVALID_POSITION_SIZE)
        kwargs.setdefault('context', {}).update({
            'symbol': symbol,
            'requested_size': requested_size,
            'max_allowed_size': max_allowed_size
        })
        super().__init__(message, **kwargs)


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================

class ValidationException(OasisException):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('safe_for_client', True)
        super().__init__(message, **kwargs)


class InvalidInputException(ValidationException):
    """Exception for invalid input data."""
    
    def __init__(
        self,
        field_name: str,
        provided_value: Any = None,
        expected_type: str = None,
        **kwargs
    ):
        if expected_type:
            message = f"Invalid input for '{field_name}': expected {expected_type}"
        else:
            message = f"Invalid input for '{field_name}'"
            
        kwargs.setdefault('error_code', ErrorCode.INVALID_INPUT)
        kwargs.setdefault('context', {}).update({
            'field_name': field_name,
            'provided_value': str(provided_value) if provided_value is not None else None,
            'expected_type': expected_type
        })
        super().__init__(message, **kwargs)


class MissingRequiredFieldException(ValidationException):
    """Exception for missing required fields."""
    
    def __init__(self, field_names: Union[str, List[str]], **kwargs):
        if isinstance(field_names, list):
            fields = "', '".join(field_names)
            message = f"Missing required fields: '{fields}'"
        else:
            message = f"Missing required field: '{field_names}'"
            field_names = [field_names]
            
        kwargs.setdefault('error_code', ErrorCode.MISSING_REQUIRED_FIELD)
        kwargs.setdefault('context', {}).update({'missing_fields': field_names})
        super().__init__(message, **kwargs)


# =============================================================================
# AUTHENTICATION AND AUTHORIZATION EXCEPTIONS
# =============================================================================

class AuthenticationException(OasisException):
    """Base exception for authentication errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('error_code', ErrorCode.INVALID_CREDENTIALS)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('safe_for_client', True)
        super().__init__(message, **kwargs)


class AuthorizationException(OasisException):
    """Exception for authorization/permission errors."""
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHORIZATION)
        kwargs.setdefault('error_code', ErrorCode.INSUFFICIENT_PERMISSIONS)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('safe_for_client', True)
        super().__init__(message, **kwargs)


class TokenExpiredException(AuthenticationException):
    """Exception for expired authentication tokens."""
    
    def __init__(self, token_type: str = "access", **kwargs):
        message = f"Authentication token expired: {token_type}"
        kwargs.setdefault('error_code', ErrorCode.TOKEN_EXPIRED)
        kwargs.setdefault('context', {}).update({'token_type': token_type})
        super().__init__(message, **kwargs)


# =============================================================================
# EXCHANGE API EXCEPTIONS
# =============================================================================

class ExchangeAPIException(OasisException):
    """Base exception for exchange API errors."""
    
    def __init__(self, message: str, exchange: str = "unknown", **kwargs):
        kwargs.setdefault('category', ErrorCategory.EXCHANGE_API)
        kwargs.setdefault('error_code', ErrorCode.API_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('context', {}).update({'exchange': exchange})
        super().__init__(message, **kwargs)


class RateLimitExceededException(ExchangeAPIException):
    """Exception when API rate limits are exceeded."""
    
    def __init__(
        self,
        exchange: str,
        retry_after_seconds: Optional[int] = None,
        **kwargs
    ):
        message = f"Rate limit exceeded for {exchange} API"
        if retry_after_seconds:
            message += f" (retry after {retry_after_seconds}s)"
            
        kwargs.setdefault('error_code', ErrorCode.RATE_LIMIT_EXCEEDED)
        kwargs.setdefault('context', {}).update({'retry_after_seconds': retry_after_seconds})
        super().__init__(message, exchange=exchange, **kwargs)


class APIKeyInvalidException(ExchangeAPIException):
    """Exception for invalid API keys."""
    
    def __init__(self, exchange: str, **kwargs):
        message = f"Invalid API key for {exchange}"
        kwargs.setdefault('error_code', ErrorCode.API_KEY_INVALID)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, exchange=exchange, **kwargs)


# =============================================================================
# EXCEPTION UTILITIES
# =============================================================================

def wrap_external_exception(
    exc: Exception,
    message: str = None,
    category: ErrorCategory = ErrorCategory.EXTERNAL_SERVICE,
    error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
    **kwargs
) -> OasisException:
    """
    Wrap external exceptions in Oasis exception format.
    
    Args:
        exc: Original exception
        message: Custom message (uses original if None)
        category: Error category
        error_code: Error code
        **kwargs: Additional context
        
    Returns:
        Wrapped OasisException
    """
    if isinstance(exc, OasisException):
        return exc
    
    wrapped_message = message or f"External error: {str(exc)}"
    
    return OasisException(
        message=wrapped_message,
        error_code=error_code,
        category=category,
        original_exception=exc,
        **kwargs
    )


def format_exception_for_api(exc: Exception) -> Dict[str, Any]:
    """
    Format exception for API response.
    
    Args:
        exc: Exception to format
        
    Returns:
        Dictionary suitable for API error response
    """
    if isinstance(exc, OasisException):
        response = {
            'error': True,
            'error_code': exc.error_code.value,
            'error_category': exc.category.value,
            'message': exc.user_message if exc.safe_for_client else "An error occurred",
            'exception_id': exc.exception_id,
        }
        
        if exc.safe_for_client and exc.context:
            response['details'] = exc.context
            
        if exc.recovery_suggestions:
            response['suggestions'] = exc.recovery_suggestions
            
        return response
    else:
        # Generic exception
        return {
            'error': True,
            'error_code': ErrorCode.SYSTEM_ERROR.value,
            'error_category': ErrorCategory.SYSTEM.value,
            'message': "An unexpected error occurred",
        }


def get_exception_summary(exc: Exception) -> Dict[str, Any]:
    """
    Get exception summary for monitoring and alerting.
    
    Args:
        exc: Exception to summarize
        
    Returns:
        Exception summary dictionary
    """
    if isinstance(exc, OasisException):
        return {
            'exception_id': exc.exception_id,
            'timestamp': exc.timestamp.isoformat(),
            'error_code': exc.error_code.value,
            'category': exc.category.value,
            'severity': exc.severity.value,
            'message': exc.message,
            'context_keys': list(exc.context.keys()) if exc.context else [],
        }
    else:
        return {
            'exception_type': exc.__class__.__name__,
            'message': str(exc),
            'timestamp': datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    """Exception system test."""
    
    # Test basic exception
    try:
        raise OasisException(
            "Test system error",
            error_code=ErrorCode.SYSTEM_ERROR,
            category=ErrorCategory.SYSTEM,
            context={'test_param': 'test_value'}
        )
    except OasisException as e:
        print(f"✅ Basic exception: {e}")
        print(f"   Exception ID: {e.exception_id}")
        print(f"   API format: {format_exception_for_api(e)}")
    
    # Test trading exception
    try:
        raise InsufficientBalanceException(
            required_amount=1000.0,
            available_balance=500.0,
            currency="USD"
        )
    except TradingException as e:
        print(f"✅ Trading exception: {e}")
        print(f"   Context: {e.context}")
    
    # Test validation exception
    try:
        raise MissingRequiredFieldException(['symbol', 'quantity'])
    except ValidationException as e:
        print(f"✅ Validation exception: {e}")
        print(f"   Safe for client: {e.safe_for_client}")
    
    # Test exception wrapping
    try:
        raise ValueError("Original error")
    except ValueError as orig_exc:
        wrapped = wrap_external_exception(
            orig_exc,
            message="Wrapped external error",
            category=ErrorCategory.EXTERNAL_SERVICE
        )
        print(f"✅ Wrapped exception: {wrapped}")
        print(f"   Original: {wrapped.original_exception}")
    
    print("\n✅ Exception system test completed")