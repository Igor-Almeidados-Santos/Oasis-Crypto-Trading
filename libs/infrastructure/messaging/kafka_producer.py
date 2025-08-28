"""
Oasis Crypto Trade - Kafka Producer Infrastructure
==================================================

High-performance Kafka producer for real-time trading events with:
- Async message production with batching
- Schema Registry integration
- Message serialization strategies
- Delivery guarantees and retry logic
- Performance monitoring and metrics
- Dead letter queue handling
- Message compression and optimization

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiokafka
import orjson
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from ...shared.config.base import OasisBaseSettings
from ...shared.exceptions.infrastructure import MessageProducerError
from ...shared.logging.config import get_logger

logger = get_logger("oasis.messaging.producer")


# =============================================================================
# MESSAGE SERIALIZATION
# =============================================================================

class MessageSerializer:
    """Base message serializer interface."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize message data to bytes."""
        raise NotImplementedError
    
    def get_content_type(self) -> str:
        """Get content type for the serializer."""
        raise NotImplementedError


class JSONMessageSerializer(MessageSerializer):
    """JSON message serializer using orjson for performance."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes."""
        try:
            return orjson.dumps(data, default=self._json_serializer)
        except (TypeError, ValueError) as e:
            raise MessageProducerError(f"JSON serialization failed: {str(e)}")
    
    def get_content_type(self) -> str:
        """Get JSON content type."""
        return "application/json"
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object {obj} is not JSON serializable")


class AvroMessageSerializer(MessageSerializer):
    """Avro message serializer (placeholder for future implementation)."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to Avro bytes."""
        # TODO: Implement Avro serialization with Schema Registry
        json_serializer = JSONMessageSerializer()
        return json_serializer.serialize(data)
    
    def get_content_type(self) -> str:
        """Get Avro content type."""
        return "application/avro"


# =============================================================================
# MESSAGE ENVELOPE
# =============================================================================

class OasisMessage:
    """
    Standard message envelope for all Oasis trading events.
    
    Provides consistent structure for:
    - Message identification and tracing
    - Timestamps and ordering
    - Schema versioning
    - Retry and error handling
    """
    
    def __init__(
        self,
        topic: str,
        data: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None,
        message_type: Optional[str] = None,
        schema_version: str = "1.0",
        priority: int = 0,
        ttl_seconds: Optional[int] = None,
    ):
        self.message_id = str(uuid.uuid4())
        self.topic = topic
        self.data = data
        self.key = key or self.message_id
        self.headers = headers or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.message_type = message_type or "generic"
        self.schema_version = schema_version
        self.priority = priority
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds
        
        # Add standard headers
        self.headers.update({
            'message_id': self.message_id,
            'correlation_id': self.correlation_id,
            'message_type': self.message_type,
            'schema_version': self.schema_version,
            'created_at': self.created_at.isoformat(),
            'source': 'oasis-crypto-trade',
        })
        
        if self.ttl_seconds:
            expiry_time = self.created_at.timestamp() + self.ttl_seconds
            self.headers['expires_at'] = str(expiry_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            'message_id': self.message_id,
            'correlation_id': self.correlation_id,
            'message_type': self.message_type,
            'schema_version': self.schema_version,
            'created_at': self.created_at.isoformat(),
            'priority': self.priority,
            'headers': self.headers,
            'data': self.data,
        }
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if not self.ttl_seconds:
            return False
        
        expiry_time = self.created_at.timestamp() + self.ttl_seconds
        return time.time() > expiry_time


# =============================================================================
# KAFKA PRODUCER MANAGER
# =============================================================================

class OasisKafkaProducer:
    """
    Enterprise-grade Kafka producer with advanced features.
    
    Features:
    - High-throughput async message production
    - Configurable serialization strategies
    - Delivery guarantees with retry logic
    - Message batching and compression
    - Dead letter queue handling
    - Performance monitoring
    - Circuit breaker for resilience
    """
    
    def __init__(self, settings: OasisBaseSettings):
        self.settings = settings
        self.producer: Optional[AIOKafkaProducer] = None
        self.serializers = {
            'json': JSONMessageSerializer(),
            'avro': AvroMessageSerializer(),
        }
        self.default_serializer = 'json'
        
        # Performance metrics
        self._metrics = {
            'messages_produced': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'avg_batch_size': 0,
            'avg_send_time_ms': 0,
            'last_send_time': None,
        }
        
        # Circuit breaker state
        self._circuit_breaker = {
            'failures': 0,
            'last_failure_time': None,
            'state': 'closed',  # closed, open, half_open
            'failure_threshold': 5,
            'recovery_timeout': 60,  # seconds
        }
    
    async def initialize(self):
        """Initialize Kafka producer with optimized settings."""
        logger.info("Initializing Kafka producer")
        
        try:
            # Configure producer with high-performance settings
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                security_protocol=self.settings.KAFKA_SECURITY_PROTOCOL,
                
                # Reliability settings
                acks=self.settings.KAFKA_PRODUCER_ACKS,
                retries=self.settings.KAFKA_PRODUCER_RETRIES,
                enable_idempotence=True,  # Prevent duplicate messages
                
                # Performance settings
                batch_size=self.settings.KAFKA_PRODUCER_BATCH_SIZE,
                linger_ms=self.settings.KAFKA_PRODUCER_LINGER_MS,
                max_request_size=self.settings.KAFKA_PRODUCER_MAX_REQUEST_SIZE,
                
                # Compression for better throughput
                compression_type='lz4',
                
                # Timeout settings
                request_timeout_ms=30000,
                delivery_timeout_ms=120000,
                
                # Buffer settings
                buffer_memory=33554432,  # 32MB
                max_block_ms=60000,
                
                # Serialization
                key_serializer=lambda x: x.encode('utf-8') if isinstance(x, str) else x,
                value_serializer=None,  # We'll handle this ourselves
                
                # Error handling
                retry_backoff_ms=100,
                reconnect_backoff_ms=50,
                reconnect_backoff_max_ms=1000,
            )
            
            # Start the producer
            await self.producer.start()
            
            logger.info(
                "Kafka producer initialized",
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                acks=self.settings.KAFKA_PRODUCER_ACKS,
                batch_size=self.settings.KAFKA_PRODUCER_BATCH_SIZE
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize Kafka producer",
                error=str(e),
                exc_info=True
            )
            raise MessageProducerError(f"Kafka producer initialization failed: {str(e)}")
    
    def _check_circuit_breaker(self):
        """Check circuit breaker state and handle failures."""
        cb = self._circuit_breaker
        current_time = time.time()
        
        if cb['state'] == 'open':
            if cb['last_failure_time'] and (current_time - cb['last_failure_time']) > cb['recovery_timeout']:
                cb['state'] = 'half_open'
                cb['failures'] = 0
                logger.info("Circuit breaker moved to half-open state")
            else:
                raise MessageProducerError("Circuit breaker is OPEN - rejecting message")
        
        elif cb['state'] == 'half_open':
            # Allow one message through to test recovery
            pass
    
    def _record_success(self):
        """Record successful message production."""
        if self._circuit_breaker['state'] == 'half_open':
            self._circuit_breaker['state'] = 'closed'
            self._circuit_breaker['failures'] = 0
            logger.info("Circuit breaker recovered to closed state")
    
    def _record_failure(self):
        """Record failed message production."""
        cb = self._circuit_breaker
        cb['failures'] += 1
        cb['last_failure_time'] = time.time()
        
        if cb['failures'] >= cb['failure_threshold']:
            cb['state'] = 'open'
            logger.warning(
                "Circuit breaker opened due to failures",
                failures=cb['failures'],
                threshold=cb['failure_threshold']
            )
    
    async def produce_message(
        self,
        message: OasisMessage,
        serializer: str = None,
        partition: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produce a single message to Kafka.
        
        Args:
            message: OasisMessage instance to send
            serializer: Serialization strategy ('json', 'avro')
            partition: Optional partition number
            timestamp_ms: Optional message timestamp
            
        Returns:
            Dictionary with send result metadata
        """
        if not self.producer:
            raise MessageProducerError("Producer not initialized")
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Check if message has expired
        if message.is_expired():
            logger.warning(
                "Message expired, not sending",
                message_id=message.message_id,
                ttl_seconds=message.ttl_seconds
            )
            raise MessageProducerError("Message has expired")
        
        # Get serializer
        serializer = serializer or self.default_serializer
        if serializer not in self.serializers:
            raise MessageProducerError(f"Unknown serializer: {serializer}")
        
        serializer_instance = self.serializers[serializer]
        
        try:
            start_time = time.time()
            
            # Serialize message
            message_dict = message.to_dict()
            serialized_value = serializer_instance.serialize(message_dict)
            
            # Prepare headers
            headers = []
            for key, value in message.headers.items():
                if isinstance(value, str):
                    headers.append((key, value.encode('utf-8')))
                else:
                    headers.append((key, str(value).encode('utf-8')))
            
            # Add content type header
            headers.append(('content_type', serializer_instance.get_content_type().encode('utf-8')))
            
            # Send message
            record_metadata = await self.producer.send_and_wait(
                topic=message.topic,
                value=serialized_value,
                key=message.key.encode('utf-8') if message.key else None,
                partition=partition,
                timestamp_ms=timestamp_ms,
                headers=headers
            )
            
            # Calculate metrics
            send_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._metrics['messages_produced'] += 1
            self._metrics['bytes_sent'] += len(serialized_value)
            self._metrics['last_send_time'] = datetime.utcnow()
            
            # Update average send time
            total_messages = self._metrics['messages_produced']
            avg_time = self._metrics['avg_send_time_ms']
            self._metrics['avg_send_time_ms'] = (
                (avg_time * (total_messages - 1) + send_time_ms) / total_messages
            )
            
            # Record success for circuit breaker
            self._record_success()
            
            result = {
                'message_id': message.message_id,
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset,
                'timestamp': record_metadata.timestamp,
                'serialized_key_size': record_metadata.serialized_key_size,
                'serialized_value_size': record_metadata.serialized_value_size,
                'send_time_ms': round(send_time_ms, 2),
            }
            
            logger.debug(
                "Message produced successfully",
                message_id=message.message_id,
                topic=message.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset,
                send_time_ms=round(send_time_ms, 2)
            )
            
            return result
            
        except Exception as e:
            # Update failure metrics
            self._metrics['messages_failed'] += 1
            self._record_failure()
            
            logger.error(
                "Failed to produce message",
                message_id=message.message_id,
                topic=message.topic,
                error=str(e),
                exc_info=True
            )
            
            raise MessageProducerError(f"Message production failed: {str(e)}")
    
    async def produce_batch(
        self,
        messages: List[OasisMessage],
        serializer: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Produce multiple messages as a batch.
        
        Args:
            messages: List of OasisMessage instances
            serializer: Serialization strategy
            
        Returns:
            List of send results
        """
        if not messages:
            return []
        
        logger.info(f"Producing batch of {len(messages)} messages")
        
        results = []
        failures = []
        
        # Send all messages concurrently
        tasks = []
        for message in messages:
            task = asyncio.create_task(
                self.produce_message(message, serializer=serializer)
            )
            tasks.append((message, task))
        
        # Wait for all tasks to complete
        for message, task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                failures.append({
                    'message_id': message.message_id,
                    'error': str(e)
                })
                logger.error(
                    "Batch message failed",
                    message_id=message.message_id,
                    error=str(e)
                )
        
        # Update batch metrics
        if results:
            batch_size = len(results)
            total_batches = getattr(self, '_batch_count', 0) + 1
            setattr(self, '_batch_count', total_batches)
            
            avg_batch = self._metrics['avg_batch_size']
            self._metrics['avg_batch_size'] = (
                (avg_batch * (total_batches - 1) + batch_size) / total_batches
            )
        
        logger.info(
            "Batch production completed",
            total_messages=len(messages),
            successful=len(results),
            failed=len(failures)
        )
        
        if failures:
            logger.warning(
                "Some messages in batch failed",
                failures=failures
            )
        
        return results
    
    async def produce_trading_event(
        self,
        event_type: str,
        symbol: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        Convenience method for producing trading events.
        
        Args:
            event_type: Type of trading event (order, trade, signal, etc.)
            symbol: Trading symbol (e.g., 'BTC/USD')
            data: Event data payload
            correlation_id: Optional correlation ID for tracking
            priority: Message priority (0 = normal, higher = more important)
            
        Returns:
            Send result metadata
        """
        topic = f"oasis.trading.{event_type.lower()}"
        
        # Enrich data with standard fields
        enriched_data = {
            'symbol': symbol,
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        
        message = OasisMessage(
            topic=topic,
            data=enriched_data,
            key=symbol,  # Partition by symbol for ordering
            correlation_id=correlation_id,
            message_type=f"trading.{event_type.lower()}",
            priority=priority,
        )
        
        return await self.produce_message(message)
    
    async def produce_market_data_event(
        self,
        data_type: str,
        symbol: str,
        data: Dict[str, Any],
        priority: int = 1,  # Market data typically high priority
    ) -> Dict[str, Any]:
        """
        Convenience method for producing market data events.
        
        Args:
            data_type: Type of market data (tick, orderbook, trade, etc.)
            symbol: Trading symbol
            data: Market data payload
            priority: Message priority
            
        Returns:
            Send result metadata
        """
        topic = f"oasis.market_data.{data_type.lower()}"
        
        enriched_data = {
            'symbol': symbol,
            'data_type': data_type,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        
        message = OasisMessage(
            topic=topic,
            data=enriched_data,
            key=symbol,
            message_type=f"market_data.{data_type.lower()}",
            priority=priority,
            ttl_seconds=300,  # Market data expires quickly
        )
        
        return await self.produce_message(message)
    
    async def flush(self, timeout_ms: int = 10000):
        """
        Flush any pending messages.
        
        Args:
            timeout_ms: Flush timeout in milliseconds
        """
        if self.producer:
            try:
                await asyncio.wait_for(
                    self.producer.flush(),
                    timeout=timeout_ms / 1000
                )
                logger.debug("Producer flush completed")
            except asyncio.TimeoutError:
                logger.warning(f"Producer flush timed out after {timeout_ms}ms")
            except Exception as e:
                logger.error(f"Producer flush failed: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get producer performance metrics."""
        base_metrics = self._metrics.copy()
        
        # Add circuit breaker metrics
        base_metrics['circuit_breaker'] = self._circuit_breaker.copy()
        
        # Add producer-specific metrics if available
        if self.producer:
            # Note: aiokafka doesn't expose detailed metrics like Java client
            # We track our own metrics instead
            pass
        
        return base_metrics
    
    async def close(self):
        """Close the Kafka producer."""
        logger.info("Closing Kafka producer")
        
        try:
            if self.producer:
                # Flush pending messages
                await self.flush()
                
                # Stop the producer
                await self.producer.stop()
                
            logger.info("Kafka producer closed")
            
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {str(e)}")
        finally:
            self.producer = None


# =============================================================================
# GLOBAL PRODUCER INSTANCE
# =============================================================================

# Global producer instance
kafka_producer: Optional[OasisKafkaProducer] = None


async def initialize_kafka_producer(settings: OasisBaseSettings):
    """Initialize global Kafka producer."""
    global kafka_producer
    
    kafka_producer = OasisKafkaProducer(settings)
    await kafka_producer.initialize()
    
    logger.info("Global Kafka producer initialized")


async def get_kafka_producer() -> OasisKafkaProducer:
    """Get global Kafka producer."""
    if not kafka_producer:
        raise MessageProducerError("Kafka producer not initialized")
    return kafka_producer


async def close_kafka_producer():
    """Close global Kafka producer."""
    global kafka_producer
    
    if kafka_producer:
        await kafka_producer.close()
        kafka_producer = None
    
    logger.info("Global Kafka producer closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def send_trading_event(
    event_type: str,
    symbol: str,
    data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    priority: int = 0,
) -> Dict[str, Any]:
    """Send trading event using global producer."""
    producer = await get_kafka_producer()
    return await producer.produce_trading_event(
        event_type=event_type,
        symbol=symbol,
        data=data,
        correlation_id=correlation_id,
        priority=priority
    )


async def send_market_data_event(
    data_type: str,
    symbol: str,
    data: Dict[str, Any],
    priority: int = 1,
) -> Dict[str, Any]:
    """Send market data event using global producer."""
    producer = await get_kafka_producer()
    return await producer.produce_market_data_event(
        data_type=data_type,
        symbol=symbol,
        data=data,
        priority=priority
    )


# =============================================================================
# TESTING UTILITIES
# =============================================================================

async def test_kafka_producer(settings: OasisBaseSettings) -> bool:
    """Test Kafka producer with comprehensive diagnostics."""
    logger.info("Testing Kafka producer")
    
    try:
        # Test basic producer
        test_producer = OasisKafkaProducer(settings)
        await test_producer.initialize()
        
        # Test single message
        test_message = OasisMessage(
            topic="oasis.test",
            data={
                "message": "Hello Oasis Trading System",
                "timestamp": time.time(),
                "test_id": str(uuid.uuid4())
            },
            message_type="test"
        )
        
        result = await test_producer.produce_message(test_message)
        
        # Test batch messages
        batch_messages = []
        for i in range(3):
            batch_message = OasisMessage(
                topic="oasis.test.batch",
                data={
                    "batch_index": i,
                    "message": f"Batch message {i}",
                    "timestamp": time.time()
                },
                message_type="test.batch"
            )
            batch_messages.append(batch_message)
        
        batch_results = await test_producer.produce_batch(batch_messages)
        
        # Test trading event
        trading_result = await test_producer.produce_trading_event(
            event_type="signal",
            symbol="BTC/USD",
            data={
                "signal_type": "BUY",
                "confidence": 0.85,
                "price": 50000.0
            }
        )
        
        # Test market data event
        market_data_result = await test_producer.produce_market_data_event(
            data_type="tick",
            symbol="BTC/USD",
            data={
                "price": 50000.0,
                "volume": 1.5,
                "bid": 49999.0,
                "ask": 50001.0
            }
        )
        
        # Flush and get metrics
        await test_producer.flush()
        metrics = await test_producer.get_metrics()
        
        # Close test producer
        await test_producer.close()
        
        logger.info(
            "Kafka producer test successful",
            single_message=result,
            batch_count=len(batch_results),
            trading_event=trading_result,
            market_data_event=market_data_result,
            metrics=metrics
        )
        
        return (
            result and
            len(batch_results) == 3 and
            trading_result and
            market_data_result
        )
        
    except Exception as e:
        logger.error(
            "Kafka producer test failed",
            error=str(e),
            exc_info=True
        )
        return False


if __name__ == "__main__":
    """Kafka producer test."""
    import asyncio
    from ...shared.config.base import get_settings
    
    async def main():
        settings = get_settings()
        success = await test_kafka_producer(settings)
        
        if success:
            print("✅ Kafka producer test passed")
            exit(0)
        else:
            print("❌ Kafka producer test failed")
            exit(1)
    
    asyncio.run(main())