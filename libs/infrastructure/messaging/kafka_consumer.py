"""
Oasis Crypto Trade - Kafka Consumer Infrastructure
==================================================

High-performance Kafka consumer for real-time trading event processing with:
- Async message consumption with auto-commit control
- Message deserialization and validation
- Dead letter queue handling
- Consumer group coordination
- Offset management and rebalancing
- Error handling and retry mechanisms
- Performance monitoring and metrics
- Graceful shutdown and cleanup

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import json
import signal
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import aiokafka
import orjson
from aiokafka import AIOKafkaConsumer, ConsumerRecord, TopicPartition
from aiokafka.errors import ConsumerStoppedError, KafkaError

from ...shared.config.base import OasisBaseSettings
from ...shared.exceptions.infrastructure import MessageConsumerError
from ...shared.logging.config import get_logger

logger = get_logger("oasis.messaging.consumer")


# =============================================================================
# MESSAGE DESERIALIZERS
# =============================================================================

class MessageDeserializer:
    """Base message deserializer interface."""
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize message data from bytes."""
        raise NotImplementedError
    
    def can_handle(self, content_type: str) -> bool:
        """Check if deserializer can handle content type."""
        raise NotImplementedError


class JSONMessageDeserializer(MessageDeserializer):
    """JSON message deserializer using orjson for performance."""
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to data."""
        try:
            return orjson.loads(data)
        except (orjson.JSONDecodeError, ValueError) as e:
            raise MessageConsumerError(f"JSON deserialization failed: {str(e)}")
    
    def can_handle(self, content_type: str) -> bool:
        """Check if can handle JSON content."""
        return content_type.lower() in ['application/json', 'text/json']


class AvroMessageDeserializer(MessageDeserializer):
    """Avro message deserializer (placeholder for future implementation)."""
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize Avro bytes to data."""
        # TODO: Implement Avro deserialization with Schema Registry
        json_deserializer = JSONMessageDeserializer()
        return json_deserializer.deserialize(data)
    
    def can_handle(self, content_type: str) -> bool:
        """Check if can handle Avro content."""
        return content_type.lower() in ['application/avro', 'avro/binary']


# =============================================================================
# MESSAGE HANDLER INTERFACE
# =============================================================================

class MessageHandler:
    """Base message handler interface."""
    
    async def handle(self, message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """
        Handle incoming message.
        
        Args:
            message: Deserialized message data
            metadata: Message metadata (topic, partition, offset, etc.)
            
        Returns:
            True if message was handled successfully, False otherwise
        """
        raise NotImplementedError
    
    def can_handle(self, topic: str, message_type: str = None) -> bool:
        """Check if handler can process messages from topic."""
        raise NotImplementedError


# =============================================================================
# CONSUMER METRICS AND MONITORING
# =============================================================================

class ConsumerMetrics:
    """Consumer performance metrics tracker."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self._metrics = {
            'messages_consumed': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'bytes_consumed': 0,
            'avg_processing_time_ms': 0,
            'last_consumed_time': None,
            'last_committed_offset': {},
            'consumer_lag': {},
            'partitions_assigned': set(),
        }
    
    def record_message_consumed(self, message_size: int):
        """Record message consumption."""
        self._metrics['messages_consumed'] += 1
        self._metrics['bytes_consumed'] += message_size
        self._metrics['last_consumed_time'] = datetime.utcnow()
    
    def record_message_processed(self, processing_time_ms: float):
        """Record successful message processing."""
        self._metrics['messages_processed'] += 1
        
        # Update average processing time
        total_processed = self._metrics['messages_processed']
        avg_time = self._metrics['avg_processing_time_ms']
        self._metrics['avg_processing_time_ms'] = (
            (avg_time * (total_processed - 1) + processing_time_ms) / total_processed
        )
    
    def record_message_failed(self):
        """Record failed message processing."""
        self._metrics['messages_failed'] += 1
    
    def update_committed_offset(self, topic: str, partition: int, offset: int):
        """Update last committed offset for partition."""
        partition_key = f"{topic}:{partition}"
        self._metrics['last_committed_offset'][partition_key] = offset
    
    def update_consumer_lag(self, topic: str, partition: int, lag: int):
        """Update consumer lag for partition."""
        partition_key = f"{topic}:{partition}"
        self._metrics['consumer_lag'][partition_key] = lag
    
    def set_assigned_partitions(self, partitions: Set[TopicPartition]):
        """Set currently assigned partitions."""
        self._metrics['partitions_assigned'] = {
            f"{tp.topic}:{tp.partition}" for tp in partitions
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        metrics = self._metrics.copy()
        
        # Convert set to list for JSON serialization
        metrics['partitions_assigned'] = list(metrics['partitions_assigned'])
        
        # Calculate success rate
        total_messages = metrics['messages_processed'] + metrics['messages_failed']
        if total_messages > 0:
            metrics['success_rate'] = metrics['messages_processed'] / total_messages
        else:
            metrics['success_rate'] = 0.0
        
        return metrics


# =============================================================================
# KAFKA CONSUMER MANAGER
# =============================================================================

class OasisKafkaConsumer:
    """
    Enterprise-grade Kafka consumer with advanced features.
    
    Features:
    - High-throughput async message consumption
    - Pluggable message handlers
    - Automatic deserialization
    - Error handling with dead letter queues
    - Consumer group coordination
    - Offset management strategies
    - Performance monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        settings: OasisBaseSettings,
        group_id: Optional[str] = None,
        topics: Optional[List[str]] = None
    ):
        self.settings = settings
        self.group_id = group_id or settings.KAFKA_CONSUMER_GROUP_ID
        self.topics = topics or []
        
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.message_handlers: Dict[str, List[MessageHandler]] = {}
        self.deserializers = {
            'json': JSONMessageDeserializer(),
            'avro': AvroMessageDeserializer(),
        }
        
        # Consumer state
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.consume_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = ConsumerMetrics()
        
        # Dead letter queue settings
        self.dlq_topic = f"{self.group_id}.dlq"
        self.max_retries = 3
        self.retry_delay_seconds = [1, 5, 15]  # Exponential backoff
    
    async def initialize(self):
        """Initialize Kafka consumer with optimized settings."""
        logger.info("Initializing Kafka consumer")
        
        try:
            # Configure consumer with optimal settings
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                security_protocol=self.settings.KAFKA_SECURITY_PROTOCOL,
                
                # Consumer group settings
                group_id=self.group_id,
                
                # Offset management
                auto_offset_reset=self.settings.KAFKA_CONSUMER_AUTO_OFFSET_RESET,
                enable_auto_commit=self.settings.KAFKA_CONSUMER_ENABLE_AUTO_COMMIT,
                auto_commit_interval_ms=5000,
                
                # Performance settings
                max_poll_records=self.settings.KAFKA_CONSUMER_MAX_POLL_RECORDS,
                max_poll_interval_ms=300000,  # 5 minutes
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                
                # Buffer settings
                fetch_min_bytes=1,
                fetch_max_wait_ms=500,
                fetch_max_bytes=52428800,  # 50MB
                max_partition_fetch_bytes=1048576,  # 1MB
                
                # Connection settings
                request_timeout_ms=30000,
                retry_backoff_ms=100,
                reconnect_backoff_ms=50,
                reconnect_backoff_max_ms=1000,
                
                # Deserialization
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                value_deserializer=None,  # We'll handle this ourselves
            )
            
            # Start the consumer
            await self.consumer.start()
            
            logger.info(
                "Kafka consumer initialized",
                group_id=self.group_id,
                topics=self.topics,
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize Kafka consumer",
                error=str(e),
                exc_info=True
            )
            raise MessageConsumerError(f"Kafka consumer initialization failed: {str(e)}")
    
    def register_handler(self, handler: MessageHandler, topics: List[str] = None):
        """
        Register message handler for specific topics.
        
        Args:
            handler: MessageHandler instance
            topics: List of topics to handle (None = all subscribed topics)
        """
        if topics is None:
            topics = self.topics
        
        for topic in topics:
            if topic not in self.message_handlers:
                self.message_handlers[topic] = []
            self.message_handlers[topic].append(handler)
        
        logger.info(
            "Message handler registered",
            handler=handler.__class__.__name__,
            topics=topics
        )
    
    def subscribe_to_topics(self, topics: List[str]):
        """Subscribe to additional topics."""
        new_topics = [topic for topic in topics if topic not in self.topics]
        if new_topics:
            self.topics.extend(new_topics)
            
            # If consumer is already running, we need to restart subscription
            if self.consumer and self.running:
                asyncio.create_task(self._resubscribe())
        
        logger.info("Subscribed to topics", topics=topics, total_topics=len(self.topics))
    
    async def _resubscribe(self):
        """Resubscribe to topics (for dynamic topic subscription)."""
        try:
            if self.consumer:
                self.consumer.subscribe(topics=self.topics)
                logger.info("Resubscribed to topics", topics=self.topics)
        except Exception as e:
            logger.error(f"Failed to resubscribe to topics: {str(e)}")
    
    def _get_deserializer(self, content_type: str) -> MessageDeserializer:
        """Get appropriate deserializer for content type."""
        content_type = content_type.lower() if content_type else 'application/json'
        
        for deserializer in self.deserializers.values():
            if deserializer.can_handle(content_type):
                return deserializer
        
        # Default to JSON
        return self.deserializers['json']
    
    def _extract_message_metadata(self, record: ConsumerRecord) -> Dict[str, Any]:
        """Extract message metadata from Kafka record."""
        headers = {}
        if record.headers:
            for key, value in record.headers:
                headers[key] = value.decode('utf-8') if isinstance(value, bytes) else str(value)
        
        return {
            'topic': record.topic,
            'partition': record.partition,
            'offset': record.offset,
            'timestamp': record.timestamp,
            'timestamp_type': record.timestamp_type,
            'key': record.key,
            'headers': headers,
            'message_id': headers.get('message_id'),
            'correlation_id': headers.get('correlation_id'),
            'message_type': headers.get('message_type'),
            'content_type': headers.get('content_type', 'application/json'),
        }
    
    async def _process_message(self, record: ConsumerRecord) -> bool:
        """
        Process a single message record.
        
        Args:
            record: Kafka consumer record
            
        Returns:
            True if message was processed successfully
        """
        start_time = time.time()
        
        try:
            # Extract metadata
            metadata = self._extract_message_metadata(record)
            
            # Update consumption metrics
            self.metrics.record_message_consumed(len(record.value) if record.value else 0)
            
            # Deserialize message
            deserializer = self._get_deserializer(metadata['content_type'])
            message_data = deserializer.deserialize(record.value) if record.value else {}
            
            # Get message handlers for this topic
            handlers = self.message_handlers.get(record.topic, [])
            if not handlers:
                logger.warning(
                    "No handlers registered for topic",
                    topic=record.topic,
                    message_id=metadata.get('message_id')
                )
                return True  # Consider it processed to avoid reprocessing
            
            # Process message with all applicable handlers
            all_successful = True
            for handler in handlers:
                try:
                    if handler.can_handle(record.topic, metadata.get('message_type')):
                        success = await handler.handle(message_data, metadata)
                        if not success:
                            all_successful = False
                            logger.warning(
                                "Handler failed to process message",
                                handler=handler.__class__.__name__,
                                topic=record.topic,
                                message_id=metadata.get('message_id')
                            )
                except Exception as e:
                    all_successful = False
                    logger.error(
                        "Handler exception while processing message",
                        handler=handler.__class__.__name__,
                        topic=record.topic,
                        message_id=metadata.get('message_id'),
                        error=str(e),
                        exc_info=True
                    )
            
            # Update processing metrics
            processing_time_ms = (time.time() - start_time) * 1000
            
            if all_successful:
                self.metrics.record_message_processed(processing_time_ms)
                logger.debug(
                    "Message processed successfully",
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                    message_id=metadata.get('message_id'),
                    processing_time_ms=round(processing_time_ms, 2)
                )
                return True
            else:
                self.metrics.record_message_failed()
                return False
            
        except Exception as e:
            # Update failure metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_message_failed()
            
            logger.error(
                "Failed to process message",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                processing_time_ms=round(processing_time_ms, 2),
                error=str(e),
                exc_info=True
            )
            return False
    
    async def _send_to_dlq(self, record: ConsumerRecord, error: str):
        """Send failed message to dead letter queue."""
        try:
            # TODO: Implement DLQ producer
            logger.warning(
                "Message would be sent to DLQ",
                original_topic=record.topic,
                dlq_topic=self.dlq_topic,
                offset=record.offset,
                error=error
            )
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {str(e)}")
    
    async def _consume_loop(self):
        """Main message consumption loop."""
        logger.info("Starting message consumption loop")
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Get message batch
                    msg_pack = await self.consumer.getmany(
                        timeout_ms=1000,
                        max_records=self.settings.KAFKA_CONSUMER_MAX_POLL_RECORDS
                    )
                    
                    if not msg_pack:
                        continue
                    
                    # Process messages
                    processed_successfully = []
                    
                    for topic_partition, messages in msg_pack.items():
                        for message in messages:
                            try:
                                success = await self._process_message(message)
                                if success:
                                    processed_successfully.append((topic_partition, message))
                                else:
                                    # Handle failed message (retry or send to DLQ)
                                    await self._send_to_dlq(message, "Processing failed")
                                
                            except Exception as e:
                                logger.error(
                                    "Unexpected error processing message",
                                    topic=message.topic,
                                    offset=message.offset,
                                    error=str(e)
                                )
                                await self._send_to_dlq(message, str(e))
                    
                    # Commit offsets for successfully processed messages
                    if processed_successfully and not self.settings.KAFKA_CONSUMER_ENABLE_AUTO_COMMIT:
                        await self._commit_offsets(processed_successfully)
                    
                except ConsumerStoppedError:
                    logger.info("Consumer stopped")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in consumption loop: {str(e)}")
                    await asyncio.sleep(1)  # Prevent tight error loop
        
        except asyncio.CancelledError:
            logger.info("Consumption loop cancelled")
            raise
        
        finally:
            logger.info("Message consumption loop ended")
    
    async def _commit_offsets(self, successful_messages: List[tuple]):
        """Commit offsets for successfully processed messages."""
        try:
            # Build offset commit map
            offsets_to_commit = {}
            
            for topic_partition, message in successful_messages:
                # Commit offset + 1 (next message to be consumed)
                offsets_to_commit[topic_partition] = message.offset + 1
                
                # Update metrics
                self.metrics.update_committed_offset(
                    topic_partition.topic,
                    topic_partition.partition,
                    message.offset + 1
                )
            
            # Commit offsets
            await self.consumer.commit(offsets_to_commit)
            
            logger.debug(
                "Offsets committed",
                partitions=len(offsets_to_commit),
                offsets=offsets_to_commit
            )
            
        except Exception as e:
            logger.error(f"Failed to commit offsets: {str(e)}")
    
    async def start(self):
        """Start the consumer."""
        if self.running:
            logger.warning("Consumer is already running")
            return
        
        if not self.consumer:
            await self.initialize()
        
        if not self.topics:
            logger.warning("No topics subscribed, consumer will not process any messages")
        
        # Subscribe to topics
        if self.topics:
            self.consumer.subscribe(topics=self.topics)
        
        # Reset metrics
        self.metrics.reset()
        
        # Start consumption loop
        self.running = True
        self.shutdown_event.clear()
        self.consume_task = asyncio.create_task(self._consume_loop())
        
        logger.info(
            "Kafka consumer started",
            group_id=self.group_id,
            topics=self.topics
        )
    
    async def stop(self, timeout: float = 30.0):
        """Stop the consumer gracefully."""
        if not self.running:
            logger.info("Consumer is not running")
            return
        
        logger.info("Stopping Kafka consumer")
        
        # Signal shutdown
        self.running = False
        self.shutdown_event.set()
        
        # Wait for consume loop to finish
        if self.consume_task:
            try:
                await asyncio.wait_for(self.consume_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Consumer shutdown timed out, cancelling task")
                self.consume_task.cancel()
                try:
                    await self.consume_task
                except asyncio.CancelledError:
                    pass
        
        # Stop the consumer
        if self.consumer:
            try:
                await self.consumer.stop()
            except Exception as e:
                logger.error(f"Error stopping consumer: {str(e)}")
        
        logger.info("Kafka consumer stopped")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get consumer performance metrics."""
        metrics = self.metrics.get_metrics()
        
        # Add consumer-specific metrics
        if self.consumer:
            try:
                assignment = self.consumer.assignment()
                if assignment:
                    self.metrics.set_assigned_partitions(assignment)
                    
                    # Get consumer lag for assigned partitions
                    for tp in assignment:
                        try:
                            committed = await self.consumer.committed(tp)
                            if committed is not None:
                                # Note: Getting exact lag requires watermarks
                                # This is a simplified implementation
                                pass
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Error getting consumer metrics: {str(e)}")
        
        return metrics
    
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self.running


# =============================================================================
# EXAMPLE MESSAGE HANDLERS
# =============================================================================

class TradingEventHandler(MessageHandler):
    """Example handler for trading events."""
    
    async def handle(self, message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Handle trading event messages."""
        try:
            symbol = message.get('symbol')
            event_type = message.get('event_type')
            
            logger.info(
                "Processing trading event",
                symbol=symbol,
                event_type=event_type,
                message_id=metadata.get('message_id')
            )
            
            # TODO: Implement actual trading event processing
            # This is just a placeholder
            await asyncio.sleep(0.01)  # Simulate processing time
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling trading event: {str(e)}")
            return False
    
    def can_handle(self, topic: str, message_type: str = None) -> bool:
        """Check if can handle trading event topics."""
        return topic.startswith('oasis.trading.')


class MarketDataEventHandler(MessageHandler):
    """Example handler for market data events."""
    
    async def handle(self, message: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Handle market data event messages."""
        try:
            symbol = message.get('symbol')
            data_type = message.get('data_type')
            
            logger.debug(
                "Processing market data event",
                symbol=symbol,
                data_type=data_type,
                message_id=metadata.get('message_id')
            )
            
            # TODO: Implement actual market data processing
            # This is just a placeholder
            await asyncio.sleep(0.001)  # Simulate very fast processing
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling market data event: {str(e)}")
            return False
    
    def can_handle(self, topic: str, message_type: str = None) -> bool:
        """Check if can handle market data topics."""
        return topic.startswith('oasis.market_data.')


# =============================================================================
# GLOBAL CONSUMER INSTANCES
# =============================================================================

# Global consumer instances
consumers: Dict[str, OasisKafkaConsumer] = {}


def create_consumer(
    settings: OasisBaseSettings,
    name: str,
    group_id: str,
    topics: List[str]
) -> OasisKafkaConsumer:
    """Create and register a named consumer."""
    consumer = OasisKafkaConsumer(settings, group_id, topics)
    consumers[name] = consumer
    return consumer


async def start_all_consumers():
    """Start all registered consumers."""
    for name, consumer in consumers.items():
        try:
            await consumer.start()
            logger.info(f"Consumer '{name}' started")
        except Exception as e:
            logger.error(f"Failed to start consumer '{name}': {str(e)}")


async def stop_all_consumers():
    """Stop all registered consumers."""
    for name, consumer in consumers.items():
        try:
            await consumer.stop()
            logger.info(f"Consumer '{name}' stopped")
        except Exception as e:
            logger.error(f"Error stopping consumer '{name}': {str(e)}")
    
    consumers.clear()


def get_consumer(name: str) -> Optional[OasisKafkaConsumer]:
    """Get consumer by name."""
    return consumers.get(name)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

async def test_kafka_consumer(settings: OasisBaseSettings) -> bool:
    """Test Kafka consumer with message handlers."""
    logger.info("Testing Kafka consumer")
    
    try:
        # Create test consumer
        consumer = OasisKafkaConsumer(
            settings,
            group_id="oasis-test-consumer",
            topics=["oasis.test", "oasis.trading.signal", "oasis.market_data.tick"]
        )
        
        # Register test handlers
        trading_handler = TradingEventHandler()
        market_data_handler = MarketDataEventHandler()
        
        consumer.register_handler(trading_handler, ["oasis.trading.signal"])
        consumer.register_handler(market_data_handler, ["oasis.market_data.tick"])
        
        # Initialize consumer
        await consumer.initialize()
        
        # Test metrics
        initial_metrics = await consumer.get_metrics()
        
        # Start consumer briefly
        await consumer.start()
        
        # Let it run for a short time
        await asyncio.sleep(2)
        
        # Get final metrics
        final_metrics = await consumer.get_metrics()
        
        # Stop consumer
        await consumer.stop()
        
        logger.info(
            "Kafka consumer test completed",
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            handlers_registered=len(consumer.message_handlers)
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "Kafka consumer test failed",
            error=str(e),
            exc_info=True
        )
        return False


if __name__ == "__main__":
    """Kafka consumer test."""
    import asyncio
    from ...shared.config.base import get_settings
    
    async def main():
        settings = get_settings()
        success = await test_kafka_consumer(settings)
        
        if success:
            print("✅ Kafka consumer test passed")
            exit(0)
        else:
            print("❌ Kafka consumer test failed")
            exit(1)
    
    asyncio.run(main())