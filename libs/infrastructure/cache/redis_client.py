"""
Oasis Crypto Trade - Redis Cache Infrastructure
===============================================

Enterprise-grade Redis cache management with:
- High-performance async operations
- Connection pooling and health monitoring
- Advanced caching patterns
- Real-time data streaming with Redis Streams
- Distributed locking mechanisms
- Cache invalidation strategies
- Performance metrics and monitoring

Author: Oasis Trading Systems
License: Proprietary
"""

import asyncio
import json
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aioredis
import orjson
from aioredis import Redis, RedisError
from aioredis.client import Pipeline

from ...shared.config.base import OasisBaseSettings
from ...shared.exceptions.infrastructure import CacheConnectionError, CacheOperationError
from ...shared.logging.config import get_logger

logger = get_logger("oasis.cache")


# =============================================================================
# SERIALIZATION STRATEGIES
# =============================================================================

class SerializationStrategy:
    """Base class for serialization strategies."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        raise NotImplementedError
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        raise NotImplementedError


class JSONSerializer(SerializationStrategy):
    """Fast JSON serialization using orjson."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize using orjson for optimal performance."""
        try:
            return orjson.dumps(data)
        except (TypeError, ValueError) as e:
            raise CacheOperationError(f"JSON serialization failed: {str(e)}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize from JSON bytes."""
        try:
            return orjson.loads(data)
        except (orjson.JSONDecodeError, ValueError) as e:
            raise CacheOperationError(f"JSON deserialization failed: {str(e)}")


class PickleSerializer(SerializationStrategy):
    """Pickle serialization for complex Python objects."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize using pickle with high protocol."""
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise CacheOperationError(f"Pickle serialization failed: {str(e)}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize from pickle bytes."""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise CacheOperationError(f"Pickle deserialization failed: {str(e)}")


# =============================================================================
# REDIS CONNECTION MANAGER
# =============================================================================

class OasisRedisManager:
    """
    Enterprise-grade Redis connection and operation manager.
    
    Features:
    - Connection pooling with health checks
    - Multiple serialization strategies
    - Distributed caching patterns
    - Real-time streaming support
    - Performance monitoring
    - Automatic failover
    """
    
    def __init__(self, settings: OasisBaseSettings):
        self.settings = settings
        self.redis: Optional[Redis] = None
        self.connection_pool: Optional[aioredis.ConnectionPool] = None
        self.serializers = {
            'json': JSONSerializer(),
            'pickle': PickleSerializer(),
        }
        self.default_serializer = 'json'
        self._connection_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_response_time': 0.0,
            'last_operation_time': None,
            'connection_errors': 0,
        }
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0
    
    async def initialize(self):
        """Initialize Redis connection with optimized settings."""
        logger.info("Initializing Redis connection")
        
        try:
            # Create connection pool with optimized settings
            self.connection_pool = aioredis.ConnectionPool(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD.get_secret_value(),
                db=self.settings.REDIS_DB,
                encoding='utf-8',
                decode_responses=False,  # Handle encoding ourselves for performance
                max_connections=self.settings.REDIS_POOL_SIZE,
                socket_timeout=self.settings.REDIS_POOL_TIMEOUT,
                socket_connect_timeout=10,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=self._health_check_interval,
                retry_on_timeout=self.settings.REDIS_RETRY_ON_TIMEOUT,
            )
            
            # Create Redis client
            self.redis = Redis(
                connection_pool=self.connection_pool,
                socket_connect_timeout=10,
                socket_timeout=self.settings.REDIS_POOL_TIMEOUT,
                retry_on_error=[ConnectionError, TimeoutError],
                retry_on_timeout=self.settings.REDIS_RETRY_ON_TIMEOUT,
            )
            
            # Test initial connection
            await self.health_check(force=True)
            
            logger.info(
                "Redis connection initialized",
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_DB,
                pool_size=self.settings.REDIS_POOL_SIZE
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize Redis connection",
                error=str(e),
                exc_info=True
            )
            raise CacheConnectionError(f"Redis initialization failed: {str(e)}")
    
    async def health_check(self, force: bool = False) -> bool:
        """Perform Redis health check."""
        if not self.redis:
            return False
        
        current_time = time.time()
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return True
        
        try:
            start_time = time.time()
            pong = await self.redis.ping()
            response_time = (time.time() - start_time) * 1000  # milliseconds
            
            if pong:
                logger.debug(
                    "Redis health check passed",
                    response_time_ms=round(response_time, 2)
                )
                self._last_health_check = current_time
                return True
            else:
                logger.warning("Redis health check failed: No pong response")
                return False
                
        except Exception as e:
            logger.error(
                "Redis health check failed",
                error=str(e)
            )
            self._connection_metrics['connection_errors'] += 1
            return False
    
    def _get_serializer(self, serializer: str = None) -> SerializationStrategy:
        """Get serializer instance."""
        serializer = serializer or self.default_serializer
        if serializer not in self.serializers:
            raise CacheOperationError(f"Unknown serializer: {serializer}")
        return self.serializers[serializer]
    
    def _build_key(self, key: str) -> str:
        """Build cache key with prefix."""
        return f"{self.settings.CACHE_PREFIX}{key}"
    
    async def _execute_operation(self, operation_name: str, operation):
        """Execute Redis operation with metrics tracking."""
        if not self.redis:
            raise CacheConnectionError("Redis not initialized")
        
        start_time = time.time()
        try:
            result = await operation()
            
            # Update success metrics
            self._connection_metrics['total_operations'] += 1
            self._connection_metrics['successful_operations'] += 1
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            total_ops = self._connection_metrics['total_operations']
            avg_time = self._connection_metrics['avg_response_time']
            
            # Update average response time
            self._connection_metrics['avg_response_time'] = (
                (avg_time * (total_ops - 1) + response_time) / total_ops
            )
            self._connection_metrics['last_operation_time'] = datetime.utcnow()
            
            logger.debug(
                f"Redis operation completed: {operation_name}",
                response_time_ms=round(response_time, 2)
            )
            
            return result
            
        except Exception as e:
            # Update failure metrics
            self._connection_metrics['total_operations'] += 1
            self._connection_metrics['failed_operations'] += 1
            
            logger.error(
                f"Redis operation failed: {operation_name}",
                error=str(e),
                response_time_ms=round((time.time() - start_time) * 1000, 2)
            )
            
            raise CacheOperationError(f"Redis operation '{operation_name}' failed: {str(e)}")
    
    # ==========================================================================
    # BASIC CACHE OPERATIONS
    # ==========================================================================
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: str = None
    ) -> bool:
        """Set cache value with optional TTL."""
        cache_key = self._build_key(key)
        ttl = ttl or self.settings.CACHE_TTL
        serializer_instance = self._get_serializer(serializer)
        
        async def _set_operation():
            serialized_value = serializer_instance.serialize(value)
            return await self.redis.setex(cache_key, ttl, serialized_value)
        
        return await self._execute_operation(f"SET {key}", _set_operation)
    
    async def get(
        self,
        key: str,
        default: Any = None,
        serializer: str = None
    ) -> Any:
        """Get cache value with deserialization."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _get_operation():
            data = await self.redis.get(cache_key)
            if data is None:
                return default
            return serializer_instance.deserialize(data)
        
        return await self._execute_operation(f"GET {key}", _get_operation)
    
    async def delete(self, key: str) -> bool:
        """Delete cache key."""
        cache_key = self._build_key(key)
        
        async def _delete_operation():
            result = await self.redis.delete(cache_key)
            return result > 0
        
        return await self._execute_operation(f"DELETE {key}", _delete_operation)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        cache_key = self._build_key(key)
        
        async def _exists_operation():
            return await self.redis.exists(cache_key)
        
        return await self._execute_operation(f"EXISTS {key}", _exists_operation)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key."""
        cache_key = self._build_key(key)
        
        async def _expire_operation():
            return await self.redis.expire(cache_key, ttl)
        
        return await self._execute_operation(f"EXPIRE {key}", _expire_operation)
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        cache_key = self._build_key(key)
        
        async def _ttl_operation():
            return await self.redis.ttl(cache_key)
        
        return await self._execute_operation(f"TTL {key}", _ttl_operation)
    
    # ==========================================================================
    # ADVANCED CACHE OPERATIONS
    # ==========================================================================
    
    async def mget(
        self,
        keys: List[str],
        serializer: str = None
    ) -> Dict[str, Any]:
        """Get multiple cache values."""
        cache_keys = [self._build_key(key) for key in keys]
        serializer_instance = self._get_serializer(serializer)
        
        async def _mget_operation():
            values = await self.redis.mget(cache_keys)
            result = {}
            
            for i, (key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    result[key] = serializer_instance.deserialize(value)
                else:
                    result[key] = None
            
            return result
        
        return await self._execute_operation(f"MGET {len(keys)} keys", _mget_operation)
    
    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        serializer: str = None
    ) -> bool:
        """Set multiple cache values."""
        serializer_instance = self._get_serializer(serializer)
        ttl = ttl or self.settings.CACHE_TTL
        
        async def _mset_operation():
            # Serialize all values
            serialized_mapping = {}
            for key, value in mapping.items():
                cache_key = self._build_key(key)
                serialized_mapping[cache_key] = serializer_instance.serialize(value)
            
            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()
            pipe.mset(serialized_mapping)
            
            # Set TTL for all keys if specified
            if ttl:
                for cache_key in serialized_mapping.keys():
                    pipe.expire(cache_key, ttl)
            
            return await pipe.execute()
        
        await self._execute_operation(f"MSET {len(mapping)} keys", _mset_operation)
        return True
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        cache_key = self._build_key(key)
        
        async def _incr_operation():
            return await self.redis.incrby(cache_key, amount)
        
        return await self._execute_operation(f"INCR {key}", _incr_operation)
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement numeric value."""
        cache_key = self._build_key(key)
        
        async def _decr_operation():
            return await self.redis.decrby(cache_key, amount)
        
        return await self._execute_operation(f"DECR {key}", _decr_operation)
    
    # ==========================================================================
    # LIST OPERATIONS
    # ==========================================================================
    
    async def lpush(self, key: str, *values: Any, serializer: str = None) -> int:
        """Push values to left of list."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _lpush_operation():
            serialized_values = [serializer_instance.serialize(v) for v in values]
            return await self.redis.lpush(cache_key, *serialized_values)
        
        return await self._execute_operation(f"LPUSH {key}", _lpush_operation)
    
    async def rpush(self, key: str, *values: Any, serializer: str = None) -> int:
        """Push values to right of list."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _rpush_operation():
            serialized_values = [serializer_instance.serialize(v) for v in values]
            return await self.redis.rpush(cache_key, *serialized_values)
        
        return await self._execute_operation(f"RPUSH {key}", _rpush_operation)
    
    async def lpop(self, key: str, serializer: str = None) -> Any:
        """Pop value from left of list."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _lpop_operation():
            data = await self.redis.lpop(cache_key)
            if data is None:
                return None
            return serializer_instance.deserialize(data)
        
        return await self._execute_operation(f"LPOP {key}", _lpop_operation)
    
    async def rpop(self, key: str, serializer: str = None) -> Any:
        """Pop value from right of list."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _rpop_operation():
            data = await self.redis.rpop(cache_key)
            if data is None:
                return None
            return serializer_instance.deserialize(data)
        
        return await self._execute_operation(f"RPOP {key}", _rpop_operation)
    
    async def llen(self, key: str) -> int:
        """Get list length."""
        cache_key = self._build_key(key)
        
        async def _llen_operation():
            return await self.redis.llen(cache_key)
        
        return await self._execute_operation(f"LLEN {key}", _llen_operation)
    
    # ==========================================================================
    # SET OPERATIONS
    # ==========================================================================
    
    async def sadd(self, key: str, *values: Any, serializer: str = None) -> int:
        """Add values to set."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _sadd_operation():
            serialized_values = [serializer_instance.serialize(v) for v in values]
            return await self.redis.sadd(cache_key, *serialized_values)
        
        return await self._execute_operation(f"SADD {key}", _sadd_operation)
    
    async def smembers(self, key: str, serializer: str = None) -> List[Any]:
        """Get all set members."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _smembers_operation():
            members = await self.redis.smembers(cache_key)
            return [serializer_instance.deserialize(member) for member in members]
        
        return await self._execute_operation(f"SMEMBERS {key}", _smembers_operation)
    
    # ==========================================================================
    # HASH OPERATIONS
    # ==========================================================================
    
    async def hset(
        self,
        key: str,
        field: str,
        value: Any,
        serializer: str = None
    ) -> bool:
        """Set hash field value."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _hset_operation():
            serialized_value = serializer_instance.serialize(value)
            return await self.redis.hset(cache_key, field, serialized_value)
        
        return await self._execute_operation(f"HSET {key} {field}", _hset_operation)
    
    async def hget(
        self,
        key: str,
        field: str,
        default: Any = None,
        serializer: str = None
    ) -> Any:
        """Get hash field value."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _hget_operation():
            data = await self.redis.hget(cache_key, field)
            if data is None:
                return default
            return serializer_instance.deserialize(data)
        
        return await self._execute_operation(f"HGET {key} {field}", _hget_operation)
    
    async def hmset(
        self,
        key: str,
        mapping: Dict[str, Any],
        serializer: str = None
    ) -> bool:
        """Set multiple hash fields."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _hmset_operation():
            serialized_mapping = {
                field: serializer_instance.serialize(value)
                for field, value in mapping.items()
            }
            return await self.redis.hmset(cache_key, serialized_mapping)
        
        return await self._execute_operation(f"HMSET {key}", _hmset_operation)
    
    async def hgetall(
        self,
        key: str,
        serializer: str = None
    ) -> Dict[str, Any]:
        """Get all hash fields and values."""
        cache_key = self._build_key(key)
        serializer_instance = self._get_serializer(serializer)
        
        async def _hgetall_operation():
            data = await self.redis.hgetall(cache_key)
            return {
                field.decode('utf-8'): serializer_instance.deserialize(value)
                for field, value in data.items()
            }
        
        return await self._execute_operation(f"HGETALL {key}", _hgetall_operation)
    
    # ==========================================================================
    # DISTRIBUTED LOCKING
    # ==========================================================================
    
    @asynccontextmanager
    async def distributed_lock(
        self,
        lock_key: str,
        timeout: int = 30,
        blocking_timeout: int = 10
    ) -> AsyncGenerator[bool, None]:
        """Distributed lock context manager."""
        cache_key = self._build_key(f"lock:{lock_key}")
        lock_value = str(time.time())
        acquired = False
        
        try:
            # Try to acquire lock
            start_time = time.time()
            while (time.time() - start_time) < blocking_timeout:
                acquired = await self.redis.set(
                    cache_key,
                    lock_value,
                    ex=timeout,
                    nx=True
                )
                if acquired:
                    break
                await asyncio.sleep(0.1)
            
            if not acquired:
                logger.warning(f"Failed to acquire distributed lock: {lock_key}")
            
            yield acquired
            
        finally:
            if acquired:
                # Release lock only if we own it
                try:
                    stored_value = await self.redis.get(cache_key)
                    if stored_value and stored_value.decode('utf-8') == lock_value:
                        await self.redis.delete(cache_key)
                        logger.debug(f"Released distributed lock: {lock_key}")
                except Exception as e:
                    logger.error(f"Error releasing lock {lock_key}: {str(e)}")
    
    # ==========================================================================
    # REDIS STREAMS (for real-time data)
    # ==========================================================================
    
    async def xadd(
        self,
        stream_key: str,
        fields: Dict[str, Any],
        message_id: str = '*',
        max_len: Optional[int] = None,
        serializer: str = None
    ) -> str:
        """Add message to Redis Stream."""
        cache_key = self._build_key(f"stream:{stream_key}")
        serializer_instance = self._get_serializer(serializer)
        
        async def _xadd_operation():
            # Serialize field values
            serialized_fields = {}
            for field, value in fields.items():
                if isinstance(value, (str, int, float)):
                    serialized_fields[field] = str(value)
                else:
                    serialized_fields[field] = serializer_instance.serialize(value).decode('utf-8')
            
            return await self.redis.xadd(
                cache_key,
                serialized_fields,
                id=message_id,
                maxlen=max_len
            )
        
        return await self._execute_operation(f"XADD {stream_key}", _xadd_operation)
    
    async def xread(
        self,
        stream_keys: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Read messages from Redis Streams."""
        cache_keys = {
            self._build_key(f"stream:{key}"): stream_id
            for key, stream_id in stream_keys.items()
        }
        
        async def _xread_operation():
            result = await self.redis.xread(
                cache_keys,
                count=count,
                block=block
            )
            
            # Process and deserialize results
            processed_result = {}
            for stream_key, messages in result.items():
                original_key = stream_key.replace(f"{self.settings.CACHE_PREFIX}stream:", "")
                processed_messages = []
                
                for message_id, fields in messages:
                    processed_fields = {}
                    for field, value in fields.items():
                        try:
                            # Try to deserialize JSON, fallback to string
                            processed_fields[field.decode('utf-8')] = orjson.loads(value)
                        except (orjson.JSONDecodeError, TypeError):
                            processed_fields[field.decode('utf-8')] = value.decode('utf-8')
                    
                    processed_messages.append({
                        'id': message_id.decode('utf-8'),
                        'fields': processed_fields
                    })
                
                processed_result[original_key] = processed_messages
            
            return processed_result
        
        return await self._execute_operation(f"XREAD {len(stream_keys)} streams", _xread_operation)
    
    # ==========================================================================
    # CACHE PATTERNS
    # ==========================================================================
    
    async def cache_aside(
        self,
        key: str,
        fetch_func,
        ttl: Optional[int] = None,
        serializer: str = None
    ) -> Any:
        """Cache-aside pattern implementation."""
        # Try to get from cache first
        cached_value = await self.get(key, serializer=serializer)
        if cached_value is not None:
            return cached_value
        
        # Fetch from source
        value = await fetch_func()
        
        # Store in cache
        if value is not None:
            await self.set(key, value, ttl=ttl, serializer=serializer)
        
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        pattern = self._build_key(pattern)
        
        async def _invalidate_operation():
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        
        return await self._execute_operation(f"DELETE pattern {pattern}", _invalidate_operation)
    
    # ==========================================================================
    # MONITORING AND METRICS
    # ==========================================================================
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        if not self.redis:
            return {}
        
        info = await self.redis.info()
        
        return {
            'connection_metrics': self._connection_metrics.copy(),
            'redis_info': {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
            }
        }
    
    async def get_info(self, section: str = None) -> Dict[str, Any]:
        """Get Redis info."""
        if not self.redis:
            return {}
        
        return await self.redis.info(section)
    
    # ==========================================================================
    # CLEANUP AND SHUTDOWN
    # ==========================================================================
    
    async def close(self):
        """Close Redis connections."""
        logger.info("Closing Redis connections")
        
        try:
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            if self.redis:
                await self.redis.close()
            
            logger.info("Redis connections closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")
        finally:
            self.redis = None
            self.connection_pool = None


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

# Global cache manager instance
cache_manager: Optional[OasisRedisManager] = None


async def initialize_cache(settings: OasisBaseSettings):
    """Initialize global cache connection."""
    global cache_manager
    
    cache_manager = OasisRedisManager(settings)
    await cache_manager.initialize()
    
    logger.info("Global cache manager initialized")


async def get_cache() -> OasisRedisManager:
    """Get global cache manager."""
    if not cache_manager:
        raise CacheConnectionError("Cache not initialized")
    return cache_manager


async def close_cache():
    """Close global cache connections."""
    global cache_manager
    
    if cache_manager:
        await cache_manager.close()
        cache_manager = None
    
    logger.info("Global cache connections closed")


# =============================================================================
# CACHE TESTING UTILITIES
# =============================================================================

async def test_cache_connection(settings: OasisBaseSettings) -> bool:
    """Test cache connection with comprehensive diagnostics."""
    logger.info("Testing cache connection")
    
    try:
        # Test basic connection
        test_manager = OasisRedisManager(settings)
        await test_manager.initialize()
        
        # Test basic operations
        test_key = "test_connection"
        test_value = {"message": "Hello Oasis", "timestamp": time.time()}
        
        # Test set/get
        await test_manager.set(test_key, test_value, ttl=60)
        retrieved_value = await test_manager.get(test_key)
        
        # Test different data types
        await test_manager.lpush("test_list", "item1", "item2")
        list_length = await test_manager.llen("test_list")
        
        await test_manager.sadd("test_set", "member1", "member2")
        set_members = await test_manager.smembers("test_set")
        
        await test_manager.hset("test_hash", "field1", "value1")
        hash_value = await test_manager.hget("test_hash", "field1")
        
        # Test distributed lock
        async with test_manager.distributed_lock("test_lock", timeout=5) as acquired:
            if not acquired:
                logger.warning("Failed to acquire test lock")
        
        # Get metrics
        metrics = await test_manager.get_metrics()
        
        # Cleanup test data
        await test_manager.delete(test_key)
        await test_manager.delete("test_list")
        await test_manager.delete("test_set")
        await test_manager.delete("test_hash")
        
        # Close test manager
        await test_manager.close()
        
        logger.info(
            "Cache connection test successful",
            retrieved_value=retrieved_value,
            list_length=list_length,
            set_members_count=len(set_members),
            hash_value=hash_value,
            metrics=metrics['connection_metrics']
        )
        
        return (
            retrieved_value == test_value and
            list_length == 2 and
            len(set_members) == 2 and
            hash_value == "value1"
        )
        
    except Exception as e:
        logger.error(
            "Cache connection test failed",
            error=str(e),
            exc_info=True
        )
        return False


if __name__ == "__main__":
    """Cache system test."""
    import asyncio
    from ...shared.config.base import get_settings
    
    async def main():
        settings = get_settings()
        success = await test_cache_connection(settings)
        
        if success:
            print("✅ Cache connection test passed")
            exit(0)
        else:
            print("❌ Cache connection test failed")
            exit(1)
    
    asyncio.run(main())