"""
Cache Manager for API Gateway.

This module provides caching functionality using Redis for performance optimization.
"""

import json
import logging
from typing import Optional, Any, Dict
import redis.asyncio as redis
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for the API Gateway."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.settings = get_settings()
        self.redis_url = self.settings.redis_url
        self.default_ttl = self.settings.cache_ttl_seconds
        self.enabled = self.settings.enable_caching
        self._redis = None
        
    async def get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis connection."""
        if not self.enabled:
            return None
            
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._redis = None
                
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return None
                
            value = await redis.get(key)
            if value:
                return json.loads(value)
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return False
                
            serialized_value = json.dumps(value, default=str)
            ttl = ttl or self.default_ttl
            
            await redis.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return False
                
            await redis.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return False
                
            result = await redis.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def flush_all(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return False
                
            await redis.flushdb()
            logger.info("Cache flushed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False
    
    async def get_user_cache_key(self, user_id: str) -> str:
        """Generate cache key for user data."""
        return f"user:{user_id}"
    
    async def get_token_cache_key(self, token_hash: str) -> str:
        """Generate cache key for token validation."""
        return f"token:{token_hash}"
    
    async def get_permission_cache_key(self, user_id: str, resource: str, action: str) -> str:
        """Generate cache key for permission check."""
        return f"perm:{user_id}:{resource}:{action}"
    
    async def ping(self) -> bool:
        """
        Ping Redis server to check connectivity.
        
        Returns:
            True if Redis is available, False otherwise
        """
        if not self.enabled:
            return True  # Consider disabled cache as "healthy"
            
        try:
            redis = await self.get_redis()
            if redis is None:
                return False
                
            await redis.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")