"""
Caching interface definitions for performance optimization.

This module defines interfaces for caching services to improve
application performance through data caching strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union
from datetime import datetime, timedelta


class ICacheService(ABC):
    """
    Interface for caching service operations.
    
    Provides abstraction for different caching backends like Redis,
    Memcached, or in-memory caches.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiration time for existing key."""
        pass
    
    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key."""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment numeric value."""
        pass
    
    @abstractmethod
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement numeric value."""
        pass
    
    # Batch operations
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        pass
    
    @abstractmethod
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache."""
        pass
    
    # Pattern operations
    @abstractmethod
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        pass
    
    @abstractmethod
    async def get_keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass
    
    # Hash operations
    @abstractmethod
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get field value from hash."""
        pass
    
    @abstractmethod
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """Set field value in hash."""
        pass
    
    @abstractmethod
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all field-value pairs from hash."""
        pass
    
    @abstractmethod
    async def hdel(self, key: str, field: str) -> bool:
        """Delete field from hash."""
        pass
    
    # List operations
    @abstractmethod
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to left of list."""
        pass
    
    @abstractmethod
    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to right of list."""
        pass
    
    @abstractmethod
    async def lpop(self, key: str) -> Optional[Any]:
        """Pop value from left of list."""
        pass
    
    @abstractmethod
    async def rpop(self, key: str) -> Optional[Any]:
        """Pop value from right of list."""
        pass
    
    @abstractmethod
    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get range of values from list."""
        pass
    
    @abstractmethod
    async def llen(self, key: str) -> int:
        """Get length of list."""
        pass
    
    # Set operations
    @abstractmethod
    async def sadd(self, key: str, *members: Any) -> int:
        """Add members to set."""
        pass
    
    @abstractmethod
    async def srem(self, key: str, *members: Any) -> int:
        """Remove members from set."""
        pass
    
    @abstractmethod
    async def smembers(self, key: str) -> set:
        """Get all members from set."""
        pass
    
    @abstractmethod
    async def sismember(self, key: str, member: Any) -> bool:
        """Check if member exists in set."""
        pass
    
    # Utility operations
    @abstractmethod
    async def ping(self) -> bool:
        """Test cache connectivity."""
        pass
    
    @abstractmethod
    async def flush_all(self) -> bool:
        """Clear all cache data."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class ISessionCache(ABC):
    """
    Interface for session-specific caching operations.
    
    Specialized caching for user sessions, temporary data,
    and short-lived application state.
    """
    
    @abstractmethod
    async def store_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """Store session data."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        pass
    
    @abstractmethod
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        pass
    
    @abstractmethod
    async def extend_session(self, session_id: str, ttl_seconds: int) -> bool:
        """Extend session TTL."""
        pass
    
    @abstractmethod
    async def get_active_sessions(self) -> List[str]:
        """Get all active session IDs."""
        pass


class IRateLimitCache(ABC):
    """
    Interface for rate limiting cache operations.
    
    Specialized caching for rate limiting, quota management,
    and usage tracking.
    """
    
    @abstractmethod
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> Dict[str, Union[bool, int]]:
        """Check if rate limit is exceeded."""
        pass
    
    @abstractmethod
    async def increment_usage(
        self,
        identifier: str,
        window_seconds: int,
        amount: int = 1
    ) -> int:
        """Increment usage counter."""
        pass
    
    @abstractmethod
    async def get_usage_count(
        self,
        identifier: str,
        window_seconds: int
    ) -> int:
        """Get current usage count."""
        pass
    
    @abstractmethod
    async def reset_usage(self, identifier: str) -> bool:
        """Reset usage counter."""
        pass
    
    @abstractmethod
    async def get_time_until_reset(
        self,
        identifier: str,
        window_seconds: int
    ) -> Optional[int]:
        """Get seconds until rate limit resets."""
        pass


class IDistributedLock(ABC):
    """
    Interface for distributed locking mechanism.
    
    Provides distributed locks for coordinating access to shared
    resources across multiple service instances.
    """
    
    @abstractmethod
    async def acquire(
        self,
        lock_key: str,
        ttl_seconds: int = 30,
        timeout_seconds: int = 10
    ) -> bool:
        """Acquire distributed lock."""
        pass
    
    @abstractmethod
    async def release(self, lock_key: str, token: str) -> bool:
        """Release distributed lock."""
        pass
    
    @abstractmethod
    async def extend(self, lock_key: str, token: str, ttl_seconds: int) -> bool:
        """Extend lock TTL."""
        pass
    
    @abstractmethod
    async def is_locked(self, lock_key: str) -> bool:
        """Check if resource is locked."""
        pass
    
    @abstractmethod
    async def force_unlock(self, lock_key: str) -> bool:
        """Force unlock (admin operation)."""
        pass
    
    @abstractmethod
    async def get_lock_info(self, lock_key: str) -> Optional[Dict[str, Any]]:
        """Get lock information."""
        pass