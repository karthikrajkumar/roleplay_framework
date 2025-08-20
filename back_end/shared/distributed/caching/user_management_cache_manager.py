"""
User Management Distributed Cache Manager

Advanced caching system specifically designed for user management operations
with multi-tenant organizations, group memberships, and permission hierarchies.

Key Features:
- Multi-layer cache hierarchy optimized for user management patterns
- Hierarchical permission caching with inheritance resolution
- Group membership cache with real-time invalidation
- Organization-scoped cache partitioning for data isolation
- Bulk operation cache warming and consistency management
- SSO identity cache with provider-aware optimization

Cache Layers:
L1 (Local): In-memory cache for frequently accessed permissions and memberships
L2 (Regional): Redis cluster for organization and group data
L3 (Global): Global cache for user profiles and cross-region data
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer identifiers"""
    L1_LOCAL = "l1_local"
    L2_REGIONAL = "l2_regional" 
    L3_GLOBAL = "l3_global"


class UserManagementDataType(Enum):
    """Data types specific to user management caching"""
    USER_PROFILE = "user_profile"
    USER_PERMISSIONS = "user_permissions"
    GROUP_MEMBERSHIP = "group_membership"
    GROUP_PERMISSIONS = "group_permissions"
    ORGANIZATION_MEMBERSHIP = "organization_membership"
    ORGANIZATION_SETTINGS = "organization_settings"
    SSO_IDENTITY = "sso_identity"
    BULK_IMPORT_STATUS = "bulk_import_status"
    PERMISSION_HIERARCHY = "permission_hierarchy"
    GROUP_HIERARCHY = "group_hierarchy"


@dataclass
class CacheEntry:
    """Cache entry with metadata for user management operations"""
    key: str
    value: Any
    data_type: UserManagementDataType
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    ttl: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    accessed_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    dependencies: Set[str] = field(default_factory=set)
    invalidation_tags: Set[str] = field(default_factory=set)


@dataclass
class CacheStats:
    """Comprehensive cache statistics"""
    layer: CacheLayer
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    memory_usage_mb: float
    entry_count: int
    avg_access_time_ms: float
    hot_keys: List[str]
    data_type_distribution: Dict[UserManagementDataType, int]


class UserManagementCacheManager:
    """
    Advanced cache manager for user management with multi-tenant support.
    
    Implements intelligent caching strategies for:
    - User permissions with role hierarchy
    - Group memberships with inheritance
    - Organization-scoped data isolation
    - SSO identity resolution
    - Bulk operation status tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = config.get('node_id', 'cache_node_1')
        self.region = config.get('region', 'us-east-1')
        
        # Cache layers configuration
        self.l1_config = config.get('l1_cache', {})
        self.l2_config = config.get('l2_cache', {})
        self.l3_config = config.get('l3_cache', {})
        
        # Initialize cache layers
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_redis_pool = None
        self.l3_redis_pool = None
        
        # Cache statistics
        self.stats = {
            CacheLayer.L1_LOCAL: CacheStats(
                layer=CacheLayer.L1_LOCAL, hit_rate=0.0, miss_rate=0.0,
                eviction_rate=0.0, memory_usage_mb=0.0, entry_count=0,
                avg_access_time_ms=0.0, hot_keys=[], data_type_distribution={}
            ),
            CacheLayer.L2_REGIONAL: CacheStats(
                layer=CacheLayer.L2_REGIONAL, hit_rate=0.0, miss_rate=0.0,
                eviction_rate=0.0, memory_usage_mb=0.0, entry_count=0,
                avg_access_time_ms=0.0, hot_keys=[], data_type_distribution={}
            ),
            CacheLayer.L3_GLOBAL: CacheStats(
                layer=CacheLayer.L3_GLOBAL, hit_rate=0.0, miss_rate=0.0,
                eviction_rate=0.0, memory_usage_mb=0.0, entry_count=0,
                avg_access_time_ms=0.0, hot_keys=[], data_type_distribution={}
            )
        }
        
        # Permission resolution cache
        self.permission_resolution_cache: Dict[str, Dict[str, Any]] = {}
        
        # Invalidation tracking
        self.invalidation_queue: List[Dict[str, Any]] = []
        
        # Initialize cache layers
        asyncio.create_task(self._initialize_cache_layers())
        
        # Start cache management tasks
        asyncio.create_task(self._cache_maintenance_loop())
        asyncio.create_task(self._invalidation_processing_loop())
        asyncio.create_task(self._stats_collection_loop())
    
    async def _initialize_cache_layers(self):
        """Initialize Redis connections for L2 and L3 cache layers"""
        try:
            # L2 Regional Redis Cluster
            if self.l2_config.get('enabled', True):
                self.l2_redis_pool = redis.ConnectionPool.from_url(
                    self.l2_config.get('url', 'redis://localhost:6379/1'),
                    decode_responses=True,
                    max_connections=self.l2_config.get('max_connections', 20)
                )
            
            # L3 Global Redis Cluster  
            if self.l3_config.get('enabled', True):
                self.l3_redis_pool = redis.ConnectionPool.from_url(
                    self.l3_config.get('url', 'redis://localhost:6379/2'),
                    decode_responses=True,
                    max_connections=self.l3_config.get('max_connections', 10)
                )
                
            logger.info(f"Cache layers initialized for region {self.region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache layers: {e}")
    
    async def get_user_permissions(self, user_id: str, organization_id: str, 
                                 group_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive user permissions with role hierarchy resolution.
        
        Implements intelligent caching with:
        - Permission inheritance from organization -> groups -> user
        - Role hierarchy resolution
        - Multi-layer cache lookup
        """
        cache_key = f"user_permissions:{organization_id}:{user_id}"
        if group_ids:
            group_hash = sha256(':'.join(sorted(group_ids)).encode()).hexdigest()[:8]
            cache_key += f":groups:{group_hash}"
        
        # Try L1 cache first
        permissions = await self._get_from_l1(cache_key)
        if permissions:
            return permissions
        
        # Try L2 cache
        permissions = await self._get_from_l2(cache_key)
        if permissions:
            await self._store_in_l1(cache_key, permissions, UserManagementDataType.USER_PERMISSIONS,
                                  organization_id=organization_id, user_id=user_id,
                                  ttl=300)  # 5 minute TTL for L1
            return permissions
        
        # Compute permissions with hierarchy resolution
        permissions = await self._compute_user_permissions(user_id, organization_id, group_ids)
        
        # Store in all cache layers
        await self._store_in_l2(cache_key, permissions, ttl=1800)  # 30 minute TTL
        await self._store_in_l1(cache_key, permissions, UserManagementDataType.USER_PERMISSIONS,
                              organization_id=organization_id, user_id=user_id, ttl=300)
        
        return permissions
    
    async def get_group_memberships(self, user_id: str, organization_id: str) -> List[Dict[str, Any]]:
        """
        Get user's group memberships with caching optimization.
        
        Returns denormalized group membership data including:
        - Group details and settings
        - User role within each group
        - Progress and participation metrics
        """
        cache_key = f"group_memberships:{organization_id}:{user_id}"
        
        # Multi-layer cache lookup
        memberships = await self._multi_layer_get(
            cache_key, 
            UserManagementDataType.GROUP_MEMBERSHIP,
            organization_id=organization_id,
            user_id=user_id
        )
        
        if memberships is None:
            # Fetch from database and cache
            memberships = await self._fetch_group_memberships_from_db(user_id, organization_id)
            
            # Store with appropriate TTLs
            await self._multi_layer_store(
                cache_key,
                memberships,
                UserManagementDataType.GROUP_MEMBERSHIP,
                organization_id=organization_id,
                user_id=user_id,
                l1_ttl=600,   # 10 minutes
                l2_ttl=3600,  # 1 hour
                l3_ttl=7200   # 2 hours
            )
        
        return memberships
    
    async def get_organization_settings(self, organization_id: str) -> Dict[str, Any]:
        """
        Get organization settings with global caching.
        
        Organization settings are cached globally since they're shared
        across regions for the same organization.
        """
        cache_key = f"org_settings:{organization_id}"
        
        # Try L1 -> L2 -> L3 -> Database
        settings = await self._get_from_l1(cache_key)
        if settings:
            return settings
            
        settings = await self._get_from_l2(cache_key)
        if settings:
            await self._store_in_l1(cache_key, settings, UserManagementDataType.ORGANIZATION_SETTINGS,
                                  organization_id=organization_id, ttl=900)  # 15 minutes
            return settings
            
        settings = await self._get_from_l3(cache_key)
        if settings:
            await self._store_in_l2(cache_key, settings, ttl=7200)  # 2 hours
            await self._store_in_l1(cache_key, settings, UserManagementDataType.ORGANIZATION_SETTINGS,
                                  organization_id=organization_id, ttl=900)
            return settings
        
        # Fetch from database
        settings = await self._fetch_organization_settings_from_db(organization_id)
        
        # Store in all layers
        await self._store_in_l3(cache_key, settings, ttl=86400)  # 24 hours
        await self._store_in_l2(cache_key, settings, ttl=7200)   # 2 hours
        await self._store_in_l1(cache_key, settings, UserManagementDataType.ORGANIZATION_SETTINGS,
                              organization_id=organization_id, ttl=900)  # 15 minutes
        
        return settings
    
    async def get_sso_identity(self, provider: str, provider_user_id: str, 
                             organization_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get SSO identity with provider-specific caching optimization.
        
        SSO identities are cached with:
        - Provider-aware key generation
        - Organization context when applicable
        - Fast lookup for authentication flows
        """
        cache_key = f"sso_identity:{provider}:{provider_user_id}"
        if organization_id:
            cache_key += f":org:{organization_id}"
        
        identity = await self._multi_layer_get(
            cache_key,
            UserManagementDataType.SSO_IDENTITY,
            organization_id=organization_id
        )
        
        if identity is None:
            identity = await self._fetch_sso_identity_from_db(provider, provider_user_id, organization_id)
            
            if identity:
                await self._multi_layer_store(
                    cache_key,
                    identity,
                    UserManagementDataType.SSO_IDENTITY,
                    organization_id=organization_id,
                    user_id=identity.get('user_id'),
                    l1_ttl=300,   # 5 minutes (frequent auth operations)
                    l2_ttl=1800,  # 30 minutes
                    l3_ttl=3600   # 1 hour
                )
        
        return identity
    
    async def get_bulk_import_status(self, import_job_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        """
        Get bulk import operation status with real-time caching.
        
        Bulk import status requires:
        - Real-time updates for progress tracking
        - Short TTL for accuracy
        - Organization-scoped access
        """
        cache_key = f"bulk_import:{organization_id}:{import_job_id}"
        
        # Primarily use L2 cache for bulk import status (real-time updates)
        status = await self._get_from_l2(cache_key)
        if status is None:
            status = await self._fetch_bulk_import_status_from_db(import_job_id, organization_id)
            
            if status:
                # Short TTL for real-time accuracy
                await self._store_in_l2(cache_key, status, ttl=30)  # 30 seconds
        
        return status
    
    async def invalidate_user_permissions(self, user_id: str, organization_id: str):
        """
        Invalidate user permissions across all cache layers.
        
        This is called when:
        - User role changes
        - Group membership changes
        - Organization permissions update
        """
        patterns = [
            f"user_permissions:{organization_id}:{user_id}*",
            f"group_memberships:{organization_id}:{user_id}",
        ]
        
        await self._invalidate_by_patterns(patterns, organization_id, user_id)
        
        # Add to invalidation queue for cross-region propagation
        self.invalidation_queue.append({
            'type': 'user_permissions',
            'user_id': user_id,
            'organization_id': organization_id,
            'timestamp': time.time()
        })
    
    async def invalidate_group_data(self, group_id: str, organization_id: str):
        """
        Invalidate group-related data across all cache layers.
        
        Called when:
        - Group settings change
        - Group membership changes
        - Group permissions update
        """
        patterns = [
            f"group_permissions:{organization_id}:{group_id}*",
            f"user_permissions:{organization_id}:*:groups:*",  # All user permissions with groups
            f"group_hierarchy:{organization_id}:*{group_id}*",
        ]
        
        await self._invalidate_by_patterns(patterns, organization_id, group_id=group_id)
        
        # Queue for cross-region invalidation
        self.invalidation_queue.append({
            'type': 'group_data',
            'group_id': group_id,
            'organization_id': organization_id,
            'timestamp': time.time()
        })
    
    async def invalidate_organization_data(self, organization_id: str):
        """
        Invalidate all organization-related data.
        
        Called for organization-wide changes:
        - Organization settings update
        - Organization subscription changes
        - Organization-wide permission changes
        """
        patterns = [
            f"org_settings:{organization_id}",
            f"user_permissions:{organization_id}:*",
            f"group_memberships:{organization_id}:*",
            f"group_permissions:{organization_id}:*",
        ]
        
        await self._invalidate_by_patterns(patterns, organization_id)
        
        # Queue for global invalidation
        self.invalidation_queue.append({
            'type': 'organization_data',
            'organization_id': organization_id,
            'timestamp': time.time()
        })
    
    async def warm_cache_for_bulk_import(self, organization_id: str, user_ids: List[str]):
        """
        Warm cache for bulk import operations to optimize performance.
        
        Pre-loads frequently accessed data:
        - Organization settings
        - Group hierarchies
        - Permission templates
        """
        logger.info(f"Warming cache for bulk import: org={organization_id}, users={len(user_ids)}")
        
        # Warm organization settings
        await self.get_organization_settings(organization_id)
        
        # Pre-compute permission hierarchies
        permission_hierarchy = await self._compute_permission_hierarchy(organization_id)
        cache_key = f"permission_hierarchy:{organization_id}"
        await self._multi_layer_store(
            cache_key, permission_hierarchy,
            UserManagementDataType.PERMISSION_HIERARCHY,
            organization_id=organization_id,
            l1_ttl=3600, l2_ttl=7200, l3_ttl=14400
        )
        
        # Pre-load group hierarchies
        group_hierarchy = await self._compute_group_hierarchy(organization_id)
        cache_key = f"group_hierarchy:{organization_id}"
        await self._multi_layer_store(
            cache_key, group_hierarchy,
            UserManagementDataType.GROUP_HIERARCHY,
            organization_id=organization_id,
            l1_ttl=3600, l2_ttl=7200, l3_ttl=14400
        )
        
        logger.info(f"Cache warming completed for organization {organization_id}")
    
    # Core cache layer operations
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get value from L1 (local) cache"""
        entry = self.l1_cache.get(key)
        if entry:
            # Check TTL
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                del self.l1_cache[key]
                return None
            
            # Update access statistics
            entry.accessed_count += 1
            entry.last_accessed = time.time()
            return entry.value
        
        return None
    
    async def _store_in_l1(self, key: str, value: Any, data_type: UserManagementDataType,
                         organization_id: Optional[str] = None, user_id: Optional[str] = None,
                         group_id: Optional[str] = None, ttl: Optional[int] = None):
        """Store value in L1 cache with metadata"""
        entry = CacheEntry(
            key=key, value=value, data_type=data_type,
            organization_id=organization_id, user_id=user_id,
            group_id=group_id, ttl=ttl
        )
        
        self.l1_cache[key] = entry
        
        # Implement LRU eviction if cache is full
        max_entries = self.l1_config.get('max_entries', 10000)
        if len(self.l1_cache) > max_entries:
            await self._evict_l1_entries(max_entries // 10)  # Evict 10%
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 (regional Redis) cache"""
        if not self.l2_redis_pool:
            return None
        
        try:
            redis_client = redis.Redis(connection_pool=self.l2_redis_pool)
            value = await redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"L2 cache get error for key {key}: {e}")
        
        return None
    
    async def _store_in_l2(self, key: str, value: Any, ttl: int = 3600):
        """Store value in L2 (regional Redis) cache"""
        if not self.l2_redis_pool:
            return
        
        try:
            redis_client = redis.Redis(connection_pool=self.l2_redis_pool)
            await redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"L2 cache store error for key {key}: {e}")
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 (global Redis) cache"""
        if not self.l3_redis_pool:
            return None
        
        try:
            redis_client = redis.Redis(connection_pool=self.l3_redis_pool)
            value = await redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"L3 cache get error for key {key}: {e}")
        
        return None
    
    async def _store_in_l3(self, key: str, value: Any, ttl: int = 7200):
        """Store value in L3 (global Redis) cache"""
        if not self.l3_redis_pool:
            return
        
        try:
            redis_client = redis.Redis(connection_pool=self.l3_redis_pool)
            await redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"L3 cache store error for key {key}: {e}")
    
    async def _multi_layer_get(self, key: str, data_type: UserManagementDataType,
                             organization_id: Optional[str] = None,
                             user_id: Optional[str] = None) -> Optional[Any]:
        """Get value using multi-layer cache strategy"""
        # Try L1 first
        value = await self._get_from_l1(key)
        if value is not None:
            return value
        
        # Try L2
        value = await self._get_from_l2(key)
        if value is not None:
            # Store in L1 for next access
            await self._store_in_l1(key, value, data_type, organization_id, user_id, ttl=300)
            return value
        
        # Try L3 for global data
        if data_type in [UserManagementDataType.ORGANIZATION_SETTINGS, 
                        UserManagementDataType.USER_PROFILE]:
            value = await self._get_from_l3(key)
            if value is not None:
                # Store in L2 and L1
                await self._store_in_l2(key, value, ttl=1800)
                await self._store_in_l1(key, value, data_type, organization_id, user_id, ttl=300)
                return value
        
        return None
    
    async def _multi_layer_store(self, key: str, value: Any, data_type: UserManagementDataType,
                               organization_id: Optional[str] = None, user_id: Optional[str] = None,
                               group_id: Optional[str] = None,
                               l1_ttl: int = 300, l2_ttl: int = 1800, l3_ttl: int = 7200):
        """Store value across appropriate cache layers"""
        # Always store in L1
        await self._store_in_l1(key, value, data_type, organization_id, user_id, group_id, l1_ttl)
        
        # Store in L2 for regional access
        await self._store_in_l2(key, value, l2_ttl)
        
        # Store in L3 for global data types
        if data_type in [UserManagementDataType.ORGANIZATION_SETTINGS,
                        UserManagementDataType.USER_PROFILE,
                        UserManagementDataType.SSO_IDENTITY]:
            await self._store_in_l3(key, value, l3_ttl)
    
    # Database integration methods (placeholders)
    
    async def _compute_user_permissions(self, user_id: str, organization_id: str,
                                      group_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Compute user permissions with role hierarchy resolution"""
        # Placeholder - would integrate with database and business logic
        return {
            'user_id': user_id,
            'organization_id': organization_id,
            'permissions': ['read', 'write'],
            'roles': ['learner'],
            'computed_at': time.time()
        }
    
    async def _fetch_group_memberships_from_db(self, user_id: str, organization_id: str) -> List[Dict[str, Any]]:
        """Fetch group memberships from database"""
        # Placeholder - would integrate with database
        return [
            {
                'group_id': 'group_1',
                'role': 'learner',
                'joined_at': '2024-01-01',
                'progress': 75.0
            }
        ]
    
    async def _fetch_organization_settings_from_db(self, organization_id: str) -> Dict[str, Any]:
        """Fetch organization settings from database"""
        # Placeholder - would integrate with database
        return {
            'organization_id': organization_id,
            'settings': {'sso_enabled': True},
            'subscription_tier': 'enterprise'
        }
    
    async def _fetch_sso_identity_from_db(self, provider: str, provider_user_id: str,
                                        organization_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch SSO identity from database"""
        # Placeholder - would integrate with database
        return {
            'provider': provider,
            'provider_user_id': provider_user_id,
            'user_id': 'user_123',
            'organization_id': organization_id
        }
    
    async def _fetch_bulk_import_status_from_db(self, import_job_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        """Fetch bulk import status from database"""
        # Placeholder - would integrate with database
        return {
            'import_job_id': import_job_id,
            'status': 'processing',
            'progress': 45.0,
            'organization_id': organization_id
        }
    
    async def _compute_permission_hierarchy(self, organization_id: str) -> Dict[str, Any]:
        """Compute permission hierarchy for organization"""
        # Placeholder - would compute role hierarchies and inheritance rules
        return {
            'organization_id': organization_id,
            'hierarchy': {
                'owner': ['admin', 'creator', 'moderator', 'learner'],
                'admin': ['creator', 'moderator', 'learner'],
                'creator': ['moderator', 'learner'],
                'moderator': ['learner']
            }
        }
    
    async def _compute_group_hierarchy(self, organization_id: str) -> Dict[str, Any]:
        """Compute group hierarchy for organization"""
        # Placeholder - would compute group parent/child relationships
        return {
            'organization_id': organization_id,
            'groups': {
                'parent_group_1': ['child_group_1', 'child_group_2']
            }
        }
    
    # Cache maintenance and monitoring
    
    async def _cache_maintenance_loop(self):
        """Background task for cache maintenance"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Evict expired L1 entries
            await self._evict_expired_l1_entries()
            
            # Update cache statistics
            await self._update_cache_stats()
            
            # Optimize cache performance
            await self._optimize_cache_performance()
    
    async def _invalidation_processing_loop(self):
        """Process invalidation queue for cross-region consistency"""
        while True:
            await asyncio.sleep(10)  # Every 10 seconds
            
            if self.invalidation_queue:
                batch = self.invalidation_queue[:100]  # Process in batches
                self.invalidation_queue = self.invalidation_queue[100:]
                
                for invalidation in batch:
                    await self._process_cross_region_invalidation(invalidation)
    
    async def _stats_collection_loop(self):
        """Collect and update cache statistics"""
        while True:
            await asyncio.sleep(60)  # Every minute
            await self._collect_cache_statistics()
    
    async def _evict_l1_entries(self, count: int):
        """Evict least recently used L1 cache entries"""
        if len(self.l1_cache) <= count:
            return
        
        # Sort by last accessed time
        entries = sorted(self.l1_cache.items(), 
                        key=lambda x: x[1].last_accessed)
        
        for i in range(count):
            key = entries[i][0]
            del self.l1_cache[key]
    
    async def _evict_expired_l1_entries(self):
        """Remove expired entries from L1 cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.l1_cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.l1_cache[key]
    
    async def _invalidate_by_patterns(self, patterns: List[str], organization_id: str,
                                   user_id: Optional[str] = None, group_id: Optional[str] = None):
        """Invalidate cache entries matching patterns"""
        # L1 invalidation
        keys_to_remove = []
        for key in self.l1_cache:
            for pattern in patterns:
                if self._key_matches_pattern(key, pattern):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.l1_cache[key]
        
        # L2 and L3 invalidation would use Redis pattern matching
        # Implementation depends on Redis cluster configuration
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches invalidation pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def _process_cross_region_invalidation(self, invalidation: Dict[str, Any]):
        """Process invalidation across regions"""
        # Placeholder - would publish invalidation messages to other regions
        logger.debug(f"Processing cross-region invalidation: {invalidation}")
    
    async def _update_cache_stats(self):
        """Update cache layer statistics"""
        # Update L1 stats
        self.stats[CacheLayer.L1_LOCAL].entry_count = len(self.l1_cache)
        
        # Calculate memory usage (rough estimate)
        memory_usage = sum(len(str(entry.value)) for entry in self.l1_cache.values()) / (1024 * 1024)
        self.stats[CacheLayer.L1_LOCAL].memory_usage_mb = memory_usage
        
        # Update data type distribution
        type_counts = {}
        for entry in self.l1_cache.values():
            type_counts[entry.data_type] = type_counts.get(entry.data_type, 0) + 1
        self.stats[CacheLayer.L1_LOCAL].data_type_distribution = type_counts
    
    async def _collect_cache_statistics(self):
        """Collect comprehensive cache statistics"""
        await self._update_cache_stats()
        
        # Log cache performance
        l1_stats = self.stats[CacheLayer.L1_LOCAL]
        logger.debug(f"L1 Cache: {l1_stats.entry_count} entries, "
                    f"{l1_stats.memory_usage_mb:.2f}MB, "
                    f"{l1_stats.hit_rate:.2f}% hit rate")
    
    async def _optimize_cache_performance(self):
        """Optimize cache performance based on access patterns"""
        # Identify hot keys that should be promoted to L1
        # Adjust TTLs based on access patterns
        # Rebalance cache distribution
        pass
    
    def get_cache_stats(self) -> Dict[CacheLayer, CacheStats]:
        """Get comprehensive cache statistics"""
        return self.stats