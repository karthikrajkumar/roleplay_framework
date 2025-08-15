"""
Intelligent Cache Manager

Multi-layer distributed caching system with machine learning-powered optimization,
adaptive cache warming, and intelligent data placement for roleplay platforms.

Features:
- Three-tier cache hierarchy (L1: Local, L2: Regional, L3: Global)
- ML-powered access pattern prediction and prefetching
- Adaptive eviction policies based on data characteristics
- Geo-aware cache placement and replication
- Real-time cache coherence and consistency
- Smart cache warming for AI models and user data
"""

import asyncio
import time
import json
import hashlib
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer types"""
    L1_LOCAL = "l1_local"        # Local in-memory cache
    L2_REGIONAL = "l2_regional"  # Regional Redis cluster
    L3_GLOBAL = "l3_global"      # Global distributed cache


class DataType(Enum):
    """Types of data for cache optimization"""
    AI_MODEL = "ai_model"
    USER_PROFILE = "user_profile"
    SCENARIO_DATA = "scenario_data"
    INTERACTION_HISTORY = "interaction_history"
    MEDIA_CONTENT = "media_content"
    REAL_TIME_STATE = "real_time_state"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    data_type: DataType
    size_bytes: int
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_pattern: List[float] = field(default_factory=list)
    importance_score: float = 1.0
    geographic_affinity: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    average_response_time: float = 0.0
    memory_utilization: float = 0.0
    network_efficiency: float = 0.0
    prediction_accuracy: float = 0.0


@dataclass
class AccessPattern:
    """User access pattern for ML prediction"""
    user_id: str
    access_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    accessed_keys: deque = field(default_factory=lambda: deque(maxlen=1000))
    session_patterns: List[List[str]] = field(default_factory=list)
    geographic_region: str = ""
    device_type: str = ""


class IntelligentCacheManager:
    """
    Multi-layer intelligent cache manager with ML-powered optimization
    for massive scalability and sub-100ms latency requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.node_id = config['node_id']
        self.region = config['region']
        
        # Cache layers
        self.cache_layers: Dict[CacheLayer, Dict[str, CacheEntry]] = {
            CacheLayer.L1_LOCAL: {},
            CacheLayer.L2_REGIONAL: {},
            CacheLayer.L3_GLOBAL: {}
        }
        
        # Cache configuration
        self.layer_config = {
            CacheLayer.L1_LOCAL: {
                'max_size_mb': config.get('l1_max_size_mb', 512),
                'ttl_seconds': config.get('l1_ttl_seconds', 300),
                'eviction_policy': 'adaptive_lru'
            },
            CacheLayer.L2_REGIONAL: {
                'max_size_mb': config.get('l2_max_size_mb', 8192),
                'ttl_seconds': config.get('l2_ttl_seconds', 3600),
                'eviction_policy': 'adaptive_lfu'
            },
            CacheLayer.L3_GLOBAL: {
                'max_size_mb': config.get('l3_max_size_mb', 32768),
                'ttl_seconds': config.get('l3_ttl_seconds', 86400),
                'eviction_policy': 'ml_optimized'
            }
        }
        
        # Performance tracking
        self.metrics: Dict[CacheLayer, CacheMetrics] = {
            layer: CacheMetrics() for layer in CacheLayer
        }
        
        # Access pattern tracking
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.global_access_patterns = deque(maxlen=10000)
        
        # ML components
        self.access_predictor = AccessPatternPredictor()
        self.cache_optimizer = CacheOptimizer()
        self.prefetch_engine = PrefetchEngine()
        
        # Cache warming
        self.warming_queue: asyncio.Queue = asyncio.Queue()
        self.is_warming = False
        
        # Performance optimization
        self.background_tasks: List[asyncio.Task] = []
        
        # Initialize components
        self._initialize_predictive_models()
        
        # Start background tasks
        self._start_background_tasks()
    
    async def get(self, key: str, user_id: str = None, 
                  region: str = None) -> Tuple[Any, CacheLayer]:
        """
        Intelligent cache get with multi-layer lookup and prediction
        
        Returns:
            Tuple of (value, cache_layer_found)
        """
        start_time = time.time()
        
        # Record access pattern
        if user_id:
            await self._record_access(user_id, key, region or self.region)
        
        # L1 Local cache lookup
        if key in self.cache_layers[CacheLayer.L1_LOCAL]:
            entry = self.cache_layers[CacheLayer.L1_LOCAL][key]
            if not self._is_expired(entry):
                await self._update_access_stats(entry, CacheLayer.L1_LOCAL)
                await self._trigger_predictive_prefetch(key, user_id)
                return entry.value, CacheLayer.L1_LOCAL
            else:
                del self.cache_layers[CacheLayer.L1_LOCAL][key]
        
        # L2 Regional cache lookup
        if key in self.cache_layers[CacheLayer.L2_REGIONAL]:
            entry = self.cache_layers[CacheLayer.L2_REGIONAL][key]
            if not self._is_expired(entry):
                await self._update_access_stats(entry, CacheLayer.L2_REGIONAL)
                # Promote to L1 if frequently accessed
                if entry.access_count > 5:
                    await self._promote_to_l1(key, entry)
                await self._trigger_predictive_prefetch(key, user_id)
                return entry.value, CacheLayer.L2_REGIONAL
            else:
                del self.cache_layers[CacheLayer.L2_REGIONAL][key]
        
        # L3 Global cache lookup
        if key in self.cache_layers[CacheLayer.L3_GLOBAL]:
            entry = self.cache_layers[CacheLayer.L3_GLOBAL][key]
            if not self._is_expired(entry):
                await self._update_access_stats(entry, CacheLayer.L3_GLOBAL)
                # Promote to L2 if regionally relevant
                if self.region in entry.geographic_affinity:
                    await self._promote_to_l2(key, entry)
                await self._trigger_predictive_prefetch(key, user_id)
                return entry.value, CacheLayer.L3_GLOBAL
            else:
                del self.cache_layers[CacheLayer.L3_GLOBAL][key]
        
        # Cache miss - record for learning
        await self._record_cache_miss(key, user_id, region)
        
        # Update metrics
        response_time = time.time() - start_time
        await self._update_metrics(None, response_time, cache_miss=True)
        
        return None, None
    
    async def put(self, key: str, value: Any, data_type: DataType, 
                  user_id: str = None, ttl: float = None, 
                  geographic_affinity: List[str] = None) -> bool:
        """
        Intelligent cache put with optimal layer placement
        
        Returns:
            Success status
        """
        try:
            # Determine optimal cache layer based on data characteristics
            optimal_layer = await self._determine_optimal_layer(
                key, value, data_type, user_id, geographic_affinity
            )
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                data_type=data_type,
                size_bytes=self._estimate_size(value),
                ttl=ttl or self.layer_config[optimal_layer]['ttl_seconds'],
                geographic_affinity=geographic_affinity or [self.region],
                importance_score=self._calculate_importance_score(data_type, user_id)
            )
            
            # Place in optimal layer
            await self._place_in_layer(optimal_layer, key, entry)
            
            # Replicate to other layers if beneficial
            await self._intelligent_replication(key, entry, optimal_layer)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache put failed for key {key}: {e}")
            return False
    
    async def invalidate(self, key: str, propagate: bool = True) -> bool:
        """
        Invalidate cache entry across all layers
        
        Args:
            key: Cache key to invalidate
            propagate: Whether to propagate invalidation to other nodes
        """
        invalidated = False
        
        for layer in CacheLayer:
            if key in self.cache_layers[layer]:
                del self.cache_layers[layer][key]
                invalidated = True
        
        # Propagate invalidation if needed
        if propagate and invalidated:
            await self._propagate_invalidation(key)
        
        return invalidated
    
    async def warm_cache(self, warming_spec: Dict[str, Any]) -> bool:
        """
        Intelligent cache warming based on predicted access patterns
        
        Args:
            warming_spec: Specification for cache warming including data types,
                         user segments, geographic regions, and priority
        """
        await self.warming_queue.put(warming_spec)
        
        if not self.is_warming:
            asyncio.create_task(self._process_warming_queue())
        
        return True
    
    async def _determine_optimal_layer(self, key: str, value: Any, 
                                      data_type: DataType, user_id: str = None,
                                      geographic_affinity: List[str] = None) -> CacheLayer:
        """Determine optimal cache layer using ML predictions"""
        
        # Real-time data goes to L1
        if data_type == DataType.REAL_TIME_STATE:
            return CacheLayer.L1_LOCAL
        
        # AI models typically go to L2/L3 for sharing
        if data_type == DataType.AI_MODEL:
            return CacheLayer.L2_REGIONAL
        
        # User-specific data analysis
        if user_id and user_id in self.access_patterns:
            pattern = self.access_patterns[user_id]
            
            # Frequently accessed user data goes to L1
            if len(pattern.access_times) > 10:
                recent_access_frequency = len([
                    t for t in pattern.access_times 
                    if time.time() - t < 300  # Last 5 minutes
                ])
                
                if recent_access_frequency > 5:
                    return CacheLayer.L1_LOCAL
        
        # Geographic affinity analysis
        if geographic_affinity:
            if self.region in geographic_affinity:
                return CacheLayer.L2_REGIONAL
            else:
                return CacheLayer.L3_GLOBAL
        
        # Default based on data type
        data_type_defaults = {
            DataType.USER_PROFILE: CacheLayer.L2_REGIONAL,
            DataType.SCENARIO_DATA: CacheLayer.L3_GLOBAL,
            DataType.INTERACTION_HISTORY: CacheLayer.L2_REGIONAL,
            DataType.MEDIA_CONTENT: CacheLayer.L3_GLOBAL
        }
        
        return data_type_defaults.get(data_type, CacheLayer.L2_REGIONAL)
    
    async def _place_in_layer(self, layer: CacheLayer, key: str, entry: CacheEntry):
        """Place entry in specified cache layer with eviction if needed"""
        
        # Check if eviction is needed
        if await self._needs_eviction(layer, entry.size_bytes):
            await self._evict_entries(layer, entry.size_bytes)
        
        # Place entry
        self.cache_layers[layer][key] = entry
        
        logger.debug(f"Placed {key} in {layer.value}")
    
    async def _needs_eviction(self, layer: CacheLayer, new_entry_size: int) -> bool:
        """Check if eviction is needed for new entry"""
        max_size_bytes = self.layer_config[layer]['max_size_mb'] * 1024 * 1024
        current_size = sum(entry.size_bytes for entry in self.cache_layers[layer].values())
        
        return (current_size + new_entry_size) > max_size_bytes
    
    async def _evict_entries(self, layer: CacheLayer, required_space: int):
        """Evict entries using adaptive policy"""
        eviction_policy = self.layer_config[layer]['eviction_policy']
        
        if eviction_policy == 'adaptive_lru':
            await self._adaptive_lru_eviction(layer, required_space)
        elif eviction_policy == 'adaptive_lfu':
            await self._adaptive_lfu_eviction(layer, required_space)
        elif eviction_policy == 'ml_optimized':
            await self._ml_optimized_eviction(layer, required_space)
    
    async def _adaptive_lru_eviction(self, layer: CacheLayer, required_space: int):
        """Adaptive LRU eviction considering data importance"""
        candidates = list(self.cache_layers[layer].values())
        
        # Sort by last access time and importance score
        candidates.sort(key=lambda e: (e.last_access_time, -e.importance_score))
        
        freed_space = 0
        for entry in candidates:
            if freed_space >= required_space:
                break
            
            del self.cache_layers[layer][entry.key]
            freed_space += entry.size_bytes
            
            # Update eviction metrics
            self.metrics[layer].eviction_rate += 1
    
    async def _ml_optimized_eviction(self, layer: CacheLayer, required_space: int):
        """ML-optimized eviction using access prediction"""
        candidates = list(self.cache_layers[layer].values())
        
        # Get eviction scores from ML model
        eviction_scores = []
        for entry in candidates:
            score = await self.cache_optimizer.calculate_eviction_score(entry)
            eviction_scores.append((score, entry))
        
        # Sort by eviction score (higher score = more likely to evict)
        eviction_scores.sort(key=lambda x: -x[0])
        
        freed_space = 0
        for score, entry in eviction_scores:
            if freed_space >= required_space:
                break
            
            del self.cache_layers[layer][entry.key]
            freed_space += entry.size_bytes
    
    async def _intelligent_replication(self, key: str, entry: CacheEntry, 
                                      primary_layer: CacheLayer):
        """Intelligently replicate entry to other layers"""
        
        # High importance entries get replicated
        if entry.importance_score > 0.8:
            # Replicate to L1 if not already there
            if primary_layer != CacheLayer.L1_LOCAL:
                if not await self._needs_eviction(CacheLayer.L1_LOCAL, entry.size_bytes):
                    self.cache_layers[CacheLayer.L1_LOCAL][key] = entry
        
        # Geographically relevant entries get regional replication
        if len(entry.geographic_affinity) > 1 and primary_layer == CacheLayer.L3_GLOBAL:
            if not await self._needs_eviction(CacheLayer.L2_REGIONAL, entry.size_bytes):
                self.cache_layers[CacheLayer.L2_REGIONAL][key] = entry
    
    async def _trigger_predictive_prefetch(self, accessed_key: str, user_id: str = None):
        """Trigger predictive prefetching based on access patterns"""
        if user_id and user_id in self.access_patterns:
            predicted_keys = await self.prefetch_engine.predict_next_accesses(
                user_id, accessed_key, self.access_patterns[user_id]
            )
            
            for key in predicted_keys:
                if key not in self.cache_layers[CacheLayer.L1_LOCAL]:
                    # Queue for prefetching
                    asyncio.create_task(self._prefetch_key(key, user_id))
    
    async def _record_access(self, user_id: str, key: str, region: str):
        """Record access pattern for ML learning"""
        if user_id not in self.access_patterns:
            self.access_patterns[user_id] = AccessPattern(
                user_id=user_id,
                geographic_region=region
            )
        
        pattern = self.access_patterns[user_id]
        pattern.access_times.append(time.time())
        pattern.accessed_keys.append(key)
        
        # Add to global patterns for aggregate learning
        self.global_access_patterns.append({
            'user_id': user_id,
            'key': key,
            'timestamp': time.time(),
            'region': region
        })
    
    def _calculate_importance_score(self, data_type: DataType, user_id: str = None) -> float:
        """Calculate importance score for cache entry"""
        base_scores = {
            DataType.AI_MODEL: 0.9,
            DataType.USER_PROFILE: 0.8,
            DataType.REAL_TIME_STATE: 1.0,
            DataType.SCENARIO_DATA: 0.7,
            DataType.INTERACTION_HISTORY: 0.6,
            DataType.MEDIA_CONTENT: 0.5
        }
        
        score = base_scores.get(data_type, 0.5)
        
        # Boost score for frequent users
        if user_id and user_id in self.access_patterns:
            pattern = self.access_patterns[user_id]
            if len(pattern.access_times) > 100:
                score *= 1.2
        
        return min(score, 1.0)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl is None:
            return False
        
        return time.time() - entry.creation_time > entry.ttl
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cache value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode('utf-8') if isinstance(value, str) else value)
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default 1KB estimate
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        self.background_tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._pattern_analyzer()),
            asyncio.create_task(self._cache_optimizer_loop()),
            asyncio.create_task(self._ttl_cleanup())
        ]
    
    async def _metrics_collector(self):
        """Collect and update cache metrics"""
        while True:
            await asyncio.sleep(10.0)  # Update every 10 seconds
            
            for layer in CacheLayer:
                await self._calculate_layer_metrics(layer)
    
    async def _calculate_layer_metrics(self, layer: CacheLayer):
        """Calculate metrics for a cache layer"""
        cache = self.cache_layers[layer]
        
        if not cache:
            return
        
        # Memory utilization
        max_size_bytes = self.layer_config[layer]['max_size_mb'] * 1024 * 1024
        current_size = sum(entry.size_bytes for entry in cache.values())
        self.metrics[layer].memory_utilization = current_size / max_size_bytes
        
        # Access patterns
        total_accesses = sum(entry.access_count for entry in cache.values())
        if total_accesses > 0:
            self.metrics[layer].average_response_time = statistics.mean([
                entry.last_access_time - entry.creation_time 
                for entry in cache.values() if entry.access_count > 0
            ])
    
    async def _ttl_cleanup(self):
        """Clean up expired entries"""
        while True:
            await asyncio.sleep(60.0)  # Check every minute
            
            for layer in CacheLayer:
                expired_keys = []
                for key, entry in self.cache_layers[layer].items():
                    if self._is_expired(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache_layers[layer][key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'layers': {},
            'access_patterns': len(self.access_patterns),
            'global_patterns': len(self.global_access_patterns)
        }
        
        for layer in CacheLayer:
            cache = self.cache_layers[layer]
            stats['layers'][layer.value] = {
                'entries': len(cache),
                'total_size_mb': sum(entry.size_bytes for entry in cache.values()) / (1024 * 1024),
                'metrics': {
                    'hit_rate': self.metrics[layer].hit_rate,
                    'memory_utilization': self.metrics[layer].memory_utilization,
                    'eviction_rate': self.metrics[layer].eviction_rate
                }
            }
        
        return stats


class AccessPatternPredictor:
    """ML-based access pattern predictor"""
    
    def __init__(self):
        self.pattern_history: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    
    async def predict_next_access(self, user_id: str, current_key: str) -> List[str]:
        """Predict next likely cache accesses"""
        # Simplified pattern matching - in production, use sophisticated ML
        if user_id not in self.pattern_history:
            return []
        
        history = self.pattern_history[user_id]
        
        # Find patterns where current_key was accessed
        next_keys = []
        for i, (timestamp, key) in enumerate(history[:-1]):
            if key == current_key and i + 1 < len(history):
                next_keys.append(history[i + 1][1])
        
        # Return most common next keys
        from collections import Counter
        common_next = Counter(next_keys).most_common(3)
        return [key for key, count in common_next if count > 1]


class CacheOptimizer:
    """ML-based cache optimization"""
    
    async def calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (higher = more likely to evict)"""
        age_factor = (time.time() - entry.last_access_time) / 3600  # Hours since last access
        size_factor = entry.size_bytes / (1024 * 1024)  # Size in MB
        importance_factor = 1.0 - entry.importance_score
        access_factor = 1.0 / max(entry.access_count, 1)
        
        return age_factor * 0.4 + size_factor * 0.2 + importance_factor * 0.3 + access_factor * 0.1


class PrefetchEngine:
    """Intelligent prefetching engine"""
    
    async def predict_next_accesses(self, user_id: str, current_key: str, 
                                   pattern: AccessPattern) -> List[str]:
        """Predict next cache accesses for prefetching"""
        if len(pattern.accessed_keys) < 3:
            return []
        
        # Simple sequence prediction
        recent_keys = list(pattern.accessed_keys)[-10:]
        
        if current_key in recent_keys:
            idx = recent_keys.index(current_key)
            if idx + 1 < len(recent_keys):
                return [recent_keys[idx + 1]]
        
        return []