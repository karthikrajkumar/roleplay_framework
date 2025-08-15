"""
Multi-Layer Intelligent Distributed Caching System

Advanced caching architecture designed for massive scalability with Redis clusters,
intelligent cache warming, predictive prefetching, and adaptive eviction policies.

Key Features:
- Hierarchical cache layers (L1: Local, L2: Regional, L3: Global)
- ML-powered cache warming and prefetching
- Adaptive eviction policies based on access patterns
- Geo-distributed cache replication with consistency guarantees
- Real-time cache coherence and invalidation
- Smart data partitioning across cache clusters
"""

from .intelligent_cache_manager import IntelligentCacheManager
from .redis_cluster_manager import RedisClusterManager
from .cache_warming_engine import CacheWarmingEngine
from .prefetch_predictor import PrefetchPredictor
from .eviction_policies import AdaptiveEvictionPolicy
from .cache_coherence import CacheCoherenceManager
from .cache_partitioner import CachePartitioner

__all__ = [
    'IntelligentCacheManager',
    'RedisClusterManager',
    'CacheWarmingEngine',
    'PrefetchPredictor',
    'AdaptiveEvictionPolicy',
    'CacheCoherenceManager',
    'CachePartitioner'
]