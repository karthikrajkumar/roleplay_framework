"""
Geo-Distributed Replication with Conflict Resolution

Advanced replication system for global deployment with intelligent conflict resolution,
adaptive consistency models, and optimized cross-region synchronization.

Key Features:
- Multi-master replication with conflict-free resolution
- Adaptive consistency models (strong, eventual, causal)
- Geo-aware replication topology optimization
- Vector clock-based conflict detection and resolution
- Bandwidth-efficient delta synchronization
- Automatic partition tolerance and recovery
"""

from .geo_replication_manager import GeoReplicationManager
from .conflict_resolver import ConflictResolver
from .vector_clock import VectorClockManager
from .delta_sync import DeltaSyncEngine
from .consistency_manager import ConsistencyManager
from .topology_optimizer import TopologyOptimizer

__all__ = [
    'GeoReplicationManager',
    'ConflictResolver',
    'VectorClockManager',
    'DeltaSyncEngine',
    'ConsistencyManager',
    'TopologyOptimizer'
]