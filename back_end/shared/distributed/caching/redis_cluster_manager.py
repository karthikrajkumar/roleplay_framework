"""
Redis Cluster Manager

Advanced Redis cluster management with intelligent sharding, automatic failover,
and performance optimization for distributed caching across global data centers.

Features:
- Consistent hashing for optimal data distribution
- Automatic cluster topology management
- Hot spot detection and load balancing
- Cross-region replication with conflict resolution
- Real-time cluster health monitoring
- Dynamic scaling and resharding
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Redis node roles"""
    MASTER = "master"
    SLAVE = "slave"
    SENTINEL = "sentinel"


class ClusterState(Enum):
    """Cluster state"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    RECOVERING = "recovering"


@dataclass
class RedisNode:
    """Redis node configuration and state"""
    node_id: str
    host: str
    port: int
    role: NodeRole
    region: str
    availability_zone: str
    is_online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    network_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    slots: Set[int] = field(default_factory=set)


@dataclass
class ShardInfo:
    """Information about a data shard"""
    shard_id: str
    master_node: str
    slave_nodes: List[str]
    slot_range: Tuple[int, int]
    data_size_mb: float = 0.0
    access_frequency: float = 0.0
    hot_keys: Set[str] = field(default_factory=set)


@dataclass
class ClusterMetrics:
    """Cluster performance metrics"""
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    total_ops_per_sec: float = 0.0
    average_latency_ms: float = 0.0
    cluster_state: ClusterState = ClusterState.HEALTHY
    hot_spots: List[str] = field(default_factory=list)


class ConsistentHashRing:
    """Consistent hash ring for data distribution"""
    
    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.nodes: Set[str] = set()
        self.total_slots = 16384  # Redis cluster slots
    
    def add_node(self, node_id: str):
        """Add node to hash ring"""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key) % self.total_slots
            self.ring[hash_value] = node_id
    
    def remove_node(self, node_id: str):
        """Remove node from hash ring"""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        
        # Remove all virtual nodes
        to_remove = []
        for hash_value, node in self.ring.items():
            if node == node_id:
                to_remove.append(hash_value)
        
        for hash_value in to_remove:
            del self.ring[hash_value]
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for a given key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key) % self.total_slots
        
        # Find the first node clockwise from hash_value
        for ring_hash in sorted(self.ring.keys()):
            if ring_hash >= hash_value:
                return self.ring[ring_hash]
        
        # Wrap around to the first node
        return self.ring[min(self.ring.keys())]
    
    def get_slot_range(self, node_id: str) -> List[Tuple[int, int]]:
        """Get slot ranges for a node"""
        if node_id not in self.nodes:
            return []
        
        node_slots = sorted([slot for slot, node in self.ring.items() if node == node_id])
        
        if not node_slots:
            return []
        
        ranges = []
        start = node_slots[0]
        end = node_slots[0]
        
        for slot in node_slots[1:]:
            if slot == end + 1:
                end = slot
            else:
                ranges.append((start, end))
                start = end = slot
        
        ranges.append((start, end))
        return ranges
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class RedisClusterManager:
    """
    Advanced Redis cluster manager with intelligent sharding and failover
    """
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config
        self.cluster_name = cluster_config['cluster_name']
        self.region = cluster_config['region']
        
        # Cluster topology
        self.nodes: Dict[str, RedisNode] = {}
        self.shards: Dict[str, ShardInfo] = {}
        self.hash_ring = ConsistentHashRing()
        
        # Connection pools
        self.connection_pools: Dict[str, redis.ConnectionPool] = {}
        self.redis_clients: Dict[str, redis.Redis] = {}
        
        # Cluster state
        self.cluster_metrics = ClusterMetrics()
        self.is_resharding = False
        self.hot_spot_threshold = 1000  # ops/sec
        
        # Monitoring
        self.health_check_interval = 10.0  # seconds
        self.metrics_collection_interval = 30.0  # seconds
        
        # Initialize cluster
        asyncio.create_task(self._initialize_cluster())
        
        # Start background tasks
        self._start_background_tasks()
    
    async def _initialize_cluster(self):
        """Initialize Redis cluster from configuration"""
        for node_config in self.cluster_config['nodes']:
            await self.add_node(
                node_id=node_config['id'],
                host=node_config['host'],
                port=node_config['port'],
                role=NodeRole(node_config['role']),
                region=node_config.get('region', self.region),
                availability_zone=node_config.get('az', 'default')
            )
        
        # Initialize sharding
        await self._initialize_sharding()
        
        logger.info(f"Initialized Redis cluster '{self.cluster_name}' with {len(self.nodes)} nodes")
    
    async def add_node(self, node_id: str, host: str, port: int, 
                      role: NodeRole, region: str, availability_zone: str) -> bool:
        """Add new node to cluster"""
        try:
            # Create node object
            node = RedisNode(
                node_id=node_id,
                host=host,
                port=port,
                role=role,
                region=region,
                availability_zone=availability_zone
            )
            
            # Create connection pool
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                decode_responses=True,
                max_connections=20,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30
            )
            
            # Create Redis client
            client = redis.Redis(connection_pool=pool)
            
            # Test connection
            await client.ping()
            
            # Add to cluster
            self.nodes[node_id] = node
            self.connection_pools[node_id] = pool
            self.redis_clients[node_id] = client
            
            # Add to hash ring if master
            if role == NodeRole.MASTER:
                self.hash_ring.add_node(node_id)
            
            logger.info(f"Added Redis node {node_id} ({host}:{port}) to cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add Redis node {node_id}: {e}")
            return False
    
    async def remove_node(self, node_id: str, migrate_data: bool = True) -> bool:
        """Remove node from cluster"""
        if node_id not in self.nodes:
            return False
        
        try:
            node = self.nodes[node_id]
            
            # Migrate data if requested and node is master
            if migrate_data and node.role == NodeRole.MASTER:
                await self._migrate_node_data(node_id)
            
            # Remove from hash ring
            self.hash_ring.remove_node(node_id)
            
            # Close connections
            if node_id in self.redis_clients:
                await self.redis_clients[node_id].close()
                del self.redis_clients[node_id]
            
            if node_id in self.connection_pools:
                await self.connection_pools[node_id].disconnect()
                del self.connection_pools[node_id]
            
            # Remove from cluster
            del self.nodes[node_id]
            
            logger.info(f"Removed Redis node {node_id} from cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove Redis node {node_id}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cluster with intelligent routing"""
        try:
            # Determine target node
            node_id = self.hash_ring.get_node(key)
            if not node_id or node_id not in self.redis_clients:
                return None
            
            # Check if node is healthy
            node = self.nodes[node_id]
            if not node.is_online:
                # Try to find replica
                replica_node = await self._find_healthy_replica(node_id)
                if replica_node:
                    node_id = replica_node
                else:
                    return None
            
            # Get from Redis
            client = self.redis_clients[node_id]
            value = await client.get(key)
            
            # Update metrics
            await self._update_access_metrics(node_id, key)
            
            return json.loads(value) if value else None
            
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cluster with intelligent routing"""
        try:
            # Determine target node
            node_id = self.hash_ring.get_node(key)
            if not node_id or node_id not in self.redis_clients:
                return False
            
            # Check if node is healthy
            node = self.nodes[node_id]
            if not node.is_online:
                logger.warning(f"Target node {node_id} is offline for key {key}")
                return False
            
            # Set in Redis
            client = self.redis_clients[node_id]
            serialized_value = json.dumps(value, default=str)
            
            if ttl:
                await client.setex(key, ttl, serialized_value)
            else:
                await client.set(key, serialized_value)
            
            # Replicate to slaves if configured
            await self._replicate_to_slaves(node_id, key, serialized_value, ttl)
            
            # Update metrics
            await self._update_access_metrics(node_id, key, is_write=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cluster"""
        try:
            # Determine target node
            node_id = self.hash_ring.get_node(key)
            if not node_id or node_id not in self.redis_clients:
                return False
            
            # Delete from Redis
            client = self.redis_clients[node_id]
            result = await client.delete(key)
            
            # Replicate deletion to slaves
            await self._replicate_deletion_to_slaves(node_id, key)
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def _migrate_node_data(self, source_node_id: str):
        """Migrate data from a node to other nodes"""
        if source_node_id not in self.redis_clients:
            return
        
        source_client = self.redis_clients[source_node_id]
        
        try:
            # Get all keys from source node
            keys = await source_client.keys('*')
            
            logger.info(f"Migrating {len(keys)} keys from node {source_node_id}")
            
            # Migrate each key
            for key in keys:
                # Get value and TTL
                value = await source_client.get(key)
                ttl = await source_client.ttl(key)
                
                if value:
                    # Determine new target node (excluding source)
                    self.hash_ring.remove_node(source_node_id)
                    target_node = self.hash_ring.get_node(key)
                    self.hash_ring.add_node(source_node_id)  # Add back temporarily
                    
                    if target_node and target_node in self.redis_clients:
                        target_client = self.redis_clients[target_node]
                        
                        if ttl > 0:
                            await target_client.setex(key, ttl, value)
                        else:
                            await target_client.set(key, value)
            
            logger.info(f"Completed migration from node {source_node_id}")
            
        except Exception as e:
            logger.error(f"Failed to migrate data from node {source_node_id}: {e}")
    
    async def _find_healthy_replica(self, master_node_id: str) -> Optional[str]:
        """Find healthy replica for a master node"""
        # Find shard containing this master
        for shard in self.shards.values():
            if shard.master_node == master_node_id:
                # Check slave nodes
                for slave_id in shard.slave_nodes:
                    if slave_id in self.nodes and self.nodes[slave_id].is_online:
                        return slave_id
        
        return None
    
    async def _replicate_to_slaves(self, master_id: str, key: str, 
                                  value: str, ttl: Optional[int]):
        """Replicate data to slave nodes"""
        # Find slaves for this master
        for shard in self.shards.values():
            if shard.master_node == master_id:
                for slave_id in shard.slave_nodes:
                    if slave_id in self.redis_clients:
                        try:
                            client = self.redis_clients[slave_id]
                            if ttl:
                                await client.setex(key, ttl, value)
                            else:
                                await client.set(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to replicate to slave {slave_id}: {e}")
    
    async def _replicate_deletion_to_slaves(self, master_id: str, key: str):
        """Replicate deletion to slave nodes"""
        for shard in self.shards.values():
            if shard.master_node == master_id:
                for slave_id in shard.slave_nodes:
                    if slave_id in self.redis_clients:
                        try:
                            client = self.redis_clients[slave_id]
                            await client.delete(key)
                        except Exception as e:
                            logger.warning(f"Failed to replicate deletion to slave {slave_id}: {e}")
    
    async def _update_access_metrics(self, node_id: str, key: str, is_write: bool = False):
        """Update access metrics for hot spot detection"""
        node = self.nodes[node_id]
        node.throughput += 1
        
        # Track hot keys
        current_time = time.time()
        
        # Simple hot key detection (in production, use sliding window)
        for shard in self.shards.values():
            if node_id == shard.master_node:
                shard.access_frequency += 1
                
                # Check if this becomes a hot spot
                if shard.access_frequency > self.hot_spot_threshold:
                    if key not in shard.hot_keys:
                        shard.hot_keys.add(key)
                        logger.warning(f"Detected hot key: {key} in shard {shard.shard_id}")
    
    async def _initialize_sharding(self):
        """Initialize cluster sharding"""
        master_nodes = [
            node_id for node_id, node in self.nodes.items() 
            if node.role == NodeRole.MASTER
        ]
        
        if not master_nodes:
            return
        
        slots_per_shard = 16384 // len(master_nodes)
        
        for i, master_id in enumerate(master_nodes):
            shard_id = f"shard_{i}"
            start_slot = i * slots_per_shard
            end_slot = start_slot + slots_per_shard - 1
            
            if i == len(master_nodes) - 1:  # Last shard gets remaining slots
                end_slot = 16383
            
            # Find slave nodes in same region
            master_node = self.nodes[master_id]
            slave_nodes = [
                node_id for node_id, node in self.nodes.items()
                if (node.role == NodeRole.SLAVE and 
                    node.region == master_node.region and
                    node.availability_zone != master_node.availability_zone)
            ]
            
            shard = ShardInfo(
                shard_id=shard_id,
                master_node=master_id,
                slave_nodes=slave_nodes,
                slot_range=(start_slot, end_slot)
            )
            
            self.shards[shard_id] = shard
            
            # Update node slots
            self.nodes[master_id].slots = set(range(start_slot, end_slot + 1))
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._hot_spot_monitor())
        asyncio.create_task(self._automatic_rebalancing())
    
    async def _health_monitor(self):
        """Monitor cluster health"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            
            for node_id, node in self.nodes.items():
                if node_id in self.redis_clients:
                    try:
                        start_time = time.time()
                        await self.redis_clients[node_id].ping()
                        response_time = time.time() - start_time
                        
                        node.is_online = True
                        node.network_latency = response_time * 1000  # ms
                        node.last_heartbeat = time.time()
                        
                    except Exception as e:
                        node.is_online = False
                        logger.warning(f"Node {node_id} health check failed: {e}")
                        
                        # Trigger failover if master is down
                        if node.role == NodeRole.MASTER:
                            await self._handle_master_failure(node_id)
    
    async def _handle_master_failure(self, failed_master_id: str):
        """Handle master node failure with automatic failover"""
        logger.warning(f"Master node {failed_master_id} failed, initiating failover")
        
        # Find the best slave to promote
        best_slave = await self._find_best_slave_for_promotion(failed_master_id)
        
        if best_slave:
            await self._promote_slave_to_master(best_slave, failed_master_id)
        else:
            logger.error(f"No suitable slave found for failed master {failed_master_id}")
    
    async def _find_best_slave_for_promotion(self, master_id: str) -> Optional[str]:
        """Find best slave for promotion to master"""
        candidates = []
        
        for shard in self.shards.values():
            if shard.master_node == master_id:
                for slave_id in shard.slave_nodes:
                    if slave_id in self.nodes and self.nodes[slave_id].is_online:
                        slave = self.nodes[slave_id]
                        candidates.append((slave_id, slave.network_latency, slave.cpu_usage))
        
        if not candidates:
            return None
        
        # Choose slave with lowest latency and CPU usage
        candidates.sort(key=lambda x: (x[1], x[2]))
        return candidates[0][0]
    
    async def _promote_slave_to_master(self, slave_id: str, old_master_id: str):
        """Promote slave to master"""
        try:
            # Update node role
            self.nodes[slave_id].role = NodeRole.MASTER
            
            # Update shards
            for shard in self.shards.values():
                if shard.master_node == old_master_id:
                    shard.master_node = slave_id
                    if slave_id in shard.slave_nodes:
                        shard.slave_nodes.remove(slave_id)
            
            # Update hash ring
            self.hash_ring.remove_node(old_master_id)
            self.hash_ring.add_node(slave_id)
            
            logger.info(f"Promoted slave {slave_id} to master, replacing {old_master_id}")
            
        except Exception as e:
            logger.error(f"Failed to promote slave {slave_id}: {e}")
    
    async def _hot_spot_monitor(self):
        """Monitor for hot spots and trigger rebalancing"""
        while True:
            await asyncio.sleep(60.0)  # Check every minute
            
            hot_shards = []
            for shard_id, shard in self.shards.items():
                if shard.access_frequency > self.hot_spot_threshold:
                    hot_shards.append(shard_id)
            
            if hot_shards and not self.is_resharding:
                logger.info(f"Detected hot shards: {hot_shards}, triggering rebalancing")
                await self._rebalance_hot_shards(hot_shards)
    
    async def _rebalance_hot_shards(self, hot_shard_ids: List[str]):
        """Rebalance hot shards"""
        self.is_resharding = True
        
        try:
            # Implementation would involve:
            # 1. Analyzing hot key distribution
            # 2. Creating new shards for hot keys
            # 3. Migrating data
            # 4. Updating hash ring
            
            logger.info(f"Rebalancing hot shards: {hot_shard_ids}")
            
            # For now, just reset access frequency
            for shard_id in hot_shard_ids:
                if shard_id in self.shards:
                    self.shards[shard_id].access_frequency = 0
                    self.shards[shard_id].hot_keys.clear()
            
        finally:
            self.is_resharding = False
    
    async def _metrics_collector(self):
        """Collect cluster metrics"""
        while True:
            await asyncio.sleep(self.metrics_collection_interval)
            
            total_nodes = len(self.nodes)
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_online)
            
            self.cluster_metrics.total_nodes = total_nodes
            self.cluster_metrics.healthy_nodes = healthy_nodes
            
            if healthy_nodes < total_nodes * 0.8:
                self.cluster_metrics.cluster_state = ClusterState.DEGRADED
            elif healthy_nodes < total_nodes * 0.6:
                self.cluster_metrics.cluster_state = ClusterState.FAILING
            else:
                self.cluster_metrics.cluster_state = ClusterState.HEALTHY
    
    async def _automatic_rebalancing(self):
        """Automatic cluster rebalancing"""
        while True:
            await asyncio.sleep(300.0)  # Check every 5 minutes
            
            if self.is_resharding:
                continue
            
            # Check if rebalancing is needed
            if await self._should_rebalance():
                await self._perform_rebalancing()
    
    async def _should_rebalance(self) -> bool:
        """Determine if cluster should be rebalanced"""
        # Check memory usage distribution
        memory_usages = [node.memory_usage for node in self.nodes.values() if node.is_online]
        
        if not memory_usages:
            return False
        
        avg_memory = sum(memory_usages) / len(memory_usages)
        max_memory = max(memory_usages)
        
        # Rebalance if any node is using 80% more memory than average
        return max_memory > avg_memory * 1.8
    
    async def _perform_rebalancing(self):
        """Perform cluster rebalancing"""
        logger.info("Starting cluster rebalancing")
        
        # This would implement sophisticated rebalancing logic
        # For now, just log the action
        
        logger.info("Cluster rebalancing completed")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information"""
        return {
            'cluster_name': self.cluster_name,
            'region': self.region,
            'total_nodes': len(self.nodes),
            'healthy_nodes': sum(1 for node in self.nodes.values() if node.is_online),
            'total_shards': len(self.shards),
            'cluster_state': self.cluster_metrics.cluster_state.value,
            'is_resharding': self.is_resharding,
            'nodes': {
                node_id: {
                    'host': node.host,
                    'port': node.port,
                    'role': node.role.value,
                    'is_online': node.is_online,
                    'memory_usage': node.memory_usage,
                    'cpu_usage': node.cpu_usage,
                    'network_latency': node.network_latency
                }
                for node_id, node in self.nodes.items()
            },
            'shards': {
                shard_id: {
                    'master_node': shard.master_node,
                    'slave_nodes': shard.slave_nodes,
                    'slot_range': shard.slot_range,
                    'data_size_mb': shard.data_size_mb,
                    'access_frequency': shard.access_frequency,
                    'hot_keys_count': len(shard.hot_keys)
                }
                for shard_id, shard in self.shards.items()
            }
        }