"""
Predictive Load Balancer

ML-powered load balancer that predicts traffic patterns, optimizes resource allocation,
and provides intelligent request routing for distributed roleplay platforms.

Features:
- Real-time traffic prediction using time series analysis
- Multi-dimensional load assessment (CPU, memory, network, AI processing)
- Adaptive routing based on performance characteristics
- Proactive scaling decisions based on predicted load
- Geographic and user-context aware routing
- Circuit breaker integration for fault tolerance
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LoadMetricType(Enum):
    """Types of load metrics"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    AI_PROCESSING_LOAD = "ai_processing_load"
    ACTIVE_CONNECTIONS = "active_connections"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


class RoutingStrategy(Enum):
    """Load balancing routing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    POWER_OF_TWO_CHOICES = "power_of_two_choices"
    CONSISTENT_HASHING = "consistent_hashing"
    ML_OPTIMIZED = "ml_optimized"


class RequestType(Enum):
    """Types of requests for specialized routing"""
    AI_INFERENCE = "ai_inference"
    USER_INTERACTION = "user_interaction"
    MEDIA_STREAMING = "media_streaming"
    DATA_QUERY = "data_query"
    REAL_TIME_UPDATE = "real_time_update"
    BACKGROUND_TASK = "background_task"


@dataclass
class ServerNode:
    """Server node with performance metrics"""
    node_id: str
    host: str
    port: int
    region: str
    availability_zone: str
    capacity: Dict[str, float]  # Max capacity for each metric
    current_load: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    active_connections: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate: float = 0.0
    specializations: Set[RequestType] = field(default_factory=set)
    weight: float = 1.0
    circuit_breaker_state: str = "closed"


@dataclass
class Request:
    """Request with metadata for intelligent routing"""
    request_id: str
    request_type: RequestType
    user_id: str
    session_id: str
    geographic_origin: str
    estimated_complexity: float
    required_resources: Dict[str, float]
    priority: int = 5  # 1-10 scale
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadPrediction:
    """Load prediction for capacity planning"""
    timestamp: float
    time_horizon: float  # Prediction horizon in seconds
    predicted_metrics: Dict[str, float]
    confidence: float
    recommendation: str


@dataclass
class RoutingDecision:
    """Routing decision with reasoning"""
    target_node: str
    confidence: float
    reasoning: str
    estimated_response_time: float
    backup_nodes: List[str] = field(default_factory=list)


class PredictiveLoadBalancer:
    """
    ML-powered load balancer with predictive capabilities and adaptive routing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cluster_id = config['cluster_id']
        self.region = config['region']
        
        # Server nodes
        self.nodes: Dict[str, ServerNode] = {}
        self.healthy_nodes: Set[str] = set()
        
        # Load balancing state
        self.round_robin_index = 0
        self.routing_strategy = RoutingStrategy(config.get('default_strategy', 'ml_optimized'))
        
        # ML components
        self.traffic_predictor = TrafficPredictor()
        self.performance_predictor = PerformancePredictor()
        self.route_optimizer = RouteOptimizer()
        
        # Metrics and monitoring
        self.request_history: deque = deque(maxlen=10000)
        self.routing_decisions: deque = deque(maxlen=1000)
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.routing_metrics = {
            'total_requests': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'average_response_time': 0.0,
            'prediction_accuracy': 0.0
        }
        
        # Configuration
        self.health_check_interval = config.get('health_check_interval', 10.0)
        self.prediction_interval = config.get('prediction_interval', 60.0)
        self.load_update_interval = config.get('load_update_interval', 5.0)
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 0.5)
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout', 60.0)
        
        # Start background tasks
        self._start_background_tasks()
    
    async def add_node(self, node_config: Dict[str, Any]) -> bool:
        """Add new server node to load balancer"""
        try:
            node = ServerNode(
                node_id=node_config['node_id'],
                host=node_config['host'],
                port=node_config['port'],
                region=node_config.get('region', self.region),
                availability_zone=node_config.get('availability_zone', 'default'),
                capacity=node_config['capacity'],
                specializations=set(RequestType(s) for s in node_config.get('specializations', [])),
                weight=node_config.get('weight', 1.0)
            )
            
            self.nodes[node.node_id] = node
            self.healthy_nodes.add(node.node_id)
            
            logger.info(f"Added server node {node.node_id} to load balancer")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add server node: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove server node from load balancer"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.healthy_nodes.discard(node_id)
            logger.info(f"Removed server node {node_id} from load balancer")
            return True
        return False
    
    async def route_request(self, request: Request) -> Optional[RoutingDecision]:
        """
        Route request to optimal server node using ML predictions
        
        Returns:
            RoutingDecision with target node and reasoning
        """
        start_time = time.time()
        
        try:
            # Filter healthy nodes
            available_nodes = [
                node_id for node_id in self.healthy_nodes
                if self._can_handle_request(self.nodes[node_id], request)
            ]
            
            if not available_nodes:
                logger.warning(f"No available nodes for request {request.request_id}")
                return None
            
            # Choose routing strategy based on request type and system state
            strategy = await self._select_routing_strategy(request, available_nodes)
            
            # Route based on strategy
            if strategy == RoutingStrategy.ML_OPTIMIZED:
                decision = await self._ml_optimized_routing(request, available_nodes)
            elif strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
                decision = await self._least_response_time_routing(request, available_nodes)
            elif strategy == RoutingStrategy.POWER_OF_TWO_CHOICES:
                decision = await self._power_of_two_routing(request, available_nodes)
            else:
                decision = await self._weighted_round_robin_routing(request, available_nodes)
            
            if decision:
                # Update metrics
                await self._update_routing_metrics(request, decision, start_time)
                
                # Record decision for learning
                self.routing_decisions.append({
                    'timestamp': time.time(),
                    'request_type': request.request_type.value,
                    'target_node': decision.target_node,
                    'strategy': strategy.value,
                    'confidence': decision.confidence
                })
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to route request {request.request_id}: {e}")
            return None
    
    async def _ml_optimized_routing(self, request: Request, 
                                   available_nodes: List[str]) -> RoutingDecision:
        """ML-optimized routing using performance predictions"""
        
        # Get performance predictions for each node
        node_scores = []
        
        for node_id in available_nodes:
            node = self.nodes[node_id]
            
            # Predict performance for this request on this node
            predicted_performance = await self.performance_predictor.predict_performance(
                node, request
            )
            
            # Calculate routing score
            score = await self._calculate_routing_score(node, request, predicted_performance)
            node_scores.append((score, node_id, predicted_performance))
        
        # Sort by score (higher is better)
        node_scores.sort(key=lambda x: -x[0])
        
        best_score, best_node, best_prediction = node_scores[0]
        
        # Create backup list
        backup_nodes = [node_id for _, node_id, _ in node_scores[1:4]]
        
        return RoutingDecision(
            target_node=best_node,
            confidence=best_score,
            reasoning=f"ML-optimized: predicted response time {best_prediction['response_time']:.2f}ms",
            estimated_response_time=best_prediction['response_time'],
            backup_nodes=backup_nodes
        )
    
    async def _calculate_routing_score(self, node: ServerNode, request: Request, 
                                      prediction: Dict[str, float]) -> float:
        """Calculate routing score for node-request pair"""
        
        # Base score from predicted performance
        response_time_score = 1.0 / max(prediction['response_time'], 1.0)
        
        # Load balancing score
        total_load = sum(node.current_load.values()) / len(node.current_load) if node.current_load else 0
        load_score = 1.0 - min(total_load, 1.0)
        
        # Health score
        health_score = node.health_score
        
        # Specialization bonus
        specialization_score = 1.2 if request.request_type in node.specializations else 1.0
        
        # Geographic affinity
        geo_score = 1.1 if node.region == request.geographic_origin else 1.0
        
        # Priority weighting
        priority_weight = request.priority / 10.0
        
        # Combine scores
        combined_score = (
            response_time_score * 0.3 +
            load_score * 0.25 +
            health_score * 0.2 +
            specialization_score * 0.15 +
            geo_score * 0.1
        ) * priority_weight
        
        return combined_score
    
    async def _least_response_time_routing(self, request: Request, 
                                          available_nodes: List[str]) -> RoutingDecision:
        """Route to node with lowest average response time"""
        
        best_node = None
        best_response_time = float('inf')
        
        for node_id in available_nodes:
            node = self.nodes[node_id]
            avg_response_time = statistics.mean(node.response_times) if node.response_times else 100.0
            
            if avg_response_time < best_response_time:
                best_response_time = avg_response_time
                best_node = node_id
        
        return RoutingDecision(
            target_node=best_node,
            confidence=0.8,
            reasoning=f"Least response time: {best_response_time:.2f}ms",
            estimated_response_time=best_response_time
        )
    
    async def _power_of_two_routing(self, request: Request, 
                                   available_nodes: List[str]) -> RoutingDecision:
        """Power of two choices algorithm for load balancing"""
        
        # Randomly select two nodes
        import random
        if len(available_nodes) < 2:
            selected_nodes = available_nodes
        else:
            selected_nodes = random.sample(available_nodes, 2)
        
        # Choose the less loaded one
        best_node = None
        best_load = float('inf')
        
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            current_load = node.active_connections + sum(node.current_load.values())
            
            if current_load < best_load:
                best_load = current_load
                best_node = node_id
        
        return RoutingDecision(
            target_node=best_node,
            confidence=0.7,
            reasoning=f"Power of two choices: load {best_load}",
            estimated_response_time=50.0
        )
    
    async def _weighted_round_robin_routing(self, request: Request, 
                                           available_nodes: List[str]) -> RoutingDecision:
        """Weighted round-robin routing"""
        
        # Create weighted list
        weighted_nodes = []
        for node_id in available_nodes:
            node = self.nodes[node_id]
            weight = max(int(node.weight * node.health_score * 10), 1)
            weighted_nodes.extend([node_id] * weight)
        
        # Select next node
        if weighted_nodes:
            selected_node = weighted_nodes[self.round_robin_index % len(weighted_nodes)]
            self.round_robin_index = (self.round_robin_index + 1) % len(weighted_nodes)
            
            return RoutingDecision(
                target_node=selected_node,
                confidence=0.6,
                reasoning="Weighted round-robin",
                estimated_response_time=100.0
            )
        
        return None
    
    async def _select_routing_strategy(self, request: Request, 
                                      available_nodes: List[str]) -> RoutingStrategy:
        """Select optimal routing strategy based on current conditions"""
        
        # High priority requests use ML optimization
        if request.priority >= 8:
            return RoutingStrategy.ML_OPTIMIZED
        
        # Real-time requests prefer low latency
        if request.request_type == RequestType.REAL_TIME_UPDATE:
            return RoutingStrategy.LEAST_RESPONSE_TIME
        
        # Background tasks can use simpler algorithms
        if request.request_type == RequestType.BACKGROUND_TASK:
            return RoutingStrategy.WEIGHTED_ROUND_ROBIN
        
        # For large clusters, use power of two choices for efficiency
        if len(available_nodes) > 20:
            return RoutingStrategy.POWER_OF_TWO_CHOICES
        
        # Default to ML optimization
        return RoutingStrategy.ML_OPTIMIZED
    
    def _can_handle_request(self, node: ServerNode, request: Request) -> bool:
        """Check if node can handle the request"""
        
        # Check circuit breaker
        if node.circuit_breaker_state == "open":
            return False
        
        # Check resource requirements
        for resource, required in request.required_resources.items():
            if resource in node.capacity:
                available = node.capacity[resource] - node.current_load.get(resource, 0)
                if available < required:
                    return False
        
        # Check specialization (if required)
        if hasattr(request, 'requires_specialization') and request.requires_specialization:
            if request.request_type not in node.specializations:
                return False
        
        return True
    
    async def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update node performance metrics"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.current_load.update(metrics)
        node.last_health_check = time.time()
        
        # Update health score based on metrics
        await self._update_health_score(node)
        
        # Record load history for prediction
        self.load_history[node_id].append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
    
    async def update_response_time(self, node_id: str, response_time: float, 
                                  success: bool):
        """Update response time metrics for a node"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.response_times.append(response_time)
        
        # Update error rate
        if success:
            node.error_rate = node.error_rate * 0.9  # Decay error rate
        else:
            node.error_rate = min(node.error_rate * 0.9 + 0.1, 1.0)
        
        # Update circuit breaker
        await self._update_circuit_breaker(node)
    
    async def _update_health_score(self, node: ServerNode):
        """Update node health score based on current metrics"""
        
        score_factors = []
        
        # CPU utilization factor
        cpu_usage = node.current_load.get('cpu_utilization', 0)
        cpu_factor = max(0, 1.0 - cpu_usage / 100.0)
        score_factors.append(cpu_factor)
        
        # Memory utilization factor
        memory_usage = node.current_load.get('memory_utilization', 0)
        memory_factor = max(0, 1.0 - memory_usage / 100.0)
        score_factors.append(memory_factor)
        
        # Response time factor
        if node.response_times:
            avg_response_time = statistics.mean(node.response_times)
            response_factor = max(0, 1.0 - avg_response_time / 1000.0)  # Normalize to 1s
            score_factors.append(response_factor)
        
        # Error rate factor
        error_factor = 1.0 - node.error_rate
        score_factors.append(error_factor)
        
        # Calculate weighted average
        if score_factors:
            node.health_score = sum(score_factors) / len(score_factors)
        
        # Update healthy nodes set
        if node.health_score > 0.3 and node.is_healthy:
            self.healthy_nodes.add(node.node_id)
        else:
            self.healthy_nodes.discard(node.node_id)
    
    async def _update_circuit_breaker(self, node: ServerNode):
        """Update circuit breaker state based on error rate"""
        
        if node.error_rate > self.circuit_breaker_threshold:
            if node.circuit_breaker_state == "closed":
                node.circuit_breaker_state = "open"
                logger.warning(f"Circuit breaker opened for node {node.node_id}")
        elif node.error_rate < self.circuit_breaker_threshold / 2:
            if node.circuit_breaker_state == "open":
                node.circuit_breaker_state = "half_open"
                logger.info(f"Circuit breaker half-opened for node {node.node_id}")
    
    async def get_load_predictions(self, time_horizon: float = 300.0) -> Dict[str, LoadPrediction]:
        """Get load predictions for all nodes"""
        predictions = {}
        
        for node_id, node in self.nodes.items():
            if node_id in self.load_history and self.load_history[node_id]:
                prediction = await self.traffic_predictor.predict_load(
                    node_id, self.load_history[node_id], time_horizon
                )
                predictions[node_id] = prediction
        
        return predictions
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._performance_optimizer())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._circuit_breaker_recovery())
    
    async def _health_monitor(self):
        """Monitor node health"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            
            for node_id, node in self.nodes.items():
                # Check if node has been updated recently
                time_since_update = time.time() - node.last_health_check
                
                if time_since_update > self.health_check_interval * 3:
                    node.is_healthy = False
                    self.healthy_nodes.discard(node_id)
                    logger.warning(f"Node {node_id} marked unhealthy due to stale metrics")
    
    async def _performance_optimizer(self):
        """Optimize performance based on ML predictions"""
        while True:
            await asyncio.sleep(self.prediction_interval)
            
            # Get load predictions
            predictions = await self.get_load_predictions()
            
            # Analyze predictions and recommend actions
            for node_id, prediction in predictions.items():
                if prediction.confidence > 0.7:
                    await self._handle_load_prediction(node_id, prediction)
    
    async def _handle_load_prediction(self, node_id: str, prediction: LoadPrediction):
        """Handle load prediction for a node"""
        
        # Check if node will be overloaded
        predicted_load = max(prediction.predicted_metrics.values())
        
        if predicted_load > 0.8:
            logger.warning(f"Predicted overload for node {node_id}: {predicted_load:.2f}")
            # Could trigger scaling actions here
        elif predicted_load < 0.2:
            logger.info(f"Predicted underutilization for node {node_id}: {predicted_load:.2f}")
            # Could trigger scale-down recommendations
    
    async def _circuit_breaker_recovery(self):
        """Recover circuit breakers after timeout"""
        while True:
            await asyncio.sleep(self.circuit_breaker_timeout)
            
            for node in self.nodes.values():
                if node.circuit_breaker_state == "half_open" and node.error_rate < 0.1:
                    node.circuit_breaker_state = "closed"
                    logger.info(f"Circuit breaker closed for node {node.node_id}")
    
    async def _metrics_collector(self):
        """Collect load balancer metrics"""
        while True:
            await asyncio.sleep(30.0)
            
            # Calculate routing success rate
            if self.routing_decisions:
                recent_decisions = [d for d in self.routing_decisions if time.time() - d['timestamp'] < 300]
                if recent_decisions:
                    self.routing_metrics['prediction_accuracy'] = sum(
                        d['confidence'] for d in recent_decisions
                    ) / len(recent_decisions)
    
    async def _update_routing_metrics(self, request: Request, decision: RoutingDecision, start_time: float):
        """Update routing performance metrics"""
        self.routing_metrics['total_requests'] += 1
        
        routing_time = time.time() - start_time
        
        # Update average response time (for routing decision time)
        current_avg = self.routing_metrics['average_response_time']
        total_requests = self.routing_metrics['total_requests']
        
        self.routing_metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + routing_time) / total_requests
        )
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        return {
            'cluster_id': self.cluster_id,
            'total_nodes': len(self.nodes),
            'healthy_nodes': len(self.healthy_nodes),
            'routing_strategy': self.routing_strategy.value,
            'routing_metrics': self.routing_metrics,
            'nodes': {
                node_id: {
                    'health_score': node.health_score,
                    'is_healthy': node.is_healthy,
                    'active_connections': node.active_connections,
                    'error_rate': node.error_rate,
                    'circuit_breaker_state': node.circuit_breaker_state,
                    'current_load': node.current_load,
                    'avg_response_time': statistics.mean(node.response_times) if node.response_times else 0
                }
                for node_id, node in self.nodes.items()
            }
        }


class TrafficPredictor:
    """ML-based traffic prediction"""
    
    async def predict_load(self, node_id: str, load_history: deque, 
                          time_horizon: float) -> LoadPrediction:
        """Predict future load for a node"""
        
        if len(load_history) < 10:
            return LoadPrediction(
                timestamp=time.time(),
                time_horizon=time_horizon,
                predicted_metrics={'cpu_utilization': 50.0},
                confidence=0.1,
                recommendation="Insufficient data for prediction"
            )
        
        # Simple trend analysis (in production, use sophisticated time series models)
        recent_data = list(load_history)[-10:]
        
        predicted_metrics = {}
        for metric in ['cpu_utilization', 'memory_utilization', 'network_io']:
            values = [data['metrics'].get(metric, 0) for data in recent_data]
            if values:
                trend = (values[-1] - values[0]) / len(values)
                predicted_value = values[-1] + trend * (time_horizon / 60.0)
                predicted_metrics[metric] = max(0, min(predicted_value, 100))
        
        return LoadPrediction(
            timestamp=time.time(),
            time_horizon=time_horizon,
            predicted_metrics=predicted_metrics,
            confidence=0.7,
            recommendation="Trend-based prediction"
        )


class PerformancePredictor:
    """ML-based performance prediction"""
    
    async def predict_performance(self, node: ServerNode, request: Request) -> Dict[str, float]:
        """Predict performance metrics for request on node"""
        
        # Base response time based on node current load
        base_response_time = 50.0  # ms
        
        # Load factor
        avg_load = sum(node.current_load.values()) / len(node.current_load) if node.current_load else 0
        load_factor = 1.0 + avg_load / 100.0
        
        # Request complexity factor
        complexity_factor = 1.0 + request.estimated_complexity
        
        # Specialization factor
        spec_factor = 0.8 if request.request_type in node.specializations else 1.0
        
        predicted_response_time = base_response_time * load_factor * complexity_factor * spec_factor
        
        return {
            'response_time': predicted_response_time,
            'cpu_usage_increase': request.required_resources.get('cpu', 5.0),
            'memory_usage_increase': request.required_resources.get('memory', 10.0)
        }


class RouteOptimizer:
    """Route optimization using ML"""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=1000)
    
    async def optimize_routes(self, current_routes: List[Dict]) -> List[Dict]:
        """Optimize current routing decisions"""
        # Placeholder for sophisticated route optimization
        return current_routes