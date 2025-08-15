"""
ML-Powered Predictive Load Balancing

Advanced load balancing system with machine learning-powered prediction,
adaptive algorithms, and real-time optimization for distributed roleplay platforms.

Key Features:
- ML-based traffic prediction and proactive scaling
- Multi-dimensional load balancing (CPU, memory, network, AI workload)
- Geo-aware request routing with latency optimization
- Adaptive algorithms that learn from system behavior
- Real-time capacity planning and resource allocation
- Circuit breaker patterns for fault tolerance
"""

from .predictive_load_balancer import PredictiveLoadBalancer
from .ml_traffic_predictor import MLTrafficPredictor
from .adaptive_routing import AdaptiveRoutingEngine
from .capacity_planner import CapacityPlanner
from .health_checker import AdvancedHealthChecker
from .circuit_breaker import CircuitBreakerManager

__all__ = [
    'PredictiveLoadBalancer',
    'MLTrafficPredictor',
    'AdaptiveRoutingEngine',
    'CapacityPlanner',
    'AdvancedHealthChecker',
    'CircuitBreakerManager'
]