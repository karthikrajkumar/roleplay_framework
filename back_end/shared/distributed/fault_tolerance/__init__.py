"""
Proactive Failure Detection and Auto-Recovery Systems

Advanced fault tolerance system with machine learning-powered failure prediction,
automatic recovery mechanisms, and cascading failure prevention for distributed
roleplay platforms.

Key Features:
- ML-based failure prediction and early warning
- Hierarchical failure detection (node, service, network, data center)
- Automatic failover and recovery orchestration
- Circuit breaker patterns with adaptive thresholds
- Cascading failure prevention and isolation
- Self-healing infrastructure with minimal downtime
"""

from .failure_detector import ProactiveFailureDetector
from .recovery_orchestrator import AutoRecoveryOrchestrator
from .circuit_breaker import AdaptiveCircuitBreaker
from .health_monitor import DistributedHealthMonitor
from .chaos_engineering import ChaosTestingEngine
from .disaster_recovery import DisasterRecoveryManager

__all__ = [
    'ProactiveFailureDetector',
    'AutoRecoveryOrchestrator', 
    'AdaptiveCircuitBreaker',
    'DistributedHealthMonitor',
    'ChaosTestingEngine',
    'DisasterRecoveryManager'
]