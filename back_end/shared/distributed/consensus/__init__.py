"""
Advanced Consensus Protocols for Global AI State Synchronization

This module implements breakthrough consensus algorithms designed for distributed
AI state management across global data centers with focus on:
- Low latency consensus for real-time AI interactions
- Byzantine fault tolerance for adversarial environments
- Adaptive quorum management for variable network conditions
- Hierarchical consensus for multi-region deployments
"""

from .hybrid_consensus import HybridConsensusProtocol
from .ai_state_consensus import AIStateConsensusManager
from .adaptive_raft import AdaptiveRaftProtocol
from .byzantine_consensus import ByzantineConsensusProtocol
from .quorum_manager import AdaptiveQuorumManager
from .consensus_coordinator import GlobalConsensusCoordinator

__all__ = [
    'HybridConsensusProtocol',
    'AIStateConsensusManager',
    'AdaptiveRaftProtocol',
    'ByzantineConsensusProtocol',
    'AdaptiveQuorumManager',
    'GlobalConsensusCoordinator'
]