"""
AI State Consensus Manager

Specialized consensus manager for AI state synchronization across distributed
roleplay platforms. Optimized for real-time AI persona interactions, memory
consistency, and collaborative AI coordination.

Key Features:
- Vector clock-based AI state versioning
- Conflict-free replicated data types (CRDTs) for AI memory
- Causal consistency for AI interaction dependencies
- Optimistic concurrency control for low-latency updates
- Intelligent state merging and conflict resolution
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod
import logging

from .hybrid_consensus import HybridConsensusProtocol

logger = logging.getLogger(__name__)


class AIStateType(Enum):
    """Types of AI state that require consensus"""
    PERSONA_MEMORY = "persona_memory"
    EMOTIONAL_STATE = "emotional_state"
    INTERACTION_CONTEXT = "interaction_context"
    LEARNING_UPDATES = "learning_updates"
    COLLABORATIVE_STATE = "collaborative_state"
    SCENARIO_STATE = "scenario_state"


@dataclass
class VectorClock:
    """Vector clock for causal ordering of AI state changes"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        """Increment clock for a node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update with another vector clock"""
        for node_id, clock_value in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock_value)
    
    def compare(self, other: 'VectorClock') -> str:
        """Compare with another vector clock"""
        less_than = False
        greater_than = False
        
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        for node_id in all_nodes:
            self_clock = self.clocks.get(node_id, 0)
            other_clock = other.clocks.get(node_id, 0)
            
            if self_clock < other_clock:
                less_than = True
            elif self_clock > other_clock:
                greater_than = True
        
        if less_than and not greater_than:
            return "before"
        elif greater_than and not less_than:
            return "after"
        elif not less_than and not greater_than:
            return "equal"
        else:
            return "concurrent"


@dataclass
class AIStateChange:
    """Represents a change to AI state"""
    change_id: str
    ai_id: str
    state_type: AIStateType
    change_data: Dict[str, Any]
    vector_clock: VectorClock
    timestamp: float
    causal_dependencies: List[str] = field(default_factory=list)
    conflict_resolution_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIMemoryEntry:
    """CRDT-based AI memory entry"""
    memory_id: str
    content: str
    importance_score: float
    access_count: int
    vector_clock: VectorClock
    tombstone: bool = False
    
    def merge(self, other: 'AIMemoryEntry') -> 'AIMemoryEntry':
        """Merge with another memory entry using CRDT semantics"""
        # Use last-writer-wins with vector clock ordering
        comparison = self.vector_clock.compare(other.vector_clock)
        
        if comparison == "before":
            return other
        elif comparison == "after":
            return self
        else:
            # Concurrent updates - merge based on importance and access
            merged_clock = VectorClock()
            merged_clock.clocks = {**self.vector_clock.clocks}
            merged_clock.update(other.vector_clock)
            
            return AIMemoryEntry(
                memory_id=self.memory_id,
                content=self.content if self.importance_score >= other.importance_score else other.content,
                importance_score=max(self.importance_score, other.importance_score),
                access_count=max(self.access_count, other.access_count),
                vector_clock=merged_clock,
                tombstone=self.tombstone or other.tombstone
            )


class AIStateCRDT:
    """Conflict-free Replicated Data Type for AI state"""
    
    def __init__(self, ai_id: str):
        self.ai_id = ai_id
        self.memories: Dict[str, AIMemoryEntry] = {}
        self.emotional_state: Dict[str, float] = {}
        self.interaction_context: Dict[str, Any] = {}
        self.vector_clock = VectorClock()
    
    def add_memory(self, memory_id: str, content: str, 
                   importance_score: float, node_id: str) -> AIMemoryEntry:
        """Add a new memory entry"""
        self.vector_clock.increment(node_id)
        
        memory_entry = AIMemoryEntry(
            memory_id=memory_id,
            content=content,
            importance_score=importance_score,
            access_count=1,
            vector_clock=VectorClock(clocks={node_id: self.vector_clock.clocks[node_id]})
        )
        
        if memory_id in self.memories:
            self.memories[memory_id] = self.memories[memory_id].merge(memory_entry)
        else:
            self.memories[memory_id] = memory_entry
        
        return self.memories[memory_id]
    
    def update_emotional_state(self, emotion: str, value: float, node_id: str):
        """Update emotional state with conflict resolution"""
        self.vector_clock.increment(node_id)
        
        # Use additive semantics for emotional states
        if emotion in self.emotional_state:
            self.emotional_state[emotion] = (self.emotional_state[emotion] + value) / 2
        else:
            self.emotional_state[emotion] = value
    
    def merge(self, other: 'AIStateCRDT') -> 'AIStateCRDT':
        """Merge with another AI state CRDT"""
        merged = AIStateCRDT(self.ai_id)
        merged.vector_clock.clocks = {**self.vector_clock.clocks}
        merged.vector_clock.update(other.vector_clock)
        
        # Merge memories
        all_memory_ids = set(self.memories.keys()) | set(other.memories.keys())
        for memory_id in all_memory_ids:
            if memory_id in self.memories and memory_id in other.memories:
                merged.memories[memory_id] = self.memories[memory_id].merge(other.memories[memory_id])
            elif memory_id in self.memories:
                merged.memories[memory_id] = self.memories[memory_id]
            else:
                merged.memories[memory_id] = other.memories[memory_id]
        
        # Merge emotional state (additive)
        all_emotions = set(self.emotional_state.keys()) | set(other.emotional_state.keys())
        for emotion in all_emotions:
            self_value = self.emotional_state.get(emotion, 0.0)
            other_value = other.emotional_state.get(emotion, 0.0)
            merged.emotional_state[emotion] = (self_value + other_value) / 2
        
        # Merge interaction context (last-writer-wins with vector clock)
        merged.interaction_context = {**self.interaction_context, **other.interaction_context}
        
        return merged


class AIStateConsensusManager:
    """
    Manages consensus for AI state changes across distributed nodes
    with specialized algorithms for AI persona consistency and collaboration.
    """
    
    def __init__(self, node_id: str, cluster_config: Dict[str, Any]):
        self.node_id = node_id
        self.cluster_config = cluster_config
        
        # Initialize consensus protocol
        self.consensus_protocol = HybridConsensusProtocol(node_id, cluster_config)
        
        # AI state management
        self.ai_states: Dict[str, AIStateCRDT] = {}
        self.pending_changes: Dict[str, AIStateChange] = {}
        self.causal_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance optimization
        self.change_batch: List[AIStateChange] = []
        self.batch_timeout = 0.005  # 5ms for AI interactions
        self.last_batch_time = time.time()
        
        # Conflict resolution
        self.conflict_resolver = AIConflictResolver()
        
        # Metrics
        self.consensus_metrics = {
            'total_changes': 0,
            'conflicts_resolved': 0,
            'average_consensus_time': 0.0,
            'ai_state_size': 0
        }
        
        # Register consensus handlers
        self.consensus_protocol.add_state_change_handler(self._handle_consensus_result)
        
        # Start background tasks
        asyncio.create_task(self._batch_processor())
        asyncio.create_task(self._causal_consistency_monitor())
        asyncio.create_task(self._ai_state_gc())
    
    async def propose_ai_memory_update(self, ai_id: str, memory_id: str, 
                                      content: str, importance_score: float) -> str:
        """Propose AI memory update for consensus"""
        change = AIStateChange(
            change_id=self._generate_change_id(),
            ai_id=ai_id,
            state_type=AIStateType.PERSONA_MEMORY,
            change_data={
                'memory_id': memory_id,
                'content': content,
                'importance_score': importance_score,
                'operation': 'update'
            },
            vector_clock=self._get_ai_vector_clock(ai_id),
            timestamp=time.time()
        )
        
        return await self._propose_change(change)
    
    async def propose_emotional_state_update(self, ai_id: str, 
                                           emotional_changes: Dict[str, float]) -> str:
        """Propose emotional state update for consensus"""
        change = AIStateChange(
            change_id=self._generate_change_id(),
            ai_id=ai_id,
            state_type=AIStateType.EMOTIONAL_STATE,
            change_data={
                'emotional_changes': emotional_changes,
                'operation': 'update'
            },
            vector_clock=self._get_ai_vector_clock(ai_id),
            timestamp=time.time()
        )
        
        return await self._propose_change(change)
    
    async def propose_collaborative_ai_sync(self, ai_ids: List[str], 
                                          sync_data: Dict[str, Any]) -> str:
        """Propose collaborative AI synchronization"""
        change = AIStateChange(
            change_id=self._generate_change_id(),
            ai_id="collaborative",
            state_type=AIStateType.COLLABORATIVE_STATE,
            change_data={
                'ai_ids': ai_ids,
                'sync_data': sync_data,
                'operation': 'sync'
            },
            vector_clock=VectorClock(clocks={self.node_id: int(time.time() * 1000)}),
            timestamp=time.time()
        )
        
        return await self._propose_change(change)
    
    async def _propose_change(self, change: AIStateChange) -> str:
        """Propose AI state change for consensus"""
        # Check for causal dependencies
        await self._analyze_causal_dependencies(change)
        
        # Add to batch
        self.change_batch.append(change)
        self.pending_changes[change.change_id] = change
        
        # High importance changes get immediate processing
        if (change.state_type == AIStateType.COLLABORATIVE_STATE or
            change.change_data.get('importance_score', 0) > 0.8):
            await self._process_change_batch()
        
        return change.change_id
    
    async def _process_change_batch(self):
        """Process batched AI state changes"""
        if not self.change_batch:
            return
        
        # Sort by causal dependencies and importance
        sorted_changes = self._sort_changes_by_causality(self.change_batch)
        self.change_batch.clear()
        
        # Convert to consensus proposals
        consensus_changes = {}
        for change in sorted_changes:
            consensus_changes[f"ai_state_{change.change_id}"] = {
                'type': 'ai_state_change',
                'change': change.__dict__,
                'vector_clock': change.vector_clock.__dict__
            }
        
        # Propose to consensus protocol
        proposal_id = await self.consensus_protocol.propose_ai_state_change(
            consensus_changes, 
            priority=self._calculate_change_priority(sorted_changes)
        )
        
        logger.debug(f"Proposed AI state changes batch: {proposal_id}")
        return proposal_id
    
    async def _handle_consensus_result(self, consensus_result: Dict[str, Any]):
        """Handle consensus result and apply AI state changes"""
        start_time = time.time()
        
        for key, value in consensus_result.items():
            if key.startswith('ai_state_') and value.get('type') == 'ai_state_change':
                change_data = value['change']
                change = AIStateChange(**change_data)
                change.vector_clock = VectorClock(**value['vector_clock'])
                
                await self._apply_ai_state_change(change)
        
        # Update metrics
        consensus_time = time.time() - start_time
        self.consensus_metrics['average_consensus_time'] = (
            (self.consensus_metrics['average_consensus_time'] * self.consensus_metrics['total_changes'] + 
             consensus_time) / (self.consensus_metrics['total_changes'] + 1)
        )
        self.consensus_metrics['total_changes'] += 1
    
    async def _apply_ai_state_change(self, change: AIStateChange):
        """Apply committed AI state change"""
        ai_id = change.ai_id
        
        # Ensure AI state exists
        if ai_id not in self.ai_states:
            self.ai_states[ai_id] = AIStateCRDT(ai_id)
        
        ai_state = self.ai_states[ai_id]
        
        try:
            if change.state_type == AIStateType.PERSONA_MEMORY:
                await self._apply_memory_change(ai_state, change)
            elif change.state_type == AIStateType.EMOTIONAL_STATE:
                await self._apply_emotional_change(ai_state, change)
            elif change.state_type == AIStateType.COLLABORATIVE_STATE:
                await self._apply_collaborative_change(change)
            
            # Update vector clock
            ai_state.vector_clock.update(change.vector_clock)
            
            # Remove from pending
            if change.change_id in self.pending_changes:
                del self.pending_changes[change.change_id]
            
            logger.debug(f"Applied AI state change {change.change_id} for AI {ai_id}")
            
        except Exception as e:
            logger.error(f"Failed to apply AI state change {change.change_id}: {e}")
    
    async def _apply_memory_change(self, ai_state: AIStateCRDT, change: AIStateChange):
        """Apply memory change to AI state"""
        change_data = change.change_data
        
        if change_data['operation'] == 'update':
            ai_state.add_memory(
                memory_id=change_data['memory_id'],
                content=change_data['content'],
                importance_score=change_data['importance_score'],
                node_id=self.node_id
            )
    
    async def _apply_emotional_change(self, ai_state: AIStateCRDT, change: AIStateChange):
        """Apply emotional state change to AI state"""
        change_data = change.change_data
        
        for emotion, value in change_data['emotional_changes'].items():
            ai_state.update_emotional_state(emotion, value, self.node_id)
    
    async def _apply_collaborative_change(self, change: AIStateChange):
        """Apply collaborative AI synchronization"""
        change_data = change.change_data
        ai_ids = change_data['ai_ids']
        sync_data = change_data['sync_data']
        
        # Synchronize states between specified AIs
        for ai_id in ai_ids:
            if ai_id in self.ai_states:
                # Apply synchronization data
                ai_state = self.ai_states[ai_id]
                for key, value in sync_data.items():
                    if key == 'shared_memory':
                        for memory_id, memory_content in value.items():
                            ai_state.add_memory(memory_id, memory_content, 0.5, self.node_id)
    
    def _sort_changes_by_causality(self, changes: List[AIStateChange]) -> List[AIStateChange]:
        """Sort changes by causal dependencies and importance"""
        # Build dependency graph
        dependency_graph = defaultdict(set)
        change_map = {change.change_id: change for change in changes}
        
        for change in changes:
            for dep in change.causal_dependencies:
                if dep in change_map:
                    dependency_graph[dep].add(change.change_id)
        
        # Topological sort with importance weighting
        in_degree = defaultdict(int)
        for change_id in change_map:
            for dependent in dependency_graph[change_id]:
                in_degree[dependent] += 1
        
        ready_queue = [change for change in changes if in_degree[change.change_id] == 0]
        ready_queue.sort(key=lambda c: -c.change_data.get('importance_score', 0))
        
        result = []
        while ready_queue:
            current = ready_queue.pop(0)
            result.append(current)
            
            for dependent_id in dependency_graph[current.change_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    dependent = change_map[dependent_id]
                    # Insert in importance order
                    inserted = False
                    for i, c in enumerate(ready_queue):
                        if (dependent.change_data.get('importance_score', 0) > 
                            c.change_data.get('importance_score', 0)):
                            ready_queue.insert(i, dependent)
                            inserted = True
                            break
                    if not inserted:
                        ready_queue.append(dependent)
        
        return result
    
    async def _analyze_causal_dependencies(self, change: AIStateChange):
        """Analyze and set causal dependencies for a change"""
        ai_id = change.ai_id
        
        # Check for dependencies on previous changes to the same AI
        for pending_id, pending_change in self.pending_changes.items():
            if (pending_change.ai_id == ai_id and 
                pending_change.timestamp < change.timestamp):
                change.causal_dependencies.append(pending_id)
        
        # Check for collaborative dependencies
        if change.state_type == AIStateType.COLLABORATIVE_STATE:
            for ai_id in change.change_data.get('ai_ids', []):
                for pending_id, pending_change in self.pending_changes.items():
                    if pending_change.ai_id == ai_id:
                        change.causal_dependencies.append(pending_id)
    
    def _calculate_change_priority(self, changes: List[AIStateChange]) -> int:
        """Calculate priority for a batch of changes"""
        max_importance = max(
            change.change_data.get('importance_score', 0) for change in changes
        )
        
        # Collaborative changes get higher priority
        has_collaborative = any(
            change.state_type == AIStateType.COLLABORATIVE_STATE 
            for change in changes
        )
        
        if has_collaborative:
            return 9
        elif max_importance > 0.8:
            return 8
        elif max_importance > 0.5:
            return 5
        else:
            return 3
    
    async def _batch_processor(self):
        """Process change batches at regular intervals"""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            if (self.change_batch and 
                time.time() - self.last_batch_time > self.batch_timeout):
                await self._process_change_batch()
                self.last_batch_time = time.time()
    
    async def _causal_consistency_monitor(self):
        """Monitor and enforce causal consistency"""
        while True:
            await asyncio.sleep(1.0)
            
            # Check for causal consistency violations
            for ai_id, ai_state in self.ai_states.items():
                await self._check_causal_consistency(ai_id, ai_state)
    
    async def _ai_state_gc(self):
        """Garbage collect old AI state data"""
        while True:
            await asyncio.sleep(60.0)  # Run every minute
            
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour retention
            
            for ai_id, ai_state in self.ai_states.items():
                # Remove old memories with low importance
                to_remove = []
                for memory_id, memory in ai_state.memories.items():
                    if (memory.importance_score < 0.3 and 
                        current_time - memory.vector_clock.clocks.get(self.node_id, 0) / 1000 > cutoff_time):
                        to_remove.append(memory_id)
                
                for memory_id in to_remove:
                    ai_state.memories[memory_id].tombstone = True
    
    def _get_ai_vector_clock(self, ai_id: str) -> VectorClock:
        """Get current vector clock for an AI"""
        if ai_id not in self.ai_states:
            return VectorClock(clocks={self.node_id: 1})
        
        clock = VectorClock(clocks={**self.ai_states[ai_id].vector_clock.clocks})
        clock.increment(self.node_id)
        return clock
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID"""
        return f"{self.node_id}_{time.time_ns()}"
    
    def get_ai_state(self, ai_id: str) -> Optional[AIStateCRDT]:
        """Get current AI state"""
        return self.ai_states.get(ai_id)
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get AI state consensus metrics"""
        total_state_size = sum(
            len(state.memories) + len(state.emotional_state) 
            for state in self.ai_states.values()
        )
        
        return {
            **self.consensus_metrics,
            'ai_state_size': total_state_size,
            'active_ais': len(self.ai_states),
            'pending_changes': len(self.pending_changes)
        }


class AIConflictResolver:
    """Resolves conflicts in AI state updates"""
    
    def __init__(self):
        self.resolution_strategies = {
            AIStateType.PERSONA_MEMORY: self._resolve_memory_conflict,
            AIStateType.EMOTIONAL_STATE: self._resolve_emotional_conflict,
            AIStateType.INTERACTION_CONTEXT: self._resolve_context_conflict
        }
    
    def resolve_conflict(self, change1: AIStateChange, 
                        change2: AIStateChange) -> AIStateChange:
        """Resolve conflict between two concurrent changes"""
        if change1.state_type != change2.state_type:
            raise ValueError("Cannot resolve conflicts between different state types")
        
        resolver = self.resolution_strategies.get(change1.state_type)
        if resolver:
            return resolver(change1, change2)
        
        # Default: use vector clock ordering
        comparison = change1.vector_clock.compare(change2.vector_clock)
        return change2 if comparison == "before" else change1
    
    def _resolve_memory_conflict(self, change1: AIStateChange, 
                                change2: AIStateChange) -> AIStateChange:
        """Resolve memory update conflicts using importance scores"""
        importance1 = change1.change_data.get('importance_score', 0)
        importance2 = change2.change_data.get('importance_score', 0)
        
        return change1 if importance1 >= importance2 else change2
    
    def _resolve_emotional_conflict(self, change1: AIStateChange, 
                                   change2: AIStateChange) -> AIStateChange:
        """Resolve emotional state conflicts by averaging"""
        merged_data = {**change1.change_data}
        
        for emotion, value2 in change2.change_data.get('emotional_changes', {}).items():
            if emotion in merged_data.get('emotional_changes', {}):
                value1 = merged_data['emotional_changes'][emotion]
                merged_data['emotional_changes'][emotion] = (value1 + value2) / 2
            else:
                merged_data.setdefault('emotional_changes', {})[emotion] = value2
        
        result = AIStateChange(
            change_id=f"merged_{change1.change_id}_{change2.change_id}",
            ai_id=change1.ai_id,
            state_type=change1.state_type,
            change_data=merged_data,
            vector_clock=change1.vector_clock,
            timestamp=max(change1.timestamp, change2.timestamp)
        )
        result.vector_clock.update(change2.vector_clock)
        
        return result
    
    def _resolve_context_conflict(self, change1: AIStateChange, 
                                 change2: AIStateChange) -> AIStateChange:
        """Resolve interaction context conflicts using timestamps"""
        return change1 if change1.timestamp >= change2.timestamp else change2