"""
Real-Time Group Consistency Manager

Advanced consistency system for managing real-time updates to group memberships,
permissions, and assignments across distributed user management nodes.

Key Features:
- Event-driven consistency with causal ordering
- Real-time propagation of group membership changes
- Conflict resolution for concurrent group assignments
- Vector clocks for causality tracking
- Eventually consistent with strong ordering guarantees
- Optimistic conflict resolution with automatic rollback
- Cross-region consistency with minimal latency impact

Consistency Models Supported:
- Strong Consistency: For critical permission changes
- Eventual Consistency: For user activity updates
- Causal Consistency: For group membership changes
- Session Consistency: For user-specific group views
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency levels for different types of operations"""
    STRONG = "strong"           # Immediate consistency across all nodes
    EVENTUAL = "eventual"       # Eventually consistent with conflict resolution
    CAUSAL = "causal"          # Causally consistent with vector clocks
    SESSION = "session"        # Consistent within user session
    WEAK = "weak"              # Best-effort consistency


class GroupEventType(Enum):
    """Types of group-related events that require consistency"""
    MEMBERSHIP_ADD = "membership_add"
    MEMBERSHIP_REMOVE = "membership_remove"
    MEMBERSHIP_UPDATE = "membership_update"
    ROLE_CHANGE = "role_change"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    GROUP_CREATED = "group_created"
    GROUP_UPDATED = "group_updated"
    GROUP_DELETED = "group_deleted"
    BULK_ASSIGNMENT = "bulk_assignment"
    PROGRESS_UPDATE = "progress_update"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts in group assignments"""
    TIMESTAMP = "timestamp"           # Last-write-wins based on timestamp
    PRIORITY = "priority"            # Based on user/admin priority
    MERGE = "merge"                  # Attempt to merge conflicting changes
    MANUAL = "manual"                # Require manual resolution
    ROLE_HIERARCHY = "role_hierarchy" # Based on organizational role hierarchy


@dataclass
class VectorClock:
    """Vector clock for tracking causal relationships"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        """Increment clock for specific node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update this vector clock with another"""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this event happens before other"""
        if not other.clocks:
            return False
        
        all_less_equal = True
        any_less = False
        
        for node_id, clock in other.clocks.items():
            our_clock = self.clocks.get(node_id, 0)
            if our_clock > clock:
                all_less_equal = False
                break
            if our_clock < clock:
                any_less = True
        
        return all_less_equal and any_less
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if events are concurrent"""
        return not self.happens_before(other) and not other.happens_before(self)
    
    def to_dict(self) -> Dict[str, int]:
        return self.clocks.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        return cls(clocks=data.copy())


@dataclass
class GroupEvent:
    """Event representing a change to group membership or permissions"""
    event_id: str
    event_type: GroupEventType
    organization_id: str
    group_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Event data
    event_data: Dict[str, Any] = field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Dict[str, Any] = field(default_factory=dict)
    
    # Consistency metadata
    consistency_level: ConsistencyLevel = ConsistencyLevel.CAUSAL
    vector_clock: VectorClock = field(default_factory=VectorClock)
    originator_node: str = ""
    
    # Timing
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    committed_at: Optional[float] = None
    
    # Conflict resolution
    conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP
    priority: int = 5  # 1-10, 10 being highest priority
    
    # Dependencies and causality
    depends_on: List[str] = field(default_factory=list)  # Event IDs this depends on
    invalidates: List[str] = field(default_factory=list)  # Event IDs this makes obsolete
    
    # Metadata
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class ConflictDetection:
    """Detected conflict between concurrent events"""
    conflict_id: str
    conflicting_events: List[GroupEvent]
    conflict_type: str
    detected_at: float = field(default_factory=time.time)
    resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.TIMESTAMP
    resolved: bool = False
    resolution_event_id: Optional[str] = None
    manual_intervention_required: bool = False


class RealTimeGroupConsistency:
    """
    Real-time consistency manager for group memberships and permissions.
    
    Provides:
    - Event-driven consistency with configurable levels
    - Causal consistency using vector clocks
    - Automatic conflict detection and resolution
    - Cross-region event propagation
    - Real-time subscription to group changes
    """
    
    def __init__(self, node_id: str, region: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.region = region
        self.config = config
        
        # Vector clock for this node
        self.vector_clock = VectorClock()
        
        # Event storage and tracking
        self.event_log: Dict[str, GroupEvent] = {}
        self.pending_events: deque = deque()
        self.applied_events: Set[str] = set()
        
        # Conflict detection and resolution
        self.detected_conflicts: Dict[str, ConflictDetection] = {}
        self.conflict_resolvers: Dict[ConflictResolutionStrategy, Callable] = {
            ConflictResolutionStrategy.TIMESTAMP: self._resolve_by_timestamp,
            ConflictResolutionStrategy.PRIORITY: self._resolve_by_priority,
            ConflictResolutionStrategy.MERGE: self._resolve_by_merge,
            ConflictResolutionStrategy.ROLE_HIERARCHY: self._resolve_by_role_hierarchy
        }
        
        # Subscriptions and callbacks
        self.event_subscribers: Dict[str, List[Callable]] = defaultdict(list)  # event_type -> callbacks
        self.group_subscribers: Dict[str, List[Callable]] = defaultdict(list)   # group_id -> callbacks
        self.user_subscribers: Dict[str, List[Callable]] = defaultdict(list)    # user_id -> callbacks
        
        # State tracking
        self.group_states: Dict[str, Dict[str, Any]] = {}  # group_id -> current state
        self.user_group_memberships: Dict[str, Set[str]] = defaultdict(set)  # user_id -> group_ids
        
        # Network and cluster management
        self.cluster_nodes: Dict[str, Dict[str, Any]] = {}
        self.node_vector_clocks: Dict[str, VectorClock] = {}
        
        # Performance metrics
        self.consistency_metrics = {
            'events_processed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'average_propagation_time': 0.0,
            'consistency_violations': 0,
            'rollbacks_executed': 0
        }
        
        # Initialize consistency system
        asyncio.create_task(self._initialize_consistency_system())
        
        # Start background tasks
        asyncio.create_task(self._event_processing_loop())
        asyncio.create_task(self._conflict_resolution_loop())
        asyncio.create_task(self._consistency_monitoring_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def _initialize_consistency_system(self):
        """Initialize the consistency system"""
        # Initialize vector clock for this node
        self.vector_clock.increment(self.node_id)
        
        # Load current state from persistent storage
        await self._load_current_state()
        
        # Connect to cluster nodes
        await self._connect_to_cluster()
        
        logger.info(f"Real-time group consistency initialized for node {self.node_id} in region {self.region}")
    
    async def apply_group_event(self, event: GroupEvent) -> bool:
        """
        Apply a group event with appropriate consistency guarantees.
        
        Returns True if event was successfully applied, False if rejected.
        """
        event.event_id = event.event_id or str(uuid.uuid4())
        event.originator_node = self.node_id
        
        # Increment our vector clock
        self.vector_clock.increment(self.node_id)
        event.vector_clock = VectorClock(clocks=self.vector_clock.clocks.copy())
        
        logger.debug(f"Applying group event {event.event_id}: {event.event_type.value}")
        
        # Validate event
        validation_result = await self._validate_event(event)
        if not validation_result['valid']:
            logger.warning(f"Event validation failed: {validation_result['reason']}")
            return False
        
        # Store event
        self.event_log[event.event_id] = event
        
        # Apply based on consistency level
        if event.consistency_level == ConsistencyLevel.STRONG:
            success = await self._apply_strong_consistency_event(event)
        elif event.consistency_level == ConsistencyLevel.CAUSAL:
            success = await self._apply_causal_consistency_event(event)
        elif event.consistency_level == ConsistencyLevel.EVENTUAL:
            success = await self._apply_eventual_consistency_event(event)
        elif event.consistency_level == ConsistencyLevel.SESSION:
            success = await self._apply_session_consistency_event(event)
        else:  # WEAK
            success = await self._apply_weak_consistency_event(event)
        
        if success:
            # Mark as applied
            self.applied_events.add(event.event_id)
            event.processed_at = time.time()
            
            # Propagate to other nodes
            await self._propagate_event_to_cluster(event)
            
            # Notify subscribers
            await self._notify_event_subscribers(event)
            
            # Update metrics
            self.consistency_metrics['events_processed'] += 1
        
        return success
    
    async def _apply_strong_consistency_event(self, event: GroupEvent) -> bool:
        """Apply event with strong consistency (synchronous across all nodes)"""
        logger.debug(f"Applying strong consistency event {event.event_id}")
        
        # Get consensus from all nodes before applying
        consensus_result = await self._get_strong_consistency_consensus(event)
        if not consensus_result:
            return False
        
        # Apply the event locally
        success = await self._apply_event_locally(event)
        if success:
            event.committed_at = time.time()
            
            # Notify all nodes to commit
            await self._broadcast_commit_event(event.event_id)
        
        return success
    
    async def _apply_causal_consistency_event(self, event: GroupEvent) -> bool:
        """Apply event with causal consistency (respects causality)"""
        logger.debug(f"Applying causal consistency event {event.event_id}")
        
        # Check if all causal dependencies are satisfied
        if not await self._check_causal_dependencies(event):
            # Add to pending events queue
            self.pending_events.append(event)
            return True  # Will be applied when dependencies are satisfied
        
        # Apply the event
        success = await self._apply_event_locally(event)
        if success:
            # Update our vector clock
            self.vector_clock.update(event.vector_clock)
            
            # Process any pending events that might now be ready
            await self._process_pending_events()
        
        return success
    
    async def _apply_eventual_consistency_event(self, event: GroupEvent) -> bool:
        """Apply event with eventual consistency (optimistic with conflict resolution)"""
        logger.debug(f"Applying eventual consistency event {event.event_id}")
        
        # Apply optimistically
        success = await self._apply_event_locally(event)
        
        if success:
            # Check for conflicts asynchronously
            asyncio.create_task(self._detect_conflicts_async(event))
        
        return success
    
    async def _apply_session_consistency_event(self, event: GroupEvent) -> bool:
        """Apply event with session consistency (consistent within user session)"""
        logger.debug(f"Applying session consistency event {event.event_id}")
        
        # For session consistency, we ensure all events from the same session
        # are applied in order
        if event.session_id:
            await self._ensure_session_ordering(event)
        
        return await self._apply_event_locally(event)
    
    async def _apply_weak_consistency_event(self, event: GroupEvent) -> bool:
        """Apply event with weak consistency (best effort)"""
        logger.debug(f"Applying weak consistency event {event.event_id}")
        
        # Apply immediately without any ordering guarantees
        return await self._apply_event_locally(event)
    
    async def _apply_event_locally(self, event: GroupEvent) -> bool:
        """Apply event to local state"""
        try:
            if event.event_type == GroupEventType.MEMBERSHIP_ADD:
                await self._handle_membership_add(event)
            elif event.event_type == GroupEventType.MEMBERSHIP_REMOVE:
                await self._handle_membership_remove(event)
            elif event.event_type == GroupEventType.MEMBERSHIP_UPDATE:
                await self._handle_membership_update(event)
            elif event.event_type == GroupEventType.ROLE_CHANGE:
                await self._handle_role_change(event)
            elif event.event_type == GroupEventType.PERMISSION_GRANT:
                await self._handle_permission_grant(event)
            elif event.event_type == GroupEventType.PERMISSION_REVOKE:
                await self._handle_permission_revoke(event)
            elif event.event_type == GroupEventType.GROUP_CREATED:
                await self._handle_group_created(event)
            elif event.event_type == GroupEventType.GROUP_UPDATED:
                await self._handle_group_updated(event)
            elif event.event_type == GroupEventType.GROUP_DELETED:
                await self._handle_group_deleted(event)
            elif event.event_type == GroupEventType.BULK_ASSIGNMENT:
                await self._handle_bulk_assignment(event)
            elif event.event_type == GroupEventType.PROGRESS_UPDATE:
                await self._handle_progress_update(event)
            else:
                logger.warning(f"Unknown event type: {event.event_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying event {event.event_id}: {e}")
            return False
    
    # Event handlers
    
    async def _handle_membership_add(self, event: GroupEvent):
        """Handle adding user to group"""
        user_id = event.user_id
        group_id = event.group_id
        
        if user_id and group_id:
            # Add to user's group memberships
            self.user_group_memberships[user_id].add(group_id)
            
            # Update group state
            if group_id not in self.group_states:
                self.group_states[group_id] = {'members': set(), 'roles': {}}
            
            self.group_states[group_id]['members'].add(user_id)
            
            # Set role if specified
            role = event.event_data.get('role', 'learner')
            self.group_states[group_id]['roles'][user_id] = role
            
            logger.debug(f"Added user {user_id} to group {group_id} with role {role}")
    
    async def _handle_membership_remove(self, event: GroupEvent):
        """Handle removing user from group"""
        user_id = event.user_id
        group_id = event.group_id
        
        if user_id and group_id:
            # Remove from user's group memberships
            self.user_group_memberships[user_id].discard(group_id)
            
            # Update group state
            if group_id in self.group_states:
                self.group_states[group_id]['members'].discard(user_id)
                self.group_states[group_id]['roles'].pop(user_id, None)
            
            logger.debug(f"Removed user {user_id} from group {group_id}")
    
    async def _handle_membership_update(self, event: GroupEvent):
        """Handle updating group membership details"""
        user_id = event.user_id
        group_id = event.group_id
        
        if user_id and group_id and group_id in self.group_states:
            # Update membership data
            update_data = event.event_data.get('updates', {})
            
            # Handle role changes
            if 'role' in update_data:
                self.group_states[group_id]['roles'][user_id] = update_data['role']
            
            # Handle other membership updates (progress, status, etc.)
            membership_key = f"{group_id}:{user_id}"
            if membership_key not in self.group_states[group_id]:
                self.group_states[group_id][membership_key] = {}
            
            self.group_states[group_id][membership_key].update(update_data)
            
            logger.debug(f"Updated membership for user {user_id} in group {group_id}")
    
    async def _handle_role_change(self, event: GroupEvent):
        """Handle user role change in group"""
        user_id = event.user_id
        group_id = event.group_id
        new_role = event.event_data.get('new_role')
        
        if user_id and group_id and new_role:
            if group_id in self.group_states and user_id in self.group_states[group_id]['members']:
                old_role = self.group_states[group_id]['roles'].get(user_id)
                self.group_states[group_id]['roles'][user_id] = new_role
                
                logger.debug(f"Changed role for user {user_id} in group {group_id}: {old_role} -> {new_role}")
    
    async def _handle_permission_grant(self, event: GroupEvent):
        """Handle granting permission to user in group"""
        user_id = event.user_id
        group_id = event.group_id
        permissions = event.event_data.get('permissions', [])
        
        if user_id and group_id and permissions:
            # Update group permissions
            if group_id not in self.group_states:
                self.group_states[group_id] = {'members': set(), 'roles': {}, 'permissions': {}}
            
            if 'permissions' not in self.group_states[group_id]:
                self.group_states[group_id]['permissions'] = {}
            
            if user_id not in self.group_states[group_id]['permissions']:
                self.group_states[group_id]['permissions'][user_id] = set()
            
            self.group_states[group_id]['permissions'][user_id].update(permissions)
            
            logger.debug(f"Granted permissions {permissions} to user {user_id} in group {group_id}")
    
    async def _handle_permission_revoke(self, event: GroupEvent):
        """Handle revoking permission from user in group"""
        user_id = event.user_id
        group_id = event.group_id
        permissions = event.event_data.get('permissions', [])
        
        if user_id and group_id and permissions:
            if (group_id in self.group_states and 
                'permissions' in self.group_states[group_id] and
                user_id in self.group_states[group_id]['permissions']):
                
                for permission in permissions:
                    self.group_states[group_id]['permissions'][user_id].discard(permission)
                
                logger.debug(f"Revoked permissions {permissions} from user {user_id} in group {group_id}")
    
    async def _handle_group_created(self, event: GroupEvent):
        """Handle group creation"""
        group_id = event.group_id
        group_data = event.event_data.get('group_data', {})
        
        if group_id:
            self.group_states[group_id] = {
                'members': set(),
                'roles': {},
                'permissions': {},
                'created_at': event.created_at,
                'created_by': event.user_id,
                **group_data
            }
            
            logger.debug(f"Created group {group_id}")
    
    async def _handle_group_updated(self, event: GroupEvent):
        """Handle group updates"""
        group_id = event.group_id
        updates = event.event_data.get('updates', {})
        
        if group_id and group_id in self.group_states:
            self.group_states[group_id].update(updates)
            
            logger.debug(f"Updated group {group_id}")
    
    async def _handle_group_deleted(self, event: GroupEvent):
        """Handle group deletion"""
        group_id = event.group_id
        
        if group_id and group_id in self.group_states:
            # Remove all user memberships
            for user_id in self.group_states[group_id]['members'].copy():
                self.user_group_memberships[user_id].discard(group_id)
            
            # Remove group state
            del self.group_states[group_id]
            
            logger.debug(f"Deleted group {group_id}")
    
    async def _handle_bulk_assignment(self, event: GroupEvent):
        """Handle bulk assignment of users to groups"""
        assignments = event.event_data.get('assignments', [])
        
        for assignment in assignments:
            user_id = assignment.get('user_id')
            group_id = assignment.get('group_id')
            role = assignment.get('role', 'learner')
            
            if user_id and group_id:
                # Create membership add event
                membership_event = GroupEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=GroupEventType.MEMBERSHIP_ADD,
                    organization_id=event.organization_id,
                    group_id=group_id,
                    user_id=user_id,
                    event_data={'role': role},
                    consistency_level=ConsistencyLevel.EVENTUAL,
                    vector_clock=VectorClock(clocks=self.vector_clock.clocks.copy())
                )
                
                await self._handle_membership_add(membership_event)
        
        logger.debug(f"Processed bulk assignment with {len(assignments)} assignments")
    
    async def _handle_progress_update(self, event: GroupEvent):
        """Handle user progress updates in groups"""
        user_id = event.user_id
        group_id = event.group_id
        progress_data = event.event_data.get('progress', {})
        
        if user_id and group_id and group_id in self.group_states:
            # Update progress data
            progress_key = f"progress:{user_id}"
            if progress_key not in self.group_states[group_id]:
                self.group_states[group_id][progress_key] = {}
            
            self.group_states[group_id][progress_key].update(progress_data)
            
            logger.debug(f"Updated progress for user {user_id} in group {group_id}")
    
    # Consistency and conflict management
    
    async def _validate_event(self, event: GroupEvent) -> Dict[str, Any]:
        """Validate event before applying"""
        if not event.organization_id:
            return {'valid': False, 'reason': 'Missing organization_id'}
        
        if event.event_type in [GroupEventType.MEMBERSHIP_ADD, GroupEventType.MEMBERSHIP_REMOVE,
                               GroupEventType.MEMBERSHIP_UPDATE, GroupEventType.ROLE_CHANGE]:
            if not event.user_id or not event.group_id:
                return {'valid': False, 'reason': 'Missing user_id or group_id for membership event'}
        
        return {'valid': True}
    
    async def _check_causal_dependencies(self, event: GroupEvent) -> bool:
        """Check if all causal dependencies for an event are satisfied"""
        for dep_event_id in event.depends_on:
            if dep_event_id not in self.applied_events:
                return False
        
        # Check vector clock causality
        for node_id, clock in event.vector_clock.clocks.items():
            our_clock = self.vector_clock.clocks.get(node_id, 0)
            if our_clock < clock - 1:  # We're missing some events from this node
                return False
        
        return True
    
    async def _process_pending_events(self):
        """Process events from the pending queue that are now ready"""
        processed_any = True
        
        while processed_any and self.pending_events:
            processed_any = False
            ready_events = []
            
            # Find events that are ready to be processed
            for event in list(self.pending_events):
                if await self._check_causal_dependencies(event):
                    ready_events.append(event)
                    self.pending_events.remove(event)
                    processed_any = True
            
            # Apply ready events
            for event in ready_events:
                await self._apply_event_locally(event)
                self.applied_events.add(event.event_id)
                event.processed_at = time.time()
                
                # Update vector clock
                self.vector_clock.update(event.vector_clock)
    
    async def _detect_conflicts_async(self, event: GroupEvent):
        """Asynchronously detect conflicts with other events"""
        # Look for concurrent events that might conflict
        conflicts = await self._find_conflicting_events(event)
        
        if conflicts:
            conflict_id = str(uuid.uuid4())
            conflict_detection = ConflictDetection(
                conflict_id=conflict_id,
                conflicting_events=conflicts + [event],
                conflict_type=self._classify_conflict(conflicts + [event]),
                resolution_strategy=event.conflict_resolution_strategy
            )
            
            self.detected_conflicts[conflict_id] = conflict_detection
            self.consistency_metrics['conflicts_detected'] += 1
            
            logger.warning(f"Conflict detected {conflict_id}: {len(conflicts + [event])} conflicting events")
            
            # Try to resolve automatically
            await self._attempt_conflict_resolution(conflict_id)
    
    async def _find_conflicting_events(self, event: GroupEvent) -> List[GroupEvent]:
        """Find events that conflict with the given event"""
        conflicts = []
        
        # Define conflict time window (events within this window might conflict)
        conflict_window = 60.0  # 60 seconds
        
        for other_event in self.event_log.values():
            if other_event.event_id == event.event_id:
                continue
            
            # Check if events are concurrent and might conflict
            if (abs(other_event.created_at - event.created_at) <= conflict_window and
                event.vector_clock.concurrent_with(other_event.vector_clock) and
                self._events_conflict(event, other_event)):
                
                conflicts.append(other_event)
        
        return conflicts
    
    def _events_conflict(self, event1: GroupEvent, event2: GroupEvent) -> bool:
        """Check if two events conflict with each other"""
        # Same organization
        if event1.organization_id != event2.organization_id:
            return False
        
        # Group membership conflicts
        if (event1.event_type in [GroupEventType.MEMBERSHIP_ADD, GroupEventType.MEMBERSHIP_REMOVE] and
            event2.event_type in [GroupEventType.MEMBERSHIP_ADD, GroupEventType.MEMBERSHIP_REMOVE] and
            event1.user_id == event2.user_id and event1.group_id == event2.group_id):
            return True
        
        # Role change conflicts
        if (event1.event_type == GroupEventType.ROLE_CHANGE and
            event2.event_type == GroupEventType.ROLE_CHANGE and
            event1.user_id == event2.user_id and event1.group_id == event2.group_id):
            return True
        
        # Permission conflicts
        if (event1.event_type in [GroupEventType.PERMISSION_GRANT, GroupEventType.PERMISSION_REVOKE] and
            event2.event_type in [GroupEventType.PERMISSION_GRANT, GroupEventType.PERMISSION_REVOKE] and
            event1.user_id == event2.user_id and event1.group_id == event2.group_id):
            
            # Check if they affect the same permissions
            permissions1 = set(event1.event_data.get('permissions', []))
            permissions2 = set(event2.event_data.get('permissions', []))
            return bool(permissions1.intersection(permissions2))
        
        return False
    
    def _classify_conflict(self, events: List[GroupEvent]) -> str:
        """Classify the type of conflict"""
        event_types = {event.event_type for event in events}
        
        if GroupEventType.MEMBERSHIP_ADD in event_types and GroupEventType.MEMBERSHIP_REMOVE in event_types:
            return "membership_add_remove_conflict"
        elif event_types == {GroupEventType.ROLE_CHANGE}:
            return "concurrent_role_change_conflict"
        elif GroupEventType.PERMISSION_GRANT in event_types and GroupEventType.PERMISSION_REVOKE in event_types:
            return "permission_grant_revoke_conflict"
        else:
            return "concurrent_modification_conflict"
    
    async def _attempt_conflict_resolution(self, conflict_id: str):
        """Attempt to automatically resolve a conflict"""
        conflict = self.detected_conflicts.get(conflict_id)
        if not conflict or conflict.resolved:
            return
        
        try:
            resolver = self.conflict_resolvers.get(conflict.resolution_strategy)
            if resolver:
                resolution_event = await resolver(conflict)
                
                if resolution_event:
                    # Apply resolution
                    await self.apply_group_event(resolution_event)
                    
                    # Mark conflict as resolved
                    conflict.resolved = True
                    conflict.resolution_event_id = resolution_event.event_id
                    
                    self.consistency_metrics['conflicts_resolved'] += 1
                    
                    logger.info(f"Resolved conflict {conflict_id} using {conflict.resolution_strategy.value}")
                else:
                    # Mark for manual intervention
                    conflict.manual_intervention_required = True
                    logger.warning(f"Conflict {conflict_id} requires manual intervention")
            else:
                conflict.manual_intervention_required = True
                logger.warning(f"No resolver for strategy {conflict.resolution_strategy.value}")
                
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            conflict.manual_intervention_required = True
    
    # Conflict resolution strategies
    
    async def _resolve_by_timestamp(self, conflict: ConflictDetection) -> Optional[GroupEvent]:
        """Resolve conflict using timestamp (last-write-wins)"""
        # Find the event with the latest timestamp
        latest_event = max(conflict.conflicting_events, key=lambda e: e.created_at)
        
        # Create a resolution event that applies the latest state
        resolution_event = GroupEvent(
            event_id=str(uuid.uuid4()),
            event_type=latest_event.event_type,
            organization_id=latest_event.organization_id,
            group_id=latest_event.group_id,
            user_id=latest_event.user_id,
            event_data=latest_event.event_data.copy(),
            consistency_level=ConsistencyLevel.STRONG,
            originator_node=self.node_id,
            priority=10  # High priority for resolution
        )
        
        return resolution_event
    
    async def _resolve_by_priority(self, conflict: ConflictDetection) -> Optional[GroupEvent]:
        """Resolve conflict using event priority"""
        # Find the event with highest priority
        highest_priority_event = max(conflict.conflicting_events, key=lambda e: e.priority)
        
        # Create resolution event based on highest priority
        resolution_event = GroupEvent(
            event_id=str(uuid.uuid4()),
            event_type=highest_priority_event.event_type,
            organization_id=highest_priority_event.organization_id,
            group_id=highest_priority_event.group_id,
            user_id=highest_priority_event.user_id,
            event_data=highest_priority_event.event_data.copy(),
            consistency_level=ConsistencyLevel.STRONG,
            originator_node=self.node_id,
            priority=10
        )
        
        return resolution_event
    
    async def _resolve_by_merge(self, conflict: ConflictDetection) -> Optional[GroupEvent]:
        """Resolve conflict by attempting to merge changes"""
        if conflict.conflict_type == "permission_grant_revoke_conflict":
            # For permission conflicts, merge all grant operations
            granted_permissions = set()
            revoked_permissions = set()
            
            for event in conflict.conflicting_events:
                permissions = set(event.event_data.get('permissions', []))
                if event.event_type == GroupEventType.PERMISSION_GRANT:
                    granted_permissions.update(permissions)
                elif event.event_type == GroupEventType.PERMISSION_REVOKE:
                    revoked_permissions.update(permissions)
            
            # Final permissions = granted - revoked
            final_permissions = granted_permissions - revoked_permissions
            
            if final_permissions:
                # Create grant event for final permissions
                resolution_event = GroupEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=GroupEventType.PERMISSION_GRANT,
                    organization_id=conflict.conflicting_events[0].organization_id,
                    group_id=conflict.conflicting_events[0].group_id,
                    user_id=conflict.conflicting_events[0].user_id,
                    event_data={'permissions': list(final_permissions)},
                    consistency_level=ConsistencyLevel.STRONG,
                    originator_node=self.node_id,
                    priority=10
                )
                return resolution_event
        
        # If merge is not possible, fall back to timestamp resolution
        return await self._resolve_by_timestamp(conflict)
    
    async def _resolve_by_role_hierarchy(self, conflict: ConflictDetection) -> Optional[GroupEvent]:
        """Resolve conflict based on organizational role hierarchy"""
        # This would require access to organization role hierarchy
        # For now, fall back to priority resolution
        return await self._resolve_by_priority(conflict)
    
    # Subscription and notification system
    
    def subscribe_to_group_events(self, group_id: str, callback: Callable):
        """Subscribe to events for a specific group"""
        self.group_subscribers[group_id].append(callback)
    
    def subscribe_to_user_events(self, user_id: str, callback: Callable):
        """Subscribe to events for a specific user"""
        self.user_subscribers[user_id].append(callback)
    
    def subscribe_to_event_type(self, event_type: GroupEventType, callback: Callable):
        """Subscribe to specific type of events"""
        self.event_subscribers[event_type.value].append(callback)
    
    async def _notify_event_subscribers(self, event: GroupEvent):
        """Notify all relevant subscribers about the event"""
        notifications = []
        
        # Notify event type subscribers
        for callback in self.event_subscribers.get(event.event_type.value, []):
            notifications.append(callback(event))
        
        # Notify group subscribers
        if event.group_id:
            for callback in self.group_subscribers.get(event.group_id, []):
                notifications.append(callback(event))
        
        # Notify user subscribers
        if event.user_id:
            for callback in self.user_subscribers.get(event.user_id, []):
                notifications.append(callback(event))
        
        # Execute notifications
        if notifications:
            await asyncio.gather(*notifications, return_exceptions=True)
    
    # Cluster and network operations
    
    async def _get_strong_consistency_consensus(self, event: GroupEvent) -> bool:
        """Get consensus from all nodes for strong consistency"""
        # Placeholder - would implement actual consensus protocol
        return True
    
    async def _broadcast_commit_event(self, event_id: str):
        """Broadcast commit message to all nodes"""
        # Placeholder - would implement actual network broadcast
        pass
    
    async def _propagate_event_to_cluster(self, event: GroupEvent):
        """Propagate event to other nodes in the cluster"""
        # Placeholder - would implement actual network propagation
        logger.debug(f"Propagating event {event.event_id} to cluster")
    
    async def _ensure_session_ordering(self, event: GroupEvent):
        """Ensure events from the same session are ordered correctly"""
        # Placeholder - would implement session ordering logic
        pass
    
    async def _load_current_state(self):
        """Load current state from persistent storage"""
        # Placeholder - would load from database
        pass
    
    async def _connect_to_cluster(self):
        """Connect to other nodes in the cluster"""
        # Placeholder - would establish cluster connections
        pass
    
    # Background maintenance tasks
    
    async def _event_processing_loop(self):
        """Background task for processing events"""
        while True:
            await asyncio.sleep(5)  # Process every 5 seconds
            
            # Process pending events
            if self.pending_events:
                await self._process_pending_events()
            
            # Clean up old events
            await self._cleanup_old_events()
    
    async def _conflict_resolution_loop(self):
        """Background task for resolving conflicts"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Try to resolve pending conflicts
            for conflict_id, conflict in list(self.detected_conflicts.items()):
                if not conflict.resolved and not conflict.manual_intervention_required:
                    await self._attempt_conflict_resolution(conflict_id)
    
    async def _consistency_monitoring_loop(self):
        """Background task for monitoring consistency"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Check for consistency violations
            await self._check_consistency_violations()
            
            # Update metrics
            await self._update_consistency_metrics()
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            
            # Cleanup old events
            await self._cleanup_old_events()
            
            # Cleanup resolved conflicts
            await self._cleanup_resolved_conflicts()
    
    async def _cleanup_old_events(self):
        """Clean up old events from memory"""
        current_time = time.time()
        retention_period = self.config.get('event_retention_hours', 24) * 3600
        
        old_event_ids = []
        for event_id, event in self.event_log.items():
            if current_time - event.created_at > retention_period:
                old_event_ids.append(event_id)
        
        for event_id in old_event_ids:
            del self.event_log[event_id]
            self.applied_events.discard(event_id)
    
    async def _cleanup_resolved_conflicts(self):
        """Clean up resolved conflicts"""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        resolved_conflicts = []
        for conflict_id, conflict in self.detected_conflicts.items():
            if conflict.resolved and (current_time - conflict.detected_at) > cleanup_age:
                resolved_conflicts.append(conflict_id)
        
        for conflict_id in resolved_conflicts:
            del self.detected_conflicts[conflict_id]
    
    async def _check_consistency_violations(self):
        """Check for consistency violations"""
        # Placeholder - would implement consistency checks
        pass
    
    async def _update_consistency_metrics(self):
        """Update consistency metrics"""
        # Calculate average propagation time
        recent_events = [e for e in self.event_log.values() 
                        if e.processed_at and (time.time() - e.created_at) < 3600]
        
        if recent_events:
            propagation_times = [e.processed_at - e.created_at for e in recent_events 
                               if e.processed_at > e.created_at]
            if propagation_times:
                self.consistency_metrics['average_propagation_time'] = sum(propagation_times) / len(propagation_times)
    
    # Public API methods
    
    def get_user_group_memberships(self, user_id: str) -> Set[str]:
        """Get all group memberships for a user"""
        return self.user_group_memberships.get(user_id, set()).copy()
    
    def get_group_members(self, group_id: str) -> Set[str]:
        """Get all members of a group"""
        group_state = self.group_states.get(group_id)
        if group_state:
            return group_state.get('members', set()).copy()
        return set()
    
    def get_user_role_in_group(self, user_id: str, group_id: str) -> Optional[str]:
        """Get user's role in a specific group"""
        group_state = self.group_states.get(group_id)
        if group_state:
            return group_state.get('roles', {}).get(user_id)
        return None
    
    def get_user_permissions_in_group(self, user_id: str, group_id: str) -> Set[str]:
        """Get user's permissions in a specific group"""
        group_state = self.group_states.get(group_id)
        if group_state and 'permissions' in group_state:
            return group_state['permissions'].get(user_id, set()).copy()
        return set()
    
    def get_consistency_metrics(self) -> Dict[str, Any]:
        """Get consistency performance metrics"""
        return self.consistency_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            'node_id': self.node_id,
            'region': self.region,
            'vector_clock': self.vector_clock.to_dict(),
            'active_events': len(self.event_log),
            'pending_events': len(self.pending_events),
            'applied_events': len(self.applied_events),
            'active_conflicts': len([c for c in self.detected_conflicts.values() if not c.resolved]),
            'group_count': len(self.group_states),
            'total_memberships': sum(len(memberships) for memberships in self.user_group_memberships.values())
        }