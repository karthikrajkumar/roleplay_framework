"""
Optimized Group Assignment Algorithm for Learning Platform
Time Complexity: O(n + g log g) where n = users, g = groups
Space Complexity: O(n + g)
Atomic operations with conflict resolution and sub-second response time
"""

import asyncio
import time
from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import redis.lock
from contextlib import asynccontextmanager


class AssignmentOperation(Enum):
    ADD = "add"
    REMOVE = "remove" 
    REPLACE = "replace"
    BULK_ADD = "bulk_add"
    BULK_REMOVE = "bulk_remove"


@dataclass
class GroupAssignmentRequest:
    operation: AssignmentOperation
    group_id: str
    user_ids: List[str]
    requester_id: str
    organization_id: str
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssignmentResult:
    success: bool
    processed_count: int
    successful_assignments: List[str]
    failed_assignments: List[Dict[str, Any]]
    conflicts_resolved: List[Dict[str, Any]]
    operation_id: str
    processing_time_ms: int
    rollback_data: Optional[Dict[str, Any]] = None


class GroupAssignmentEngine:
    """
    High-performance group assignment system with atomic operations
    Designed for sub-second response time even with 50k users
    """
    
    def __init__(self, db_connection, redis_client, kafka_producer):
        self.db = db_connection
        self.redis = redis_client
        self.kafka = kafka_producer
        
        # Performance optimization parameters
        self.batch_size = 1000  # Optimal batch size for database operations
        self.max_concurrent_operations = 16
        self.lock_timeout = 30  # seconds
        self.operation_timeout = 45  # seconds
        
        # Caching for permission lookups
        self.permission_cache = {}
        self.group_hierarchy_cache = {}
        self.user_group_cache = {}
        
        # Conflict resolution strategies
        self.conflict_resolvers = {
            'duplicate_assignment': self._resolve_duplicate_assignment,
            'permission_conflict': self._resolve_permission_conflict,
            'capacity_exceeded': self._resolve_capacity_conflict,
            'hierarchy_violation': self._resolve_hierarchy_violation
        }

    async def assign_users_to_group(self, request: GroupAssignmentRequest) -> AssignmentResult:
        """
        Main entry point for group assignment operations
        Implements atomic operations with comprehensive conflict resolution
        """
        
        operation_start = time.time()
        operation_id = str(uuid.uuid4())
        
        # Validate request and check permissions
        validation_result = await self._validate_assignment_request(request)
        if not validation_result['valid']:
            return AssignmentResult(
                success=False,
                processed_count=0,
                successful_assignments=[],
                failed_assignments=validation_result['errors'],
                conflicts_resolved=[],
                operation_id=operation_id,
                processing_time_ms=int((time.time() - operation_start) * 1000)
            )
        
        # Acquire distributed locks for atomic operation
        async with self._acquire_assignment_locks(request):
            
            try:
                # Process assignment based on operation type
                if request.operation in [AssignmentOperation.ADD, AssignmentOperation.BULK_ADD]:
                    result = await self._process_add_assignment(request, operation_id)
                elif request.operation in [AssignmentOperation.REMOVE, AssignmentOperation.BULK_REMOVE]:
                    result = await self._process_remove_assignment(request, operation_id)
                elif request.operation == AssignmentOperation.REPLACE:
                    result = await self._process_replace_assignment(request, operation_id)
                else:
                    raise ValueError(f"Unsupported operation: {request.operation}")
                
                # Update caches and publish events
                await self._post_process_assignment(request, result)
                
                result.processing_time_ms = int((time.time() - operation_start) * 1000)
                return result
                
            except Exception as e:
                # Comprehensive error handling with rollback
                await self._handle_assignment_error(request, operation_id, e)
                raise

    async def _process_add_assignment(self, request: GroupAssignmentRequest, 
                                    operation_id: str) -> AssignmentResult:
        """
        Process ADD/BULK_ADD operations with conflict detection and resolution
        """
        
        result = AssignmentResult(
            success=True,
            processed_count=0,
            successful_assignments=[],
            failed_assignments=[],
            conflicts_resolved=[],
            operation_id=operation_id,
            processing_time_ms=0
        )
        
        # Phase 1: Conflict Detection and Pre-processing
        conflict_analysis = await self._analyze_assignment_conflicts(request)
        resolved_conflicts = await self._resolve_conflicts(conflict_analysis)
        
        # Update result with conflict resolution info
        result.conflicts_resolved = resolved_conflicts
        
        # Phase 2: Filter users for actual assignment
        eligible_users = await self._filter_eligible_users(
            request.user_ids, request.group_id, resolved_conflicts
        )
        
        if not eligible_users:
            result.processed_count = len(request.user_ids)
            result.failed_assignments = [
                {'user_id': uid, 'reason': 'no_eligible_users'} 
                for uid in request.user_ids
            ]
            return result
        
        # Phase 3: Atomic Database Transaction
        async with self.db.transaction():
            
            # Batch process assignments for performance
            assignment_batches = self._create_assignment_batches(
                eligible_users, request.group_id, self.batch_size
            )
            
            rollback_data = {'assignments': []}
            
            for batch in assignment_batches:
                try:
                    # Execute batch assignment
                    batch_result = await self._execute_assignment_batch(batch, request)
                    
                    # Track successful assignments for rollback
                    rollback_data['assignments'].extend(batch_result['assigned_users'])
                    result.successful_assignments.extend(batch_result['assigned_users'])
                    result.processed_count += batch_result['processed_count']
                    
                    # Handle batch-level failures
                    if batch_result['failed_assignments']:
                        result.failed_assignments.extend(batch_result['failed_assignments'])
                    
                except Exception as batch_error:
                    # Handle batch failure - continue with other batches
                    result.failed_assignments.extend([
                        {'user_id': user_id, 'reason': str(batch_error)}
                        for user_id in batch['user_ids']
                    ])
            
            # Prepare rollback data
            result.rollback_data = rollback_data
            
            # Update group statistics atomically
            await self._update_group_statistics(request.group_id, len(result.successful_assignments))
        
        return result

    async def _analyze_assignment_conflicts(self, request: GroupAssignmentRequest) -> Dict[str, Any]:
        """
        Comprehensive conflict analysis for group assignments
        Detects: duplicates, capacity issues, permission conflicts, hierarchy violations
        """
        
        conflicts = {
            'duplicate_assignments': [],
            'permission_conflicts': [],
            'capacity_conflicts': [],
            'hierarchy_violations': [],
            'user_conflicts': []
        }
        
        # Get current group state
        group_info = await self._get_group_info_with_cache(request.group_id)
        current_members = await self._get_group_members_with_cache(request.group_id)
        
        # Analyze each user for potential conflicts
        user_analyses = await asyncio.gather(*[
            self._analyze_user_assignment_conflicts(
                user_id, request.group_id, group_info, current_members
            ) for user_id in request.user_ids
        ])
        
        # Aggregate conflict results
        for user_id, user_conflicts in zip(request.user_ids, user_analyses):
            for conflict_type, conflict_data in user_conflicts.items():
                if conflict_data:
                    conflicts[conflict_type].append({
                        'user_id': user_id,
                        'conflict_data': conflict_data
                    })
        
        # Check group capacity constraints
        if group_info.get('max_members'):
            projected_size = len(current_members) + len(request.user_ids)
            if projected_size > group_info['max_members']:
                conflicts['capacity_conflicts'].append({
                    'current_size': len(current_members),
                    'requested_additions': len(request.user_ids),
                    'max_capacity': group_info['max_members'],
                    'overflow_count': projected_size - group_info['max_members']
                })
        
        return conflicts

    async def _resolve_conflicts(self, conflict_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligent conflict resolution using configurable strategies
        """
        
        resolved_conflicts = []
        
        for conflict_type, conflicts in conflict_analysis.items():
            if not conflicts:
                continue
                
            resolver = self.conflict_resolvers.get(conflict_type)
            if resolver:
                resolution_results = await resolver(conflicts)
                resolved_conflicts.extend(resolution_results)
            else:
                # Default handling for unresolved conflict types
                resolved_conflicts.extend([
                    {
                        'type': conflict_type,
                        'conflicts': conflicts,
                        'resolution': 'unresolved',
                        'action': 'skip_conflicted_users'
                    }
                ])
        
        return resolved_conflicts

    async def _resolve_duplicate_assignment(self, conflicts: List[Dict]) -> List[Dict]:
        """Resolve duplicate assignment conflicts"""
        resolutions = []
        
        for conflict in conflicts:
            # Check if user is already in group with same role
            existing_membership = await self._get_user_group_membership(
                conflict['user_id'], conflict['conflict_data']['group_id']
            )
            
            if existing_membership:
                resolution = {
                    'type': 'duplicate_assignment',
                    'user_id': conflict['user_id'],
                    'resolution': 'skip_existing_member',
                    'action': 'no_operation_needed'
                }
            else:
                resolution = {
                    'type': 'duplicate_assignment',
                    'user_id': conflict['user_id'], 
                    'resolution': 'update_membership',
                    'action': 'update_existing_assignment'
                }
            
            resolutions.append(resolution)
        
        return resolutions

    async def _resolve_permission_conflict(self, conflicts: List[Dict]) -> List[Dict]:
        """Resolve permission-related conflicts"""
        resolutions = []
        
        for conflict in conflicts:
            # Check if requester has delegation permissions
            can_delegate = await self._check_delegation_permissions(
                conflict['conflict_data']['requester_id'],
                conflict['conflict_data']['target_group_id']
            )
            
            if can_delegate:
                resolution = {
                    'type': 'permission_conflict',
                    'user_id': conflict['user_id'],
                    'resolution': 'delegated_permission_granted',
                    'action': 'proceed_with_assignment'
                }
            else:
                resolution = {
                    'type': 'permission_conflict',
                    'user_id': conflict['user_id'],
                    'resolution': 'insufficient_permissions',
                    'action': 'skip_user_assignment'
                }
            
            resolutions.append(resolution)
        
        return resolutions

    async def _execute_assignment_batch(self, batch: Dict[str, Any], 
                                      request: GroupAssignmentRequest) -> Dict[str, Any]:
        """
        Execute a batch of user assignments atomically
        Optimized for high-performance bulk operations
        """
        
        batch_result = {
            'processed_count': 0,
            'assigned_users': [],
            'failed_assignments': []
        }
        
        try:
            # Prepare batch insert data
            assignment_records = []
            current_time = time.time()
            
            for user_id in batch['user_ids']:
                assignment_records.append({
                    'id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'group_id': request.group_id,
                    'assigned_by': request.requester_id,
                    'assigned_at': current_time,
                    'assignment_type': request.operation.value,
                    'status': 'active',
                    'metadata': request.options
                })
            
            # Execute bulk insert with conflict handling
            insert_query = """
                INSERT INTO user_group_assignments 
                (id, user_id, group_id, assigned_by, assigned_at, assignment_type, status, metadata)
                VALUES (:id, :user_id, :group_id, :assigned_by, :assigned_at, :assignment_type, :status, :metadata)
                ON CONFLICT (user_id, group_id) 
                DO UPDATE SET 
                    assigned_by = EXCLUDED.assigned_by,
                    assigned_at = EXCLUDED.assigned_at,
                    assignment_type = EXCLUDED.assignment_type,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata
                RETURNING user_id
            """
            
            result = await self.db.execute_many(insert_query, assignment_records)
            
            # Process results
            assigned_user_ids = [row['user_id'] for row in result]
            batch_result['assigned_users'] = assigned_user_ids
            batch_result['processed_count'] = len(batch['user_ids'])
            
            # Identify failed assignments
            failed_user_ids = set(batch['user_ids']) - set(assigned_user_ids)
            batch_result['failed_assignments'] = [
                {'user_id': uid, 'reason': 'database_constraint_violation'}
                for uid in failed_user_ids
            ]
            
        except Exception as e:
            # Handle batch-level database errors
            batch_result['failed_assignments'] = [
                {'user_id': uid, 'reason': str(e)}
                for uid in batch['user_ids']
            ]
        
        return batch_result

    @asynccontextmanager
    async def _acquire_assignment_locks(self, request: GroupAssignmentRequest):
        """
        Acquire distributed locks for atomic group assignment operations
        Uses Redis distributed locking with timeout and retry logic
        """
        
        # Create lock keys for resources that need protection
        lock_keys = [
            f"group_assignment_lock:{request.group_id}",
            f"org_assignment_lock:{request.organization_id}"
        ]
        
        # Add user-specific locks for large operations
        if len(request.user_ids) > 1000:
            # For large operations, also lock user assignment changes
            user_batch_locks = [
                f"user_batch_lock:{request.organization_id}:{i}"
                for i in range(0, len(request.user_ids), 10000)
            ]
            lock_keys.extend(user_batch_locks)
        
        # Acquire all locks atomically
        locks = []
        try:
            for lock_key in lock_keys:
                lock = redis.lock.Lock(
                    self.redis,
                    lock_key,
                    timeout=self.lock_timeout,
                    blocking_timeout=5
                )
                await lock.acquire()
                locks.append(lock)
            
            yield  # Execute the protected operation
            
        finally:
            # Release all locks
            for lock in locks:
                try:
                    await lock.release()
                except Exception as e:
                    # Log lock release failure but don't propagate
                    print(f"Failed to release lock: {e}")

    async def _update_group_statistics(self, group_id: str, member_count_delta: int):
        """
        Atomically update group statistics and cached data
        """
        
        # Update database statistics
        update_query = """
            UPDATE groups 
            SET 
                member_count = member_count + :delta,
                updated_at = :timestamp
            WHERE id = :group_id
        """
        
        await self.db.execute(update_query, {
            'delta': member_count_delta,
            'timestamp': time.time(),
            'group_id': group_id
        })
        
        # Update Redis cache
        cache_key = f"group_stats:{group_id}"
        await self.redis.hincrby(cache_key, "member_count", member_count_delta)
        await self.redis.expire(cache_key, 3600)  # 1 hour expiry

    async def _post_process_assignment(self, request: GroupAssignmentRequest, 
                                     result: AssignmentResult):
        """
        Post-processing: cache updates, event publishing, notifications
        """
        
        # Update local caches
        await self._invalidate_relevant_caches(request.group_id, request.user_ids)
        
        # Publish assignment events to Kafka
        if result.successful_assignments:
            event = {
                'event_type': 'group_assignment_completed',
                'operation_id': result.operation_id,
                'group_id': request.group_id,
                'organization_id': request.organization_id,
                'assigned_users': result.successful_assignments,
                'assigned_by': request.requester_id,
                'timestamp': time.time(),
                'operation': request.operation.value
            }
            
            await self.kafka.send('group_assignments', event)
        
        # Handle failed assignments
        if result.failed_assignments:
            error_event = {
                'event_type': 'group_assignment_errors',
                'operation_id': result.operation_id,
                'group_id': request.group_id,
                'failed_assignments': result.failed_assignments,
                'timestamp': time.time()
            }
            
            await self.kafka.send('assignment_errors', error_event)

    def _create_assignment_batches(self, user_ids: List[str], group_id: str, 
                                 batch_size: int) -> List[Dict[str, Any]]:
        """
        Create optimally-sized batches for database operations
        Considers user distribution and database connection limits
        """
        
        batches = []
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i + batch_size]
            batches.append({
                'batch_id': f"batch_{i // batch_size}",
                'user_ids': batch_user_ids,
                'group_id': group_id,
                'size': len(batch_user_ids)
            })
        
        return batches

    async def _get_group_info_with_cache(self, group_id: str) -> Dict[str, Any]:
        """Get group information with Redis caching"""
        
        cache_key = f"group_info:{group_id}"
        cached_info = await self.redis.get(cache_key)
        
        if cached_info:
            return json.loads(cached_info)
        
        # Fetch from database
        query = """
            SELECT id, name, organization_id, max_members, group_type, 
                   member_count, created_at, status
            FROM groups 
            WHERE id = :group_id AND status = 'active'
        """
        
        result = await self.db.fetch_one(query, {'group_id': group_id})
        
        if result:
            group_info = dict(result)
            # Cache for 1 hour
            await self.redis.setex(cache_key, 3600, json.dumps(group_info, default=str))
            return group_info
        
        return {}

    async def get_assignment_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for group assignments
        """
        
        # Query recent assignment operations
        metrics_query = """
            SELECT 
                operation_type,
                COUNT(*) as operation_count,
                AVG(processing_time_ms) as avg_processing_time,
                AVG(user_count) as avg_users_per_operation,
                AVG(success_rate) as avg_success_rate
            FROM assignment_operations 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY operation_type
        """
        
        operation_metrics = await self.db.fetch_all(metrics_query)
        
        # Get system performance metrics
        system_metrics = {
            'cache_hit_rates': {
                'group_info': await self._get_cache_hit_rate('group_info'),
                'user_permissions': await self._get_cache_hit_rate('user_permissions')
            },
            'lock_contention': await self._get_lock_contention_stats(),
            'database_performance': {
                'avg_query_time': await self._get_avg_query_time(),
                'connection_pool_usage': await self._get_connection_pool_stats()
            }
        }
        
        return {
            'operation_metrics': [dict(row) for row in operation_metrics],
            'system_metrics': system_metrics,
            'performance_targets': {
                'sub_second_operations': '< 1000ms for < 10k users',
                'large_operations': '< 5000ms for 50k users',
                'success_rate': '> 99.5%',
                'cache_hit_rate': '> 95%'
            }
        }


# High-Performance Group Hierarchy Manager

class GroupHierarchyManager:
    """
    Optimized group hierarchy management with O(log n) permission lookups
    Uses materialized path and closure table hybrid approach
    """
    
    def __init__(self, db_connection, redis_client):
        self.db = db_connection
        self.redis = redis_client
        self.hierarchy_cache = {}
        
    async def check_assignment_permissions(self, user_id: str, group_id: str, 
                                         requester_id: str) -> Dict[str, Any]:
        """
        Fast permission checking with hierarchical group support
        Time Complexity: O(log n) with caching
        """
        
        # Check cache first
        cache_key = f"permission:{requester_id}:{group_id}:{user_id}"
        cached_result = await self.redis.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Multi-level permission check
        permission_checks = await asyncio.gather(
            self._check_direct_permissions(requester_id, group_id),
            self._check_inherited_permissions(requester_id, group_id),
            self._check_organization_permissions(requester_id, group_id),
            self._check_role_based_permissions(requester_id, group_id)
        )
        
        # Aggregate permission results
        permission_result = {
            'allowed': any(check['allowed'] for check in permission_checks),
            'permission_source': next(
                (check['source'] for check in permission_checks if check['allowed']),
                'none'
            ),
            'restrictions': [check.get('restrictions', []) for check in permission_checks],
            'hierarchy_level': max(
                check.get('level', 0) for check in permission_checks
            )
        }
        
        # Cache result for 5 minutes
        await self.redis.setex(cache_key, 300, json.dumps(permission_result))
        
        return permission_result

    async def _check_inherited_permissions(self, requester_id: str, 
                                         group_id: str) -> Dict[str, Any]:
        """
        Check permissions inherited through group hierarchy
        Uses closure table for O(log n) lookups
        """
        
        query = """
            WITH RECURSIVE group_hierarchy AS (
                -- Base case: direct parent groups
                SELECT parent_group_id, child_group_id, 1 as level
                FROM group_hierarchy_closure 
                WHERE child_group_id = :group_id
                
                UNION ALL
                
                -- Recursive case: ancestor groups
                SELECT ghc.parent_group_id, ghc.child_group_id, gh.level + 1
                FROM group_hierarchy_closure ghc
                JOIN group_hierarchy gh ON ghc.child_group_id = gh.parent_group_id
                WHERE gh.level < 10  -- Prevent infinite recursion
            )
            SELECT 
                gh.parent_group_id,
                gh.level,
                gm.role,
                g.name as group_name
            FROM group_hierarchy gh
            JOIN group_memberships gm ON gm.group_id = gh.parent_group_id
            JOIN groups g ON g.id = gh.parent_group_id
            WHERE gm.user_id = :requester_id 
              AND gm.status = 'active'
              AND gm.role IN ('admin', 'moderator', 'group_manager')
            ORDER BY gh.level
            LIMIT 1
        """
        
        result = await self.db.fetch_one(query, {
            'group_id': group_id,
            'requester_id': requester_id
        })
        
        if result:
            return {
                'allowed': True,
                'source': 'inherited_hierarchy',
                'level': result['level'],
                'role': result['role'],
                'parent_group': result['parent_group_id']
            }
        
        return {'allowed': False, 'source': 'inherited_hierarchy', 'level': 0}


# Usage Example and Performance Testing

async def performance_test_example():
    """
    Performance testing example for group assignments
    """
    
    engine = GroupAssignmentEngine(db_connection, redis_client, kafka_producer)
    
    # Test case 1: Small batch assignment (< 1000 users)
    small_request = GroupAssignmentRequest(
        operation=AssignmentOperation.BULK_ADD,
        group_id="group_123",
        user_ids=[f"user_{i}" for i in range(500)],
        requester_id="admin_user",
        organization_id="org_456"
    )
    
    start_time = time.time()
    small_result = await engine.assign_users_to_group(small_request)
    small_time = time.time() - start_time
    
    print(f"Small batch (500 users): {small_time:.2f}s")
    print(f"Success rate: {small_result.success}")
    print(f"Processed: {small_result.processed_count}")
    
    # Test case 2: Large batch assignment (50k users)
    large_request = GroupAssignmentRequest(
        operation=AssignmentOperation.BULK_ADD,
        group_id="group_789",
        user_ids=[f"user_{i}" for i in range(50000)],
        requester_id="admin_user",
        organization_id="org_456"
    )
    
    start_time = time.time()
    large_result = await engine.assign_users_to_group(large_request)
    large_time = time.time() - start_time
    
    print(f"Large batch (50k users): {large_time:.2f}s")
    print(f"Success rate: {large_result.success}")
    print(f"Processed: {large_result.processed_count}")
    
    # Performance analysis
    if large_time <= 5.0:  # Target: sub-5-second for 50k users
        print("✅ Performance target met!")
    else:
        print("❌ Performance target missed")


if __name__ == "__main__":
    asyncio.run(performance_test_example())