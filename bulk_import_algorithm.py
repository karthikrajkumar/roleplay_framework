"""
Optimized Bulk User Import Algorithm
Time Complexity: O(n log n), Space Complexity: O(b + d)
Designed for 10K-100K user imports with fault tolerance and progress tracking
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from redis.exceptions import RedisError

@dataclass
class ImportBatch:
    batch_id: str
    start_row: int
    end_row: int
    data: List[Dict[str, Any]]
    priority: int = 0
    retry_count: int = 0
    dependencies: List[str] = None

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_data: Dict[str, Any]
    similarity_score: float = 0.0

class AdaptiveBatchProcessor:
    """
    Intelligent batch processing with dynamic sizing and error recovery
    """
    
    def __init__(self, db_connection, redis_client, kafka_producer):
        self.db = db_connection
        self.redis = redis_client
        self.kafka = kafka_producer
        
        # Performance-tuned parameters
        self.base_batch_size = 1000
        self.min_batch_size = 100
        self.max_batch_size = 5000
        self.max_workers = min(32, (os.cpu_count() or 1) * 4)
        
        # Caching and deduplication
        self.duplicate_cache = {}
        self.validation_cache = {}
        self.bloom_filter_size = 10000000  # 10M elements
        
        # Error handling
        self.max_retries = 3
        self.circuit_breaker_threshold = 0.1  # 10% error rate
        self.backoff_multiplier = 2.0
        
        # Performance tracking
        self.processing_rates = deque(maxlen=100)
        self.error_rates = deque(maxlen=100)
        
    async def process_bulk_import(self, job_id: str, file_path: str, 
                                import_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for bulk import processing
        
        Args:
            job_id: Unique identifier for import job
            file_path: Path to CSV/Excel file
            import_config: Configuration including validation rules, field mapping
            
        Returns:
            Dictionary with processing results and statistics
        """
        
        start_time = time.time()
        
        try:
            # Phase 1: File Analysis and Preparation
            file_metadata = await self._analyze_file(file_path)
            optimal_batch_size = self._calculate_optimal_batch_size(file_metadata)
            
            # Phase 2: Batch Creation with Dependency Resolution
            batches = await self._create_intelligent_batches(
                file_path, optimal_batch_size, import_config
            )
            
            # Phase 3: Parallel Processing with Circuit Breaker
            results = await self._process_batches_parallel(job_id, batches)
            
            # Phase 4: Results Aggregation and Rollback Preparation
            final_results = await self._finalize_import(job_id, results)
            
            processing_time = time.time() - start_time
            await self._update_performance_metrics(job_id, processing_time, final_results)
            
            return final_results
            
        except Exception as e:
            await self._handle_critical_error(job_id, e)
            raise

    async def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze file characteristics to optimize processing strategy
        """
        file_size = os.path.getsize(file_path)
        
        # Sample first 1000 rows for analysis
        sample_data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            sample_data = list(itertools.islice(reader, 1000))
        
        # Analyze data complexity
        avg_row_size = file_size / len(sample_data) if sample_data else 1024
        estimated_total_rows = file_size // avg_row_size
        
        # Detect data patterns
        complexity_score = self._calculate_data_complexity(sample_data)
        duplicate_likelihood = self._estimate_duplicate_rate(sample_data)
        
        return {
            'file_size': file_size,
            'estimated_rows': estimated_total_rows,
            'avg_row_size': avg_row_size,
            'complexity_score': complexity_score,
            'duplicate_likelihood': duplicate_likelihood,
            'sample_data': sample_data[:10]  # Keep small sample for optimization
        }

    def _calculate_optimal_batch_size(self, file_metadata: Dict[str, Any]) -> int:
        """
        Dynamic batch sizing based on data characteristics and system resources
        
        Algorithm: Adaptive sizing using multiple factors
        - Base size adjusted by data complexity
        - Memory constraints consideration
        - Historical performance data
        """
        
        complexity_factor = 1.0 / (1.0 + file_metadata['complexity_score'])
        memory_factor = self._get_available_memory_factor()
        performance_factor = self._get_historical_performance_factor()
        
        # Calculate optimal batch size
        optimal_size = int(
            self.base_batch_size * 
            complexity_factor * 
            memory_factor * 
            performance_factor
        )
        
        # Ensure within bounds
        return max(
            self.min_batch_size, 
            min(self.max_batch_size, optimal_size)
        )

    async def _create_intelligent_batches(self, file_path: str, batch_size: int, 
                                        config: Dict[str, Any]) -> List[ImportBatch]:
        """
        Create batches with intelligent ordering and dependency resolution
        """
        batches = []
        batch_id = 0
        current_batch_data = []
        current_row = 0
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row_data in reader:
                # Pre-process and validate row
                processed_row = await self._preprocess_row(row_data, config)
                current_batch_data.append(processed_row)
                current_row += 1
                
                # Create batch when size reached
                if len(current_batch_data) >= batch_size:
                    batch = ImportBatch(
                        batch_id=f"batch_{batch_id}",
                        start_row=current_row - len(current_batch_data),
                        end_row=current_row - 1,
                        data=current_batch_data.copy(),
                        priority=self._calculate_batch_priority(current_batch_data)
                    )
                    batches.append(batch)
                    
                    batch_id += 1
                    current_batch_data.clear()
            
            # Handle remaining data
            if current_batch_data:
                batch = ImportBatch(
                    batch_id=f"batch_{batch_id}",
                    start_row=current_row - len(current_batch_data),
                    end_row=current_row - 1,
                    data=current_batch_data,
                    priority=self._calculate_batch_priority(current_batch_data)
                )
                batches.append(batch)
        
        # Sort batches by priority for optimal processing order
        batches.sort(key=lambda x: x.priority)
        
        return batches

    async def _process_batches_parallel(self, job_id: str, 
                                      batches: List[ImportBatch]) -> Dict[str, Any]:
        """
        Process batches in parallel with intelligent error handling
        """
        
        # Initialize progress tracking
        total_batches = len(batches)
        completed_batches = 0
        failed_batches = []
        results = {
            'processed_count': 0,
            'success_count': 0,
            'error_count': 0,
            'duplicate_count': 0,
            'batch_results': []
        }
        
        # Circuit breaker for error rate monitoring
        circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker_threshold,
            recovery_timeout=30,
            expected_exception=ImportProcessingError
        )
        
        # Process batches with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_batch(batch: ImportBatch):
            async with semaphore:
                try:
                    with circuit_breaker:
                        batch_result = await self._process_batch_with_retry(
                            job_id, batch
                        )
                        return batch_result
                except Exception as e:
                    return self._create_error_result(batch, e)
        
        # Execute all batches
        batch_tasks = [process_single_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Aggregate results
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                failed_batches.append((batches[i], batch_result))
                continue
                
            results['processed_count'] += batch_result['processed_count']
            results['success_count'] += batch_result['success_count']
            results['error_count'] += batch_result['error_count']
            results['duplicate_count'] += batch_result['duplicate_count']
            results['batch_results'].append(batch_result)
            
            completed_batches += 1
            
            # Update progress
            progress = (completed_batches / total_batches) * 100
            await self._update_progress(job_id, progress, batch_result)
        
        # Handle failed batches with retry logic
        if failed_batches:
            retry_results = await self._handle_failed_batches(job_id, failed_batches)
            results.update(retry_results)
        
        return results

    async def _process_batch_with_retry(self, job_id: str, 
                                      batch: ImportBatch) -> Dict[str, Any]:
        """
        Process individual batch with exponential backoff retry
        """
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self._process_single_batch_core(job_id, batch)
                
            except (DatabaseError, RedisError) as e:
                if attempt == self.max_retries:
                    raise
                
                # Exponential backoff with jitter
                delay = (self.backoff_multiplier ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
                
                # Mark batch for retry
                batch.retry_count += 1
                
        raise ImportProcessingError(f"Max retries exceeded for batch {batch.batch_id}")

    async def _process_single_batch_core(self, job_id: str, 
                                       batch: ImportBatch) -> Dict[str, Any]:
        """
        Core batch processing logic with validation and deduplication
        """
        batch_start_time = time.time()
        batch_result = {
            'batch_id': batch.batch_id,
            'processed_count': 0,
            'success_count': 0,
            'error_count': 0,
            'duplicate_count': 0,
            'created_user_ids': [],
            'errors': [],
            'processing_time_ms': 0
        }
        
        # Phase 1: Validation and Deduplication
        validated_users = []
        for row_index, user_data in enumerate(batch.data):
            try:
                # Validate user data
                validation_result = await self._validate_user_data(user_data)
                
                if not validation_result.is_valid:
                    batch_result['errors'].extend([
                        {
                            'row': batch.start_row + row_index,
                            'errors': validation_result.errors
                        }
                    ])
                    batch_result['error_count'] += 1
                    continue
                
                # Check for duplicates
                is_duplicate, duplicate_info = await self._check_duplicate(
                    validation_result.normalized_data
                )
                
                if is_duplicate:
                    batch_result['duplicate_count'] += 1
                    continue
                
                validated_users.append({
                    'row_index': batch.start_row + row_index,
                    'data': validation_result.normalized_data,
                    'validation_result': validation_result
                })
                
            except Exception as e:
                batch_result['errors'].append({
                    'row': batch.start_row + row_index,
                    'errors': [str(e)]
                })
                batch_result['error_count'] += 1
        
        # Phase 2: Database Transaction
        if validated_users:
            try:
                async with self.db.transaction():
                    created_users = await self._create_users_bulk(validated_users)
                    
                    # Update caches
                    await self._update_caches(created_users)
                    
                    # Prepare rollback data
                    await self._prepare_rollback_data(job_id, created_users)
                    
                    batch_result['success_count'] = len(created_users)
                    batch_result['created_user_ids'] = [str(u.id) for u in created_users]
                    
            except Exception as e:
                # Transaction failed - mark all as errors
                batch_result['error_count'] += len(validated_users)
                batch_result['errors'].append({
                    'type': 'transaction_error',
                    'message': str(e)
                })
        
        batch_result['processed_count'] = len(batch.data)
        batch_result['processing_time_ms'] = int((time.time() - batch_start_time) * 1000)
        
        # Update processing rate for optimization
        self._update_processing_rate(batch_result)
        
        return batch_result


class DuplicateDetectionEngine:
    """
    High-performance duplicate detection with configurable similarity thresholds
    Time Complexity: O(1) average for hash lookups, O(n) worst case for similarity matching
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.email_hash_index = {}
        self.phone_hash_index = {}
        self.fuzzy_matcher = FuzzyMatcher()
        
    async def check_duplicate(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        Multi-stage duplicate detection algorithm
        
        Stage 1: Exact hash matching (O(1))
        Stage 2: Fuzzy similarity matching (O(n) worst case, optimized with indexing)
        Stage 3: Composite similarity scoring
        """
        
        # Stage 1: Exact matching on normalized hashes
        email_hash = self._generate_email_hash(user_data.get('email', ''))
        phone_hash = self._generate_phone_hash(user_data.get('phone', ''))
        
        # Check email hash index
        if email_hash in self.email_hash_index:
            return True, {
                'type': 'exact_email',
                'match_id': self.email_hash_index[email_hash],
                'confidence': 1.0
            }
        
        # Check phone hash index
        if phone_hash and phone_hash in self.phone_hash_index:
            return True, {
                'type': 'exact_phone', 
                'match_id': self.phone_hash_index[phone_hash],
                'confidence': 1.0
            }
        
        # Stage 2: Fuzzy similarity matching
        similarity_matches = await self._find_similarity_matches(user_data)
        
        if similarity_matches:
            best_match = max(similarity_matches, key=lambda x: x['score'])
            if best_match['score'] >= self.similarity_threshold:
                return True, {
                    'type': 'fuzzy_match',
                    'match_id': best_match['user_id'],
                    'confidence': best_match['score'],
                    'matched_fields': best_match['fields']
                }
        
        return False, None

    def _generate_email_hash(self, email: str) -> str:
        """Generate normalized hash for email"""
        if not email:
            return ""
        
        # Normalize email: lowercase, remove dots from Gmail addresses
        normalized = email.lower().strip()
        if '@gmail.com' in normalized:
            local, domain = normalized.split('@')
            local = local.replace('.', '')
            normalized = f"{local}@{domain}"
        
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def _find_similarity_matches(self, user_data: Dict[str, Any]) -> List[Dict]:
        """
        Find potential matches using similarity algorithms
        Optimized with pre-computed similarity indices
        """
        matches = []
        
        # Search similar emails using phonetic matching
        email_matches = await self._find_similar_emails(user_data.get('email', ''))
        
        # Search similar names using edit distance
        name_matches = await self._find_similar_names(
            user_data.get('first_name', ''),
            user_data.get('last_name', '')
        )
        
        # Combine and score matches
        all_candidates = set(email_matches + name_matches)
        
        for candidate_id in all_candidates:
            candidate_data = await self._get_user_data(candidate_id)
            similarity_score = self._calculate_composite_similarity(
                user_data, candidate_data
            )
            
            if similarity_score > 0.5:  # Minimum threshold for consideration
                matches.append({
                    'user_id': candidate_id,
                    'score': similarity_score,
                    'fields': self._get_matching_fields(user_data, candidate_data)
                })
        
        return matches


# Performance Analysis and Benchmarks

class PerformanceBenchmarks:
    """
    Performance benchmarks and optimization metrics
    """
    
    @staticmethod
    def analyze_time_complexity():
        return {
            'bulk_import_overall': 'O(n log n)',
            'file_analysis': 'O(k) where k = sample size',
            'batch_creation': 'O(n)',
            'duplicate_detection_exact': 'O(1) average',
            'duplicate_detection_fuzzy': 'O(n * m) worst case, O(log n) with indexing',
            'validation': 'O(n * v) where v = validation rules',
            'database_insertion': 'O(n) with batch inserts',
            'progress_tracking': 'O(1)'
        }
    
    @staticmethod
    def analyze_space_complexity():
        return {
            'batch_processing': 'O(b) where b = batch size',
            'duplicate_cache': 'O(d) where d = unique records processed',
            'validation_cache': 'O(v) bounded by LRU cache size',
            'error_tracking': 'O(e) where e = number of errors',
            'rollback_data': 'O(r) where r = successfully imported records'
        }
    
    @staticmethod
    def get_performance_targets():
        return {
            'throughput': {
                '10k_users': '< 2 minutes',
                '50k_users': '< 8 minutes', 
                '100k_users': '< 15 minutes'
            },
            'memory_usage': {
                'peak_memory': '< 2GB for 100k imports',
                'memory_per_batch': '< 50MB'
            },
            'error_rates': {
                'duplicate_detection_accuracy': '> 99.5%',
                'false_positive_rate': '< 0.1%',
                'validation_accuracy': '> 99.9%'
            },
            'scalability': {
                'concurrent_imports': '10 simultaneous jobs',
                'worker_efficiency': '> 85% CPU utilization'
            }
        }


# Error Handling and Recovery Strategies

class ErrorRecoveryManager:
    """
    Comprehensive error handling with partial retry capabilities
    """
    
    def __init__(self):
        self.error_classifications = {
            'retryable': ['connection_error', 'timeout', 'temporary_failure'],
            'non_retryable': ['validation_error', 'duplicate', 'permission_denied'],
            'critical': ['system_error', 'corruption', 'security_breach']
        }
    
    async def handle_batch_errors(self, batch_errors: List[Dict]) -> Dict[str, Any]:
        """
        Intelligent error handling with classification and recovery
        """
        recovery_strategy = {
            'retryable_batches': [],
            'failed_batches': [],
            'partial_recovery_batches': [],
            'rollback_required': False
        }
        
        for error_info in batch_errors:
            error_type = self._classify_error(error_info['error'])
            
            if error_type == 'retryable':
                recovery_strategy['retryable_batches'].append(error_info)
            elif error_type == 'critical':
                recovery_strategy['rollback_required'] = True
                recovery_strategy['failed_batches'].append(error_info)
            else:
                # Attempt partial recovery
                partial_result = await self._attempt_partial_recovery(error_info)
                if partial_result['recoverable_items']:
                    recovery_strategy['partial_recovery_batches'].append(partial_result)
                else:
                    recovery_strategy['failed_batches'].append(error_info)
        
        return recovery_strategy

# Usage Example and Integration Points

async def example_usage():
    """
    Example of how to use the bulk import system
    """
    
    # Initialize processor
    processor = AdaptiveBatchProcessor(db_connection, redis_client, kafka_producer)
    
    # Configure import job
    import_config = {
        'validation_rules': {
            'email': {'required': True, 'format': 'email'},
            'username': {'required': True, 'min_length': 3, 'max_length': 30}
        },
        'field_mapping': {
            'email_address': 'email',
            'user_name': 'username',
            'first': 'first_name',
            'last': 'last_name'
        },
        'duplicate_detection': {
            'enabled': True,
            'threshold': 0.85,
            'fields': ['email', 'username', 'phone']
        },
        'batch_options': {
            'auto_size': True,
            'max_errors_per_batch': 10,
            'continue_on_error': True
        }
    }
    
    # Process bulk import
    job_id = "import_job_123"
    file_path = "/path/to/users.csv"
    
    try:
        results = await processor.process_bulk_import(job_id, file_path, import_config)
        
        print(f"Import completed successfully:")
        print(f"- Processed: {results['processed_count']} users")
        print(f"- Successful: {results['success_count']} users")
        print(f"- Errors: {results['error_count']} users")
        print(f"- Duplicates: {results['duplicate_count']} users")
        
    except ImportProcessingError as e:
        print(f"Import failed: {e}")
        # Trigger rollback if needed
        await processor.rollback_import(job_id)

if __name__ == "__main__":
    asyncio.run(example_usage())