"""
Advanced Duplicate Detection Algorithm for Learning Platform
Time Complexity: O(n) average case, O(n log n) worst case with similarity matching
Space Complexity: O(n) for hash storage + O(k) for similarity indices
Configurable similarity thresholds with minimal false positives
"""

import asyncio
import hashlib
import json
import time
from typing import List, Dict, Set, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import re
import unicodedata
from difflib import SequenceMatcher
import jellyfish  # For phonetic matching
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis.exceptions


class MatchType(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    PHONETIC = "phonetic"
    COMPOSITE = "composite"


@dataclass
class DuplicateMatch:
    match_type: MatchType
    similarity_score: float
    matched_user_id: str
    matched_fields: List[str]
    confidence: float
    match_details: Dict[str, Any]


@dataclass
class DetectionConfig:
    similarity_threshold: float = 0.85
    fuzzy_threshold: float = 0.80
    phonetic_threshold: float = 0.90
    composite_threshold: float = 0.85
    enable_fuzzy_matching: bool = True
    enable_phonetic_matching: bool = True
    enable_composite_scoring: bool = True
    max_candidates: int = 100
    cache_ttl: int = 3600  # 1 hour


class AdvancedDuplicateDetector:
    """
    High-performance duplicate detection with multiple algorithms
    Optimized for minimal false positives and configurable similarity
    """
    
    def __init__(self, db_connection, redis_client, config: DetectionConfig = None):
        self.db = db_connection
        self.redis = redis_client
        self.config = config or DetectionConfig()
        
        # Multi-level hash indices
        self.exact_hash_index = {}  # SHA-256 hashes for exact matching
        self.fuzzy_hash_index = {}  # MinHash for fuzzy matching
        self.phonetic_index = {}   # Soundex/Metaphone for phonetic matching
        
        # Similarity matching components
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(2, 3),
            analyzer='char',
            max_features=10000
        )
        self.similarity_cache = {}
        
        # Performance tracking
        self.detection_stats = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        
        # Normalized field weights for composite scoring
        self.field_weights = {
            'email': 0.35,
            'username': 0.25,
            'phone': 0.20,
            'full_name': 0.15,
            'birth_date': 0.05
        }

    async def detect_duplicates(self, user_data: Dict[str, Any], 
                              existing_users: Optional[List[Dict]] = None) -> List[DuplicateMatch]:
        """
        Main entry point for duplicate detection
        Returns ranked list of potential duplicates with confidence scores
        """
        
        detection_start = time.time()
        
        try:
            # Normalize input data
            normalized_data = self._normalize_user_data(user_data)
            
            # Multi-stage detection pipeline
            exact_matches = await self._detect_exact_matches(normalized_data)
            fuzzy_matches = []
            phonetic_matches = []
            composite_matches = []
            
            if not exact_matches:
                # Only proceed with expensive matching if no exact matches found
                if self.config.enable_fuzzy_matching:
                    fuzzy_matches = await self._detect_fuzzy_matches(normalized_data)
                
                if self.config.enable_phonetic_matching:
                    phonetic_matches = await self._detect_phonetic_matches(normalized_data)
                
                if self.config.enable_composite_scoring:
                    composite_matches = await self._detect_composite_matches(
                        normalized_data, existing_users
                    )
            
            # Combine and rank all matches
            all_matches = exact_matches + fuzzy_matches + phonetic_matches + composite_matches
            ranked_matches = self._rank_and_deduplicate_matches(all_matches)
            
            # Update statistics
            processing_time = time.time() - detection_start
            self.processing_times.append(processing_time)
            self.detection_stats['total_detections'] += 1
            self.detection_stats['total_matches'] += len(ranked_matches)
            
            return ranked_matches[:self.config.max_candidates]
            
        except Exception as e:
            self.detection_stats['errors'] += 1
            raise DuplicateDetectionError(f"Detection failed: {str(e)}")

    def _normalize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive data normalization for consistent matching
        """
        
        normalized = {}
        
        # Email normalization
        if 'email' in user_data and user_data['email']:
            email = user_data['email'].lower().strip()
            
            # Handle Gmail dot notation and plus addressing
            if '@gmail.com' in email:
                local_part, domain = email.split('@', 1)
                # Remove dots from local part
                local_part = local_part.replace('.', '')
                # Remove plus addressing
                if '+' in local_part:
                    local_part = local_part.split('+')[0]
                email = f"{local_part}@{domain}"
            
            normalized['email'] = email
            normalized['email_domain'] = email.split('@')[-1]
            normalized['email_local'] = email.split('@')[0]
        
        # Phone normalization
        if 'phone' in user_data and user_data['phone']:
            phone = re.sub(r'[^\d]', '', str(user_data['phone']))
            # Normalize to international format assumptions
            if len(phone) == 10:  # US number without country code
                phone = '1' + phone
            normalized['phone'] = phone
            normalized['phone_last_4'] = phone[-4:] if len(phone) >= 4 else phone
        
        # Name normalization
        first_name = user_data.get('first_name', '').strip().lower()
        last_name = user_data.get('last_name', '').strip().lower()
        
        if first_name:
            normalized['first_name'] = self._normalize_name_component(first_name)
            normalized['first_name_metaphone'] = jellyfish.metaphone(first_name)
        
        if last_name:
            normalized['last_name'] = self._normalize_name_component(last_name)
            normalized['last_name_metaphone'] = jellyfish.metaphone(last_name)
        
        if first_name and last_name:
            full_name = f"{first_name} {last_name}"
            normalized['full_name'] = full_name
            normalized['full_name_reversed'] = f"{last_name} {first_name}"
            normalized['initials'] = f"{first_name[0]}{last_name[0]}" if len(first_name) > 0 and len(last_name) > 0 else ""
        
        # Username normalization
        if 'username' in user_data and user_data['username']:
            username = user_data['username'].lower().strip()
            normalized['username'] = username
            normalized['username_alphanum'] = re.sub(r'[^a-z0-9]', '', username)
        
        # Date normalization
        if 'birth_date' in user_data and user_data['birth_date']:
            # Normalize date format
            birth_date = str(user_data['birth_date'])
            normalized['birth_date'] = birth_date
            # Extract year for partial matching
            year_match = re.search(r'\b(19|20)\d{2}\b', birth_date)
            if year_match:
                normalized['birth_year'] = year_match.group()
        
        return normalized

    def _normalize_name_component(self, name: str) -> str:
        """
        Normalize name component with Unicode handling and common variations
        """
        # Remove diacritics and convert to ASCII
        name = unicodedata.normalize('NFD', name)
        name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
        
        # Handle common name variations
        name_variations = {
            'michael': ['mike', 'mick'],
            'william': ['bill', 'will'],
            'robert': ['bob', 'rob'],
            'elizabeth': ['liz', 'beth', 'betty'],
            'richard': ['rick', 'dick'],
            'christopher': ['chris'],
            'matthew': ['matt'],
            'anthony': ['tony'],
            'jennifer': ['jen', 'jenny']
        }
        
        # Check if name matches any variations
        for canonical, variations in name_variations.items():
            if name in variations:
                return canonical
            elif name == canonical:
                return canonical
        
        return name

    async def _detect_exact_matches(self, normalized_data: Dict[str, Any]) -> List[DuplicateMatch]:
        """
        Lightning-fast exact matching using hash indices
        Time Complexity: O(1) average case
        """
        
        matches = []
        
        # Email exact match
        if 'email' in normalized_data:
            email_hash = self._generate_deterministic_hash(normalized_data['email'])
            existing_user_id = await self._lookup_hash_index('email', email_hash)
            
            if existing_user_id:
                matches.append(DuplicateMatch(
                    match_type=MatchType.EXACT,
                    similarity_score=1.0,
                    matched_user_id=existing_user_id,
                    matched_fields=['email'],
                    confidence=1.0,
                    match_details={'hash': email_hash, 'field': 'email'}
                ))
        
        # Phone exact match
        if 'phone' in normalized_data:
            phone_hash = self._generate_deterministic_hash(normalized_data['phone'])
            existing_user_id = await self._lookup_hash_index('phone', phone_hash)
            
            if existing_user_id:
                matches.append(DuplicateMatch(
                    match_type=MatchType.EXACT,
                    similarity_score=1.0,
                    matched_user_id=existing_user_id,
                    matched_fields=['phone'],
                    confidence=1.0,
                    match_details={'hash': phone_hash, 'field': 'phone'}
                ))
        
        # Username exact match
        if 'username' in normalized_data:
            username_hash = self._generate_deterministic_hash(normalized_data['username'])
            existing_user_id = await self._lookup_hash_index('username', username_hash)
            
            if existing_user_id:
                matches.append(DuplicateMatch(
                    match_type=MatchType.EXACT,
                    similarity_score=1.0,
                    matched_user_id=existing_user_id,
                    matched_fields=['username'],
                    confidence=1.0,
                    match_details={'hash': username_hash, 'field': 'username'}
                ))
        
        return matches

    async def _detect_fuzzy_matches(self, normalized_data: Dict[str, Any]) -> List[DuplicateMatch]:
        """
        Fuzzy string matching using edit distance and token similarity
        Optimized with pre-computed similarity indices
        """
        
        matches = []
        
        # Email domain-based fuzzy matching
        if 'email' in normalized_data:
            email_candidates = await self._get_similar_emails(normalized_data['email'])
            
            for candidate_id, candidate_email in email_candidates:
                similarity = self._calculate_string_similarity(
                    normalized_data['email'], candidate_email
                )
                
                if similarity >= self.config.fuzzy_threshold:
                    matches.append(DuplicateMatch(
                        match_type=MatchType.FUZZY,
                        similarity_score=similarity,
                        matched_user_id=candidate_id,
                        matched_fields=['email'],
                        confidence=similarity * 0.9,  # Slight confidence penalty for fuzzy
                        match_details={
                            'algorithm': 'sequence_matcher',
                            'original': normalized_data['email'],
                            'candidate': candidate_email
                        }
                    ))
        
        # Name-based fuzzy matching
        if 'full_name' in normalized_data:
            name_candidates = await self._get_similar_names(normalized_data['full_name'])
            
            for candidate_id, candidate_name in name_candidates:
                # Use multiple similarity algorithms
                seq_similarity = SequenceMatcher(None, normalized_data['full_name'], candidate_name).ratio()
                token_similarity = self._calculate_token_similarity(
                    normalized_data['full_name'], candidate_name
                )
                
                # Combined similarity score
                combined_similarity = (seq_similarity * 0.6) + (token_similarity * 0.4)
                
                if combined_similarity >= self.config.fuzzy_threshold:
                    matches.append(DuplicateMatch(
                        match_type=MatchType.FUZZY,
                        similarity_score=combined_similarity,
                        matched_user_id=candidate_id,
                        matched_fields=['full_name'],
                        confidence=combined_similarity * 0.85,
                        match_details={
                            'seq_similarity': seq_similarity,
                            'token_similarity': token_similarity,
                            'original': normalized_data['full_name'],
                            'candidate': candidate_name
                        }
                    ))
        
        return matches

    async def _detect_phonetic_matches(self, normalized_data: Dict[str, Any]) -> List[DuplicateMatch]:
        """
        Phonetic matching using Soundex, Metaphone, and Double Metaphone
        Effective for name variations and typos
        """
        
        matches = []
        
        # First name phonetic matching
        if 'first_name_metaphone' in normalized_data:
            metaphone_key = normalized_data['first_name_metaphone']
            if metaphone_key:
                candidates = await self._get_phonetic_candidates('first_name', metaphone_key)
                
                for candidate_id, candidate_name in candidates:
                    # Additional verification with original names
                    name_similarity = self._calculate_string_similarity(
                        normalized_data.get('first_name', ''), candidate_name
                    )
                    
                    if name_similarity >= 0.6:  # Lower threshold for phonetic matching
                        confidence = min(name_similarity * 1.1, 1.0)  # Boost phonetic matches slightly
                        
                        matches.append(DuplicateMatch(
                            match_type=MatchType.PHONETIC,
                            similarity_score=confidence,
                            matched_user_id=candidate_id,
                            matched_fields=['first_name'],
                            confidence=confidence * 0.8,  # Phonetic matches are less certain
                            match_details={
                                'metaphone_key': metaphone_key,
                                'name_similarity': name_similarity,
                                'original': normalized_data.get('first_name', ''),
                                'candidate': candidate_name
                            }
                        ))
        
        # Last name phonetic matching
        if 'last_name_metaphone' in normalized_data:
            metaphone_key = normalized_data['last_name_metaphone']
            if metaphone_key:
                candidates = await self._get_phonetic_candidates('last_name', metaphone_key)
                
                for candidate_id, candidate_name in candidates:
                    name_similarity = self._calculate_string_similarity(
                        normalized_data.get('last_name', ''), candidate_name
                    )
                    
                    if name_similarity >= 0.6:
                        confidence = min(name_similarity * 1.1, 1.0)
                        
                        matches.append(DuplicateMatch(
                            match_type=MatchType.PHONETIC,
                            similarity_score=confidence,
                            matched_user_id=candidate_id,
                            matched_fields=['last_name'],
                            confidence=confidence * 0.8,
                            match_details={
                                'metaphone_key': metaphone_key,
                                'name_similarity': name_similarity,
                                'original': normalized_data.get('last_name', ''),
                                'candidate': candidate_name
                            }
                        ))
        
        return matches

    async def _detect_composite_matches(self, normalized_data: Dict[str, Any], 
                                      existing_users: Optional[List[Dict]] = None) -> List[DuplicateMatch]:
        """
        Advanced composite matching using weighted field similarity
        Considers multiple fields together for higher confidence detection
        """
        
        matches = []
        
        if not existing_users:
            # If no specific users provided, get recent candidates from database
            existing_users = await self._get_recent_user_candidates(limit=1000)
        
        for existing_user in existing_users:
            existing_normalized = self._normalize_user_data(existing_user)
            
            # Calculate field-wise similarities
            field_similarities = {}
            total_weighted_score = 0.0
            matched_fields = []
            
            for field, weight in self.field_weights.items():
                if field in normalized_data and field in existing_normalized:
                    similarity = self._calculate_field_similarity(
                        field, normalized_data[field], existing_normalized[field]
                    )
                    
                    field_similarities[field] = similarity
                    total_weighted_score += similarity * weight
                    
                    if similarity > 0.7:  # Field contributes significantly
                        matched_fields.append(field)
            
            # Only consider as potential match if multiple fields match
            if len(matched_fields) >= 2 and total_weighted_score >= self.config.composite_threshold:
                
                # Boost confidence for multiple field matches
                confidence_boost = min(len(matched_fields) * 0.1, 0.3)
                final_confidence = min(total_weighted_score + confidence_boost, 1.0)
                
                matches.append(DuplicateMatch(
                    match_type=MatchType.COMPOSITE,
                    similarity_score=total_weighted_score,
                    matched_user_id=existing_user.get('id'),
                    matched_fields=matched_fields,
                    confidence=final_confidence,
                    match_details={
                        'field_similarities': field_similarities,
                        'weighted_score': total_weighted_score,
                        'field_count': len(matched_fields)
                    }
                ))
        
        return matches

    def _calculate_field_similarity(self, field: str, value1: str, value2: str) -> float:
        """
        Field-specific similarity calculation with appropriate algorithms
        """
        
        if field == 'email':
            # Exact match for email domains, fuzzy for local parts
            if '@' in value1 and '@' in value2:
                local1, domain1 = value1.split('@', 1)
                local2, domain2 = value2.split('@', 1)
                
                if domain1 == domain2:
                    # Same domain, compare local parts
                    return self._calculate_string_similarity(local1, local2)
                else:
                    # Different domains, much lower similarity
                    return 0.2
            else:
                return self._calculate_string_similarity(value1, value2)
        
        elif field == 'phone':
            # Phone number similarity based on suffix matching
            if len(value1) >= 4 and len(value2) >= 4:
                suffix_match = value1[-4:] == value2[-4:]
                if suffix_match:
                    # Check more digits
                    full_match = self._calculate_string_similarity(value1, value2)
                    return max(0.7, full_match)  # Minimum 70% for suffix match
                else:
                    return 0.0
            else:
                return self._calculate_string_similarity(value1, value2)
        
        elif field in ['first_name', 'last_name', 'full_name']:
            # Name similarity with phonetic consideration
            string_sim = self._calculate_string_similarity(value1, value2)
            
            # Add phonetic similarity bonus
            try:
                phonetic_sim = 1.0 if jellyfish.metaphone(value1) == jellyfish.metaphone(value2) else 0.0
                return min((string_sim * 0.8) + (phonetic_sim * 0.2), 1.0)
            except:
                return string_sim
        
        elif field == 'username':
            # Username similarity with alphanumeric normalization
            alphanum1 = re.sub(r'[^a-z0-9]', '', value1.lower())
            alphanum2 = re.sub(r'[^a-z0-9]', '', value2.lower())
            return self._calculate_string_similarity(alphanum1, alphanum2)
        
        else:
            # Default string similarity
            return self._calculate_string_similarity(value1, value2)

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Optimized string similarity using multiple algorithms
        """
        
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Use SequenceMatcher for primary similarity
        primary_sim = SequenceMatcher(None, str1, str2).ratio()
        
        # Add Jaro-Winkler for additional accuracy on names
        try:
            jaro_sim = jellyfish.jaro_winkler_similarity(str1, str2)
            # Weighted combination
            return (primary_sim * 0.7) + (jaro_sim * 0.3)
        except:
            return primary_sim

    def _calculate_token_similarity(self, str1: str, str2: str) -> float:
        """
        Token-based similarity for handling word reordering
        """
        
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0

    def _rank_and_deduplicate_matches(self, matches: List[DuplicateMatch]) -> List[DuplicateMatch]:
        """
        Rank matches by confidence and remove duplicates
        Implements sophisticated ranking algorithm
        """
        
        if not matches:
            return []
        
        # Group matches by user_id
        user_matches = defaultdict(list)
        for match in matches:
            user_matches[match.matched_user_id].append(match)
        
        # Select best match per user and rank globally
        best_matches = []
        
        for user_id, user_match_list in user_matches.items():
            # Sort by confidence, then by similarity score
            user_match_list.sort(key=lambda m: (m.confidence, m.similarity_score), reverse=True)
            best_match = user_match_list[0]
            
            # Boost confidence if multiple match types agree
            if len(user_match_list) > 1:
                match_type_diversity = len(set(m.match_type for m in user_match_list))
                if match_type_diversity > 1:
                    # Multiple algorithms agree - boost confidence
                    best_match.confidence = min(best_match.confidence * 1.2, 1.0)
                    best_match.match_details['multi_algorithm_agreement'] = True
            
            best_matches.append(best_match)
        
        # Final ranking by confidence
        best_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return best_matches

    async def _lookup_hash_index(self, field: str, hash_value: str) -> Optional[str]:
        """
        Fast Redis-based hash index lookup
        """
        
        cache_key = f"hash_index:{field}:{hash_value}"
        
        try:
            cached_user_id = await self.redis.get(cache_key)
            if cached_user_id:
                return cached_user_id.decode('utf-8')
        except redis.exceptions.RedisError:
            pass
        
        # Fallback to database lookup
        query = f"""
            SELECT user_id FROM duplicate_detection_cache 
            WHERE {field}_hash = :hash_value 
            LIMIT 1
        """
        
        result = await self.db.fetch_one(query, {'hash_value': hash_value})
        
        if result:
            user_id = result['user_id']
            # Cache for future lookups
            try:
                await self.redis.setex(cache_key, self.config.cache_ttl, user_id)
            except redis.exceptions.RedisError:
                pass
            
            return user_id
        
        return None

    def _generate_deterministic_hash(self, value: str) -> str:
        """
        Generate deterministic SHA-256 hash for consistent indexing
        """
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    async def get_detection_performance_metrics(self) -> Dict[str, Any]:
        """
        Comprehensive performance metrics for duplicate detection
        """
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'detection_stats': dict(self.detection_stats),
            'performance_metrics': {
                'average_processing_time_ms': avg_processing_time * 1000,
                'total_detections': len(self.processing_times),
                'cache_efficiency': await self._calculate_cache_hit_rate()
            },
            'accuracy_metrics': {
                'false_positive_estimate': await self._estimate_false_positive_rate(),
                'detection_coverage': await self._estimate_detection_coverage()
            },
            'configuration': {
                'similarity_threshold': self.config.similarity_threshold,
                'fuzzy_threshold': self.config.fuzzy_threshold,
                'phonetic_threshold': self.config.phonetic_threshold,
                'composite_threshold': self.config.composite_threshold
            }
        }

    async def update_detection_indices(self, user_data: Dict[str, Any], user_id: str):
        """
        Update all detection indices when new user is created
        Should be called after successful user creation
        """
        
        normalized_data = self._normalize_user_data(user_data)
        
        # Update hash indices
        hash_updates = []
        
        if 'email' in normalized_data:
            email_hash = self._generate_deterministic_hash(normalized_data['email'])
            hash_updates.append(('email', email_hash, user_id))
        
        if 'phone' in normalized_data:
            phone_hash = self._generate_deterministic_hash(normalized_data['phone'])
            hash_updates.append(('phone', phone_hash, user_id))
        
        if 'username' in normalized_data:
            username_hash = self._generate_deterministic_hash(normalized_data['username'])
            hash_updates.append(('username', username_hash, user_id))
        
        # Batch update database
        if hash_updates:
            await self._batch_update_hash_indices(hash_updates)
        
        # Update Redis caches
        for field, hash_value, uid in hash_updates:
            cache_key = f"hash_index:{field}:{hash_value}"
            try:
                await self.redis.setex(cache_key, self.config.cache_ttl, uid)
            except redis.exceptions.RedisError:
                pass


# Usage Example and Testing

async def duplicate_detection_example():
    """
    Example usage of the duplicate detection system
    """
    
    detector = AdvancedDuplicateDetector(
        db_connection=db_connection,
        redis_client=redis_client,
        config=DetectionConfig(
            similarity_threshold=0.85,
            enable_fuzzy_matching=True,
            enable_phonetic_matching=True,
            enable_composite_scoring=True
        )
    )
    
    # Test user data
    new_user = {
        'email': 'john.doe@gmail.com',
        'username': 'johndoe123',
        'first_name': 'John',
        'last_name': 'Doe',
        'phone': '+1-555-123-4567'
    }
    
    # Detect duplicates
    start_time = time.time()
    matches = await detector.detect_duplicates(new_user)
    detection_time = time.time() - start_time
    
    print(f"Duplicate detection completed in {detection_time:.3f}s")
    print(f"Found {len(matches)} potential matches:")
    
    for match in matches:
        print(f"  - User ID: {match.matched_user_id}")
        print(f"    Type: {match.match_type.value}")
        print(f"    Similarity: {match.similarity_score:.3f}")
        print(f"    Confidence: {match.confidence:.3f}")
        print(f"    Matched fields: {', '.join(match.matched_fields)}")
        print()
    
    # Performance metrics
    metrics = await detector.get_detection_performance_metrics()
    print("Performance metrics:")
    print(f"  - Average processing time: {metrics['performance_metrics']['average_processing_time_ms']:.2f}ms")
    print(f"  - Total detections: {metrics['performance_metrics']['total_detections']}")


if __name__ == "__main__":
    asyncio.run(duplicate_detection_example())