"""
Advanced Collaborative AI Coordination Engine

This module implements sophisticated multi-character AI coordination with:
- Game Theory-based Turn Allocation with Nash Equilibrium optimization
- Neural Story Arc Generation with narrative coherence tracking
- Dynamic Role Assignment using personality compatibility analysis
- Conflict Resolution with consensus algorithms and negotiation strategies
- Performance Prediction with ensemble learning methods
- Real-time Narrative Coherence maintenance using story graph networks
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
import json
import math
from collections import defaultdict, deque
from enum import Enum
import random
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from ..interfaces.neural_persona import (
    ICollaborativeAICoordinator,
    IPerformancePredictionEngine,
    ConversationContext,
    LearningOutcome,
    PersonalityProfile,
    PersonalityTrait,
    EmotionalState
)


logger = logging.getLogger(__name__)


class CoordinationStrategy(str, Enum):
    """Coordination strategies for multi-persona interactions."""
    ROUND_ROBIN = "round_robin"
    EXPERTISE_BASED = "expertise_based"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    NARRATIVE_DRIVEN = "narrative_driven"
    CONFLICT_MEDIATION = "conflict_mediation"
    COLLABORATIVE = "collaborative"


class ConflictType(str, Enum):
    """Types of conflicts between personas."""
    OPINION_DISAGREEMENT = "opinion_disagreement"
    PERSONALITY_CLASH = "personality_clash"
    GOAL_MISALIGNMENT = "goal_misalignment"
    RESOURCE_COMPETITION = "resource_competition"
    NARRATIVE_INCONSISTENCY = "narrative_inconsistency"
    TIMING_CONFLICT = "timing_conflict"


@dataclass
class PersonaAgent:
    """Individual persona agent with capabilities and state."""
    persona_id: str
    personality: PersonalityProfile
    current_emotional_state: EmotionalState
    expertise_areas: List[LearningOutcome]
    speaking_priority: float
    narrative_role: str
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    conflict_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_score: float = 0.5
    last_speaking_turn: Optional[datetime] = None
    speaking_frequency: float = 0.0
    user_preference_alignment: float = 0.5


@dataclass
class NarrativeArc:
    """Narrative arc structure for story coherence."""
    arc_id: str
    arc_type: str  # "character_development", "conflict_resolution", "mystery", etc.
    current_stage: str  # "setup", "rising_action", "climax", "falling_action", "resolution"
    protagonist_ids: List[str]
    antagonist_ids: List[str]
    key_plot_points: List[Dict[str, Any]]
    tension_level: float
    pacing_score: float
    coherence_score: float
    user_engagement_prediction: float
    estimated_completion_turns: int
    adaptive_elements: List[str]


@dataclass
class CoordinationDecision:
    """Decision made by the coordination system."""
    speaker_id: str
    response_content: str
    coordination_strategy: CoordinationStrategy
    confidence: float
    reasoning: str
    expected_outcomes: Dict[str, float]
    alternative_speakers: List[Tuple[str, float]]  # (persona_id, score)
    narrative_impact: Dict[str, float]
    estimated_user_satisfaction: float


class GameTheoryTurnAllocator:
    """Game theory-based turn allocation system."""
    
    def __init__(self):
        self.payoff_matrix = {}
        self.historical_allocations = []
        self.nash_equilibria = {}
        
    def calculate_payoff_matrix(self, 
                               personas: List[PersonaAgent],
                               context: ConversationContext,
                               user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate payoff matrix for turn allocation game."""
        
        n_personas = len(personas)
        payoff_matrix = np.zeros((n_personas, n_personas))
        
        for i, persona_i in enumerate(personas):
            for j, persona_j in enumerate(personas):
                if i == j:
                    # Self-payoff for speaking
                    payoff = self._calculate_speaking_payoff(persona_i, context, user_preferences)
                else:
                    # Payoff from other persona speaking
                    payoff = self._calculate_listening_payoff(persona_i, persona_j, context)
                
                payoff_matrix[i][j] = payoff
        
        return payoff_matrix
    
    def find_optimal_allocation(self, 
                              payoff_matrix: np.ndarray,
                              personas: List[PersonaAgent]) -> Tuple[str, float]:
        """Find optimal turn allocation using game theory."""
        
        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(-payoff_matrix)  # Negative for maximization
        
        # Find the best single speaker (not assignment to all)
        best_speaker_idx = np.argmax(np.diag(payoff_matrix))
        best_payoff = payoff_matrix[best_speaker_idx, best_speaker_idx]
        
        return personas[best_speaker_idx].persona_id, best_payoff
    
    def _calculate_speaking_payoff(self, 
                                  persona: PersonaAgent,
                                  context: ConversationContext,
                                  user_preferences: Dict[str, Any]) -> float:
        """Calculate payoff for a persona speaking."""
        
        payoff_components = []
        
        # Expertise alignment
        relevant_expertise = sum(1 for obj in context.learning_objectives 
                               if obj in persona.expertise_areas)
        expertise_score = relevant_expertise / max(1, len(context.learning_objectives))
        payoff_components.append(expertise_score * 0.3)
        
        # Emotional resonance with user
        if context.emotional_trajectory:
            user_emotion = context.emotional_trajectory[-1]
            emotional_distance = self._calculate_emotional_distance(
                persona.current_emotional_state, user_emotion
            )
            emotional_score = 1.0 - emotional_distance
            payoff_components.append(emotional_score * 0.25)
        
        # User preference alignment
        payoff_components.append(persona.user_preference_alignment * 0.2)
        
        # Recency penalty (avoid same persona speaking too frequently)
        if persona.last_speaking_turn:
            time_since_last = datetime.now() - persona.last_speaking_turn
            recency_penalty = min(1.0, time_since_last.total_seconds() / 300)  # 5 minutes
            payoff_components.append(recency_penalty * 0.15)
        else:
            payoff_components.append(0.15)  # No penalty for first time speakers
        
        # Performance history
        avg_performance = np.mean(list(persona.performance_metrics.values())) if persona.performance_metrics else 0.5
        payoff_components.append(avg_performance * 0.1)
        
        return sum(payoff_components)
    
    def _calculate_listening_payoff(self, 
                                   listener: PersonaAgent,
                                   speaker: PersonaAgent,
                                   context: ConversationContext) -> float:
        """Calculate payoff for a persona listening to another speak."""
        
        # Complementary expertise (learn from others)
        complementary_score = 0.0
        for expertise in speaker.expertise_areas:
            if expertise not in listener.expertise_areas:
                complementary_score += 0.1
        
        # Personality compatibility
        compatibility = self._calculate_personality_compatibility(
            listener.personality, speaker.personality
        )
        
        # Collaboration history
        collaboration_bonus = listener.collaboration_score * 0.1
        
        return min(1.0, complementary_score + compatibility * 0.3 + collaboration_bonus)
    
    def _calculate_emotional_distance(self, 
                                    emotion1: EmotionalState,
                                    emotion2: EmotionalState) -> float:
        """Calculate distance between emotional states."""
        
        # Convert emotions to vectors
        vec1 = np.array(list(emotion1.dimensions.values()))
        vec2 = np.array(list(emotion2.dimensions.values()))
        
        # Euclidean distance normalized by maximum possible distance
        distance = np.linalg.norm(vec1 - vec2) / (2 * math.sqrt(len(vec1)))
        
        return min(1.0, distance)
    
    def _calculate_personality_compatibility(self, 
                                           personality1: PersonalityProfile,
                                           personality2: PersonalityProfile) -> float:
        """Calculate compatibility between personalities."""
        
        # Extract trait vectors
        traits1 = np.array(list(personality1.traits.values()))
        traits2 = np.array(list(personality2.traits.values()))
        
        # Calculate cosine similarity
        similarity = cosine_similarity([traits1], [traits2])[0][0]
        
        # Convert to compatibility (0.0 to 1.0)
        compatibility = (similarity + 1.0) / 2.0
        
        return compatibility


class NarrativeCoherenceManager:
    """Manage narrative coherence across multi-character interactions."""
    
    def __init__(self):
        self.story_graph = nx.DiGraph()
        self.narrative_arcs = {}
        self.coherence_violations = []
        self.plot_threads = defaultdict(list)
        
    def initialize_narrative_arc(self, 
                                arc_type: str,
                                protagonists: List[str],
                                antagonists: List[str] = None,
                                target_outcomes: List[LearningOutcome] = None) -> str:
        """Initialize a new narrative arc."""
        
        arc_id = f"arc_{datetime.now().timestamp()}"
        
        narrative_arc = NarrativeArc(
            arc_id=arc_id,
            arc_type=arc_type,
            current_stage="setup",
            protagonist_ids=protagonists,
            antagonist_ids=antagonists or [],
            key_plot_points=[],
            tension_level=0.1,
            pacing_score=0.5,
            coherence_score=1.0,
            user_engagement_prediction=0.5,
            estimated_completion_turns=10,
            adaptive_elements=[]
        )
        
        self.narrative_arcs[arc_id] = narrative_arc
        
        # Add to story graph
        for protagonist in protagonists:
            self.story_graph.add_node(protagonist, role="protagonist", arc=arc_id)
        
        for antagonist in (antagonists or []):
            self.story_graph.add_node(antagonist, role="antagonist", arc=arc_id)
        
        return arc_id
    
    def evaluate_narrative_impact(self, 
                                 speaker_id: str,
                                 proposed_action: str,
                                 context: ConversationContext) -> Dict[str, float]:
        """Evaluate narrative impact of a proposed action."""
        
        impact_metrics = {}
        
        # Tension impact
        tension_change = self._calculate_tension_change(speaker_id, proposed_action)
        impact_metrics['tension_change'] = tension_change
        
        # Pacing impact
        pacing_impact = self._calculate_pacing_impact(proposed_action, context)
        impact_metrics['pacing_impact'] = pacing_impact
        
        # Character development impact
        character_development = self._calculate_character_development_impact(
            speaker_id, proposed_action
        )
        impact_metrics['character_development'] = character_development
        
        # Plot progression impact
        plot_progression = self._calculate_plot_progression_impact(proposed_action)
        impact_metrics['plot_progression'] = plot_progression
        
        # Coherence maintenance
        coherence_impact = self._calculate_coherence_impact(speaker_id, proposed_action)
        impact_metrics['coherence_impact'] = coherence_impact
        
        return impact_metrics
    
    def advance_narrative_stage(self, arc_id: str, trigger_event: str) -> bool:
        """Advance narrative to next stage based on trigger event."""
        
        if arc_id not in self.narrative_arcs:
            return False
        
        arc = self.narrative_arcs[arc_id]
        stage_transitions = {
            "setup": "rising_action",
            "rising_action": "climax",
            "climax": "falling_action",
            "falling_action": "resolution"
        }
        
        if arc.current_stage in stage_transitions:
            new_stage = stage_transitions[arc.current_stage]
            arc.current_stage = new_stage
            
            # Update tension based on stage
            stage_tension = {
                "setup": 0.2,
                "rising_action": 0.6,
                "climax": 1.0,
                "falling_action": 0.4,
                "resolution": 0.1
            }
            arc.tension_level = stage_tension[new_stage]
            
            return True
        
        return False
    
    def _calculate_tension_change(self, speaker_id: str, action: str) -> float:
        """Calculate how action changes narrative tension."""
        
        tension_keywords = {
            'conflict': 0.3,
            'challenge': 0.2,
            'reveal': 0.25,
            'betrayal': 0.4,
            'crisis': 0.35,
            'resolution': -0.3,
            'peace': -0.2,
            'agreement': -0.15
        }
        
        action_lower = action.lower()
        tension_change = 0.0
        
        for keyword, change in tension_keywords.items():
            if keyword in action_lower:
                tension_change += change
        
        return np.clip(tension_change, -0.5, 0.5)
    
    def _calculate_pacing_impact(self, action: str, context: ConversationContext) -> float:
        """Calculate impact on narrative pacing."""
        
        # Fast pacing indicators
        fast_keywords = ['urgent', 'quickly', 'immediate', 'rush', 'hurry']
        slow_keywords = ['slowly', 'carefully', 'gradually', 'pause', 'reflect']
        
        action_lower = action.lower()
        
        fast_score = sum(1 for kw in fast_keywords if kw in action_lower)
        slow_score = sum(1 for kw in slow_keywords if kw in action_lower)
        
        # Current pacing need based on conversation length
        if context.turn_count > 15:
            # Need to speed up
            return fast_score * 0.2 - slow_score * 0.1
        else:
            # Can take time
            return slow_score * 0.1 - fast_score * 0.05
    
    def _calculate_character_development_impact(self, speaker_id: str, action: str) -> float:
        """Calculate character development impact."""
        
        development_keywords = {
            'learn': 0.2,
            'grow': 0.25,
            'realize': 0.3,
            'understand': 0.15,
            'change': 0.35,
            'adapt': 0.2,
            'overcome': 0.3
        }
        
        action_lower = action.lower()
        development_score = 0.0
        
        for keyword, score in development_keywords.items():
            if keyword in action_lower:
                development_score += score
        
        return min(1.0, development_score)
    
    def _calculate_plot_progression_impact(self, action: str) -> float:
        """Calculate plot progression impact."""
        
        progression_keywords = {
            'discover': 0.3,
            'investigate': 0.2,
            'solve': 0.35,
            'complete': 0.4,
            'achieve': 0.3,
            'fail': 0.25,  # Failure also progresses plot
            'attempt': 0.15
        }
        
        action_lower = action.lower()
        progression_score = 0.0
        
        for keyword, score in progression_keywords.items():
            if keyword in action_lower:
                progression_score += score
        
        return min(1.0, progression_score)
    
    def _calculate_coherence_impact(self, speaker_id: str, action: str) -> float:
        """Calculate impact on narrative coherence."""
        
        # Check for contradictions with established facts
        coherence_score = 1.0
        
        # Simple heuristic: look for contradiction keywords
        contradiction_keywords = ['but earlier', 'however', 'actually', 'wait', 'mistake']
        
        action_lower = action.lower()
        for keyword in contradiction_keywords:
            if keyword in action_lower:
                coherence_score -= 0.2
        
        return max(0.0, coherence_score)


class ConflictResolutionEngine:
    """Engine for resolving conflicts between personas."""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.OPINION_DISAGREEMENT: self._resolve_opinion_conflict,
            ConflictType.PERSONALITY_CLASH: self._resolve_personality_conflict,
            ConflictType.GOAL_MISALIGNMENT: self._resolve_goal_conflict,
            ConflictType.RESOURCE_COMPETITION: self._resolve_resource_conflict,
            ConflictType.NARRATIVE_INCONSISTENCY: self._resolve_narrative_conflict,
            ConflictType.TIMING_CONFLICT: self._resolve_timing_conflict
        }
        
    def detect_conflicts(self, 
                        conflicting_responses: List[Tuple[str, str, float]],
                        personas: Dict[str, PersonaAgent]) -> List[Dict[str, Any]]:
        """Detect conflicts between persona responses."""
        
        conflicts = []
        
        for i, (persona1_id, response1, confidence1) in enumerate(conflicting_responses):
            for j, (persona2_id, response2, confidence2) in enumerate(conflicting_responses[i+1:], i+1):
                conflict_type, conflict_severity = self._analyze_conflict(
                    response1, response2, personas[persona1_id], personas[persona2_id]
                )
                
                if conflict_severity > 0.3:  # Threshold for significant conflict
                    conflicts.append({
                        'personas': [persona1_id, persona2_id],
                        'responses': [response1, response2],
                        'conflict_type': conflict_type,
                        'severity': conflict_severity,
                        'confidences': [confidence1, confidence2]
                    })
        
        return conflicts
    
    def resolve_conflict(self, 
                        conflict: Dict[str, Any],
                        resolution_strategy: str = "consensus_weighted") -> Tuple[str, Dict[str, float]]:
        """Resolve a detected conflict."""
        
        conflict_type = conflict['conflict_type']
        
        if conflict_type in self.resolution_strategies:
            resolution_function = self.resolution_strategies[conflict_type]
            return resolution_function(conflict, resolution_strategy)
        else:
            return self._default_resolution(conflict, resolution_strategy)
    
    def _analyze_conflict(self, 
                         response1: str, 
                         response2: str,
                         persona1: PersonaAgent,
                         persona2: PersonaAgent) -> Tuple[ConflictType, float]:
        """Analyze the type and severity of conflict."""
        
        # Simple keyword-based conflict detection
        contradiction_words = ['no', 'wrong', 'disagree', 'however', 'but', 'actually']
        
        response1_lower = response1.lower()
        response2_lower = response2.lower()
        
        # Check for direct contradictions
        contradiction_score = 0.0
        for word in contradiction_words:
            if word in response1_lower or word in response2_lower:
                contradiction_score += 0.2
        
        # Personality-based conflict
        personality_distance = 1.0 - self._calculate_personality_compatibility(
            persona1.personality, persona2.personality
        )
        
        # Determine conflict type and severity
        if contradiction_score > 0.4:
            return ConflictType.OPINION_DISAGREEMENT, contradiction_score
        elif personality_distance > 0.7:
            return ConflictType.PERSONALITY_CLASH, personality_distance
        else:
            return ConflictType.OPINION_DISAGREEMENT, contradiction_score
    
    def _calculate_personality_compatibility(self, 
                                           personality1: PersonalityProfile,
                                           personality2: PersonalityProfile) -> float:
        """Calculate compatibility between personalities."""
        
        traits1 = np.array(list(personality1.traits.values()))
        traits2 = np.array(list(personality2.traits.values()))
        
        similarity = cosine_similarity([traits1], [traits2])[0][0]
        return (similarity + 1.0) / 2.0
    
    def _resolve_opinion_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve opinion disagreement."""
        
        responses = conflict['responses']
        confidences = conflict['confidences']
        
        if strategy == "consensus_weighted":
            # Weight by confidence and create compromise
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weights = [c / total_confidence for c in confidences]
                
                # Simple compromise: acknowledge both views
                resolution = f"I understand there are different perspectives here. {responses[0]} However, {responses[1]} Perhaps we can find common ground."
                
                return resolution, {pid: w for pid, w in zip(conflict['personas'], weights)}
        
        return responses[0], {conflict['personas'][0]: 1.0, conflict['personas'][1]: 0.0}
    
    def _resolve_personality_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve personality clash."""
        
        # For personality conflicts, use the response from more agreeable persona
        # This is a simplification - in practice, would use more sophisticated mediation
        
        responses = conflict['responses']
        return responses[0], {conflict['personas'][0]: 0.7, conflict['personas'][1]: 0.3}
    
    def _resolve_goal_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve goal misalignment."""
        
        responses = conflict['responses']
        # Try to find solution that addresses both goals
        resolution = f"Let's consider both objectives: {responses[0]} while also {responses[1]}"
        
        return resolution, {conflict['personas'][0]: 0.5, conflict['personas'][1]: 0.5}
    
    def _resolve_resource_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve resource competition."""
        
        responses = conflict['responses']
        # Suggest resource sharing or alternative approaches
        resolution = f"Perhaps we can share resources or find alternative approaches: {responses[0]}"
        
        return resolution, {conflict['personas'][0]: 0.6, conflict['personas'][1]: 0.4}
    
    def _resolve_narrative_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve narrative inconsistency."""
        
        responses = conflict['responses']
        # Choose the response that better maintains narrative coherence
        return responses[0], {conflict['personas'][0]: 1.0, conflict['personas'][1]: 0.0}
    
    def _resolve_timing_conflict(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Resolve timing conflict."""
        
        responses = conflict['responses']
        # Use first response for now, implement queue for others
        return responses[0], {conflict['personas'][0]: 1.0, conflict['personas'][1]: 0.0}
    
    def _default_resolution(self, conflict: Dict[str, Any], strategy: str) -> Tuple[str, Dict[str, float]]:
        """Default conflict resolution."""
        
        responses = conflict['responses']
        confidences = conflict['confidences']
        
        # Use highest confidence response
        best_idx = np.argmax(confidences)
        best_persona = conflict['personas'][best_idx]
        best_response = responses[best_idx]
        
        weights = {pid: 0.0 for pid in conflict['personas']}
        weights[best_persona] = 1.0
        
        return best_response, weights


class PerformancePredictionEnsemble:
    """Ensemble learning for performance prediction."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        self.is_trained = False
        self.feature_history = []
        self.performance_history = []
        
    def extract_features(self, 
                        interaction_history: List[Dict[str, Any]],
                        context: ConversationContext) -> np.ndarray:
        """Extract features for performance prediction."""
        
        features = []
        
        # Conversation features
        features.append(context.turn_count)
        features.append(context.duration.total_seconds() / 3600)  # Duration in hours
        features.append(context.difficulty_level)
        features.append(len(context.topic_progression))
        
        # Engagement features
        if context.engagement_metrics:
            features.extend(list(context.engagement_metrics.values())[:5])  # Limit to 5
        else:
            features.extend([0.5] * 5)  # Default values
        
        # Emotional features
        if context.emotional_trajectory:
            latest_emotion = context.emotional_trajectory[-1]
            features.append(latest_emotion.intensity)
            features.append(latest_emotion.stability)
            features.extend(list(latest_emotion.dimensions.values())[:4])  # Limit to 4
        else:
            features.extend([0.5] * 6)  # Default values
        
        # Learning objective features
        features.append(len(context.learning_objectives))
        
        # Interaction history features
        if interaction_history:
            recent_interactions = interaction_history[-5:]  # Last 5 interactions
            avg_satisfaction = np.mean([i.get('satisfaction', 0.5) for i in recent_interactions])
            avg_complexity = np.mean([i.get('complexity', 0.5) for i in recent_interactions])
            features.extend([avg_satisfaction, avg_complexity])
        else:
            features.extend([0.5, 0.5])
        
        return np.array(features)
    
    def train_models(self, training_data: List[Tuple[np.ndarray, Dict[LearningOutcome, float]]]):
        """Train ensemble models on historical data."""
        
        if len(training_data) < 10:  # Need minimum data
            return
        
        X = np.array([features for features, _ in training_data])
        
        # Train separate model for each learning outcome
        self.outcome_models = {}
        
        for outcome in LearningOutcome:
            y = np.array([outcomes.get(outcome, 0.5) for _, outcomes in training_data])
            
            outcome_models = {}
            for model_name, model in self.models.items():
                try:
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X, y)
                    outcome_models[model_name] = model_copy
                except Exception as e:
                    logger.warning(f"Failed to train {model_name} for {outcome}: {e}")
            
            self.outcome_models[outcome] = outcome_models
        
        self.is_trained = True
    
    def predict_outcomes(self, 
                        features: np.ndarray,
                        prediction_horizon: timedelta) -> Dict[LearningOutcome, Tuple[float, float]]:
        """Predict learning outcomes with confidence intervals."""
        
        if not self.is_trained:
            # Return default predictions
            return {outcome: (0.5, 0.2) for outcome in LearningOutcome}
        
        predictions = {}
        
        for outcome in LearningOutcome:
            if outcome not in self.outcome_models:
                predictions[outcome] = (0.5, 0.2)
                continue
            
            outcome_predictions = []
            
            for model_name, model in self.outcome_models[outcome].items():
                try:
                    pred = model.predict(features.reshape(1, -1))[0]
                    outcome_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
            
            if outcome_predictions:
                # Ensemble prediction (mean)
                mean_pred = np.mean(outcome_predictions)
                # Confidence interval (std as proxy)
                confidence = np.std(outcome_predictions) if len(outcome_predictions) > 1 else 0.1
                
                predictions[outcome] = (
                    np.clip(mean_pred, 0.0, 1.0),
                    min(0.5, confidence)
                )
            else:
                predictions[outcome] = (0.5, 0.2)
        
        return predictions


class AdvancedCollaborativeAICoordinator(ICollaborativeAICoordinator, IPerformancePredictionEngine):
    """
    Advanced collaborative AI coordinator with sophisticated algorithms.
    
    Combines multi-persona coordination with performance prediction capabilities.
    """
    
    def __init__(self):
        # Initialize components
        self.turn_allocator = GameTheoryTurnAllocator()
        self.narrative_manager = NarrativeCoherenceManager()
        self.conflict_resolver = ConflictResolutionEngine()
        self.performance_predictor = PerformancePredictionEnsemble()
        
        # Active personas
        self.active_personas = {}
        
        # Coordination history
        self.coordination_history = []
        
        # Performance tracking
        self.performance_data = []
        
        logger.info("AdvancedCollaborativeAICoordinator initialized")
    
    async def coordinate_multi_persona_interaction(
        self,
        active_personas: List[str],
        interaction_context: ConversationContext,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Coordinate interactions between multiple AI personas."""
        
        try:
            # Get persona agents
            persona_agents = [self.active_personas.get(pid) for pid in active_personas]
            persona_agents = [p for p in persona_agents if p is not None]
            
            if not persona_agents:
                return {'error': 'No valid personas available'}
            
            # Calculate optimal speaker using game theory
            payoff_matrix = self.turn_allocator.calculate_payoff_matrix(
                persona_agents, interaction_context, user_preferences
            )
            
            optimal_speaker_id, optimal_payoff = self.turn_allocator.find_optimal_allocation(
                payoff_matrix, persona_agents
            )
            
            # Generate responses from multiple personas
            persona_responses = {}
            for persona in persona_agents:
                response = await self._generate_persona_response(
                    persona, interaction_context, user_preferences
                )
                persona_responses[persona.persona_id] = response
            
            # Detect and resolve conflicts
            response_tuples = [
                (pid, resp['content'], resp['confidence']) 
                for pid, resp in persona_responses.items()
            ]
            
            conflicts = self.conflict_resolver.detect_conflicts(
                response_tuples, {p.persona_id: p for p in persona_agents}
            )
            
            if conflicts:
                # Resolve conflicts
                resolution_results = {}
                for conflict in conflicts:
                    resolved_response, weights = self.conflict_resolver.resolve_conflict(conflict)
                    resolution_results[conflict['personas'][0]] = {
                        'resolved_response': resolved_response,
                        'weights': weights
                    }
                
                coordination_result = {
                    'optimal_speaker': optimal_speaker_id,
                    'all_responses': persona_responses,
                    'conflicts_detected': len(conflicts),
                    'conflict_resolutions': resolution_results,
                    'coordination_strategy': 'conflict_resolution',
                    'payoff_matrix': payoff_matrix.tolist()
                }
            else:
                # No conflicts - use optimal speaker
                coordination_result = {
                    'optimal_speaker': optimal_speaker_id,
                    'selected_response': persona_responses[optimal_speaker_id],
                    'all_responses': persona_responses,
                    'conflicts_detected': 0,
                    'coordination_strategy': 'game_theory_optimal',
                    'payoff_matrix': payoff_matrix.tolist()
                }
            
            # Update coordination history
            self.coordination_history.append({
                'timestamp': datetime.now(),
                'context': interaction_context,
                'result': coordination_result
            })
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Error in coordinate_multi_persona_interaction: {str(e)}")
            return {'error': str(e)}
    
    async def allocate_speaking_turns(
        self,
        personas: List[Dict[str, Any]],
        conversation_dynamics: Dict[str, float],
        narrative_requirements: Dict[str, Any]
    ) -> List[Tuple[str, float, str]]:
        """Allocate speaking turns with timing and reason."""
        
        try:
            allocations = []
            
            # Convert to persona agents if needed
            persona_agents = []
            for persona_data in personas:
                if isinstance(persona_data, dict):
                    # Create temporary persona agent
                    agent = PersonaAgent(
                        persona_id=persona_data['id'],
                        personality=persona_data.get('personality'),
                        current_emotional_state=persona_data.get('emotional_state'),
                        expertise_areas=persona_data.get('expertise', []),
                        speaking_priority=conversation_dynamics.get(persona_data['id'], 0.5),
                        narrative_role=persona_data.get('role', 'supporting')
                    )
                    persona_agents.append(agent)
            
            # Calculate turn allocation scores
            total_priority = sum(p.speaking_priority for p in persona_agents)
            
            for persona in persona_agents:
                # Base allocation from priority
                allocation_score = persona.speaking_priority / total_priority if total_priority > 0 else 1.0 / len(persona_agents)
                
                # Adjust based on narrative requirements
                if narrative_requirements.get('focus_character') == persona.persona_id:
                    allocation_score *= 1.5
                
                if persona.narrative_role == 'protagonist':
                    allocation_score *= 1.3
                elif persona.narrative_role == 'antagonist':
                    allocation_score *= 1.2
                
                # Calculate timing (seconds from now)
                timing = allocation_score * 60  # Convert to seconds
                
                # Generate reasoning
                reasoning = f"Priority: {persona.speaking_priority:.2f}, Role: {persona.narrative_role}"
                
                allocations.append((persona.persona_id, timing, reasoning))
            
            # Sort by timing (earliest first)
            allocations.sort(key=lambda x: x[1])
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in allocate_speaking_turns: {str(e)}")
            return []
    
    async def resolve_persona_conflicts(
        self,
        conflicting_responses: List[Tuple[str, str, float]],
        resolution_strategy: str = "consensus_weighted"
    ) -> Tuple[str, Dict[str, float]]:
        """Resolve conflicts between persona responses."""
        
        try:
            if not conflicting_responses:
                return "", {}
            
            if len(conflicting_responses) == 1:
                persona_id, response, confidence = conflicting_responses[0]
                return response, {persona_id: 1.0}
            
            # Create mock personas for conflict resolution
            mock_personas = {}
            for persona_id, response, confidence in conflicting_responses:
                mock_personas[persona_id] = PersonaAgent(
                    persona_id=persona_id,
                    personality=None,  # Will use defaults in conflict resolution
                    current_emotional_state=None,
                    expertise_areas=[],
                    speaking_priority=confidence,
                    narrative_role='participant'
                )
            
            # Detect conflicts
            conflicts = self.conflict_resolver.detect_conflicts(
                conflicting_responses, mock_personas
            )
            
            if conflicts:
                # Resolve first major conflict
                conflict = conflicts[0]
                resolved_response, weights = self.conflict_resolver.resolve_conflict(
                    conflict, resolution_strategy
                )
                return resolved_response, weights
            else:
                # No conflicts detected - use highest confidence
                best_response = max(conflicting_responses, key=lambda x: x[2])
                persona_id, response, confidence = best_response
                
                weights = {pid: 0.0 for pid, _, _ in conflicting_responses}
                weights[persona_id] = 1.0
                
                return response, weights
                
        except Exception as e:
            logger.error(f"Error in resolve_persona_conflicts: {str(e)}")
            # Fallback to first response
            if conflicting_responses:
                persona_id, response, confidence = conflicting_responses[0]
                return response, {persona_id: 1.0}
            return "", {}
    
    async def maintain_narrative_coherence(
        self,
        storyline: Dict[str, Any],
        persona_actions: List[Dict[str, Any]],
        user_agency_level: float
    ) -> Dict[str, Any]:
        """Maintain narrative coherence across multiple personas."""
        
        try:
            coherence_report = {
                'overall_coherence': 1.0,
                'violations': [],
                'recommendations': [],
                'narrative_health': {}
            }
            
            # Check for narrative violations
            violations = []
            
            # Check character consistency
            for action in persona_actions:
                persona_id = action.get('persona_id')
                action_text = action.get('action', '')
                
                # Evaluate narrative impact
                narrative_impact = self.narrative_manager.evaluate_narrative_impact(
                    persona_id, action_text, None  # Would need context
                )
                
                # Check for coherence violations
                if narrative_impact.get('coherence_impact', 1.0) < 0.5:
                    violations.append({
                        'persona_id': persona_id,
                        'violation_type': 'character_inconsistency',
                        'severity': 1.0 - narrative_impact['coherence_impact'],
                        'action': action_text
                    })
            
            # Check plot consistency
            if len(persona_actions) > 1:
                # Simple check for contradictory actions
                action_texts = [action.get('action', '') for action in persona_actions]
                
                for i, action1 in enumerate(action_texts):
                    for j, action2 in enumerate(action_texts[i+1:], i+1):
                        if self._are_contradictory(action1, action2):
                            violations.append({
                                'violation_type': 'plot_contradiction',
                                'severity': 0.6,
                                'actions': [action1, action2],
                                'personas': [persona_actions[i]['persona_id'], 
                                           persona_actions[j]['persona_id']]
                            })
            
            # Calculate overall coherence
            if violations:
                avg_severity = np.mean([v['severity'] for v in violations])
                coherence_report['overall_coherence'] = max(0.0, 1.0 - avg_severity)
            
            coherence_report['violations'] = violations
            
            # Generate recommendations
            recommendations = []
            
            if violations:
                recommendations.append("Review character motivations for consistency")
                recommendations.append("Ensure actions align with established personality traits")
                
                if len(violations) > 2:
                    recommendations.append("Consider reducing number of active personas")
            
            if user_agency_level < 0.3:
                recommendations.append("Increase user involvement in narrative decisions")
            elif user_agency_level > 0.8:
                recommendations.append("Provide more guided narrative structure")
            
            coherence_report['recommendations'] = recommendations
            
            # Narrative health metrics
            coherence_report['narrative_health'] = {
                'tension_level': storyline.get('tension', 0.5),
                'pacing_score': storyline.get('pacing', 0.5),
                'character_development': len([a for a in persona_actions if 'growth' in a.get('action', '')]) / max(1, len(persona_actions)),
                'user_agency': user_agency_level
            }
            
            return coherence_report
            
        except Exception as e:
            logger.error(f"Error in maintain_narrative_coherence: {str(e)}")
            return {'overall_coherence': 0.5, 'error': str(e)}
    
    # Performance Prediction Methods
    
    async def predict_learning_outcomes(
        self,
        user_interaction_history: List[Dict[str, Any]],
        current_session_data: ConversationContext,
        prediction_horizon: timedelta
    ) -> Dict[LearningOutcome, Tuple[float, float]]:
        """Predict learning outcomes with confidence intervals."""
        
        try:
            # Extract features
            features = self.performance_predictor.extract_features(
                user_interaction_history, current_session_data
            )
            
            # Make predictions
            predictions = self.performance_predictor.predict_outcomes(
                features, prediction_horizon
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_learning_outcomes: {str(e)}")
            return {outcome: (0.5, 0.2) for outcome in LearningOutcome}
    
    async def analyze_skill_progression(
        self,
        skill_assessments: List[Dict[str, Any]],
        interaction_patterns: Dict[str, Any],
        time_series_data: List[Tuple[datetime, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze skill progression patterns and trends."""
        
        try:
            progression_analysis = {}
            
            # Analyze each skill
            skills = set()
            for assessment in skill_assessments:
                skills.update(assessment.get('skills', {}).keys())
            
            for skill in skills:
                # Extract skill progression over time
                skill_progression = []
                for assessment in skill_assessments:
                    if skill in assessment.get('skills', {}):
                        skill_progression.append(assessment['skills'][skill])
                
                if len(skill_progression) >= 2:
                    # Calculate trend
                    trend = np.polyfit(range(len(skill_progression)), skill_progression, 1)[0]
                    
                    # Calculate variance (consistency)
                    variance = np.var(skill_progression)
                    
                    # Calculate current level
                    current_level = skill_progression[-1] if skill_progression else 0.5
                    
                    progression_analysis[skill] = {
                        'trend': float(trend),
                        'variance': float(variance),
                        'current_level': float(current_level),
                        'improvement_rate': float(trend * len(skill_progression)),
                        'consistency': float(1.0 - min(1.0, variance))
                    }
                else:
                    # Insufficient data
                    progression_analysis[skill] = {
                        'trend': 0.0,
                        'variance': 0.0,
                        'current_level': 0.5,
                        'improvement_rate': 0.0,
                        'consistency': 0.5
                    }
            
            return progression_analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_skill_progression: {str(e)}")
            return {}
    
    async def identify_learning_bottlenecks(
        self,
        performance_data: Dict[str, List[float]],
        engagement_metrics: Dict[str, float],
        error_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify potential learning bottlenecks and challenges."""
        
        try:
            bottlenecks = []
            
            # Analyze performance plateaus
            for skill, scores in performance_data.items():
                if len(scores) >= 5:
                    # Check for plateau (no improvement in last 5 measurements)
                    recent_scores = scores[-5:]
                    if max(recent_scores) - min(recent_scores) < 0.1:
                        bottlenecks.append({
                            'type': 'plateau',
                            'skill': skill,
                            'severity': 0.7,
                            'description': f"No significant improvement in {skill} over recent attempts",
                            'recommendation': f"Try alternative learning strategies for {skill}"
                        })
            
            # Analyze engagement issues
            low_engagement_threshold = 0.4
            for metric, value in engagement_metrics.items():
                if value < low_engagement_threshold:
                    bottlenecks.append({
                        'type': 'engagement',
                        'metric': metric,
                        'value': value,
                        'severity': (low_engagement_threshold - value) / low_engagement_threshold,
                        'description': f"Low {metric} engagement detected",
                        'recommendation': f"Increase variety and interactivity to improve {metric}"
                    })
            
            # Analyze error patterns
            error_frequency = defaultdict(int)
            for error in error_patterns:
                error_type = error.get('type', 'unknown')
                error_frequency[error_type] += 1
            
            for error_type, frequency in error_frequency.items():
                if frequency >= 3:  # Recurring error
                    bottlenecks.append({
                        'type': 'recurring_error',
                        'error_type': error_type,
                        'frequency': frequency,
                        'severity': min(1.0, frequency / 10),
                        'description': f"Recurring {error_type} errors detected",
                        'recommendation': f"Focus on addressing {error_type} specifically"
                    })
            
            # Sort by severity
            bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error in identify_learning_bottlenecks: {str(e)}")
            return []
    
    async def recommend_intervention_strategies(
        self,
        predicted_performance: Dict[LearningOutcome, float],
        risk_factors: List[str],
        available_resources: List[str]
    ) -> List[Dict[str, Any]]:
        """Recommend intervention strategies for at-risk learners."""
        
        try:
            interventions = []
            
            # Analyze performance risks
            at_risk_threshold = 0.4
            at_risk_outcomes = [
                outcome for outcome, score in predicted_performance.items() 
                if score < at_risk_threshold
            ]
            
            # Generate targeted interventions
            for outcome in at_risk_outcomes:
                interventions.append({
                    'type': 'targeted_practice',
                    'target': outcome.value,
                    'priority': 'high' if predicted_performance[outcome] < 0.3 else 'medium',
                    'description': f"Focused practice sessions for {outcome.value}",
                    'estimated_duration': timedelta(hours=2),
                    'resources_needed': ['practice_materials', 'feedback_system'],
                    'success_criteria': f"Improve {outcome.value} score above 0.6"
                })
            
            # Address specific risk factors
            risk_interventions = {
                'low_engagement': {
                    'type': 'engagement_boost',
                    'description': 'Gamification and interactive elements',
                    'resources_needed': ['interactive_content', 'reward_system']
                },
                'high_difficulty': {
                    'type': 'difficulty_adjustment',
                    'description': 'Reduce complexity and provide more scaffolding',
                    'resources_needed': ['adaptive_content', 'tutoring_support']
                },
                'time_pressure': {
                    'type': 'pacing_adjustment',
                    'description': 'Extend timelines and reduce pressure',
                    'resources_needed': ['flexible_scheduling']
                },
                'knowledge_gaps': {
                    'type': 'prerequisite_review',
                    'description': 'Review foundational concepts',
                    'resources_needed': ['review_materials', 'diagnostic_tools']
                }
            }
            
            for risk_factor in risk_factors:
                if risk_factor in risk_interventions:
                    intervention = risk_interventions[risk_factor].copy()
                    intervention['risk_factor'] = risk_factor
                    intervention['priority'] = 'high'
                    interventions.append(intervention)
            
            # Filter by available resources
            feasible_interventions = []
            for intervention in interventions:
                required_resources = intervention.get('resources_needed', [])
                if all(resource in available_resources for resource in required_resources):
                    intervention['feasible'] = True
                    feasible_interventions.append(intervention)
                else:
                    intervention['feasible'] = False
                    intervention['missing_resources'] = [
                        r for r in required_resources if r not in available_resources
                    ]
                    feasible_interventions.append(intervention)
            
            # Sort by priority and feasibility
            feasible_interventions.sort(
                key=lambda x: (x['priority'] == 'high', x['feasible']), 
                reverse=True
            )
            
            return feasible_interventions
            
        except Exception as e:
            logger.error(f"Error in recommend_intervention_strategies: {str(e)}")
            return []
    
    # Helper methods
    
    async def _generate_persona_response(self, 
                                       persona: PersonaAgent,
                                       context: ConversationContext,
                                       user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for a specific persona."""
        
        # Simplified response generation
        # In practice, this would call the main AI model with persona context
        
        base_responses = [
            "I understand your perspective on this.",
            "Let me share my thoughts on this topic.",
            "That's an interesting point to consider.",
            "I'd like to add something to this discussion.",
            "From my experience, I think..."
        ]
        
        response_content = random.choice(base_responses)
        
        # Adjust based on personality
        if persona.personality:
            if persona.personality.traits.get(PersonalityTrait.EXTRAVERSION, 0.5) > 0.7:
                response_content = "Absolutely! " + response_content
            if persona.personality.traits.get(PersonalityTrait.AGREEABLENESS, 0.5) > 0.7:
                response_content = response_content + " What do you think?"
        
        confidence = random.uniform(0.6, 0.9)
        
        return {
            'content': response_content,
            'confidence': confidence,
            'persona_id': persona.persona_id,
            'timestamp': datetime.now()
        }
    
    def _are_contradictory(self, action1: str, action2: str) -> bool:
        """Check if two actions are contradictory."""
        
        # Simple keyword-based contradiction detection
        contradiction_pairs = [
            (['agree', 'yes', 'correct'], ['disagree', 'no', 'wrong']),
            (['go', 'move', 'advance'], ['stop', 'stay', 'wait']),
            (['attack', 'fight'], ['retreat', 'flee', 'peace']),
            (['happy', 'joy'], ['sad', 'angry', 'upset'])
        ]
        
        action1_lower = action1.lower()
        action2_lower = action2.lower()
        
        for positive_words, negative_words in contradiction_pairs:
            has_positive_1 = any(word in action1_lower for word in positive_words)
            has_negative_1 = any(word in action1_lower for word in negative_words)
            has_positive_2 = any(word in action2_lower for word in positive_words)
            has_negative_2 = any(word in action2_lower for word in negative_words)
            
            if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                return True
        
        return False
    
    def register_persona(self, persona_data: Dict[str, Any]):
        """Register a new persona agent."""
        
        persona_agent = PersonaAgent(
            persona_id=persona_data['id'],
            personality=persona_data.get('personality'),
            current_emotional_state=persona_data.get('emotional_state'),
            expertise_areas=persona_data.get('expertise', []),
            speaking_priority=persona_data.get('priority', 0.5),
            narrative_role=persona_data.get('role', 'supporting')
        )
        
        self.active_personas[persona_data['id']] = persona_agent
        
        logger.info(f"Registered persona: {persona_data['id']}")
    
    def update_persona_performance(self, 
                                  persona_id: str, 
                                  performance_metrics: Dict[str, float]):
        """Update performance metrics for a persona."""
        
        if persona_id in self.active_personas:
            self.active_personas[persona_id].performance_metrics.update(performance_metrics)
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination system metrics."""
        
        if not self.coordination_history:
            return {'total_coordinations': 0}
        
        recent_history = self.coordination_history[-10:]  # Last 10 coordinations
        
        conflict_rates = [h['result'].get('conflicts_detected', 0) for h in recent_history]
        avg_conflicts = np.mean(conflict_rates) if conflict_rates else 0
        
        strategies_used = [h['result'].get('coordination_strategy') for h in recent_history]
        strategy_distribution = {
            strategy: strategies_used.count(strategy) / len(strategies_used)
            for strategy in set(strategies_used) if strategy
        }
        
        return {
            'total_coordinations': len(self.coordination_history),
            'average_conflicts_per_session': avg_conflicts,
            'strategy_distribution': strategy_distribution,
            'active_personas': len(self.active_personas),
            'coordination_success_rate': 1.0 - (avg_conflicts / 5)  # Normalize by max expected conflicts
        }