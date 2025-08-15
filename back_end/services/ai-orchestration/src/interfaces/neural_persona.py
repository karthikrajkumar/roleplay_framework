"""
Neural Persona Architecture Interfaces

Advanced interfaces for the neural persona system with emotional intelligence,
multi-modal processing, and adaptive learning capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol, TypeVar, Generic
from datetime import datetime, timedelta
from uuid import UUID
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field

# Type variables for generic interfaces
T = TypeVar('T')
PersonaState = TypeVar('PersonaState')
EmotionalVector = TypeVar('EmotionalVector')


class ModalityType(str, Enum):
    """Types of input modalities supported by the system."""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    GESTURE = "gesture"
    BIOMETRIC = "biometric"


class EmotionalDimension(str, Enum):
    """Emotional dimensions for multi-dimensional emotional modeling."""
    VALENCE = "valence"  # Positive/Negative
    AROUSAL = "arousal"  # Active/Passive
    DOMINANCE = "dominance"  # Dominant/Submissive
    ENGAGEMENT = "engagement"  # Engaged/Disengaged
    AUTHENTICITY = "authenticity"  # Genuine/Artificial
    EMPATHY = "empathy"  # Empathetic/Detached


class PersonalityTrait(str, Enum):
    """Big Five personality traits with additional roleplay-specific traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    # Extended traits for roleplay
    CREATIVITY = "creativity"
    ADAPTABILITY = "adaptability"
    CHARISMA = "charisma"
    INTELLIGENCE = "intelligence"
    HUMOR = "humor"


class ConversationPhase(str, Enum):
    """Phases of conversation for path optimization."""
    INTRODUCTION = "introduction"
    EXPLORATION = "exploration"
    CONFLICT = "conflict"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    REFLECTION = "reflection"
    TRANSITION = "transition"


class LearningOutcome(str, Enum):
    """Potential learning outcomes for prediction algorithms."""
    LANGUAGE_SKILLS = "language_skills"
    CULTURAL_AWARENESS = "cultural_awareness"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVITY = "creativity"
    CONFIDENCE = "confidence"
    EMPATHY = "empathy"
    COMMUNICATION = "communication"


@dataclass(frozen=True)
class EmotionalState:
    """Immutable emotional state representation with multi-dimensional modeling."""
    dimensions: Dict[EmotionalDimension, float]  # Values between -1.0 and 1.0
    intensity: float  # Overall emotional intensity 0.0 to 1.0
    stability: float  # Emotional stability measure 0.0 to 1.0
    timestamp: datetime
    confidence: float = 1.0  # Confidence in the emotional assessment
    
    def __post_init__(self):
        """Validate emotional state values."""
        for dim, value in self.dimensions.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"Emotional dimension {dim} must be between -1.0 and 1.0")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.stability <= 1.0:
            raise ValueError("Stability must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PersonalityProfile:
    """Immutable personality profile with trait vectors."""
    traits: Dict[PersonalityTrait, float]  # Values between 0.0 and 1.0
    consistency_score: float  # How consistent the personality is
    adaptability_range: Dict[PersonalityTrait, Tuple[float, float]]  # Min/max adaptation ranges
    last_updated: datetime
    
    def __post_init__(self):
        """Validate personality profile values."""
        for trait, value in self.traits.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Personality trait {trait} must be between 0.0 and 1.0")


@dataclass
class MultiModalInput:
    """Multi-modal input container with temporal alignment."""
    modalities: Dict[ModalityType, Any]
    timestamp: datetime
    session_id: UUID
    user_id: UUID
    confidence_scores: Dict[ModalityType, float] = None
    preprocessing_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {modality: 1.0 for modality in self.modalities.keys()}


@dataclass
class ConversationContext:
    """Rich conversation context with temporal and relational information."""
    session_id: UUID
    user_id: UUID
    persona_id: UUID
    current_phase: ConversationPhase
    turn_count: int
    duration: timedelta
    emotional_trajectory: List[EmotionalState]
    topic_progression: List[str]
    engagement_metrics: Dict[str, float]
    learning_objectives: List[LearningOutcome]
    difficulty_level: float  # 0.0 to 1.0
    user_preferences: Dict[str, Any]
    relationship_dynamics: Dict[str, float]


class IEmotionalIntelligenceEngine(ABC):
    """
    Advanced emotional intelligence engine for dynamic emotional state tracking
    and personality-consistent responses.
    """
    
    @abstractmethod
    async def analyze_emotional_state(
        self,
        input_data: MultiModalInput,
        context: ConversationContext
    ) -> EmotionalState:
        """Analyze current emotional state from multi-modal input."""
        pass
    
    @abstractmethod
    async def predict_emotional_transition(
        self,
        current_state: EmotionalState,
        proposed_response: str,
        context: ConversationContext
    ) -> EmotionalState:
        """Predict emotional state after a proposed response."""
        pass
    
    @abstractmethod
    async def generate_emotionally_consistent_response(
        self,
        target_emotional_state: EmotionalState,
        personality: PersonalityProfile,
        context: ConversationContext
    ) -> Tuple[str, float]:
        """Generate response consistent with target emotional state and personality."""
        pass
    
    @abstractmethod
    async def calculate_emotional_resonance(
        self,
        persona_emotion: EmotionalState,
        user_emotion: EmotionalState
    ) -> float:
        """Calculate emotional resonance between persona and user."""
        pass


class IMultiModalFusionEngine(ABC):
    """
    Advanced multi-modal fusion engine for real-time processing of text, audio,
    and video inputs with temporal alignment and cross-modal attention.
    """
    
    @abstractmethod
    async def fuse_modalities(
        self,
        input_data: MultiModalInput,
        fusion_strategy: str = "attention_weighted"
    ) -> Dict[str, Any]:
        """Fuse multiple modalities into unified representation."""
        pass
    
    @abstractmethod
    async def extract_cross_modal_features(
        self,
        input_data: MultiModalInput
    ) -> Dict[str, np.ndarray]:
        """Extract cross-modal features with attention mechanisms."""
        pass
    
    @abstractmethod
    async def temporal_alignment(
        self,
        modality_streams: Dict[ModalityType, List[Any]],
        alignment_window: timedelta
    ) -> Dict[ModalityType, List[Any]]:
        """Align modalities temporally for synchronized processing."""
        pass
    
    @abstractmethod
    async def quality_assessment(
        self,
        input_data: MultiModalInput
    ) -> Dict[ModalityType, float]:
        """Assess quality of each modality for adaptive processing."""
        pass


class IPredictiveLearningEngine(ABC):
    """
    Predictive learning engine with conversation path optimization using
    reinforcement learning and neural architecture search.
    """
    
    @abstractmethod
    async def predict_conversation_paths(
        self,
        context: ConversationContext,
        num_paths: int = 5,
        lookahead_turns: int = 3
    ) -> List[Tuple[List[str], float]]:
        """Predict optimal conversation paths with confidence scores."""
        pass
    
    @abstractmethod
    async def optimize_learning_trajectory(
        self,
        user_profile: Dict[str, Any],
        learning_objectives: List[LearningOutcome],
        current_context: ConversationContext
    ) -> Dict[str, Any]:
        """Optimize learning trajectory for maximum educational impact."""
        pass
    
    @abstractmethod
    async def adaptive_curriculum_generation(
        self,
        user_progress: Dict[LearningOutcome, float],
        difficulty_preferences: Dict[str, float],
        time_constraints: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Generate adaptive curriculum based on user progress and preferences."""
        pass
    
    @abstractmethod
    async def predict_engagement_decay(
        self,
        context: ConversationContext,
        current_engagement: float
    ) -> Tuple[float, timedelta]:
        """Predict when user engagement will start to decay."""
        pass


class IAdaptiveDifficultyEngine(ABC):
    """
    Adaptive difficulty engine that adjusts challenge level based on user
    performance, emotional state, and learning objectives.
    """
    
    @abstractmethod
    async def calculate_optimal_difficulty(
        self,
        user_performance: Dict[str, float],
        emotional_state: EmotionalState,
        learning_objectives: List[LearningOutcome],
        context: ConversationContext
    ) -> float:
        """Calculate optimal difficulty level for current context."""
        pass
    
    @abstractmethod
    async def adjust_complexity_dynamically(
        self,
        current_difficulty: float,
        user_feedback: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """Dynamically adjust complexity based on real-time feedback."""
        pass
    
    @abstractmethod
    async def generate_scaffolding_strategy(
        self,
        difficulty_gap: float,
        user_strengths: List[str],
        learning_style: str
    ) -> Dict[str, Any]:
        """Generate scaffolding strategy to bridge difficulty gaps."""
        pass
    
    @abstractmethod
    async def predict_frustration_threshold(
        self,
        user_profile: Dict[str, Any],
        current_emotional_state: EmotionalState,
        session_history: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Predict user's frustration threshold and confidence interval."""
        pass


class IMemoryConsolidationEngine(ABC):
    """
    Advanced memory consolidation system for long-term context retention
    and relationship building with hierarchical memory structures.
    """
    
    @abstractmethod
    async def consolidate_episodic_memory(
        self,
        conversation_session: ConversationContext,
        key_moments: List[Dict[str, Any]],
        emotional_highlights: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Consolidate episodic memories from conversation sessions."""
        pass
    
    @abstractmethod
    async def update_semantic_memory(
        self,
        learned_concepts: List[str],
        relationship_updates: Dict[str, Any],
        personality_adaptations: Dict[PersonalityTrait, float]
    ) -> None:
        """Update long-term semantic memory with new information."""
        pass
    
    @abstractmethod
    async def retrieve_relevant_memories(
        self,
        query_context: ConversationContext,
        memory_types: List[str],
        recency_weight: float = 0.3,
        relevance_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using multi-criteria retrieval."""
        pass
    
    @abstractmethod
    async def calculate_memory_importance(
        self,
        memory_item: Dict[str, Any],
        emotional_intensity: float,
        frequency_accessed: int,
        recency: timedelta
    ) -> float:
        """Calculate importance score for memory consolidation."""
        pass
    
    @abstractmethod
    async def forget_obsolete_memories(
        self,
        forgetting_curve_params: Dict[str, float],
        importance_threshold: float = 0.1
    ) -> List[UUID]:
        """Implement forgetting mechanism for obsolete memories."""
        pass


class ICollaborativeAICoordinator(ABC):
    """
    Collaborative AI coordination system for multi-character scenarios
    with dynamic role allocation and conflict resolution.
    """
    
    @abstractmethod
    async def coordinate_multi_persona_interaction(
        self,
        active_personas: List[UUID],
        interaction_context: ConversationContext,
        user_preferences: Dict[str, Any]
    ) -> Dict[UUID, Dict[str, Any]]:
        """Coordinate interactions between multiple AI personas."""
        pass
    
    @abstractmethod
    async def allocate_speaking_turns(
        self,
        personas: List[Dict[str, Any]],
        conversation_dynamics: Dict[str, float],
        narrative_requirements: Dict[str, Any]
    ) -> List[Tuple[UUID, float, str]]:
        """Allocate speaking turns with timing and reason."""
        pass
    
    @abstractmethod
    async def resolve_persona_conflicts(
        self,
        conflicting_responses: List[Tuple[UUID, str, float]],
        resolution_strategy: str = "consensus_weighted"
    ) -> Tuple[str, Dict[UUID, float]]:
        """Resolve conflicts between persona responses."""
        pass
    
    @abstractmethod
    async def maintain_narrative_coherence(
        self,
        storyline: Dict[str, Any],
        persona_actions: List[Dict[str, Any]],
        user_agency_level: float
    ) -> Dict[str, Any]:
        """Maintain narrative coherence across multiple personas."""
        pass


class IPerformancePredictionEngine(ABC):
    """
    Performance prediction engine for learning outcomes using advanced
    machine learning and time-series forecasting.
    """
    
    @abstractmethod
    async def predict_learning_outcomes(
        self,
        user_interaction_history: List[Dict[str, Any]],
        current_session_data: ConversationContext,
        prediction_horizon: timedelta
    ) -> Dict[LearningOutcome, Tuple[float, float]]:
        """Predict learning outcomes with confidence intervals."""
        pass
    
    @abstractmethod
    async def analyze_skill_progression(
        self,
        skill_assessments: List[Dict[str, Any]],
        interaction_patterns: Dict[str, Any],
        time_series_data: List[Tuple[datetime, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze skill progression patterns and trends."""
        pass
    
    @abstractmethod
    async def identify_learning_bottlenecks(
        self,
        performance_data: Dict[str, List[float]],
        engagement_metrics: Dict[str, float],
        error_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify potential learning bottlenecks and challenges."""
        pass
    
    @abstractmethod
    async def recommend_intervention_strategies(
        self,
        predicted_performance: Dict[LearningOutcome, float],
        risk_factors: List[str],
        available_resources: List[str]
    ) -> List[Dict[str, Any]]:
        """Recommend intervention strategies for at-risk learners."""
        pass


class IEthicalAIFramework(ABC):
    """
    Ethical AI framework with bias detection, mitigation, and fairness
    monitoring for responsible AI persona interactions.
    """
    
    @abstractmethod
    async def detect_bias_patterns(
        self,
        interaction_history: List[Dict[str, Any]],
        demographic_data: Dict[str, Any],
        response_patterns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Detect bias patterns in AI responses and interactions."""
        pass
    
    @abstractmethod
    async def mitigate_identified_bias(
        self,
        bias_report: Dict[str, Dict[str, float]],
        mitigation_strategy: str,
        intervention_strength: float = 0.5
    ) -> Dict[str, Any]:
        """Apply bias mitigation strategies to AI responses."""
        pass
    
    @abstractmethod
    async def monitor_fairness_metrics(
        self,
        user_interactions: List[Dict[str, Any]],
        outcome_metrics: Dict[str, List[float]],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Monitor fairness metrics across different user groups."""
        pass
    
    @abstractmethod
    async def ensure_consent_and_privacy(
        self,
        data_collection_request: Dict[str, Any],
        user_privacy_preferences: Dict[str, Any],
        regulatory_requirements: List[str]
    ) -> Tuple[bool, List[str]]:
        """Ensure consent and privacy compliance."""
        pass
    
    @abstractmethod
    async def audit_decision_transparency(
        self,
        decision_context: Dict[str, Any],
        explanation_level: str = "detailed"
    ) -> Dict[str, Any]:
        """Provide transparent explanations for AI decisions."""
        pass


class INeuralPersonaOrchestrator(ABC):
    """
    Main orchestrator interface that coordinates all neural persona subsystems
    for cohesive, intelligent, and ethical AI interactions.
    """
    
    @abstractmethod
    async def process_user_interaction(
        self,
        input_data: MultiModalInput,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Process complete user interaction through all subsystems."""
        pass
    
    @abstractmethod
    async def generate_persona_response(
        self,
        processed_input: Dict[str, Any],
        persona_id: UUID,
        response_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive persona response with all enhancements."""
        pass
    
    @abstractmethod
    async def optimize_interaction_quality(
        self,
        interaction_session: ConversationContext,
        quality_metrics: Dict[str, float],
        optimization_goals: List[str]
    ) -> Dict[str, Any]:
        """Optimize overall interaction quality using all subsystems."""
        pass
    
    @abstractmethod
    async def monitor_system_health(
        self,
        performance_metrics: Dict[str, float],
        error_rates: Dict[str, float],
        user_satisfaction: Dict[str, float]
    ) -> Dict[str, Any]:
        """Monitor overall system health and performance."""
        pass


# Protocol for dependency injection
class NeuralPersonaServiceProvider(Protocol):
    """Protocol defining the service provider interface for dependency injection."""
    
    def get_emotional_intelligence_engine(self) -> IEmotionalIntelligenceEngine:
        """Get emotional intelligence engine instance."""
        ...
    
    def get_multimodal_fusion_engine(self) -> IMultiModalFusionEngine:
        """Get multi-modal fusion engine instance."""
        ...
    
    def get_predictive_learning_engine(self) -> IPredictiveLearningEngine:
        """Get predictive learning engine instance."""
        ...
    
    def get_adaptive_difficulty_engine(self) -> IAdaptiveDifficultyEngine:
        """Get adaptive difficulty engine instance."""
        ...
    
    def get_memory_consolidation_engine(self) -> IMemoryConsolidationEngine:
        """Get memory consolidation engine instance."""
        ...
    
    def get_collaborative_ai_coordinator(self) -> ICollaborativeAICoordinator:
        """Get collaborative AI coordinator instance."""
        ...
    
    def get_performance_prediction_engine(self) -> IPerformancePredictionEngine:
        """Get performance prediction engine instance."""
        ...
    
    def get_ethical_ai_framework(self) -> IEthicalAIFramework:
        """Get ethical AI framework instance."""
        ...
    
    def get_neural_persona_orchestrator(self) -> INeuralPersonaOrchestrator:
        """Get neural persona orchestrator instance."""
        ...