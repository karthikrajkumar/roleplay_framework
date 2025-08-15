"""
Advanced Adaptive Difficulty Engine

This module implements sophisticated adaptive difficulty algorithms including:
- Dynamic Difficulty Adjustment (DDA) using reinforcement learning
- Emotional State-Aware Difficulty Scaling
- Zone of Proximal Development (ZPD) optimization
- Flow Theory-based engagement optimization
- Bayesian Knowledge Tracing for skill assessment
- Multi-objective optimization for difficulty balancing
- Frustration-Flow model for optimal challenge levels
"""

import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from collections import deque, defaultdict

from ..interfaces.neural_persona import (
    IAdaptiveDifficultyEngine,
    EmotionalState,
    EmotionalDimension,
    ConversationContext,
    LearningOutcome
)


logger = logging.getLogger(__name__)


@dataclass
class DifficultyProfile:
    """User's difficulty profile and preferences."""
    user_id: str
    preferred_difficulty: float  # 0.0 to 1.0
    tolerance_range: Tuple[float, float]  # (min, max) acceptable difficulty
    adaptation_speed: float  # How quickly to adapt (0.0 to 1.0)
    challenge_preference: str  # "gradual", "steep", "variable"
    frustration_threshold: float  # Emotional threshold before backing off
    flow_indicators: Dict[str, float]  # Indicators of flow state
    mastery_confidence: Dict[LearningOutcome, float]  # Confidence in each skill
    historical_performance: List[Dict[str, Any]]  # Performance history


@dataclass
class FlowMetrics:
    """Metrics for measuring flow state."""
    challenge_skill_balance: float  # How well challenge matches skill
    clear_goals: float  # Clarity of objectives
    immediate_feedback: float  # Quality of feedback loop
    concentration: float  # Level of focus/concentration
    sense_of_control: float  # User's sense of control
    loss_of_self_consciousness: float  # Immersion level
    time_transformation: float  # Altered perception of time
    overall_flow_score: float  # Combined flow measure


@dataclass
class ZPDAnalysis:
    """Zone of Proximal Development analysis."""
    current_skill_level: float
    independent_capability: float  # What user can do alone
    assisted_capability: float  # What user can do with help
    zpd_range: Tuple[float, float]  # (lower_bound, upper_bound)
    optimal_difficulty: float  # Sweet spot within ZPD
    scaffolding_needs: List[str]  # Required support types
    confidence_interval: Tuple[float, float]  # Uncertainty bounds


class BayesianKnowledgeTracer:
    """Bayesian Knowledge Tracing for skill assessment."""
    
    def __init__(self, 
                 prior_knowledge: float = 0.1,
                 learning_rate: float = 0.1,
                 guess_rate: float = 0.1,
                 slip_rate: float = 0.1):
        self.prior_knowledge = prior_knowledge
        self.learning_rate = learning_rate
        self.guess_rate = guess_rate
        self.slip_rate = slip_rate
        
        # Track knowledge state for each skill
        self.knowledge_states = {}
        self.performance_history = defaultdict(list)
    
    def update_knowledge_state(
        self,
        user_id: str,
        skill: LearningOutcome,
        performance: bool,
        difficulty: float
    ) -> float:
        """
        Update knowledge state based on performance.
        
        Args:
            user_id: User identifier
            skill: Learning outcome/skill
            performance: Whether user succeeded (True/False)
            difficulty: Difficulty level of the task
        
        Returns:
            Updated knowledge probability
        """
        
        # Initialize if first time
        key = f"{user_id}_{skill.value}"
        if key not in self.knowledge_states:
            self.knowledge_states[key] = self.prior_knowledge
        
        current_knowledge = self.knowledge_states[key]
        
        # Adjust parameters based on difficulty
        adjusted_guess = self.guess_rate * (1.0 - difficulty)
        adjusted_slip = self.slip_rate * difficulty
        
        # Calculate probability of correct response given knowledge state
        if performance:
            # Correct response
            p_correct_given_known = 1.0 - adjusted_slip
            p_correct_given_unknown = adjusted_guess
            
            # Bayes' theorem
            numerator = current_knowledge * p_correct_given_known
            denominator = (current_knowledge * p_correct_given_known + 
                          (1 - current_knowledge) * p_correct_given_unknown)
        else:
            # Incorrect response
            p_incorrect_given_known = adjusted_slip
            p_incorrect_given_unknown = 1.0 - adjusted_guess
            
            # Bayes' theorem
            numerator = current_knowledge * p_incorrect_given_known
            denominator = (current_knowledge * p_incorrect_given_known + 
                          (1 - current_knowledge) * p_incorrect_given_unknown)
        
        # Update knowledge state
        if denominator > 0:
            updated_knowledge = numerator / denominator
        else:
            updated_knowledge = current_knowledge
        
        # Apply learning rate (knowledge can increase over time)
        if performance:
            updated_knowledge = min(1.0, updated_knowledge + self.learning_rate * (1 - updated_knowledge))
        
        self.knowledge_states[key] = updated_knowledge
        
        # Store performance history
        self.performance_history[key].append({
            'timestamp': datetime.now(),
            'performance': performance,
            'difficulty': difficulty,
            'knowledge_state': updated_knowledge
        })
        
        return updated_knowledge
    
    def get_knowledge_state(self, user_id: str, skill: LearningOutcome) -> float:
        """Get current knowledge state for a skill."""
        key = f"{user_id}_{skill.value}"
        return self.knowledge_states.get(key, self.prior_knowledge)
    
    def predict_performance(
        self,
        user_id: str,
        skill: LearningOutcome,
        difficulty: float
    ) -> float:
        """Predict probability of successful performance."""
        
        knowledge = self.get_knowledge_state(user_id, skill)
        
        # Adjust parameters based on difficulty
        adjusted_guess = self.guess_rate * (1.0 - difficulty)
        adjusted_slip = self.slip_rate * difficulty
        
        # P(correct) = P(known) * P(correct|known) + P(unknown) * P(correct|unknown)
        p_correct = (knowledge * (1.0 - adjusted_slip) + 
                    (1.0 - knowledge) * adjusted_guess)
        
        return p_correct


class FlowStateDetector:
    """Detect and measure flow state from user behavior."""
    
    def __init__(self):
        self.response_time_history = deque(maxlen=10)
        self.error_rate_history = deque(maxlen=10)
        self.engagement_history = deque(maxlen=10)
        
    def analyze_flow_state(
        self,
        response_times: List[float],
        error_rates: List[float],
        engagement_metrics: Dict[str, float],
        emotional_state: EmotionalState,
        difficulty_level: float,
        skill_level: float
    ) -> FlowMetrics:
        """Analyze current flow state from behavioral indicators."""
        
        # Challenge-skill balance (core of flow theory)
        challenge_skill_balance = self._calculate_challenge_skill_balance(
            difficulty_level, skill_level
        )
        
        # Concentration (from response time consistency)
        concentration = self._calculate_concentration(response_times)
        
        # Sense of control (from error rates and emotional state)
        sense_of_control = self._calculate_sense_of_control(
            error_rates, emotional_state
        )
        
        # Clear goals (from engagement with objectives)
        clear_goals = engagement_metrics.get('goal_clarity', 0.5)
        
        # Immediate feedback (from responsiveness to feedback)
        immediate_feedback = engagement_metrics.get('feedback_responsiveness', 0.5)
        
        # Loss of self-consciousness (from immersion indicators)
        loss_of_self_consciousness = self._calculate_immersion(
            response_times, engagement_metrics
        )
        
        # Time transformation (from session duration vs. perceived time)
        time_transformation = engagement_metrics.get('time_distortion', 0.5)
        
        # Overall flow score (weighted combination)
        overall_flow_score = (
            challenge_skill_balance * 0.25 +
            concentration * 0.20 +
            sense_of_control * 0.15 +
            clear_goals * 0.10 +
            immediate_feedback * 0.10 +
            loss_of_self_consciousness * 0.10 +
            time_transformation * 0.10
        )
        
        return FlowMetrics(
            challenge_skill_balance=challenge_skill_balance,
            clear_goals=clear_goals,
            immediate_feedback=immediate_feedback,
            concentration=concentration,
            sense_of_control=sense_of_control,
            loss_of_self_consciousness=loss_of_self_consciousness,
            time_transformation=time_transformation,
            overall_flow_score=overall_flow_score
        )
    
    def _calculate_challenge_skill_balance(
        self,
        difficulty: float,
        skill: float
    ) -> float:
        """Calculate challenge-skill balance (optimal when closely matched)."""
        
        # Optimal flow occurs when challenge slightly exceeds skill
        optimal_ratio = 1.1  # Challenge should be 10% higher than skill
        actual_ratio = difficulty / (skill + 1e-8)
        
        # Distance from optimal ratio (closer = better flow)
        distance = abs(actual_ratio - optimal_ratio)
        
        # Convert to 0-1 scale (closer to optimal = higher score)
        balance_score = max(0.0, 1.0 - distance)
        
        return balance_score
    
    def _calculate_concentration(self, response_times: List[float]) -> float:
        """Calculate concentration from response time consistency."""
        
        if len(response_times) < 3:
            return 0.5
        
        # Concentration indicated by consistent response times
        mean_time = np.mean(response_times)
        std_time = np.std(response_times)
        
        # Coefficient of variation (lower = more consistent = better concentration)
        if mean_time > 0:
            cv = std_time / mean_time
            concentration = max(0.0, 1.0 - cv)
        else:
            concentration = 0.5
        
        return concentration
    
    def _calculate_sense_of_control(
        self,
        error_rates: List[float],
        emotional_state: EmotionalState
    ) -> float:
        """Calculate sense of control from performance and emotional state."""
        
        # Low error rate indicates control
        if error_rates:
            avg_error_rate = np.mean(error_rates)
            performance_control = 1.0 - avg_error_rate
        else:
            performance_control = 0.5
        
        # Emotional indicators of control
        dominance = emotional_state.dimensions.get(EmotionalDimension.DOMINANCE, 0.0)
        confidence = emotional_state.stability
        
        # Combine indicators
        sense_of_control = (
            performance_control * 0.4 +
            (dominance + 1.0) / 2.0 * 0.3 +  # Convert from [-1,1] to [0,1]
            confidence * 0.3
        )
        
        return sense_of_control
    
    def _calculate_immersion(
        self,
        response_times: List[float],
        engagement_metrics: Dict[str, float]
    ) -> float:
        """Calculate immersion/loss of self-consciousness."""
        
        # Fast, automatic responses indicate immersion
        if response_times:
            avg_response_time = np.mean(response_times)
            # Optimal response time for immersion (not too fast, not too slow)
            optimal_time = 3.0  # seconds
            time_score = max(0.0, 1.0 - abs(avg_response_time - optimal_time) / optimal_time)
        else:
            time_score = 0.5
        
        # High engagement indicates immersion
        engagement_score = engagement_metrics.get('overall_engagement', 0.5)
        
        # Combine indicators
        immersion = (time_score * 0.6 + engagement_score * 0.4)
        
        return immersion


class ZPDOptimizer:
    """Optimize difficulty within the Zone of Proximal Development."""
    
    def __init__(self):
        self.skill_assessments = {}
        self.zpd_history = {}
    
    def analyze_zpd(
        self,
        user_id: str,
        skill: LearningOutcome,
        performance_history: List[Dict[str, Any]],
        current_emotional_state: EmotionalState
    ) -> ZPDAnalysis:
        """Analyze user's Zone of Proximal Development for a skill."""
        
        # Store user and skill for tracking
        self.zpd_history[f"{user_id}_{skill.value}"] = {
            'timestamp': datetime.now(),
            'analysis_requested': True
        }
        
        # Assess current skill level from performance history
        current_skill_level = self._assess_current_skill_level(performance_history)
        
        # Determine independent capability (what user can do alone)
        independent_capability = self._determine_independent_capability(
            performance_history, assistance_threshold=0.1
        )
        
        # Determine assisted capability (what user can do with help)
        assisted_capability = self._determine_assisted_capability(
            performance_history, assistance_threshold=0.9
        )
        
        # Define ZPD range
        zpd_lower = max(independent_capability, current_skill_level - 0.1)
        zpd_upper = min(assisted_capability, current_skill_level + 0.3)
        zpd_range = (zpd_lower, zpd_upper)
        
        # Calculate optimal difficulty within ZPD
        optimal_difficulty = self._calculate_optimal_difficulty_in_zpd(
            zpd_range, current_emotional_state, performance_history
        )
        
        # Identify scaffolding needs
        scaffolding_needs = self._identify_scaffolding_needs(
            current_skill_level, assisted_capability, performance_history
        )
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            performance_history, optimal_difficulty
        )
        
        return ZPDAnalysis(
            current_skill_level=current_skill_level,
            independent_capability=independent_capability,
            assisted_capability=assisted_capability,
            zpd_range=zpd_range,
            optimal_difficulty=optimal_difficulty,
            scaffolding_needs=scaffolding_needs,
            confidence_interval=confidence_interval
        )
    
    def _assess_current_skill_level(
        self,
        performance_history: List[Dict[str, Any]]
    ) -> float:
        """Assess current skill level from performance history."""
        
        if not performance_history:
            return 0.3  # Default starting level
        
        # Weight recent performances more heavily
        weights = np.exp(-0.1 * np.arange(len(performance_history))[::-1])
        weights = weights / weights.sum()
        
        # Calculate weighted performance score
        performances = []
        difficulties = []
        
        for record in performance_history:
            success = record.get('success', False)
            difficulty = record.get('difficulty', 0.5)
            
            performances.append(1.0 if success else 0.0)
            difficulties.append(difficulty)
        
        if not performances:
            return 0.3
        
        # Weighted average of successful difficulties
        weighted_performance = np.average(performances, weights=weights)
        avg_difficulty = np.average(difficulties, weights=weights)
        
        # Skill level is estimated as the difficulty at which user performs at 50% success
        skill_level = avg_difficulty * weighted_performance
        
        return np.clip(skill_level, 0.0, 1.0)
    
    def _determine_independent_capability(
        self,
        performance_history: List[Dict[str, Any]],
        assistance_threshold: float
    ) -> float:
        """Determine what user can do independently."""
        
        independent_performances = [
            record for record in performance_history
            if record.get('assistance_level', 0.0) <= assistance_threshold
        ]
        
        if not independent_performances:
            return 0.2  # Conservative estimate
        
        # Find highest difficulty level achieved independently
        successful_independent = [
            record['difficulty'] for record in independent_performances
            if record.get('success', False)
        ]
        
        if successful_independent:
            return max(successful_independent)
        else:
            return 0.1
    
    def _determine_assisted_capability(
        self,
        performance_history: List[Dict[str, Any]],
        assistance_threshold: float
    ) -> float:
        """Determine what user can do with assistance."""
        
        assisted_performances = [
            record for record in performance_history
            if record.get('assistance_level', 0.0) >= assistance_threshold
        ]
        
        if not assisted_performances:
            return 0.8  # Optimistic estimate
        
        # Find highest difficulty level achieved with assistance
        successful_assisted = [
            record['difficulty'] for record in assisted_performances
            if record.get('success', False)
        ]
        
        if successful_assisted:
            return max(successful_assisted)
        else:
            return 0.6
    
    def _calculate_optimal_difficulty_in_zpd(
        self,
        zpd_range: Tuple[float, float],
        emotional_state: EmotionalState,
        performance_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate optimal difficulty within ZPD."""
        
        zpd_lower, zpd_upper = zpd_range
        
        if zpd_upper <= zpd_lower:
            return zpd_lower
        
        # Consider emotional state
        valence = emotional_state.dimensions.get(EmotionalDimension.VALENCE, 0.0)
        arousal = emotional_state.dimensions.get(EmotionalDimension.AROUSAL, 0.0)
        
        # Positive emotional state -> can handle more challenge
        emotional_adjustment = (valence + arousal) / 4.0  # Normalize to [-0.5, 0.5]
        
        # Recent performance trend
        if len(performance_history) >= 3:
            recent_success_rate = np.mean([
                1.0 if record.get('success', False) else 0.0
                for record in performance_history[-3:]
            ])
            performance_adjustment = (recent_success_rate - 0.5) * 0.2
        else:
            performance_adjustment = 0.0
        
        # Calculate optimal point in ZPD
        zpd_center = (zpd_lower + zpd_upper) / 2.0
        optimal_difficulty = zpd_center + emotional_adjustment + performance_adjustment
        
        # Ensure within ZPD bounds
        optimal_difficulty = np.clip(optimal_difficulty, zpd_lower, zpd_upper)
        
        return optimal_difficulty
    
    def _identify_scaffolding_needs(
        self,
        current_level: float,
        assisted_level: float,
        performance_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify what scaffolding support is needed."""
        
        scaffolding_needs = []
        
        # Gap between independent and assisted performance
        support_gap = assisted_level - current_level
        
        if support_gap > 0.3:
            scaffolding_needs.append("substantial_guidance")
        elif support_gap > 0.1:
            scaffolding_needs.append("moderate_guidance")
        else:
            scaffolding_needs.append("minimal_guidance")
        
        # Analyze error patterns
        recent_failures = [
            record for record in performance_history[-5:]
            if not record.get('success', True)
        ]
        
        if len(recent_failures) > 2:
            scaffolding_needs.append("error_correction_support")
        
        # Check for confidence issues
        if current_level < 0.3:
            scaffolding_needs.append("confidence_building")
        
        # Check for consistency issues
        if len(performance_history) >= 5:
            recent_difficulties = [r.get('difficulty', 0.5) for r in performance_history[-5:]]
            if np.std(recent_difficulties) > 0.2:
                scaffolding_needs.append("consistency_support")
        
        return scaffolding_needs
    
    def _calculate_confidence_interval(
        self,
        performance_history: List[Dict[str, Any]],
        optimal_difficulty: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the optimal difficulty."""
        
        if len(performance_history) < 3:
            # Wide confidence interval for insufficient data
            return (max(0.0, optimal_difficulty - 0.2), min(1.0, optimal_difficulty + 0.2))
        
        # Use performance variance to estimate confidence
        recent_performances = [
            1.0 if record.get('success', False) else 0.0
            for record in performance_history[-10:]
        ]
        
        performance_variance = np.var(recent_performances)
        
        # Higher variance = wider confidence interval
        confidence_width = 0.1 + performance_variance * 0.3
        
        lower_bound = max(0.0, optimal_difficulty - confidence_width)
        upper_bound = min(1.0, optimal_difficulty + confidence_width)
        
        return (lower_bound, upper_bound)


class DynamicDifficultyAdjuster:
    """Main class for dynamic difficulty adjustment."""
    
    def __init__(self):
        self.adjustment_history = defaultdict(list)
        self.performance_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_model_trained = False
        
    def calculate_difficulty_adjustment(
        self,
        current_difficulty: float,
        user_performance: Dict[str, float],
        emotional_state: EmotionalState,
        flow_metrics: FlowMetrics,
        zpd_analysis: ZPDAnalysis
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate optimal difficulty adjustment."""
        
        adjustments = {}
        
        # Performance-based adjustment
        performance_avg = np.mean(list(user_performance.values()))
        if performance_avg > 0.8:
            adjustments['performance'] = 0.1  # Increase difficulty
        elif performance_avg < 0.4:
            adjustments['performance'] = -0.15  # Decrease difficulty
        else:
            adjustments['performance'] = 0.0
        
        # Emotional state adjustment
        frustration_level = self._calculate_frustration_level(emotional_state)
        if frustration_level > 0.7:
            adjustments['emotional'] = -0.2  # Reduce difficulty when frustrated
        elif frustration_level < 0.3:
            adjustments['emotional'] = 0.05  # Can handle more challenge
        else:
            adjustments['emotional'] = 0.0
        
        # Flow state adjustment
        if flow_metrics.overall_flow_score > 0.7:
            adjustments['flow'] = 0.02  # Slight increase to maintain challenge
        elif flow_metrics.overall_flow_score < 0.3:
            adjustments['flow'] = -0.1  # Significant decrease to restore flow
        else:
            adjustments['flow'] = 0.0
        
        # ZPD-based adjustment
        if current_difficulty < zpd_analysis.zpd_range[0]:
            adjustments['zpd'] = zpd_analysis.zpd_range[0] - current_difficulty
        elif current_difficulty > zpd_analysis.zpd_range[1]:
            adjustments['zpd'] = zpd_analysis.zpd_range[1] - current_difficulty
        else:
            # Move toward optimal difficulty within ZPD
            adjustments['zpd'] = (zpd_analysis.optimal_difficulty - current_difficulty) * 0.3
        
        # Weighted combination of adjustments
        weights = {
            'performance': 0.3,
            'emotional': 0.25,
            'flow': 0.2,
            'zpd': 0.25
        }
        
        total_adjustment = sum(adjustments[key] * weights[key] for key in adjustments)
        
        # Apply adjustment bounds
        total_adjustment = np.clip(total_adjustment, -0.3, 0.3)
        
        # Calculate new difficulty
        new_difficulty = np.clip(current_difficulty + total_adjustment, 0.0, 1.0)
        
        # Metadata about the adjustment
        adjustment_metadata = {
            'adjustments': adjustments,
            'weights': weights,
            'total_adjustment': total_adjustment,
            'reasoning': self._generate_adjustment_reasoning(adjustments),
            'confidence': self._calculate_adjustment_confidence(adjustments, emotional_state)
        }
        
        return new_difficulty, adjustment_metadata
    
    def _calculate_frustration_level(self, emotional_state: EmotionalState) -> float:
        """Calculate frustration level from emotional state."""
        
        # Frustration indicators
        negative_valence = max(0.0, -emotional_state.dimensions.get(EmotionalDimension.VALENCE, 0.0))
        high_arousal = emotional_state.dimensions.get(EmotionalDimension.AROUSAL, 0.0)
        low_dominance = max(0.0, -emotional_state.dimensions.get(EmotionalDimension.DOMINANCE, 0.0))
        
        # Combine indicators
        frustration = (negative_valence * 0.4 + high_arousal * 0.3 + low_dominance * 0.3)
        
        # Normalize to 0-1 range
        frustration = np.clip(frustration, 0.0, 1.0)
        
        return frustration
    
    def _generate_adjustment_reasoning(self, adjustments: Dict[str, float]) -> str:
        """Generate human-readable reasoning for difficulty adjustment."""
        
        reasons = []
        
        if adjustments['performance'] > 0.05:
            reasons.append("User performing well, increasing challenge")
        elif adjustments['performance'] < -0.05:
            reasons.append("User struggling, reducing difficulty")
        
        if adjustments['emotional'] < -0.1:
            reasons.append("High frustration detected, backing off")
        elif adjustments['emotional'] > 0.03:
            reasons.append("Positive emotional state, can handle more challenge")
        
        if adjustments['flow'] < -0.05:
            reasons.append("Flow state disrupted, adjusting for optimal experience")
        elif adjustments['flow'] > 0.01:
            reasons.append("Good flow state, slight increase to maintain engagement")
        
        if abs(adjustments['zpd']) > 0.05:
            reasons.append("Adjusting to optimal Zone of Proximal Development")
        
        return "; ".join(reasons) if reasons else "No significant adjustment needed"
    
    def _calculate_adjustment_confidence(
        self,
        adjustments: Dict[str, float],
        emotional_state: EmotionalState
    ) -> float:
        """Calculate confidence in the difficulty adjustment."""
        
        # Higher confidence when:
        # 1. Adjustments are consistent across factors
        # 2. Emotional state is stable
        # 3. Sufficient data available
        
        adjustment_values = list(adjustments.values())
        adjustment_consistency = 1.0 - np.std(adjustment_values) if len(adjustment_values) > 1 else 1.0
        
        emotional_stability = emotional_state.stability
        
        # Simple confidence calculation
        confidence = (adjustment_consistency * 0.5 + emotional_stability * 0.5)
        
        return np.clip(confidence, 0.0, 1.0)


class AdvancedAdaptiveDifficultyEngine(IAdaptiveDifficultyEngine):
    """
    Advanced adaptive difficulty engine with sophisticated algorithms.
    
    Features:
    - Bayesian Knowledge Tracing for skill assessment
    - Flow Theory-based optimization
    - Zone of Proximal Development analysis
    - Emotional state-aware adjustments
    - Multi-objective optimization
    """
    
    def __init__(self):
        # Initialize components
        self.knowledge_tracer = BayesianKnowledgeTracer()
        self.flow_detector = FlowStateDetector()
        self.zpd_optimizer = ZPDOptimizer()
        self.difficulty_adjuster = DynamicDifficultyAdjuster()
        
        # User profiles
        self.user_profiles = {}
        
        # Performance tracking
        self.performance_tracker = defaultdict(list)
        
        logger.info("AdvancedAdaptiveDifficultyEngine initialized")
    
    async def calculate_optimal_difficulty(
        self,
        user_performance: Dict[str, float],
        emotional_state: EmotionalState,
        learning_objectives: List[LearningOutcome],
        context: ConversationContext
    ) -> float:
        """Calculate optimal difficulty level for current context."""
        
        try:
            user_id = str(context.user_id)
            
            # Initialize user profile if needed
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._create_default_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Update profile usage for tracking
            profile.historical_performance.append({
                'timestamp': datetime.now(),
                'interaction_type': 'difficulty_calculation',
                'context_phase': context.current_phase.value
            })
            
            # Analyze flow state
            response_times = context.user_preferences.get('response_times', [3.0])
            error_rates = [1.0 - performance for performance in user_performance.values()]
            
            flow_metrics = self.flow_detector.analyze_flow_state(
                response_times=response_times,
                error_rates=error_rates,
                engagement_metrics=context.engagement_metrics,
                emotional_state=emotional_state,
                difficulty_level=context.difficulty_level,
                skill_level=np.mean(list(user_performance.values()))
            )
            
            # Analyze ZPD for primary learning objective
            primary_objective = learning_objectives[0] if learning_objectives else LearningOutcome.LANGUAGE_SKILLS
            performance_history = self.performance_tracker[f"{user_id}_{primary_objective.value}"]
            
            zpd_analysis = self.zpd_optimizer.analyze_zpd(
                user_id=user_id,
                skill=primary_objective,
                performance_history=performance_history,
                current_emotional_state=emotional_state
            )
            
            # Calculate difficulty adjustment
            new_difficulty, adjustment_metadata = self.difficulty_adjuster.calculate_difficulty_adjustment(
                current_difficulty=context.difficulty_level,
                user_performance=user_performance,
                emotional_state=emotional_state,
                flow_metrics=flow_metrics,
                zpd_analysis=zpd_analysis
            )
            
            # Update user profile
            self._update_user_profile(user_id, new_difficulty, flow_metrics, adjustment_metadata)
            
            return new_difficulty
            
        except Exception as e:
            logger.error(f"Error in calculate_optimal_difficulty: {str(e)}")
            # Fallback to slight adjustment of current difficulty
            return np.clip(context.difficulty_level * 1.05, 0.0, 1.0)
    
    async def adjust_complexity_dynamically(
        self,
        current_difficulty: float,
        user_feedback: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """Dynamically adjust complexity based on real-time feedback."""
        
        try:
            adjustments = {}
            
            # Process user feedback
            if 'too_easy' in user_feedback and user_feedback['too_easy']:
                adjustments['user_feedback'] = 0.2
            elif 'too_hard' in user_feedback and user_feedback['too_hard']:
                adjustments['user_feedback'] = -0.2
            else:
                adjustments['user_feedback'] = 0.0
            
            # Process performance metrics
            avg_performance = np.mean(list(performance_metrics.values()))
            if avg_performance > 0.85:
                adjustments['performance'] = 0.1
            elif avg_performance < 0.4:
                adjustments['performance'] = -0.15
            else:
                adjustments['performance'] = 0.0
            
            # Combine adjustments
            total_adjustment = adjustments['user_feedback'] * 0.6 + adjustments['performance'] * 0.4
            
            # Apply bounds
            total_adjustment = np.clip(total_adjustment, -0.3, 0.3)
            
            new_difficulty = np.clip(current_difficulty + total_adjustment, 0.0, 1.0)
            
            metadata = {
                'adjustments': adjustments,
                'total_adjustment': total_adjustment,
                'confidence': 0.8,  # High confidence for direct user feedback
                'reasoning': self._generate_dynamic_adjustment_reasoning(adjustments)
            }
            
            return new_difficulty, metadata
            
        except Exception as e:
            logger.error(f"Error in adjust_complexity_dynamically: {str(e)}")
            return current_difficulty, {'error': str(e)}
    
    async def generate_scaffolding_strategy(
        self,
        difficulty_gap: float,
        user_strengths: List[str],
        learning_style: str
    ) -> Dict[str, Any]:
        """Generate scaffolding strategy to bridge difficulty gaps."""
        
        try:
            strategy = {
                'scaffolding_type': self._determine_scaffolding_type(difficulty_gap),
                'support_level': self._calculate_support_level(difficulty_gap),
                'techniques': [],
                'duration': self._estimate_scaffolding_duration(difficulty_gap),
                'success_criteria': [],
                'fade_out_plan': []
            }
            
            # Generate techniques based on strengths and learning style
            if 'language' in user_strengths:
                strategy['techniques'].append('verbal_scaffolding')
                strategy['techniques'].append('linguistic_cues')
            
            if 'visual' in user_strengths:
                strategy['techniques'].append('visual_aids')
                strategy['techniques'].append('graphic_organizers')
            
            # Learning style adaptations
            if learning_style == 'visual':
                strategy['techniques'].extend(['diagrams', 'mind_maps', 'color_coding'])
            elif learning_style == 'auditory':
                strategy['techniques'].extend(['verbal_explanations', 'discussion', 'verbal_rehearsal'])
            elif learning_style == 'kinesthetic':
                strategy['techniques'].extend(['hands_on_practice', 'role_playing', 'simulation'])
            
            # Success criteria
            strategy['success_criteria'] = [
                'independent_task_completion',
                'reduced_error_rate',
                'increased_confidence',
                'faster_response_times'
            ]
            
            # Fade-out plan
            strategy['fade_out_plan'] = self._create_fade_out_plan(difficulty_gap)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error in generate_scaffolding_strategy: {str(e)}")
            return {
                'scaffolding_type': 'moderate',
                'support_level': 0.5,
                'techniques': ['encouragement', 'hints'],
                'error': str(e)
            }
    
    async def predict_frustration_threshold(
        self,
        user_profile: Dict[str, Any],
        current_emotional_state: EmotionalState,
        session_history: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Predict user's frustration threshold and confidence interval."""
        
        try:
            # Base frustration threshold from user profile
            base_threshold = user_profile.get('frustration_threshold', 0.7)
            
            # Current emotional state influence
            current_frustration = self.difficulty_adjuster._calculate_frustration_level(current_emotional_state)
            
            # Historical pattern analysis
            if session_history:
                historical_frustrations = [
                    session.get('peak_frustration', 0.5) for session in session_history[-5:]
                ]
                std_historical = np.std(historical_frustrations)
            else:
                std_historical = 0.2
            
            # Adjust threshold based on current state
            if current_frustration > 0.5:
                # Already elevated, threshold is lower
                adjustment = -0.1 * (current_frustration - 0.5)
            else:
                # Good state, can tolerate more
                adjustment = 0.05 * (0.5 - current_frustration)
            
            predicted_threshold = base_threshold + adjustment
            predicted_threshold = np.clip(predicted_threshold, 0.1, 0.9)
            
            # Confidence interval based on historical variance
            confidence_width = min(0.2, std_historical * 2)
            confidence_interval = (
                max(0.1, predicted_threshold - confidence_width),
                min(0.9, predicted_threshold + confidence_width)
            )
            
            return predicted_threshold, confidence_interval[1] - confidence_interval[0]
            
        except Exception as e:
            logger.error(f"Error in predict_frustration_threshold: {str(e)}")
            return 0.6, 0.2  # Default values
    
    def _create_default_profile(self, user_id: str) -> DifficultyProfile:
        """Create a default difficulty profile for a new user."""
        
        return DifficultyProfile(
            user_id=user_id,
            preferred_difficulty=0.5,
            tolerance_range=(0.3, 0.8),
            adaptation_speed=0.3,
            challenge_preference="gradual",
            frustration_threshold=0.7,
            flow_indicators={},
            mastery_confidence={outcome: 0.3 for outcome in LearningOutcome},
            historical_performance=[]
        )
    
    def _update_user_profile(
        self,
        user_id: str,
        new_difficulty: float,
        flow_metrics: FlowMetrics,
        adjustment_metadata: Dict[str, Any]
    ):
        """Update user profile with new information."""
        
        profile = self.user_profiles[user_id]
        
        # Update preferred difficulty (gradual adaptation)
        alpha = 0.1  # Learning rate
        profile.preferred_difficulty = (
            (1 - alpha) * profile.preferred_difficulty + 
            alpha * new_difficulty
        )
        
        # Update flow indicators
        profile.flow_indicators['last_flow_score'] = flow_metrics.overall_flow_score
        profile.flow_indicators['challenge_skill_balance'] = flow_metrics.challenge_skill_balance
        
        # Add to historical performance
        profile.historical_performance.append({
            'timestamp': datetime.now(),
            'difficulty': new_difficulty,
            'flow_score': flow_metrics.overall_flow_score,
            'adjustment_reasoning': adjustment_metadata.get('reasoning', '')
        })
        
        # Keep only recent history
        if len(profile.historical_performance) > 50:
            profile.historical_performance = profile.historical_performance[-50:]
    
    def _determine_scaffolding_type(self, difficulty_gap: float) -> str:
        """Determine appropriate scaffolding type based on difficulty gap."""
        
        if difficulty_gap > 0.5:
            return "intensive"
        elif difficulty_gap > 0.3:
            return "moderate"
        elif difficulty_gap > 0.1:
            return "light"
        else:
            return "minimal"
    
    def _calculate_support_level(self, difficulty_gap: float) -> float:
        """Calculate appropriate support level (0.0 to 1.0)."""
        
        # Linear mapping from gap to support level
        support_level = min(1.0, difficulty_gap * 2.0)
        return support_level
    
    def _estimate_scaffolding_duration(self, difficulty_gap: float) -> timedelta:
        """Estimate how long scaffolding should be maintained."""
        
        # Larger gaps require longer scaffolding
        base_minutes = 10
        additional_minutes = difficulty_gap * 30
        
        total_minutes = base_minutes + additional_minutes
        return timedelta(minutes=total_minutes)
    
    def _create_fade_out_plan(self, difficulty_gap: float) -> List[Dict[str, Any]]:
        """Create a plan for gradually removing scaffolding."""
        
        num_phases = max(3, int(difficulty_gap * 10))
        
        fade_out_plan = []
        for phase in range(num_phases):
            support_reduction = (phase + 1) / num_phases
            
            phase_plan = {
                'phase': phase + 1,
                'support_level': 1.0 - support_reduction,
                'target_independence': support_reduction,
                'success_criteria': f"Achieve {support_reduction * 100:.0f}% independence",
                'duration': timedelta(minutes=5 + phase * 2)
            }
            
            fade_out_plan.append(phase_plan)
        
        return fade_out_plan
    
    def _generate_dynamic_adjustment_reasoning(self, adjustments: Dict[str, float]) -> str:
        """Generate reasoning for dynamic adjustments."""
        
        reasons = []
        
        if adjustments.get('user_feedback', 0) > 0.1:
            reasons.append("User reported content too easy")
        elif adjustments.get('user_feedback', 0) < -0.1:
            reasons.append("User reported content too difficult")
        
        if adjustments.get('performance', 0) > 0.05:
            reasons.append("Performance metrics indicate readiness for more challenge")
        elif adjustments.get('performance', 0) < -0.05:
            reasons.append("Performance metrics suggest need for easier content")
        
        return "; ".join(reasons) if reasons else "Minor optimization based on patterns"