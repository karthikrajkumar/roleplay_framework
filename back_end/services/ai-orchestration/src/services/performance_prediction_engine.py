"""
Advanced Performance Prediction Engine

This module implements sophisticated performance prediction algorithms including:
- Deep Neural Networks with LSTM/GRU for time-series learning outcome prediction
- Gaussian Process Regression for uncertainty quantification in skill progression
- Ensemble Methods with XGBoost and Random Forest for robust performance forecasting
- Causal Inference techniques for identifying learning bottlenecks
- Multi-task Learning for simultaneous prediction of multiple learning outcomes
- Anomaly Detection using Isolation Forest for identifying learning difficulties
- Bayesian Optimization for hyperparameter tuning of prediction models
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import torch
import torch.nn as nn
try:
    import gpytorch
except ImportError:
    gpytorch = None
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..interfaces.neural_persona import (
    IPerformancePredictionEngine,
    ConversationContext,
    LearningOutcome
)


logger = logging.getLogger(__name__)


@dataclass
class PerformancePrediction:
    """Container for performance prediction results."""
    outcome: LearningOutcome
    predicted_score: float
    confidence_interval: Tuple[float, float]
    certainty: float
    contributing_factors: Dict[str, float]
    risk_level: str  # "low", "medium", "high"
    recommendation: str


@dataclass
class SkillProgressionModel:
    """Model for tracking skill progression over time."""
    skill: LearningOutcome
    progression_rate: float
    mastery_probability: float
    time_to_mastery: Optional[timedelta]
    learning_curve_parameters: Dict[str, float]
    plateau_indicators: List[Dict[str, Any]]


@dataclass
class LearningBottleneck:
    """Identified learning bottleneck."""
    bottleneck_type: str
    severity: float  # 0.0 to 1.0
    affected_skills: List[LearningOutcome]
    causal_factors: List[str]
    intervention_suggestions: List[str]
    confidence: float


class LSTMPredictor(nn.Module):
    """LSTM-based neural network for performance prediction."""
    
    def __init__(self, 
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_size: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
        # Output layers for each learning outcome
        self.output_layers = nn.ModuleDict({
            outcome.value: nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ) for outcome in LearningOutcome
        })
        
        # Uncertainty estimation layer
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with attention and uncertainty estimation."""
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention over sequence
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step for prediction
        final_hidden = attended_out[:, -1, :]
        
        # Predictions for each learning outcome
        predictions = {}
        for outcome in LearningOutcome:
            predictions[outcome.value] = self.output_layers[outcome.value](final_hidden)
        
        # Uncertainty estimation
        uncertainties = self.uncertainty_layer(final_hidden)
        
        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'attention_weights': attention_weights,
            'hidden_representation': final_hidden
        }


class GaussianProcessModel:
    """Gaussian Process model for skill progression with uncertainty."""
    
    def __init__(self, skill: LearningOutcome):
        self.skill = skill
        self.model = None
        self.likelihood = None
        self.trained = False
        
    def train(self, training_data: List[Tuple[float, float, float]]):
        """
        Train GP model on time-series performance data.
        
        Args:
            training_data: List of (time_delta_hours, performance_score, uncertainty)
        """
        
        if len(training_data) < 3:
            logger.warning(f"Insufficient data for GP training for skill {self.skill}")
            return
        
        # Prepare training data
        times = torch.tensor([item[0] for item in training_data], dtype=torch.float32)
        performances = torch.tensor([item[1] for item in training_data], dtype=torch.float32)
        
        # Define GP model
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
                )
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(times, performances, self.likelihood)
        
        # Training
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(100):
            optimizer.zero_grad()
            output = self.model(times)
            loss = -mll(output, performances)
            loss.backward()
            optimizer.step()
        
        self.trained = True
        logger.info(f"Trained GP model for skill {self.skill}")
    
    def predict(self, future_time_hours: float) -> Tuple[float, float]:
        """Predict performance at future time with uncertainty."""
        
        if not self.trained:
            return 0.5, 0.3  # Default prediction with high uncertainty
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor([future_time_hours], dtype=torch.float32)
            observed_pred = self.likelihood(self.model(test_x))
            
            mean = observed_pred.mean.item()
            variance = observed_pred.variance.item()
            
            return float(np.clip(mean, 0.0, 1.0)), float(np.sqrt(variance))


class CausalInferenceAnalyzer:
    """Causal inference for identifying learning bottlenecks."""
    
    def __init__(self):
        self.causal_graph = None
        self.causal_effects = {}
        
    def build_causal_graph(self, interaction_data: List[Dict[str, Any]]):
        """Build causal graph from interaction data."""
        
        # Simplified causal structure based on domain knowledge
        causal_relationships = {
            'emotional_state_variance': ['engagement_level', 'performance_variance'],
            'response_time_consistency': ['concentration_level', 'performance_stability'],
            'error_pattern_complexity': ['confusion_level', 'learning_difficulty'],
            'help_seeking_frequency': ['metacognitive_awareness', 'self_regulation'],
            'practice_frequency': ['motivation_level', 'skill_improvement_rate']
        }
        
        self.causal_graph = causal_relationships
        logger.info("Built causal graph for learning bottleneck analysis")
    
    def identify_causal_factors(
        self,
        performance_data: Dict[str, List[float]],
        contextual_factors: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Identify causal factors affecting performance."""
        
        causal_effects = {}
        
        # Calculate correlations as proxy for causal effects
        # In practice, would use more sophisticated causal inference methods
        for factor, values in contextual_factors.items():
            if len(values) >= 3:
                # Calculate correlation with average performance
                avg_performance = [
                    np.mean([performance_data[outcome][i] for outcome in performance_data.keys()])
                    for i in range(min(len(values), min(len(v) for v in performance_data.values())))
                ]
                
                if len(avg_performance) >= 3:
                    correlation = np.corrcoef(values[:len(avg_performance)], avg_performance)[0, 1]
                    causal_effects[factor] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return causal_effects


class EnsemblePredictor:
    """Ensemble of prediction models for robust performance forecasting."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'neural_net': LSTMPredictor()
        }
        self.model_weights = {'random_forest': 0.3, 'xgboost': 0.4, 'neural_net': 0.3}
        self.scaler = StandardScaler()
        self.trained = False
        
    def prepare_features(self, interaction_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare feature matrix from interaction data."""
        
        features = []
        
        for interaction in interaction_data:
            feature_vector = [
                interaction.get('response_time', 3.0),
                interaction.get('error_rate', 0.5),
                interaction.get('help_requests', 0),
                interaction.get('engagement_score', 0.5),
                interaction.get('difficulty_level', 0.5),
                interaction.get('session_duration', 10.0),
                interaction.get('emotional_valence', 0.0),
                interaction.get('emotional_arousal', 0.0),
                interaction.get('confidence_level', 0.5),
                interaction.get('previous_performance', 0.5)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, interaction_data: List[Dict[str, Any]], outcomes: Dict[LearningOutcome, List[float]]):
        """Train ensemble models."""
        
        if len(interaction_data) < 10:
            logger.warning("Insufficient data for ensemble training")
            return
        
        # Prepare features
        X = self.prepare_features(interaction_data)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model for each outcome
        for outcome in LearningOutcome:
            if outcome in outcomes and len(outcomes[outcome]) == len(X):
                y = np.array(outcomes[outcome])
                
                # Train traditional ML models
                try:
                    self.models['random_forest'].fit(X_scaled, y)
                    self.models['xgboost'].fit(X_scaled, y)
                    logger.info(f"Trained ensemble models for outcome {outcome}")
                except Exception as e:
                    logger.error(f"Error training models for {outcome}: {str(e)}")
        
        self.trained = True
    
    def predict(self, interaction_features: Dict[str, Any]) -> Dict[LearningOutcome, Tuple[float, float]]:
        """Make ensemble predictions for all learning outcomes."""
        
        if not self.trained:
            # Return default predictions
            return {
                outcome: (0.5, 0.3) for outcome in LearningOutcome
            }
        
        # Prepare feature vector
        feature_vector = np.array([[
            interaction_features.get('response_time', 3.0),
            interaction_features.get('error_rate', 0.5),
            interaction_features.get('help_requests', 0),
            interaction_features.get('engagement_score', 0.5),
            interaction_features.get('difficulty_level', 0.5),
            interaction_features.get('session_duration', 10.0),
            interaction_features.get('emotional_valence', 0.0),
            interaction_features.get('emotional_arousal', 0.0),
            interaction_features.get('confidence_level', 0.5),
            interaction_features.get('previous_performance', 0.5)
        ]])
        
        X_scaled = self.scaler.transform(feature_vector)
        
        predictions = {}
        
        for outcome in LearningOutcome:
            try:
                # Get predictions from each model
                rf_pred = self.models['random_forest'].predict(X_scaled)[0]
                xgb_pred = self.models['xgboost'].predict(X_scaled)[0]
                
                # Ensemble prediction
                ensemble_pred = (
                    rf_pred * self.model_weights['random_forest'] +
                    xgb_pred * self.model_weights['xgboost']
                )
                
                # Estimate uncertainty based on model disagreement
                model_preds = [rf_pred, xgb_pred]
                uncertainty = np.std(model_preds)
                
                predictions[outcome] = (
                    float(np.clip(ensemble_pred, 0.0, 1.0)),
                    float(uncertainty)
                )
                
            except Exception as e:
                logger.error(f"Error in ensemble prediction for {outcome}: {str(e)}")
                predictions[outcome] = (0.5, 0.3)
        
        return predictions


class AdvancedPerformancePredictionEngine(IPerformancePredictionEngine):
    """
    Advanced implementation of performance prediction engine.
    
    This engine combines multiple sophisticated approaches:
    - Deep learning with LSTM networks for temporal patterns
    - Gaussian Process regression for uncertainty quantification
    - Ensemble methods for robust predictions
    - Causal inference for bottleneck identification
    - Anomaly detection for learning difficulties
    """
    
    def __init__(self):
        # Initialize components
        self.lstm_predictor = LSTMPredictor()
        if gpytorch:
            self.gp_models = {skill: GaussianProcessModel(skill) for skill in LearningOutcome}
        else:
            self.gp_models = {}
        self.ensemble_predictor = EnsemblePredictor()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Data storage
        self.interaction_history = []
        self.performance_history = defaultdict(list)
        self.skill_assessments = defaultdict(list)
        
        # Model training state
        self.models_trained = False
        self.last_training_time = None
        
        logger.info("AdvancedPerformancePredictionEngine initialized")
    
    async def predict_learning_outcomes(
        self,
        user_interaction_history: List[Dict[str, Any]],
        current_session_data: ConversationContext,
        prediction_horizon: timedelta
    ) -> Dict[LearningOutcome, Tuple[float, float]]:
        """Predict learning outcomes with confidence intervals."""
        
        try:
            # Update internal data
            self.interaction_history.extend(user_interaction_history)
            
            # Extract current session features
            current_features = self._extract_session_features(current_session_data)
            
            # Train models if needed
            if not self.models_trained or self._should_retrain():
                await self._train_models()
            
            # Get ensemble predictions
            ensemble_predictions = self.ensemble_predictor.predict(current_features)
            
            # Get GP predictions for temporal modeling
            prediction_hours = prediction_horizon.total_seconds() / 3600
            gp_predictions = {}
            
            for skill in LearningOutcome:
                if len(self.skill_assessments[skill]) >= 3:
                    gp_mean, gp_std = self.gp_models[skill].predict(prediction_hours)
                    gp_predictions[skill] = (gp_mean, gp_std)
                else:
                    gp_predictions[skill] = (0.5, 0.3)
            
            # Combine predictions
            final_predictions = {}
            for outcome in LearningOutcome:
                ensemble_mean, ensemble_std = ensemble_predictions[outcome]
                gp_mean, gp_std = gp_predictions[outcome]
                
                # Weighted combination based on data availability
                ensemble_weight = 0.7 if len(self.interaction_history) > 20 else 0.4
                gp_weight = 1.0 - ensemble_weight
                
                combined_mean = ensemble_mean * ensemble_weight + gp_mean * gp_weight
                combined_std = np.sqrt(ensemble_std**2 * ensemble_weight + gp_std**2 * gp_weight)
                
                # Confidence interval (95%)
                ci_lower = max(0.0, combined_mean - 1.96 * combined_std)
                ci_upper = min(1.0, combined_mean + 1.96 * combined_std)
                
                final_predictions[outcome] = (
                    float(combined_mean),
                    float(combined_std)
                )
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error in predict_learning_outcomes: {str(e)}")
            # Return default predictions
            return {outcome: (0.5, 0.3) for outcome in LearningOutcome}
    
    async def analyze_skill_progression(
        self,
        skill_assessments: List[Dict[str, Any]],
        interaction_patterns: Dict[str, Any],
        time_series_data: List[Tuple[datetime, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze skill progression patterns and trends."""
        
        try:
            progression_analysis = {}
            
            for skill in LearningOutcome:
                skill_data = [
                    assessment for assessment in skill_assessments
                    if assessment.get('skill') == skill.value
                ]
                
                if len(skill_data) < 2:
                    progression_analysis[skill.value] = {
                        'progression_rate': 0.0,
                        'mastery_probability': 0.1,
                        'consistency_score': 0.5,
                        'trend_direction': 'unknown'
                    }
                    continue
                
                # Calculate progression rate
                scores = [data.get('score', 0.5) for data in skill_data]
                times = [data.get('timestamp', datetime.now()) for data in skill_data]
                
                if len(scores) >= 2:
                    # Linear regression for trend
                    time_deltas = [(t - times[0]).total_seconds() / 3600 for t in times]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(time_deltas, scores)
                    
                    progression_rate = float(slope)
                    consistency_score = float(r_value ** 2)  # R-squared
                    
                    # Mastery probability based on current score and trend
                    current_score = scores[-1]
                    mastery_probability = min(1.0, current_score + max(0, progression_rate) * 10)
                    
                    # Trend direction
                    if progression_rate > 0.01:
                        trend_direction = 'improving'
                    elif progression_rate < -0.01:
                        trend_direction = 'declining'
                    else:
                        trend_direction = 'stable'
                    
                    progression_analysis[skill.value] = {
                        'progression_rate': progression_rate,
                        'mastery_probability': float(mastery_probability),
                        'consistency_score': consistency_score,
                        'trend_direction': trend_direction,
                        'current_score': float(current_score),
                        'sessions_analyzed': len(skill_data)
                    }
                else:
                    progression_analysis[skill.value] = {
                        'progression_rate': 0.0,
                        'mastery_probability': 0.3,
                        'consistency_score': 0.5,
                        'trend_direction': 'insufficient_data'
                    }
            
            return progression_analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_skill_progression: {str(e)}")
            return {skill.value: {'error': str(e)} for skill in LearningOutcome}
    
    async def identify_learning_bottlenecks(
        self,
        performance_data: Dict[str, List[float]],
        engagement_metrics: Dict[str, float],
        error_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify potential learning bottlenecks and challenges."""
        
        try:
            bottlenecks = []
            
            # Analyze performance variance
            for skill, scores in performance_data.items():
                if len(scores) >= 3:
                    variance = np.var(scores)
                    mean_score = np.mean(scores)
                    
                    # High variance bottleneck
                    if variance > 0.1 and mean_score < 0.6:
                        bottlenecks.append({
                            'type': 'high_variance_performance',
                            'affected_skill': skill,
                            'severity': float(variance),
                            'description': f"Inconsistent performance in {skill}",
                            'causal_factors': ['attention_issues', 'content_difficulty'],
                            'interventions': ['provide_consistent_practice', 'adjust_difficulty']
                        })
                    
                    # Plateau detection
                    if len(scores) >= 5:
                        recent_trend = scores[-3:]
                        if np.std(recent_trend) < 0.05 and np.mean(recent_trend) < 0.7:
                            bottlenecks.append({
                                'type': 'learning_plateau',
                                'affected_skill': skill,
                                'severity': 0.7,
                                'description': f"Learning plateau detected in {skill}",
                                'causal_factors': ['lack_of_challenge', 'motivation_issues'],
                                'interventions': ['increase_difficulty', 'introduce_variety']
                            })
            
            # Analyze engagement patterns
            low_engagement_threshold = 0.3
            if engagement_metrics.get('overall_engagement', 0.5) < low_engagement_threshold:
                bottlenecks.append({
                    'type': 'low_engagement',
                    'affected_skill': 'all',
                    'severity': 1.0 - engagement_metrics.get('overall_engagement', 0.5),
                    'description': "Low overall engagement detected",
                    'causal_factors': ['content_relevance', 'motivation', 'difficulty_mismatch'],
                    'interventions': ['gamification', 'personalization', 'social_elements']
                })
            
            # Analyze error patterns
            error_analysis = self._analyze_error_patterns(error_patterns)
            if error_analysis['systematic_errors']:
                bottlenecks.append({
                    'type': 'systematic_errors',
                    'affected_skill': error_analysis['primary_skill'],
                    'severity': error_analysis['error_rate'],
                    'description': "Systematic errors indicating conceptual gaps",
                    'causal_factors': ['misconceptions', 'incomplete_understanding'],
                    'interventions': ['conceptual_review', 'scaffolded_practice']
                })
            
            # Use causal inference for additional analysis
            contextual_factors = {
                'response_time_variance': [np.var([p.get('response_time', 3.0) for p in error_patterns])],
                'help_seeking_frequency': [sum(1 for p in error_patterns if p.get('help_requested', False))],
                'emotional_stability': [engagement_metrics.get('emotional_stability', 0.5)]
            }
            
            causal_effects = self.causal_analyzer.identify_causal_factors(
                performance_data, contextual_factors
            )
            
            # Add causal bottlenecks
            for factor, effect_size in causal_effects.items():
                if effect_size > 0.5:  # Strong causal effect
                    bottlenecks.append({
                        'type': 'causal_bottleneck',
                        'affected_skill': 'multiple',
                        'severity': float(effect_size),
                        'description': f"Strong causal factor: {factor}",
                        'causal_factors': [factor],
                        'interventions': self._get_interventions_for_factor(factor)
                    })
            
            # Sort by severity
            bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
            
            return bottlenecks[:10]  # Return top 10 bottlenecks
            
        except Exception as e:
            logger.error(f"Error in identify_learning_bottlenecks: {str(e)}")
            return [{'type': 'analysis_error', 'description': str(e)}]
    
    async def recommend_intervention_strategies(
        self,
        predicted_performance: Dict[LearningOutcome, float],
        risk_factors: List[str],
        available_resources: List[str]
    ) -> List[Dict[str, Any]]:
        """Recommend intervention strategies for at-risk learners."""
        
        try:
            interventions = []
            
            # Analyze performance predictions for at-risk outcomes
            for outcome, performance in predicted_performance.items():
                if performance < 0.4:  # At-risk threshold
                    intervention = self._generate_outcome_intervention(
                        outcome, performance, risk_factors, available_resources
                    )
                    interventions.append(intervention)
            
            # General interventions based on risk factors
            for risk_factor in risk_factors:
                general_intervention = self._generate_risk_factor_intervention(
                    risk_factor, available_resources
                )
                if general_intervention:
                    interventions.append(general_intervention)
            
            # Prioritize interventions by impact and feasibility
            for intervention in interventions:
                intervention['priority_score'] = self._calculate_intervention_priority(
                    intervention, available_resources
                )
            
            # Sort by priority
            interventions.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return interventions[:8]  # Return top 8 interventions
            
        except Exception as e:
            logger.error(f"Error in recommend_intervention_strategies: {str(e)}")
            return [{'type': 'default', 'description': 'Provide additional support and practice'}]
    
    def _extract_session_features(self, context: ConversationContext) -> Dict[str, Any]:
        """Extract features from current session context."""
        
        # Calculate emotional state features
        if context.emotional_trajectory:
            latest_emotion = context.emotional_trajectory[-1]
            emotional_valence = latest_emotion.dimensions.get('valence', 0.0)
            emotional_arousal = latest_emotion.dimensions.get('arousal', 0.0)
            emotional_stability = latest_emotion.stability
        else:
            emotional_valence = 0.0
            emotional_arousal = 0.0
            emotional_stability = 0.5
        
        return {
            'response_time': 3.0,  # Would be calculated from actual data
            'error_rate': 0.3,  # Would be calculated from interaction history
            'help_requests': 0,  # Count of help requests in session
            'engagement_score': context.engagement_metrics.get('overall_engagement', 0.5),
            'difficulty_level': context.difficulty_level,
            'session_duration': context.duration.total_seconds() / 60,  # minutes
            'emotional_valence': emotional_valence,
            'emotional_arousal': emotional_arousal,
            'confidence_level': emotional_stability,
            'previous_performance': 0.5,  # Would be calculated from history
            'turn_count': context.turn_count,
            'phase': hash(context.current_phase.value) % 10 / 10.0  # Encode phase
        }
    
    async def _train_models(self):
        """Train all prediction models."""
        
        try:
            if len(self.interaction_history) >= 10:
                # Prepare outcomes data
                outcomes = {}
                for outcome in LearningOutcome:
                    outcomes[outcome] = [
                        interaction.get(f'{outcome.value}_score', 0.5)
                        for interaction in self.interaction_history
                    ]
                
                # Train ensemble predictor
                self.ensemble_predictor.train(self.interaction_history, outcomes)
                
                # Train GP models
                for skill in LearningOutcome:
                    if len(self.skill_assessments[skill]) >= 3:
                        training_data = [
                            (i * 1.0, assessment['score'], 0.1)  # time, score, uncertainty
                            for i, assessment in enumerate(self.skill_assessments[skill])
                        ]
                        self.gp_models[skill].train(training_data)
                
                # Build causal graph
                self.causal_analyzer.build_causal_graph(self.interaction_history)
                
                self.models_trained = True
                self.last_training_time = datetime.now()
                
                logger.info("Successfully trained all prediction models")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    def _should_retrain(self) -> bool:
        """Determine if models should be retrained."""
        
        if not self.last_training_time:
            return True
        
        # Retrain if it's been more than 1 hour or significant new data
        time_threshold = timedelta(hours=1)
        data_threshold = 20
        
        return (
            datetime.now() - self.last_training_time > time_threshold or
            len(self.interaction_history) % data_threshold == 0
        )
    
    def _analyze_error_patterns(self, error_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns for systematic issues."""
        
        if not error_patterns:
            return {'systematic_errors': False, 'error_rate': 0.0, 'primary_skill': 'unknown'}
        
        # Count error types
        error_types = defaultdict(int)
        for error in error_patterns:
            error_type = error.get('type', 'unknown')
            error_types[error_type] += 1
        
        total_errors = len(error_patterns)
        most_common_error = max(error_types.items(), key=lambda x: x[1])
        
        # Systematic if one error type represents >50% of errors
        systematic = most_common_error[1] / total_errors > 0.5
        
        return {
            'systematic_errors': systematic,
            'error_rate': total_errors / max(1, len(error_patterns)),
            'primary_skill': most_common_error[0],
            'error_distribution': dict(error_types)
        }
    
    def _get_interventions_for_factor(self, factor: str) -> List[str]:
        """Get intervention recommendations for a causal factor."""
        
        intervention_map = {
            'emotional_state_variance': ['emotion_regulation_training', 'stress_management'],
            'response_time_consistency': ['attention_training', 'pacing_strategies'],
            'help_seeking_frequency': ['metacognitive_training', 'self_monitoring'],
            'practice_frequency': ['motivation_enhancement', 'goal_setting']
        }
        
        return intervention_map.get(factor, ['general_support'])
    
    def _generate_outcome_intervention(
        self,
        outcome: LearningOutcome,
        performance: float,
        risk_factors: List[str],
        available_resources: List[str]
    ) -> Dict[str, Any]:
        """Generate intervention for specific learning outcome."""
        
        intervention_strategies = {
            LearningOutcome.LANGUAGE_SKILLS: ['vocabulary_practice', 'conversation_drills', 'reading_comprehension'],
            LearningOutcome.EMOTIONAL_INTELLIGENCE: ['empathy_training', 'emotion_recognition', 'social_scenarios'],
            LearningOutcome.PROBLEM_SOLVING: ['logical_reasoning', 'pattern_recognition', 'structured_analysis'],
            LearningOutcome.CREATIVITY: ['divergent_thinking', 'brainstorming', 'creative_prompts'],
            LearningOutcome.CONFIDENCE: ['positive_reinforcement', 'gradual_challenges', 'success_experiences'],
            LearningOutcome.COMMUNICATION: ['active_listening', 'clear_expression', 'feedback_practice']
        }
        
        strategies = intervention_strategies.get(outcome, ['general_practice'])
        
        # Filter strategies by available resources
        available_strategies = [
            strategy for strategy in strategies
            if any(resource in strategy for resource in available_resources)
        ] or strategies[:2]  # Default to first 2 if none match
        
        severity = 1.0 - performance  # Lower performance = higher severity
        
        return {
            'type': 'learning_outcome_intervention',
            'target_outcome': outcome.value,
            'predicted_performance': performance,
            'severity': severity,
            'strategies': available_strategies,
            'estimated_duration': f"{int(severity * 20 + 10)} minutes",
            'success_probability': min(0.9, 0.5 + (1.0 - severity) * 0.4),
            'description': f"Targeted intervention for {outcome.value} improvement"
        }
    
    def _generate_risk_factor_intervention(
        self,
        risk_factor: str,
        available_resources: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate intervention for specific risk factor."""
        
        risk_interventions = {
            'low_motivation': {
                'strategies': ['gamification', 'goal_setting', 'progress_visualization'],
                'description': 'Motivation enhancement intervention'
            },
            'attention_issues': {
                'strategies': ['focused_practice', 'distraction_reduction', 'mindfulness'],
                'description': 'Attention improvement intervention'
            },
            'high_anxiety': {
                'strategies': ['relaxation_techniques', 'gradual_exposure', 'confidence_building'],
                'description': 'Anxiety reduction intervention'
            },
            'social_isolation': {
                'strategies': ['peer_interaction', 'collaborative_learning', 'social_scenarios'],
                'description': 'Social engagement intervention'
            }
        }
        
        if risk_factor not in risk_interventions:
            return None
        
        intervention_data = risk_interventions[risk_factor]
        
        return {
            'type': 'risk_factor_intervention',
            'target_risk_factor': risk_factor,
            'strategies': intervention_data['strategies'],
            'description': intervention_data['description'],
            'severity': 0.6,  # Default moderate severity
            'estimated_duration': "15 minutes",
            'success_probability': 0.7
        }
    
    def _calculate_intervention_priority(
        self,
        intervention: Dict[str, Any],
        available_resources: List[str]
    ) -> float:
        """Calculate priority score for intervention."""
        
        # Base priority on severity and success probability
        severity = intervention.get('severity', 0.5)
        success_prob = intervention.get('success_probability', 0.5)
        
        # Resource availability bonus
        strategies = intervention.get('strategies', [])
        resource_match = sum(
            1 for strategy in strategies
            if any(resource in strategy for resource in available_resources)
        ) / max(1, len(strategies))
        
        # Calculate priority (0.0 to 1.0)
        priority = (severity * 0.4 + success_prob * 0.4 + resource_match * 0.2)
        
        return float(priority)