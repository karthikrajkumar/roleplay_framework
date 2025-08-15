"""
Proactive Failure Detector

ML-powered failure detection system that predicts and detects failures before
they impact system availability. Uses multi-dimensional monitoring and 
sophisticated anomaly detection algorithms.

Features:
- Multi-level failure detection (hardware, software, network, application)
- ML-based failure prediction using historical patterns
- Real-time anomaly detection with adaptive baselines
- Cascading failure detection and prevention
- Integration with auto-recovery systems
- Configurable detection sensitivity and response strategies
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be detected"""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CRASH = "software_crash"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    CASCADING_FAILURE = "cascading_failure"


class FailureSeverity(Enum):
    """Severity levels for failures"""
    CRITICAL = "critical"     # Immediate action required
    HIGH = "high"            # Action required soon
    MEDIUM = "medium"        # Monitor closely
    LOW = "low"             # Informational
    PREDICTED = "predicted"  # Predicted future failure


class ComponentType(Enum):
    """Types of system components"""
    APPLICATION_SERVICE = "application_service"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCER = "load_balancer"
    STORAGE = "storage"
    NETWORK = "network"
    COMPUTE_NODE = "compute_node"


@dataclass
class HealthMetric:
    """Health metric data point"""
    metric_name: str
    value: float
    timestamp: float
    component_id: str
    component_type: ComponentType
    thresholds: Dict[str, float] = field(default_factory=dict)
    is_anomaly: bool = False


@dataclass
class FailureSignal:
    """Signal indicating potential or actual failure"""
    signal_id: str
    component_id: str
    component_type: ComponentType
    failure_type: FailureType
    severity: FailureSeverity
    confidence: float  # 0.0 to 1.0
    timestamp: float
    description: str
    contributing_metrics: List[str] = field(default_factory=list)
    predicted_impact: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_id: str
    component_type: ComponentType
    health_score: float  # 0.0 to 1.0
    status: str  # healthy, degraded, failing, failed
    last_updated: float
    metrics: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))
    failure_predictions: List[FailureSignal] = field(default_factory=list)
    dependency_graph: Set[str] = field(default_factory=set)


class ProactiveFailureDetector:
    """
    ML-powered failure detector that predicts and detects failures
    across distributed system components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector_id = config['detector_id']
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Failure detection
        self.active_signals: Dict[str, FailureSignal] = {}
        self.signal_history: deque = deque(maxlen=10000)
        self.anomaly_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # ML Models
        self.failure_predictors: Dict[ComponentType, FailurePredictor] = {}
        self.anomaly_detector = AnomalyDetector()
        self.cascading_analyzer = CascadingFailureAnalyzer()
        
        # Configuration
        self.detection_interval = config.get('detection_interval', 5.0)  # seconds
        self.prediction_horizon = config.get('prediction_horizon', 300.0)  # seconds
        self.anomaly_sensitivity = config.get('anomaly_sensitivity', 0.8)
        self.cascade_detection_enabled = config.get('cascade_detection_enabled', True)
        
        # Thresholds
        self.default_thresholds = {
            'cpu_utilization': {'warning': 70.0, 'critical': 90.0},
            'memory_utilization': {'warning': 80.0, 'critical': 95.0},
            'disk_utilization': {'warning': 85.0, 'critical': 95.0},
            'network_latency': {'warning': 100.0, 'critical': 500.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'response_time': {'warning': 1000.0, 'critical': 5000.0}
        }
        
        # Callbacks
        self.failure_callbacks: List[callable] = []
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Start monitoring
        self._start_monitoring()
    
    def _initialize_ml_models(self):
        """Initialize ML models for different component types"""
        for component_type in ComponentType:
            self.failure_predictors[component_type] = FailurePredictor(component_type)
    
    async def register_component(self, component_id: str, component_type: ComponentType,
                                dependencies: List[str] = None) -> bool:
        """Register a component for monitoring"""
        try:
            component_health = ComponentHealth(
                component_id=component_id,
                component_type=component_type,
                health_score=1.0,
                status="healthy",
                last_updated=time.time(),
                dependency_graph=set(dependencies or [])
            )
            
            self.components[component_id] = component_health
            
            # Update dependency graph
            if dependencies:
                self.dependency_graph[component_id].update(dependencies)
                
                # Add reverse dependencies
                for dep in dependencies:
                    if dep in self.components:
                        self.components[dep].dependency_graph.add(component_id)
            
            logger.info(f"Registered component {component_id} for failure detection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False
    
    async def update_metrics(self, component_id: str, metrics: Dict[str, float]) -> bool:
        """Update health metrics for a component"""
        if component_id not in self.components:
            logger.warning(f"Component {component_id} not registered")
            return False
        
        component = self.components[component_id]
        current_time = time.time()
        
        try:
            # Store metrics
            for metric_name, value in metrics.items():
                health_metric = HealthMetric(
                    metric_name=metric_name,
                    value=value,
                    timestamp=current_time,
                    component_id=component_id,
                    component_type=component.component_type,
                    thresholds=self.default_thresholds.get(metric_name, {})
                )
                
                # Check for anomalies
                is_anomaly = await self._detect_metric_anomaly(component_id, health_metric)
                health_metric.is_anomaly = is_anomaly
                
                # Store in component metrics
                component.metrics[metric_name].append(health_metric)
                
                # Generate failure signals if needed
                if is_anomaly or self._is_threshold_violated(health_metric):
                    await self._generate_failure_signal(component_id, health_metric)
            
            # Update component health score
            await self._update_component_health_score(component_id)
            
            # Check for failure predictions
            await self._check_failure_predictions(component_id)
            
            component.last_updated = current_time
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for component {component_id}: {e}")
            return False
    
    async def _detect_metric_anomaly(self, component_id: str, metric: HealthMetric) -> bool:
        """Detect if a metric value is anomalous"""
        
        component = self.components[component_id]
        metric_history = component.metrics.get(metric.metric_name, deque())
        
        if len(metric_history) < 10:  # Need sufficient history
            return False
        
        # Get recent values
        recent_values = [m.value for m in list(metric_history)[-100:]]
        
        # Use anomaly detector
        is_anomaly = await self.anomaly_detector.detect_anomaly(
            metric.metric_name, metric.value, recent_values, self.anomaly_sensitivity
        )
        
        return is_anomaly
    
    def _is_threshold_violated(self, metric: HealthMetric) -> bool:
        """Check if metric violates configured thresholds"""
        thresholds = metric.thresholds
        
        if 'critical' in thresholds and metric.value >= thresholds['critical']:
            return True
        
        if 'warning' in thresholds and metric.value >= thresholds['warning']:
            return True
        
        return False
    
    async def _generate_failure_signal(self, component_id: str, metric: HealthMetric):
        """Generate failure signal based on metric anomaly or threshold violation"""
        
        component = self.components[component_id]
        
        # Determine failure type and severity
        failure_type = self._determine_failure_type(metric)
        severity = self._determine_severity(metric)
        
        # Calculate confidence based on metric history and patterns
        confidence = await self._calculate_signal_confidence(component_id, metric)
        
        # Generate signal
        signal = FailureSignal(
            signal_id=self._generate_signal_id(),
            component_id=component_id,
            component_type=component.component_type,
            failure_type=failure_type,
            severity=severity,
            confidence=confidence,
            timestamp=time.time(),
            description=f"Anomaly detected in {metric.metric_name}: {metric.value}",
            contributing_metrics=[metric.metric_name],
            predicted_impact=await self._predict_failure_impact(component_id, failure_type),
            recommended_actions=self._generate_recommended_actions(failure_type, severity)
        )
        
        # Store signal
        self.active_signals[signal.signal_id] = signal
        self.signal_history.append(signal)
        component.failure_predictions.append(signal)
        
        # Trigger callbacks
        await self._trigger_failure_callbacks(signal)
        
        logger.warning(f"Generated failure signal {signal.signal_id} for component {component_id}")
    
    def _determine_failure_type(self, metric: HealthMetric) -> FailureType:
        """Determine failure type based on metric"""
        
        metric_to_failure_type = {
            'cpu_utilization': FailureType.RESOURCE_EXHAUSTION,
            'memory_utilization': FailureType.RESOURCE_EXHAUSTION,
            'disk_utilization': FailureType.RESOURCE_EXHAUSTION,
            'network_latency': FailureType.NETWORK_PARTITION,
            'error_rate': FailureType.SOFTWARE_CRASH,
            'response_time': FailureType.PERFORMANCE_DEGRADATION
        }
        
        return metric_to_failure_type.get(metric.metric_name, FailureType.PERFORMANCE_DEGRADATION)
    
    def _determine_severity(self, metric: HealthMetric) -> FailureSeverity:
        """Determine severity based on metric value and thresholds"""
        
        thresholds = metric.thresholds
        
        if 'critical' in thresholds and metric.value >= thresholds['critical']:
            return FailureSeverity.CRITICAL
        elif 'warning' in thresholds and metric.value >= thresholds['warning']:
            return FailureSeverity.HIGH
        elif metric.is_anomaly:
            return FailureSeverity.MEDIUM
        else:
            return FailureSeverity.LOW
    
    async def _calculate_signal_confidence(self, component_id: str, metric: HealthMetric) -> float:
        """Calculate confidence score for failure signal"""
        
        confidence_factors = []
        
        # Threshold violation factor
        if self._is_threshold_violated(metric):
            thresholds = metric.thresholds
            if 'critical' in thresholds and metric.value >= thresholds['critical']:
                confidence_factors.append(0.9)
            elif 'warning' in thresholds and metric.value >= thresholds['warning']:
                confidence_factors.append(0.7)
        
        # Anomaly detection factor
        if metric.is_anomaly:
            confidence_factors.append(0.6)
        
        # Historical pattern factor
        component = self.components[component_id]
        metric_history = component.metrics.get(metric.metric_name, deque())
        
        if len(metric_history) > 50:
            recent_anomalies = sum(1 for m in list(metric_history)[-50:] if m.is_anomaly)
            if recent_anomalies > 5:  # Pattern of anomalies
                confidence_factors.append(0.8)
        
        # Combine confidence factors
        if confidence_factors:
            return min(sum(confidence_factors) / len(confidence_factors), 1.0)
        else:
            return 0.5  # Default confidence
    
    async def _predict_failure_impact(self, component_id: str, failure_type: FailureType) -> Dict[str, Any]:
        """Predict impact of potential failure"""
        
        component = self.components[component_id]
        impact = {
            'affected_components': [],
            'estimated_downtime': 0,
            'service_impact': 'low',
            'user_impact': 'minimal'
        }
        
        # Analyze dependency impact
        dependent_components = [
            comp_id for comp_id, comp in self.components.items()
            if component_id in comp.dependency_graph
        ]
        
        impact['affected_components'] = dependent_components
        
        # Estimate impact based on component type and failure type
        if component.component_type == ComponentType.DATABASE:
            impact['service_impact'] = 'high'
            impact['user_impact'] = 'significant'
            impact['estimated_downtime'] = 300  # 5 minutes
        elif component.component_type == ComponentType.LOAD_BALANCER:
            impact['service_impact'] = 'critical'
            impact['user_impact'] = 'severe'
            impact['estimated_downtime'] = 60  # 1 minute
        
        return impact
    
    def _generate_recommended_actions(self, failure_type: FailureType, 
                                    severity: FailureSeverity) -> List[str]:
        """Generate recommended actions for failure"""
        
        actions = []
        
        if failure_type == FailureType.RESOURCE_EXHAUSTION:
            actions.extend([
                "Scale up resources",
                "Optimize resource usage",
                "Implement load shedding"
            ])
        elif failure_type == FailureType.NETWORK_PARTITION:
            actions.extend([
                "Check network connectivity",
                "Failover to backup network",
                "Enable partition tolerance mode"
            ])
        elif failure_type == FailureType.SOFTWARE_CRASH:
            actions.extend([
                "Restart service",
                "Check error logs",
                "Deploy hotfix if available"
            ])
        
        if severity in [FailureSeverity.CRITICAL, FailureSeverity.HIGH]:
            actions.insert(0, "Initiate emergency response protocol")
        
        return actions
    
    async def _update_component_health_score(self, component_id: str):
        """Update overall health score for component"""
        
        component = self.components[component_id]
        health_factors = []
        
        # Analyze recent metrics
        for metric_name, metric_history in component.metrics.items():
            if metric_history:
                recent_metrics = list(metric_history)[-10:]  # Last 10 measurements
                
                # Calculate metric health factor
                anomaly_rate = sum(1 for m in recent_metrics if m.is_anomaly) / len(recent_metrics)
                threshold_violations = sum(1 for m in recent_metrics if self._is_threshold_violated(m))
                
                metric_health = 1.0 - (anomaly_rate * 0.3 + threshold_violations * 0.5 / len(recent_metrics))
                health_factors.append(max(metric_health, 0.0))
        
        # Calculate overall health score
        if health_factors:
            component.health_score = sum(health_factors) / len(health_factors)
        else:
            component.health_score = 1.0
        
        # Update status based on health score
        if component.health_score >= 0.8:
            component.status = "healthy"
        elif component.health_score >= 0.6:
            component.status = "degraded"
        elif component.health_score >= 0.3:
            component.status = "failing"
        else:
            component.status = "failed"
    
    async def _check_failure_predictions(self, component_id: str):
        """Check for failure predictions using ML models"""
        
        component = self.components[component_id]
        predictor = self.failure_predictors.get(component.component_type)
        
        if not predictor:
            return
        
        # Prepare metric data for prediction
        metric_data = {}
        for metric_name, metric_history in component.metrics.items():
            if metric_history:
                recent_values = [m.value for m in list(metric_history)[-100:]]
                metric_data[metric_name] = recent_values
        
        if not metric_data:
            return
        
        # Get failure prediction
        prediction = await predictor.predict_failure(
            component_id, metric_data, self.prediction_horizon
        )
        
        if prediction and prediction.confidence > 0.7:
            # Generate predictive failure signal
            signal = FailureSignal(
                signal_id=self._generate_signal_id(),
                component_id=component_id,
                component_type=component.component_type,
                failure_type=prediction.failure_type,
                severity=FailureSeverity.PREDICTED,
                confidence=prediction.confidence,
                timestamp=time.time(),
                description=f"Predicted {prediction.failure_type.value} in {prediction.time_to_failure:.0f} seconds",
                predicted_impact=await self._predict_failure_impact(component_id, prediction.failure_type),
                recommended_actions=["Prepare failover", "Schedule maintenance", "Alert operations team"]
            )
            
            self.active_signals[signal.signal_id] = signal
            self.signal_history.append(signal)
            component.failure_predictions.append(signal)
            
            await self._trigger_failure_callbacks(signal)
            
            logger.info(f"Predicted failure for component {component_id}: {prediction.failure_type.value}")
    
    async def _trigger_failure_callbacks(self, signal: FailureSignal):
        """Trigger registered failure callbacks"""
        for callback in self.failure_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._cascading_failure_monitor())
        asyncio.create_task(self._signal_cleanup())
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            await asyncio.sleep(self.detection_interval)
            
            # Check component health
            for component_id, component in self.components.items():
                # Check if component is stale
                time_since_update = time.time() - component.last_updated
                if time_since_update > 60:  # 1 minute
                    # Generate stale component signal
                    await self._generate_stale_component_signal(component_id, time_since_update)
    
    async def _cascading_failure_monitor(self):
        """Monitor for cascading failures"""
        if not self.cascade_detection_enabled:
            return
        
        while True:
            await asyncio.sleep(10.0)  # Check every 10 seconds
            
            # Analyze recent failures for cascading patterns
            recent_signals = [
                signal for signal in list(self.signal_history)[-50:]
                if time.time() - signal.timestamp < 300  # Last 5 minutes
            ]
            
            if len(recent_signals) > 5:  # Multiple recent failures
                cascade_risk = await self.cascading_analyzer.analyze_cascade_risk(
                    recent_signals, self.dependency_graph
                )
                
                if cascade_risk.risk_score > 0.7:
                    await self._handle_cascade_risk(cascade_risk)
    
    async def _signal_cleanup(self):
        """Clean up old signals"""
        while True:
            await asyncio.sleep(300.0)  # Every 5 minutes
            
            current_time = time.time()
            expired_signals = []
            
            for signal_id, signal in self.active_signals.items():
                # Remove signals older than 1 hour
                if current_time - signal.timestamp > 3600:
                    expired_signals.append(signal_id)
            
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return f"signal_{self.detector_id}_{time.time_ns()}"
    
    def add_failure_callback(self, callback: callable):
        """Add callback for failure notifications"""
        self.failure_callbacks.append(callback)
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status of a component"""
        return self.components.get(component_id)
    
    def get_active_signals(self, severity_filter: FailureSeverity = None) -> List[FailureSignal]:
        """Get active failure signals"""
        signals = list(self.active_signals.values())
        
        if severity_filter:
            signals = [s for s in signals if s.severity == severity_filter]
        
        return sorted(signals, key=lambda x: x.timestamp, reverse=True)
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get failure detector statistics"""
        return {
            'detector_id': self.detector_id,
            'monitored_components': len(self.components),
            'active_signals': len(self.active_signals),
            'total_signals_generated': len(self.signal_history),
            'component_health_scores': {
                comp_id: comp.health_score 
                for comp_id, comp in self.components.items()
            },
            'severity_distribution': {
                severity.value: len([s for s in self.active_signals.values() if s.severity == severity])
                for severity in FailureSeverity
            }
        }


@dataclass
class FailurePrediction:
    """ML-based failure prediction"""
    failure_type: FailureType
    confidence: float
    time_to_failure: float  # seconds
    contributing_factors: List[str]


class FailurePredictor:
    """ML-based failure predictor for specific component types"""
    
    def __init__(self, component_type: ComponentType):
        self.component_type = component_type
        self.model_trained = False
        self.prediction_history = deque(maxlen=1000)
    
    async def predict_failure(self, component_id: str, metric_data: Dict[str, List[float]], 
                            horizon: float) -> Optional[FailurePrediction]:
        """Predict failure for component"""
        
        # Simplified prediction logic (in production, use sophisticated ML models)
        
        # Check CPU utilization trend
        if 'cpu_utilization' in metric_data:
            cpu_values = metric_data['cpu_utilization']
            if len(cpu_values) > 10:
                recent_trend = np.mean(cpu_values[-5:]) - np.mean(cpu_values[-10:-5])
                
                if recent_trend > 10 and np.mean(cpu_values[-5:]) > 80:
                    return FailurePrediction(
                        failure_type=FailureType.RESOURCE_EXHAUSTION,
                        confidence=0.8,
                        time_to_failure=horizon * 0.3,  # 30% of horizon
                        contributing_factors=['cpu_utilization_trend']
                    )
        
        # Check memory utilization
        if 'memory_utilization' in metric_data:
            memory_values = metric_data['memory_utilization']
            if len(memory_values) > 5 and np.mean(memory_values[-5:]) > 90:
                return FailurePrediction(
                    failure_type=FailureType.RESOURCE_EXHAUSTION,
                    confidence=0.75,
                    time_to_failure=horizon * 0.5,
                    contributing_factors=['memory_utilization']
                )
        
        return None


class AnomalyDetector:
    """Statistical anomaly detection"""
    
    async def detect_anomaly(self, metric_name: str, value: float, 
                           history: List[float], sensitivity: float) -> bool:
        """Detect if value is anomalous compared to history"""
        
        if len(history) < 10:
            return False
        
        # Use simple statistical method
        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0
        
        if stdev == 0:
            return value != mean
        
        # Z-score based detection
        z_score = abs(value - mean) / stdev
        threshold = 2.0 + (1.0 - sensitivity) * 2.0  # Adaptive threshold
        
        return z_score > threshold


@dataclass
class CascadeRisk:
    """Cascading failure risk assessment"""
    risk_score: float
    affected_components: List[str]
    failure_path: List[str]
    estimated_propagation_time: float


class CascadingFailureAnalyzer:
    """Analyzes risk of cascading failures"""
    
    async def analyze_cascade_risk(self, recent_signals: List[FailureSignal], 
                                 dependency_graph: Dict[str, Set[str]]) -> CascadeRisk:
        """Analyze risk of cascading failure"""
        
        # Identify failure clusters
        failed_components = set(signal.component_id for signal in recent_signals)
        
        # Calculate cascade risk based on dependency graph
        at_risk_components = set()
        for failed_comp in failed_components:
            if failed_comp in dependency_graph:
                at_risk_components.update(dependency_graph[failed_comp])
        
        # Calculate risk score
        risk_score = min(len(at_risk_components) / 10.0, 1.0)  # Normalize to max 10 components
        
        return CascadeRisk(
            risk_score=risk_score,
            affected_components=list(at_risk_components),
            failure_path=list(failed_components),
            estimated_propagation_time=len(at_risk_components) * 30.0  # 30 seconds per component
        )