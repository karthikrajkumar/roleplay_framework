"""
AI domain entities for the AI roleplay platform.

This module contains AI-related domain entities including providers,
models, and responses for managing multiple AI services.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pydantic import Field, field_validator
from enum import Enum

from .base import AggregateRoot, ValueObject, EntityStatus, DomainEvent


class AIProviderType(str, Enum):
    """AI provider type enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """AI model capability enumeration."""
    TEXT_GENERATION = "text_generation"
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    FUNCTION_CALLING = "function_calling"
    EMBEDDING = "embedding"


class ResponseStatus(str, Enum):
    """AI response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class AIModelUsedEvent(DomainEvent):
    """Event raised when an AI model is used."""
    event_type: str = "ai.model.used"
    provider_id: UUID
    model_id: UUID
    tokens_used: int
    response_time_ms: int


class ModelConfiguration(ValueObject):
    """AI model configuration value object."""
    
    temperature: float = Field(default=0.8, ge=0, le=2, description="Creativity level")
    max_tokens: int = Field(default=1000, ge=1, le=8000, description="Maximum response tokens")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Nucleus sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2, le=2, description="Presence penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    
    @field_validator('stop_sequences')
    @classmethod
    def validate_stop_sequences(cls, v):
        """Validate stop sequences."""
        if len(v) > 10:
            raise ValueError("Cannot have more than 10 stop sequences")
        return v


class UsageStatistics(ValueObject):
    """Usage statistics value object."""
    
    total_requests: int = Field(default=0, description="Total requests made")
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    average_response_time: float = Field(default=0.0, description="Average response time in ms")
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Success rate")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")


class AIProvider(AggregateRoot):
    """
    AI Provider aggregate representing an AI service provider.
    
    Manages connection details, authentication, and configuration
    for different AI service providers.
    """
    
    name: str = Field(..., min_length=1, max_length=100, description="Provider name")
    provider_type: AIProviderType = Field(..., description="Provider type")
    description: str = Field(default="", max_length=500, description="Provider description")
    
    # Connection details
    base_url: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    
    # Configuration
    default_config: ModelConfiguration = Field(
        default_factory=ModelConfiguration, 
        description="Default model configuration"
    )
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    
    # Status and monitoring
    is_enabled: bool = Field(default=True, description="Provider enabled status")
    health_status: str = Field(default="unknown", description="Health check status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check")
    
    # Usage tracking
    usage_stats: UsageStatistics = Field(
        default_factory=UsageStatistics, 
        description="Usage statistics"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate provider name."""
        return v.strip()
    
    def update_health_status(self, status: str) -> None:
        """Update provider health status."""
        self.health_status = status
        self.last_health_check = datetime.now()
        self.mark_as_modified()
    
    def record_usage(self, tokens: int, cost: float, response_time_ms: int, success: bool) -> None:
        """Record usage statistics."""
        stats = self.usage_stats.model_dump()
        
        stats['total_requests'] += 1
        stats['total_tokens'] += tokens
        stats['total_cost'] += cost
        stats['last_used'] = datetime.now()
        
        # Update average response time
        if stats['total_requests'] > 1:
            stats['average_response_time'] = (
                (stats['average_response_time'] * (stats['total_requests'] - 1) + response_time_ms) 
                / stats['total_requests']
            )
        else:
            stats['average_response_time'] = response_time_ms
        
        # Update success rate
        if success:
            current_successes = stats['success_rate'] * (stats['total_requests'] - 1)
            stats['success_rate'] = (current_successes + 1) / stats['total_requests']
        else:
            current_successes = stats['success_rate'] * (stats['total_requests'] - 1)
            stats['success_rate'] = current_successes / stats['total_requests']
        
        self.usage_stats = UsageStatistics(**stats)
        self.mark_as_modified()


class AIModel(AggregateRoot):
    """
    AI Model aggregate representing a specific AI model.
    
    Contains model metadata, capabilities, and configuration
    for different AI models from various providers.
    """
    
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    model_id: str = Field(..., description="Model identifier from provider")
    provider_id: UUID = Field(..., description="Associated provider ID")
    
    # Model details
    description: str = Field(default="", max_length=1000, description="Model description")
    version: str = Field(default="1.0", description="Model version")
    capabilities: List[ModelCapability] = Field(default_factory=list, description="Model capabilities")
    
    # Specifications
    context_length: int = Field(default=4096, description="Maximum context length")
    max_output_tokens: int = Field(default=1000, description="Maximum output tokens")
    supports_streaming: bool = Field(default=False, description="Streaming support")
    supports_functions: bool = Field(default=False, description="Function calling support")
    
    # Pricing (per 1K tokens)
    input_cost_per_1k: float = Field(default=0.0, description="Input cost per 1K tokens")
    output_cost_per_1k: float = Field(default=0.0, description="Output cost per 1K tokens")
    
    # Configuration constraints
    min_temperature: float = Field(default=0.0, description="Minimum temperature")
    max_temperature: float = Field(default=2.0, description="Maximum temperature")
    default_config: ModelConfiguration = Field(
        default_factory=ModelConfiguration,
        description="Default configuration"
    )
    
    # Status
    is_available: bool = Field(default=True, description="Model availability")
    is_deprecated: bool = Field(default=False, description="Deprecation status")
    
    # Usage tracking
    usage_stats: UsageStatistics = Field(
        default_factory=UsageStatistics,
        description="Usage statistics"
    )
    
    @field_validator('capabilities')
    @classmethod
    def validate_capabilities(cls, v):
        """Validate model capabilities."""
        if not v:
            raise ValueError("Model must have at least one capability")
        return v
    
    def can_handle_capability(self, capability: ModelCapability) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost
    
    def record_usage(self, input_tokens: int, output_tokens: int, response_time_ms: int, success: bool) -> None:
        """Record model usage."""
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        stats = self.usage_stats.model_dump()
        
        stats['total_requests'] += 1
        stats['total_tokens'] += total_tokens
        stats['total_cost'] += cost
        stats['last_used'] = datetime.now()
        
        # Update average response time
        if stats['total_requests'] > 1:
            stats['average_response_time'] = (
                (stats['average_response_time'] * (stats['total_requests'] - 1) + response_time_ms) 
                / stats['total_requests']
            )
        else:
            stats['average_response_time'] = response_time_ms
        
        # Update success rate
        if success:
            current_successes = stats['success_rate'] * (stats['total_requests'] - 1)
            stats['success_rate'] = (current_successes + 1) / stats['total_requests']
        else:
            current_successes = stats['success_rate'] * (stats['total_requests'] - 1)
            stats['success_rate'] = current_successes / stats['total_requests']
        
        self.usage_stats = UsageStatistics(**stats)
        self.mark_as_modified()
        
        # Emit usage event
        self.add_domain_event(AIModelUsedEvent(
            aggregate_id=self.id,
            version=self.version,
            provider_id=self.provider_id,
            model_id=self.id,
            tokens_used=total_tokens,
            response_time_ms=response_time_ms
        ))


class AIResponse(ValueObject):
    """
    AI Response value object representing a response from an AI model.
    
    Contains the response content, metadata, and usage information.
    """
    
    id: UUID = Field(..., description="Response ID")
    model_id: UUID = Field(..., description="Model used")
    provider_id: UUID = Field(..., description="Provider used")
    
    # Request details
    prompt: str = Field(..., description="Input prompt")
    config: ModelConfiguration = Field(..., description="Configuration used")
    
    # Response content
    content: str = Field(..., description="Response content")
    finish_reason: str = Field(default="completed", description="Why generation stopped")
    
    # Metadata
    status: ResponseStatus = Field(..., description="Response status")
    created_at: datetime = Field(..., description="Response timestamp")
    
    # Usage tracking
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens")
    cost: float = Field(default=0.0, description="Response cost")
    
    # Performance
    response_time_ms: int = Field(default=0, description="Response time in milliseconds")
    queue_time_ms: int = Field(default=0, description="Queue time in milliseconds")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
    
    def is_successful(self) -> bool:
        """Check if response was successful."""
        return self.status == ResponseStatus.SUCCESS
    
    def get_total_time_ms(self) -> int:
        """Get total time including queue time."""
        return self.response_time_ms + self.queue_time_ms