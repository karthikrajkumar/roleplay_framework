"""
Configuration settings for the API Gateway service.

This module manages all configuration settings using Pydantic
for validation and environment variable loading.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    """API Gateway configuration settings."""
    
    # Application settings
    app_name: str = Field(default="API Gateway", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Security settings
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="Secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute per IP")
    rate_limit_burst: int = Field(default=20, description="Rate limit burst capacity")
    
    # Service discovery
    consul_url: str = Field(default="http://localhost:8500", description="Consul server URL")
    service_name: str = Field(default="api-gateway", description="Service name for discovery")
    
    # Microservice endpoints
    user_service_url: str = Field(default="http://localhost:8001", description="User service URL")
    roleplay_service_url: str = Field(default="http://localhost:8002", description="Roleplay service URL")
    ai_service_url: str = Field(default="http://localhost:8003", description="AI service URL")
    notification_service_url: str = Field(default="http://localhost:8004", description="Notification service URL")
    analytics_service_url: str = Field(default="http://localhost:8005", description="Analytics service URL")
    
    # Database settings (for caching and sessions)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_pool_size: int = Field(default=10, description="Redis connection pool size")
    
    # Caching settings
    cache_ttl_seconds: int = Field(default=300, description="Default cache TTL")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    
    # Monitoring and observability
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    
    # Request timeout settings
    request_timeout_seconds: int = Field(default=30, description="Default request timeout")
    upstream_timeout_seconds: int = Field(default=25, description="Upstream service timeout")
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_breaker_timeout_seconds: int = Field(default=60, description="Circuit breaker timeout")
    circuit_breaker_recovery_timeout: int = Field(default=30, description="Circuit breaker recovery timeout")
    
    # Health check settings
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    health_check_timeout: int = Field(default=5, description="Health check timeout in seconds")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {allowed}')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'Log level must be one of: {allowed}')
        return v.upper()
    
    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format."""
        allowed = ['json', 'text']
        if v not in allowed:
            raise ValueError(f'Log format must be one of: {allowed}')
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == 'development'


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()