from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Configuration
    title: str = "Analytics Service"
    description: str = "Advanced AI roleplay analytics and insights service"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration  
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8005, description="Server port")
    reload: bool = Field(default=False, description="Auto reload")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://roleplay_user:password@postgres:5432/roleplay_platform",
        description="Database connection URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://:password@redis:6379/0",
        description="Redis connection URL"
    )
    
    # Analytics Configuration
    analytics_retention_days: int = Field(default=90, description="Analytics data retention in days")
    batch_size: int = Field(default=1000, description="Analytics batch processing size")
    
    # External Services
    external_api_timeout: int = Field(default=30, description="External API timeout in seconds")
    
    # Security
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:30000"],
        description="CORS allowed origins"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.startswith(('postgresql', 'sqlite')):
            raise ValueError('database_url must be a valid PostgreSQL or SQLite URL')
        return v
    
    @field_validator('redis_url')
    @classmethod 
    def validate_redis_url(cls, v: str) -> str:
        if not v or not v.startswith('redis://'):
            raise ValueError('redis_url must be a valid Redis URL')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings with environment variable overrides."""
    return Settings()