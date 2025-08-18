from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Configuration
    title: str = "User Management Service"
    description: str = "Advanced AI roleplay user management and authentication service"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration  
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
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
    
    # Authentication Configuration
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time")
    refresh_token_expire_days: int = Field(default=30, description="Refresh token expiration time")
    
    # Password Configuration
    password_min_length: int = Field(default=8, description="Minimum password length")
    password_require_special: bool = Field(default=True, description="Require special characters")
    
    # Email Configuration
    smtp_server: str = Field(default="localhost", description="SMTP server")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")
    
    # External Services
    external_api_timeout: int = Field(default=30, description="External API timeout in seconds")
    
    # Security
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:30000"],
        description="CORS allowed origins"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute per user")
    
    # User Management
    max_login_attempts: int = Field(default=5, description="Max login attempts before lockout")
    account_lockout_minutes: int = Field(default=15, description="Account lockout duration")
    
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