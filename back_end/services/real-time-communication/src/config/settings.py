from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # API Configuration
    title: str = "Real-time Communication Service"
    description: str = "Advanced AI roleplay real-time communication service"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration  
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8006, description="Server port")
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
    
    # WebSocket Configuration
    websocket_max_connections: int = Field(default=10000, description="Max WebSocket connections")
    websocket_heartbeat_interval: int = Field(default=30, description="WebSocket heartbeat interval")
    websocket_timeout: int = Field(default=60, description="WebSocket connection timeout")
    
    # Real-time Communication
    message_queue_max_size: int = Field(default=1000, description="Message queue max size")
    broadcast_timeout: int = Field(default=5, description="Broadcast timeout in seconds")
    
    # WebRTC Configuration
    ice_servers: List[str] = Field(
        default=["stun:stun.l.google.com:19302"],
        description="ICE servers for WebRTC"
    )
    
    # External Services
    external_api_timeout: int = Field(default=30, description="External API timeout in seconds")
    
    # Security
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:30000"],
        description="CORS allowed origins"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute per user")
    
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