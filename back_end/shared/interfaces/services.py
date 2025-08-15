"""
Service interface definitions for business logic layer.

This module defines abstract interfaces for application services,
providing consistent business logic abstractions across all services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID
from datetime import datetime

from ..domain.entities import (
    User, RoleplaySession, Character, Scenario, Message,
    AIProvider, AIModel, AIResponse, ModelConfiguration
)


class IAIService(ABC):
    """
    AI Service interface for managing AI interactions.
    
    Provides abstraction for different AI providers and models,
    handling model selection, request routing, and response processing.
    """
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        model_id: UUID,
        config: Optional[ModelConfiguration] = None,
        user_id: Optional[UUID] = None
    ) -> AIResponse:
        """Generate AI response using specified model."""
        pass
    
    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        model_id: UUID,
        config: Optional[ModelConfiguration] = None,
        user_id: Optional[UUID] = None
    ) -> AsyncGenerator[str, None]:
        """Stream AI response in real-time."""
        pass
    
    @abstractmethod
    async def get_available_models(self, capability: Optional[str] = None) -> List[AIModel]:
        """Get available AI models, optionally filtered by capability."""
        pass
    
    @abstractmethod
    async def get_model_recommendations(
        self,
        character_id: Optional[UUID] = None,
        scenario_id: Optional[UUID] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[AIModel]:
        """Get recommended models based on context."""
        pass
    
    @abstractmethod
    async def validate_model_config(self, model_id: UUID, config: ModelConfiguration) -> bool:
        """Validate model configuration against model constraints."""
        pass
    
    @abstractmethod
    async def estimate_cost(
        self,
        prompt: str,
        model_id: UUID,
        estimated_response_tokens: int = 500
    ) -> float:
        """Estimate cost for a request."""
        pass


class ICharacterService(ABC):
    """
    Character Service interface for managing roleplay characters.
    
    Handles character creation, updates, validation, and AI persona generation.
    """
    
    @abstractmethod
    async def create_character(self, character_data: Dict[str, Any], creator_id: UUID) -> Character:
        """Create a new character."""
        pass
    
    @abstractmethod
    async def update_character(
        self, 
        character_id: UUID, 
        updates: Dict[str, Any], 
        user_id: UUID
    ) -> Character:
        """Update an existing character."""
        pass
    
    @abstractmethod
    async def get_character(self, character_id: UUID, user_id: Optional[UUID] = None) -> Optional[Character]:
        """Get character with access validation."""
        pass
    
    @abstractmethod
    async def delete_character(self, character_id: UUID, user_id: UUID) -> bool:
        """Delete a character."""
        pass
    
    @abstractmethod
    async def search_characters(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        user_id: Optional[UUID] = None
    ) -> List[Character]:
        """Search characters with filters."""
        pass
    
    @abstractmethod
    async def generate_character_from_description(
        self,
        description: str,
        creator_id: UUID
    ) -> Character:
        """Generate character using AI from description."""
        pass
    
    @abstractmethod
    async def validate_character_access(self, character_id: UUID, user_id: UUID) -> bool:
        """Validate user access to character."""
        pass


class IScenarioService(ABC):
    """
    Scenario Service interface for managing roleplay scenarios.
    
    Handles scenario creation, updates, and AI-generated scenarios.
    """
    
    @abstractmethod
    async def create_scenario(self, scenario_data: Dict[str, Any], creator_id: UUID) -> Scenario:
        """Create a new scenario."""
        pass
    
    @abstractmethod
    async def update_scenario(
        self, 
        scenario_id: UUID, 
        updates: Dict[str, Any], 
        user_id: UUID
    ) -> Scenario:
        """Update an existing scenario."""
        pass
    
    @abstractmethod
    async def get_scenario(self, scenario_id: UUID, user_id: Optional[UUID] = None) -> Optional[Scenario]:
        """Get scenario with access validation."""
        pass
    
    @abstractmethod
    async def search_scenarios(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Scenario]:
        """Search scenarios with filters."""
        pass
    
    @abstractmethod
    async def generate_scenario(
        self,
        theme: str,
        character_ids: List[UUID],
        creator_id: UUID
    ) -> Scenario:
        """Generate scenario using AI."""
        pass


class IRoleplayService(ABC):
    """
    Roleplay Service interface for managing roleplay sessions.
    
    Handles session management, message processing, and AI character interactions.
    """
    
    @abstractmethod
    async def create_session(
        self,
        title: str,
        user_id: UUID,
        character_ids: List[UUID],
        scenario_id: Optional[UUID] = None
    ) -> RoleplaySession:
        """Create a new roleplay session."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: UUID, user_id: UUID) -> Optional[RoleplaySession]:
        """Get session with access validation."""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        session_id: UUID,
        content: str,
        sender_id: UUID,
        message_type: str = "text"
    ) -> Message:
        """Send a user message to the session."""
        pass
    
    @abstractmethod
    async def generate_ai_response(
        self,
        session_id: UUID,
        character_id: UUID,
        user_id: UUID
    ) -> Message:
        """Generate AI character response."""
        pass
    
    @abstractmethod
    async def stream_ai_response(
        self,
        session_id: UUID,
        character_id: UUID,
        user_id: UUID
    ) -> AsyncGenerator[str, None]:
        """Stream AI character response."""
        pass
    
    @abstractmethod
    async def pause_session(self, session_id: UUID, user_id: UUID) -> bool:
        """Pause a session."""
        pass
    
    @abstractmethod
    async def resume_session(self, session_id: UUID, user_id: UUID) -> bool:
        """Resume a session."""
        pass
    
    @abstractmethod
    async def end_session(
        self, 
        session_id: UUID, 
        user_id: UUID, 
        summary: Optional[str] = None
    ) -> bool:
        """End a session."""
        pass
    
    @abstractmethod
    async def get_user_sessions(
        self,
        user_id: UUID,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 20
    ) -> List[RoleplaySession]:
        """Get user's roleplay sessions."""
        pass


class INotificationService(ABC):
    """
    Notification Service interface for managing user notifications.
    
    Handles various notification channels including email, push, SMS, and in-app.
    """
    
    @abstractmethod
    async def send_notification(
        self,
        user_id: UUID,
        title: str,
        message: str,
        notification_type: str,
        channels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification to user."""
        pass
    
    @abstractmethod
    async def send_bulk_notification(
        self,
        user_ids: List[UUID],
        title: str,
        message: str,
        notification_type: str,
        channels: Optional[List[str]] = None
    ) -> Dict[UUID, bool]:
        """Send notification to multiple users."""
        pass
    
    @abstractmethod
    async def get_user_notifications(
        self,
        user_id: UUID,
        unread_only: bool = False,
        offset: int = 0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get user notifications."""
        pass
    
    @abstractmethod
    async def mark_as_read(self, notification_id: UUID, user_id: UUID) -> bool:
        """Mark notification as read."""
        pass
    
    @abstractmethod
    async def subscribe_to_topic(self, user_id: UUID, topic: str) -> bool:
        """Subscribe user to notification topic."""
        pass
    
    @abstractmethod
    async def unsubscribe_from_topic(self, user_id: UUID, topic: str) -> bool:
        """Unsubscribe user from notification topic."""
        pass


class IAnalyticsService(ABC):
    """
    Analytics Service interface for tracking usage and generating insights.
    
    Provides analytics for user behavior, AI usage, and system performance.
    """
    
    @abstractmethod
    async def track_event(
        self,
        event_name: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a custom event."""
        pass
    
    @abstractmethod
    async def track_user_action(
        self,
        action: str,
        user_id: UUID,
        resource_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track user action."""
        pass
    
    @abstractmethod
    async def get_user_analytics(
        self,
        user_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        pass
    
    @abstractmethod
    async def get_platform_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get platform-wide analytics."""
        pass
    
    @abstractmethod
    async def get_ai_usage_analytics(
        self,
        provider_id: Optional[UUID] = None,
        model_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get AI usage analytics."""
        pass
    
    @abstractmethod
    async def generate_user_insights(self, user_id: UUID) -> Dict[str, Any]:
        """Generate personalized insights for user."""
        pass


class IUserService(ABC):
    """
    User Service interface for managing user operations.
    
    Handles user registration, authentication, profile management, and preferences.
    """
    
    @abstractmethod
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        profile_data: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user account."""
        pass
    
    @abstractmethod
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        pass
    
    @abstractmethod
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def update_user_profile(
        self,
        user_id: UUID,
        profile_updates: Dict[str, Any]
    ) -> User:
        """Update user profile."""
        pass
    
    @abstractmethod
    async def update_user_preferences(
        self,
        user_id: UUID,
        preferences: Dict[str, Any]
    ) -> User:
        """Update user preferences."""
        pass
    
    @abstractmethod
    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change user password."""
        pass
    
    @abstractmethod
    async def verify_email(self, user_id: UUID, verification_token: str) -> bool:
        """Verify user email address."""
        pass
    
    @abstractmethod
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate user account."""
        pass