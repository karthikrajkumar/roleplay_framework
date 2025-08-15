"""
Repository interface definitions for data access layer.

This module defines abstract interfaces for repository patterns,
providing consistent data access abstractions across all services.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any
from uuid import UUID

from ..domain.entities import (
    BaseEntity, User, RoleplaySession, Character, Scenario, Message,
    AIProvider, AIModel, AIResponse
)

T = TypeVar('T', bound=BaseEntity)


class IRepository(Generic[T], ABC):
    """
    Generic repository interface providing standard CRUD operations.
    
    All domain-specific repositories should inherit from this interface
    to ensure consistent data access patterns.
    """
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination."""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters."""
        pass
    
    @abstractmethod
    async def find_by_criteria(
        self, 
        filters: Dict[str, Any], 
        offset: int = 0, 
        limit: int = 100
    ) -> List[T]:
        """Find entities by criteria with pagination."""
        pass


class IUserRepository(IRepository[User], ABC):
    """User repository interface with user-specific operations."""
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        pass
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        pass
    
    @abstractmethod
    async def email_exists(self, email: str) -> bool:
        """Check if email is already registered."""
        pass
    
    @abstractmethod
    async def username_exists(self, username: str) -> bool:
        """Check if username is already taken."""
        pass
    
    @abstractmethod
    async def get_by_subscription_tier(self, tier: str) -> List[User]:
        """Get users by subscription tier."""
        pass
    
    @abstractmethod
    async def get_active_users_since(self, since_date) -> List[User]:
        """Get users active since a specific date."""
        pass
    
    @abstractmethod
    async def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp."""
        pass


class IRoleplayRepository(IRepository[RoleplaySession], ABC):
    """Roleplay repository interface with roleplay-specific operations."""
    
    @abstractmethod
    async def get_sessions_by_user(self, user_id: UUID) -> List[RoleplaySession]:
        """Get all sessions for a user."""
        pass
    
    @abstractmethod
    async def get_active_sessions_by_user(self, user_id: UUID) -> List[RoleplaySession]:
        """Get active sessions for a user."""
        pass
    
    @abstractmethod
    async def get_sessions_by_character(self, character_id: UUID) -> List[RoleplaySession]:
        """Get sessions using a specific character."""
        pass
    
    @abstractmethod
    async def get_sessions_by_scenario(self, scenario_id: UUID) -> List[RoleplaySession]:
        """Get sessions using a specific scenario."""
        pass
    
    @abstractmethod
    async def add_message_to_session(self, session_id: UUID, message: Message) -> None:
        """Add a message to a session."""
        pass
    
    @abstractmethod
    async def get_session_messages(
        self, 
        session_id: UUID, 
        offset: int = 0, 
        limit: int = 100
    ) -> List[Message]:
        """Get messages for a session with pagination."""
        pass
    
    @abstractmethod
    async def update_session_status(self, session_id: UUID, status: str) -> None:
        """Update session status."""
        pass


class ICharacterRepository(IRepository[Character], ABC):
    """Character repository interface with character-specific operations."""
    
    @abstractmethod
    async def get_by_creator(self, creator_id: UUID) -> List[Character]:
        """Get characters created by a user."""
        pass
    
    @abstractmethod
    async def get_public_characters(self) -> List[Character]:
        """Get all public characters."""
        pass
    
    @abstractmethod
    async def get_by_tags(self, tags: List[str]) -> List[Character]:
        """Get characters by tags."""
        pass
    
    @abstractmethod
    async def search_characters(self, query: str) -> List[Character]:
        """Search characters by name or description."""
        pass
    
    @abstractmethod
    async def get_popular_characters(self, limit: int = 20) -> List[Character]:
        """Get most popular characters by usage."""
        pass
    
    @abstractmethod
    async def increment_usage_count(self, character_id: UUID) -> None:
        """Increment character usage counter."""
        pass


class IScenarioRepository(IRepository[Scenario], ABC):
    """Scenario repository interface with scenario-specific operations."""
    
    @abstractmethod
    async def get_by_creator(self, creator_id: UUID) -> List[Scenario]:
        """Get scenarios created by a user."""
        pass
    
    @abstractmethod
    async def get_public_scenarios(self) -> List[Scenario]:
        """Get all public scenarios."""
        pass
    
    @abstractmethod
    async def get_by_category(self, category: str) -> List[Scenario]:
        """Get scenarios by category."""
        pass
    
    @abstractmethod
    async def get_by_tags(self, tags: List[str]) -> List[Scenario]:
        """Get scenarios by tags."""
        pass
    
    @abstractmethod
    async def search_scenarios(self, query: str) -> List[Scenario]:
        """Search scenarios by title or description."""
        pass
    
    @abstractmethod
    async def get_popular_scenarios(self, limit: int = 20) -> List[Scenario]:
        """Get most popular scenarios by usage."""
        pass


class IAIRepository(ABC):
    """AI repository interface for AI-related data operations."""
    
    # Provider operations
    @abstractmethod
    async def get_provider_by_id(self, provider_id: UUID) -> Optional[AIProvider]:
        """Get AI provider by ID."""
        pass
    
    @abstractmethod
    async def get_all_providers(self) -> List[AIProvider]:
        """Get all AI providers."""
        pass
    
    @abstractmethod
    async def get_enabled_providers(self) -> List[AIProvider]:
        """Get all enabled providers."""
        pass
    
    @abstractmethod
    async def create_provider(self, provider: AIProvider) -> AIProvider:
        """Create a new provider."""
        pass
    
    @abstractmethod
    async def update_provider(self, provider: AIProvider) -> AIProvider:
        """Update provider."""
        pass
    
    # Model operations
    @abstractmethod
    async def get_model_by_id(self, model_id: UUID) -> Optional[AIModel]:
        """Get AI model by ID."""
        pass
    
    @abstractmethod
    async def get_models_by_provider(self, provider_id: UUID) -> List[AIModel]:
        """Get models for a specific provider."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[AIModel]:
        """Get all available models."""
        pass
    
    @abstractmethod
    async def get_models_by_capability(self, capability: str) -> List[AIModel]:
        """Get models by capability."""
        pass
    
    @abstractmethod
    async def create_model(self, model: AIModel) -> AIModel:
        """Create a new model."""
        pass
    
    @abstractmethod
    async def update_model(self, model: AIModel) -> AIModel:
        """Update model."""
        pass
    
    # Response operations
    @abstractmethod
    async def save_response(self, response: AIResponse) -> None:
        """Save AI response for analytics."""
        pass
    
    @abstractmethod
    async def get_responses_by_model(
        self, 
        model_id: UUID, 
        limit: int = 100
    ) -> List[AIResponse]:
        """Get recent responses for a model."""
        pass
    
    @abstractmethod
    async def get_usage_statistics(
        self, 
        provider_id: Optional[UUID] = None,
        model_id: Optional[UUID] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Get usage statistics with optional filters."""
        pass


class IUnitOfWork(ABC):
    """
    Unit of Work pattern interface for managing transactions.
    
    Provides a way to group multiple repository operations into
    a single transaction with rollback capability.
    """
    
    @abstractmethod
    async def __aenter__(self):
        """Enter async context manager."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass
    
    @property
    @abstractmethod
    def users(self) -> IUserRepository:
        """Get user repository."""
        pass
    
    @property
    @abstractmethod
    def sessions(self) -> IRoleplayRepository:
        """Get roleplay session repository."""
        pass
    
    @property
    @abstractmethod
    def characters(self) -> ICharacterRepository:
        """Get character repository."""
        pass
    
    @property
    @abstractmethod
    def scenarios(self) -> IScenarioRepository:
        """Get scenario repository."""
        pass
    
    @property
    @abstractmethod
    def ai(self) -> IAIRepository:
        """Get AI repository."""
        pass