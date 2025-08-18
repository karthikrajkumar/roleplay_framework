"""
Domain entities package.

This package contains all domain entities used across the roleplay platform,
including user entities, roleplay entities, AI entities, and more.
"""

from .base import BaseEntity, AggregateRoot, ValueObject, DomainEvent, EntityStatus
from .user import User, UserProfile, UserPreferences, UserRole, SubscriptionTier
from .roleplay import RoleplaySession, Character, Scenario, Message
from .ai import AIProvider, AIModel, AIResponse, ModelConfiguration

__all__ = [
    "BaseEntity",
    "AggregateRoot", 
    "ValueObject",
    "DomainEvent",
    "EntityStatus",
    "User",
    "UserProfile",
    "UserPreferences",
    "UserRole",
    "SubscriptionTier",
    "RoleplaySession",
    "Character",
    "Scenario",
    "Message",
    "AIProvider",
    "AIModel",
    "AIResponse",
    "ModelConfiguration"
]