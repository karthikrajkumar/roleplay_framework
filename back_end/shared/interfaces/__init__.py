"""
Interfaces package for the AI Roleplay Platform.

This package contains all interface definitions for repositories, services,
and other abstractions used throughout the system.
"""

from .repository import IRepository, IUserRepository, IRoleplayRepository, IAIRepository
from .services import IAIService, INotificationService, IAnalyticsService
from .messaging import IEventPublisher, IEventHandler, IMessageBus
from .caching import ICacheService
from .storage import IFileStorage, IVectorStore

__all__ = [
    "IRepository",
    "IUserRepository", 
    "IRoleplayRepository",
    "IAIRepository",
    "IAIService",
    "INotificationService",
    "IAnalyticsService",
    "IEventPublisher",
    "IEventHandler",
    "IMessageBus",
    "ICacheService",
    "IFileStorage",
    "IVectorStore"
]