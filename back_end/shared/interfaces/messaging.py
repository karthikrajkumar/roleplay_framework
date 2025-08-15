"""
Messaging interface definitions for event-driven architecture.

This module defines interfaces for event publishing, handling, and message bus
operations to support decoupled microservices communication.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional, Type
from uuid import UUID

from ..domain.entities.base import DomainEvent


class IEventHandler(ABC):
    """
    Interface for domain event handlers.
    
    Event handlers process domain events and execute business logic
    in response to domain changes.
    """
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can process the event type."""
        pass


class IEventPublisher(ABC):
    """
    Interface for publishing domain events.
    
    Publishers are responsible for dispatching events to registered
    handlers and external message brokers.
    """
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a single domain event."""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events."""
        pass
    
    @abstractmethod
    async def publish_to_external(
        self, 
        event: DomainEvent, 
        topic: Optional[str] = None
    ) -> None:
        """Publish event to external message broker."""
        pass


class IMessageBus(ABC):
    """
    Interface for message bus operations.
    
    Message bus coordinates event publishing and handler registration
    for decoupled communication between services.
    """
    
    @abstractmethod
    def register_handler(
        self, 
        event_type: str, 
        handler: IEventHandler
    ) -> None:
        """Register an event handler for a specific event type."""
        pass
    
    @abstractmethod
    def unregister_handler(
        self, 
        event_type: str, 
        handler: IEventHandler
    ) -> None:
        """Unregister an event handler."""
        pass
    
    @abstractmethod
    async def dispatch(self, event: DomainEvent) -> None:
        """Dispatch event to registered handlers."""
        pass
    
    @abstractmethod
    async def dispatch_batch(self, events: List[DomainEvent]) -> None:
        """Dispatch multiple events."""
        pass
    
    @abstractmethod
    def get_handlers(self, event_type: str) -> List[IEventHandler]:
        """Get registered handlers for event type."""
        pass


class IMessageQueue(ABC):
    """
    Interface for message queue operations.
    
    Provides abstraction for message queuing systems like RabbitMQ,
    Redis, or cloud-based queues.
    """
    
    @abstractmethod
    async def send_message(
        self,
        queue_name: str,
        message: Dict[str, Any],
        priority: int = 0,
        delay_seconds: int = 0
    ) -> bool:
        """Send message to queue."""
        pass
    
    @abstractmethod
    async def receive_message(
        self,
        queue_name: str,
        timeout_seconds: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Receive message from queue."""
        pass
    
    @abstractmethod
    async def receive_messages(
        self,
        queue_name: str,
        max_messages: int = 10,
        timeout_seconds: int = 30
    ) -> List[Dict[str, Any]]:
        """Receive multiple messages from queue."""
        pass
    
    @abstractmethod
    async def acknowledge_message(self, queue_name: str, receipt_handle: str) -> bool:
        """Acknowledge message processing completion."""
        pass
    
    @abstractmethod
    async def create_queue(
        self,
        queue_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new queue."""
        pass
    
    @abstractmethod
    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue."""
        pass
    
    @abstractmethod
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics."""
        pass


class IWebSocketManager(ABC):
    """
    Interface for WebSocket connection management.
    
    Handles real-time communication for live roleplay sessions
    and notifications.
    """
    
    @abstractmethod
    async def connect_user(self, user_id: UUID, connection_id: str) -> None:
        """Register user WebSocket connection."""
        pass
    
    @abstractmethod
    async def disconnect_user(self, user_id: UUID, connection_id: str) -> None:
        """Unregister user WebSocket connection."""
        pass
    
    @abstractmethod
    async def send_to_user(
        self,
        user_id: UUID,
        message: Dict[str, Any]
    ) -> bool:
        """Send message to specific user."""
        pass
    
    @abstractmethod
    async def send_to_session(
        self,
        session_id: UUID,
        message: Dict[str, Any],
        exclude_user: Optional[UUID] = None
    ) -> int:
        """Send message to all users in a session."""
        pass
    
    @abstractmethod
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected users."""
        pass
    
    @abstractmethod
    async def join_session(self, user_id: UUID, session_id: UUID) -> None:
        """Join user to session room."""
        pass
    
    @abstractmethod
    async def leave_session(self, user_id: UUID, session_id: UUID) -> None:
        """Remove user from session room."""
        pass
    
    @abstractmethod
    async def get_session_participants(self, session_id: UUID) -> List[UUID]:
        """Get active participants in session."""
        pass
    
    @abstractmethod
    async def get_user_connections(self, user_id: UUID) -> List[str]:
        """Get active connections for user."""
        pass
    
    @abstractmethod
    async def is_user_online(self, user_id: UUID) -> bool:
        """Check if user has active connections."""
        pass