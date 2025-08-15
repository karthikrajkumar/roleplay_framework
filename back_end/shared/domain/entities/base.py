"""
Base entity classes providing common functionality for all domain entities.

This module implements the foundation for domain entities with audit trails,
validation, and event sourcing capabilities.
"""

from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from enum import Enum


class EntityStatus(str, Enum):
    """Common entity status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    PENDING = "pending"


class DomainEvent(BaseModel):
    """Base class for domain events."""
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    aggregate_id: UUID
    version: int
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


T = TypeVar('T', bound='BaseEntity')


class BaseEntity(BaseModel, ABC):
    """
    Base entity class providing common functionality for all domain entities.
    
    Features:
    - Unique identifier
    - Audit trail (created_at, updated_at)
    - Soft delete capability
    - Event sourcing support
    - Validation framework
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    created_by: Optional[UUID] = Field(None, description="ID of creator")
    updated_by: Optional[UUID] = Field(None, description="ID of last updater")
    version: int = Field(default=1, description="Entity version for optimistic locking")
    status: EntityStatus = Field(default=EntityStatus.ACTIVE, description="Entity status")
    
    # Event sourcing
    _domain_events: List[DomainEvent] = Field(default_factory=list, exclude=True)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if 'updated_at' not in data:
            self.updated_at = self.created_at
    
    def add_domain_event(self, event: DomainEvent) -> None:
        """Add a domain event to be published."""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> List[DomainEvent]:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def mark_as_modified(self, modified_by: Optional[UUID] = None) -> None:
        """Mark entity as modified with timestamp and user."""
        self.updated_at = datetime.utcnow()
        self.version += 1
        if modified_by:
            self.updated_by = modified_by
    
    def soft_delete(self, deleted_by: Optional[UUID] = None) -> None:
        """Perform soft delete on entity."""
        self.status = EntityStatus.DELETED
        self.mark_as_modified(deleted_by)
    
    def is_active(self) -> bool:
        """Check if entity is active."""
        return self.status == EntityStatus.ACTIVE
    
    def is_deleted(self) -> bool:
        """Check if entity is soft deleted."""
        return self.status == EntityStatus.DELETED


class AggregateRoot(BaseEntity, ABC):
    """
    Base aggregate root class for domain aggregates.
    
    Aggregate roots are the only entities that can be referenced from outside
    the aggregate boundary and are responsible for maintaining consistency.
    """
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def apply_event(self, event: DomainEvent) -> None:
        """Apply a domain event to the aggregate."""
        self.add_domain_event(event)
        self.version = event.version
        self.updated_at = event.occurred_at


class ValueObject(BaseModel, ABC):
    """
    Base value object class.
    
    Value objects are immutable objects that are defined by their attributes
    rather than their identity.
    """
    
    class Config:
        """Pydantic configuration for value objects."""
        frozen = True
        validate_assignment = True