"""
Roleplay domain entities for the AI roleplay platform.

This module contains all roleplay-related domain entities including sessions,
characters, scenarios, and messages.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pydantic import Field, validator
from enum import Enum

from .base import AggregateRoot, ValueObject, EntityStatus, DomainEvent


class MessageType(str, Enum):
    """Message type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SYSTEM = "system"
    ACTION = "action"


class SessionStatus(str, Enum):
    """Roleplay session status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class CharacterType(str, Enum):
    """Character type enumeration."""
    USER_CREATED = "user_created"
    TEMPLATE = "template"
    AI_GENERATED = "ai_generated"
    COMMUNITY = "community"


class ScenarioCategory(str, Enum):
    """Scenario category enumeration."""
    FANTASY = "fantasy"
    SCI_FI = "sci_fi"
    MODERN = "modern"
    HISTORICAL = "historical"
    ROMANCE = "romance"
    ADVENTURE = "adventure"
    MYSTERY = "mystery"
    HORROR = "horror"
    EDUCATIONAL = "educational"
    CUSTOM = "custom"


class ContentRating(str, Enum):
    """Content rating enumeration."""
    GENERAL = "general"
    TEEN = "teen"
    MATURE = "mature"
    ADULT = "adult"


class SessionStartedEvent(DomainEvent):
    """Event raised when a roleplay session starts."""
    event_type: str = "session.started"
    scenario_id: UUID
    character_ids: List[UUID]


class MessageSentEvent(DomainEvent):
    """Event raised when a message is sent."""
    event_type: str = "message.sent"
    session_id: UUID
    sender_type: str
    message_type: MessageType


class CharacterPersonality(ValueObject):
    """Character personality traits value object."""
    
    traits: List[str] = Field(default_factory=list, description="Personality traits")
    goals: List[str] = Field(default_factory=list, description="Character goals")
    fears: List[str] = Field(default_factory=list, description="Character fears")
    speech_style: str = Field(default="neutral", description="Speaking style")
    mannerisms: List[str] = Field(default_factory=list, description="Character mannerisms")
    
    @validator('traits', 'goals', 'fears', 'mannerisms')
    def validate_lists(cls, v):
        """Validate list length."""
        if len(v) > 10:
            raise ValueError("Lists cannot exceed 10 items")
        return v


class CharacterAppearance(ValueObject):
    """Character appearance description value object."""
    
    age: Optional[str] = Field(None, description="Character age")
    height: Optional[str] = Field(None, description="Character height")
    build: Optional[str] = Field(None, description="Character build")
    hair: Optional[str] = Field(None, description="Hair description")
    eyes: Optional[str] = Field(None, description="Eye description")
    skin: Optional[str] = Field(None, description="Skin description")
    clothing: Optional[str] = Field(None, description="Typical clothing")
    distinguishing_features: List[str] = Field(default_factory=list, description="Unique features")


class Message(ValueObject):
    """Message value object representing a single conversation message."""
    
    id: UUID = Field(..., description="Message ID")
    session_id: UUID = Field(..., description="Session this message belongs to")
    sender_id: Optional[UUID] = Field(None, description="User ID if human sender")
    sender_type: str = Field(..., description="Type of sender (user, ai, system)")
    sender_name: str = Field(..., description="Display name of sender")
    
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message data")
    
    timestamp: datetime = Field(..., description="Message timestamp")
    edited: bool = Field(default=False, description="Whether message was edited")
    edited_at: Optional[datetime] = Field(None, description="Edit timestamp")
    
    # AI-specific fields
    ai_model: Optional[str] = Field(None, description="AI model used")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    generation_time_ms: Optional[int] = Field(None, description="Generation time in milliseconds")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate message content."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        if len(v) > 10000:
            raise ValueError("Message content too long")
        return v


class Character(AggregateRoot):
    """
    Character aggregate representing an AI character in roleplay scenarios.
    
    Characters can be user-created, templates, or generated by AI.
    They contain personality, appearance, and behavior definitions.
    """
    
    name: str = Field(..., min_length=1, max_length=100, description="Character name")
    description: str = Field(..., max_length=1000, description="Character description")
    
    character_type: CharacterType = Field(default=CharacterType.USER_CREATED, description="Character type")
    creator_id: Optional[UUID] = Field(None, description="Creator user ID")
    
    # Character definition
    personality: CharacterPersonality = Field(..., description="Personality traits")
    appearance: CharacterAppearance = Field(..., description="Physical appearance")
    background: str = Field(default="", max_length=2000, description="Character background")
    
    # Behavior settings
    system_prompt: str = Field(..., max_length=5000, description="AI system prompt")
    example_dialogue: List[str] = Field(default_factory=list, description="Example conversations")
    forbidden_topics: List[str] = Field(default_factory=list, description="Topics to avoid")
    
    # Metadata
    content_rating: ContentRating = Field(default=ContentRating.GENERAL, description="Content rating")
    tags: List[str] = Field(default_factory=list, description="Character tags")
    is_public: bool = Field(default=False, description="Available to other users")
    usage_count: int = Field(default=0, description="Times used in sessions")
    rating: float = Field(default=0.0, ge=0, le=5, description="User rating")
    
    # AI settings
    temperature: float = Field(default=0.8, ge=0, le=2, description="AI creativity level")
    max_tokens: int = Field(default=500, ge=1, le=2000, description="Max response length")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate character name."""
        if not v.strip():
            raise ValueError("Character name cannot be empty")
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags."""
        if len(v) > 20:
            raise ValueError("Cannot have more than 20 tags")
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    def increment_usage(self) -> None:
        """Increment usage counter."""
        self.usage_count += 1
        self.mark_as_modified()
    
    def update_rating(self, new_rating: float) -> None:
        """Update character rating."""
        if 0 <= new_rating <= 5:
            self.rating = new_rating
            self.mark_as_modified()


class Scenario(AggregateRoot):
    """
    Scenario aggregate representing a roleplay scenario/setting.
    
    Scenarios define the context, setting, and initial conditions
    for roleplay sessions.
    """
    
    title: str = Field(..., min_length=1, max_length=200, description="Scenario title")
    description: str = Field(..., max_length=2000, description="Scenario description")
    
    category: ScenarioCategory = Field(..., description="Scenario category")
    content_rating: ContentRating = Field(default=ContentRating.GENERAL, description="Content rating")
    
    # Scenario content
    setting: str = Field(..., max_length=1000, description="Setting description")
    initial_situation: str = Field(..., max_length=1000, description="Initial situation")
    suggested_characters: List[UUID] = Field(default_factory=list, description="Recommended characters")
    
    # Metadata
    creator_id: Optional[UUID] = Field(None, description="Creator user ID")
    is_public: bool = Field(default=False, description="Available to other users")
    tags: List[str] = Field(default_factory=list, description="Scenario tags")
    usage_count: int = Field(default=0, description="Times used")
    rating: float = Field(default=0.0, ge=0, le=5, description="User rating")
    
    # Configuration
    max_participants: int = Field(default=2, ge=1, le=10, description="Max participants")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags."""
        if len(v) > 15:
            raise ValueError("Cannot have more than 15 tags")
        return [tag.lower().strip() for tag in v if tag.strip()]
    
    def increment_usage(self) -> None:
        """Increment usage counter."""
        self.usage_count += 1
        self.mark_as_modified()


class RoleplaySession(AggregateRoot):
    """
    Roleplay session aggregate root managing an active roleplay conversation.
    
    Sessions coordinate between users, AI characters, and scenarios to
    create immersive roleplay experiences.
    """
    
    title: str = Field(..., min_length=1, max_length=200, description="Session title")
    
    # Participants
    user_id: UUID = Field(..., description="Primary user")
    scenario_id: Optional[UUID] = Field(None, description="Scenario being used")
    character_ids: List[UUID] = Field(default_factory=list, description="Active characters")
    
    # Session state
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    started_at: datetime = Field(..., description="Session start time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    ended_at: Optional[datetime] = Field(None, description="Session end time")
    
    # Content
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    summary: Optional[str] = Field(None, max_length=1000, description="Session summary")
    
    # Settings
    content_rating: ContentRating = Field(default=ContentRating.GENERAL, description="Content rating")
    auto_save: bool = Field(default=True, description="Auto-save enabled")
    
    # Analytics
    total_messages: int = Field(default=0, description="Total message count")
    user_messages: int = Field(default=0, description="User message count")
    ai_messages: int = Field(default=0, description="AI message count")
    total_tokens: int = Field(default=0, description="Total tokens used")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not hasattr(self, 'started_at'):
            self.started_at = datetime.utcnow()
        if not hasattr(self, 'last_activity'):
            self.last_activity = self.started_at
        
        if self.is_new():
            self.add_domain_event(SessionStartedEvent(
                aggregate_id=self.id,
                version=self.version,
                scenario_id=self.scenario_id,
                character_ids=self.character_ids
            ))
    
    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.total_messages += 1
        self.last_activity = message.timestamp
        
        if message.sender_type == "user":
            self.user_messages += 1
        elif message.sender_type == "ai":
            self.ai_messages += 1
            if message.tokens_used:
                self.total_tokens += message.tokens_used
        
        self.mark_as_modified()
        
        # Emit message sent event
        self.add_domain_event(MessageSentEvent(
            aggregate_id=self.id,
            version=self.version,
            session_id=self.id,
            sender_type=message.sender_type,
            message_type=message.message_type
        ))
    
    def pause(self) -> None:
        """Pause the session."""
        if self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.PAUSED
            self.mark_as_modified()
    
    def resume(self) -> None:
        """Resume the session."""
        if self.status == SessionStatus.PAUSED:
            self.status = SessionStatus.ACTIVE
            self.last_activity = datetime.utcnow()
            self.mark_as_modified()
    
    def end(self, summary: Optional[str] = None) -> None:
        """End the session."""
        if self.status in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            self.status = SessionStatus.COMPLETED
            self.ended_at = datetime.utcnow()
            if summary:
                self.summary = summary
            self.mark_as_modified()
    
    def archive(self) -> None:
        """Archive the session."""
        if self.status == SessionStatus.COMPLETED:
            self.status = SessionStatus.ARCHIVED
            self.mark_as_modified()
    
    def get_duration_minutes(self) -> Optional[int]:
        """Get session duration in minutes."""
        if self.ended_at:
            delta = self.ended_at - self.started_at
            return int(delta.total_seconds() / 60)
        return None
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE
    
    def get_last_messages(self, count: int = 10) -> List[Message]:
        """Get the last N messages."""
        return self.messages[-count:] if len(self.messages) >= count else self.messages