"""
User domain entities for the AI roleplay platform.

This module contains all user-related domain entities including User, UserProfile,
and UserPreferences with proper validation and business logic.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import Field, EmailStr, validator
from enum import Enum

from .base import AggregateRoot, ValueObject, EntityStatus, DomainEvent


class UserRole(str, Enum):
    """User role enumeration."""
    USER = "user"
    PREMIUM = "premium"
    MODERATOR = "moderator"
    ADMIN = "admin"


class SubscriptionTier(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class NotificationPreference(str, Enum):
    """Notification preference enumeration."""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    IN_APP = "in_app"


class UserRegisteredEvent(DomainEvent):
    """Event raised when a user registers."""
    event_type: str = "user.registered"
    email: EmailStr
    username: str


class UserProfileUpdatedEvent(DomainEvent):
    """Event raised when user profile is updated."""
    event_type: str = "user.profile.updated"
    fields_updated: List[str]


class UserPreferences(ValueObject):
    """User preferences value object."""
    
    language: str = Field(default="en", description="Preferred language")
    timezone: str = Field(default="UTC", description="User timezone")
    theme: str = Field(default="light", description="UI theme preference")
    notifications: List[NotificationPreference] = Field(
        default=[NotificationPreference.EMAIL, NotificationPreference.IN_APP],
        description="Notification preferences"
    )
    ai_interaction_style: str = Field(default="balanced", description="AI interaction style")
    content_rating: str = Field(default="general", description="Content rating preference")
    auto_save: bool = Field(default=True, description="Auto-save conversations")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        if len(v) != 2:
            raise ValueError("Language must be a 2-character ISO code")
        return v.lower()


class UserProfile(ValueObject):
    """User profile value object containing non-authentication data."""
    
    display_name: str = Field(..., min_length=1, max_length=50)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None)
    date_of_birth: Optional[datetime] = Field(None)
    location: Optional[str] = Field(None, max_length=100)
    website: Optional[str] = Field(None)
    social_links: Dict[str, str] = Field(default_factory=dict)
    
    @validator('social_links')
    def validate_social_links(cls, v):
        """Validate social links structure."""
        allowed_platforms = {'twitter', 'github', 'linkedin', 'discord'}
        for platform in v.keys():
            if platform not in allowed_platforms:
                raise ValueError(f"Unsupported social platform: {platform}")
        return v


class User(AggregateRoot):
    """
    User aggregate root representing a platform user.
    
    The User entity manages authentication, profile, preferences, and
    subscription information while maintaining consistency across the aggregate.
    """
    
    # Authentication fields
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr = Field(..., description="User email address")
    password_hash: str = Field(..., description="Hashed password")
    email_verified: bool = Field(default=False, description="Email verification status")
    phone: Optional[str] = Field(None, description="Phone number")
    phone_verified: bool = Field(default=False, description="Phone verification status")
    
    # Role and permissions
    role: UserRole = Field(default=UserRole.USER, description="User role")
    is_active: bool = Field(default=True, description="Account active status")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    login_count: int = Field(default=0, description="Total login count")
    
    # Subscription and billing
    subscription_tier: SubscriptionTier = Field(
        default=SubscriptionTier.FREE, 
        description="Current subscription tier"
    )
    subscription_expires_at: Optional[datetime] = Field(None, description="Subscription expiry")
    trial_ends_at: Optional[datetime] = Field(None, description="Trial period end")
    
    # Profile and preferences
    profile: UserProfile = Field(..., description="User profile information")
    preferences: UserPreferences = Field(
        default_factory=UserPreferences, 
        description="User preferences"
    )
    
    # Usage tracking
    total_conversations: int = Field(default=0, description="Total conversations count")
    total_messages: int = Field(default=0, description="Total messages count")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.is_new():
            self.add_domain_event(UserRegisteredEvent(
                aggregate_id=self.id,
                version=self.version,
                email=self.email,
                username=self.username
            ))
    
    def is_new(self) -> bool:
        """Check if this is a newly created user."""
        return self.version == 1 and not self.last_login
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v.lower()
    
    @validator('email')
    def validate_email_domain(cls, v):
        """Additional email validation."""
        # Add custom domain validation if needed
        return v.lower()
    
    def update_profile(self, profile_data: Dict[str, Any], updated_by: Optional[UUID] = None) -> None:
        """Update user profile with validation."""
        old_profile = self.profile.dict()
        new_profile_dict = {**old_profile, **profile_data}
        
        try:
            self.profile = UserProfile(**new_profile_dict)
            self.mark_as_modified(updated_by)
            
            # Emit profile updated event
            fields_updated = list(profile_data.keys())
            self.add_domain_event(UserProfileUpdatedEvent(
                aggregate_id=self.id,
                version=self.version,
                fields_updated=fields_updated
            ))
            
        except ValueError as e:
            raise ValueError(f"Invalid profile data: {e}")
    
    def update_preferences(self, preferences_data: Dict[str, Any]) -> None:
        """Update user preferences."""
        current_prefs = self.preferences.dict()
        new_prefs_dict = {**current_prefs, **preferences_data}
        
        try:
            self.preferences = UserPreferences(**new_prefs_dict)
            self.mark_as_modified()
        except ValueError as e:
            raise ValueError(f"Invalid preferences data: {e}")
    
    def record_login(self) -> None:
        """Record user login."""
        self.last_login = datetime.utcnow()
        self.login_count += 1
        self.last_activity = self.last_login
        self.mark_as_modified()
    
    def record_activity(self) -> None:
        """Record user activity."""
        self.last_activity = datetime.utcnow()
        # Don't increment version for activity updates
    
    def upgrade_subscription(self, tier: SubscriptionTier, expires_at: datetime) -> None:
        """Upgrade user subscription."""
        self.subscription_tier = tier
        self.subscription_expires_at = expires_at
        self.mark_as_modified()
    
    def is_premium(self) -> bool:
        """Check if user has premium subscription."""
        if self.subscription_tier in [SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE]:
            if self.subscription_expires_at and self.subscription_expires_at > datetime.utcnow():
                return True
        return False
    
    def can_access_feature(self, feature: str) -> bool:
        """Check if user can access a specific feature."""
        feature_map = {
            "unlimited_conversations": [SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE],
            "custom_characters": [SubscriptionTier.BASIC, SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE],
            "priority_support": [SubscriptionTier.PREMIUM, SubscriptionTier.ENTERPRISE],
            "advanced_ai_models": [SubscriptionTier.ENTERPRISE],
        }
        
        required_tiers = feature_map.get(feature, [])
        return self.subscription_tier in required_tiers or self.role in [UserRole.MODERATOR, UserRole.ADMIN]
    
    def increment_usage(self, conversations: int = 0, messages: int = 0) -> None:
        """Increment usage counters."""
        self.total_conversations += conversations
        self.total_messages += messages
        self.record_activity()