"""
User mapper for converting between User domain entities and database models.

This module handles bidirectional mapping between the User domain entity
and UserModel SQLAlchemy model, ensuring proper data transformation.
"""

from typing import Optional
from datetime import datetime

from shared.domain.entities import User, UserProfile, UserPreferences, UserRole, SubscriptionTier
from ..models.user_models import UserModel


class UserMapper:
    """
    Mapper class for User entity and UserModel conversions.
    
    Handles the conversion between domain entities and database models,
    ensuring proper data transformation and validation.
    """
    
    async def model_to_entity(self, model: UserModel) -> User:
        """Convert UserModel to User domain entity."""
        
        # Extract profile data
        profile_data = model.profile_data or {}
        profile = UserProfile(
            display_name=profile_data.get("display_name", model.username),
            bio=profile_data.get("bio"),
            avatar_url=profile_data.get("avatar_url"),
            date_of_birth=self._parse_datetime(profile_data.get("date_of_birth")),
            location=profile_data.get("location"),
            website=profile_data.get("website"),
            social_links=profile_data.get("social_links", {})
        )
        
        # Extract preferences
        preferences_data = model.preferences or {}
        preferences = UserPreferences(
            language=preferences_data.get("language", "en"),
            timezone=preferences_data.get("timezone", "UTC"),
            theme=preferences_data.get("theme", "light"),
            notifications=preferences_data.get("notifications", ["email", "in_app"]),
            ai_interaction_style=preferences_data.get("ai_interaction_style", "balanced"),
            content_rating=preferences_data.get("content_rating", "general"),
            auto_save=preferences_data.get("auto_save", True)
        )
        
        # Create User entity
        user = User(
            id=model.id,
            username=model.username,
            email=model.email,
            password_hash=model.password_hash,
            email_verified=model.email_verified,
            phone=model.phone,
            phone_verified=model.phone_verified,
            role=UserRole(model.role),
            is_active=model.is_active,
            last_login=model.last_login,
            login_count=model.login_count,
            subscription_tier=SubscriptionTier(model.subscription_tier),
            subscription_expires_at=model.subscription_expires_at,
            trial_ends_at=model.trial_ends_at,
            profile=profile,
            preferences=preferences,
            total_conversations=model.total_conversations,
            total_messages=model.total_messages,
            last_activity=model.last_activity,
            created_at=model.created_at,
            updated_at=model.updated_at,
            created_by=model.created_by,
            updated_by=model.updated_by,
            version=model.version,
            status=model.status
        )
        
        return user
    
    async def entity_to_model(self, entity: User, existing_model: Optional[UserModel] = None) -> UserModel:
        """Convert User domain entity to UserModel."""
        
        # Prepare profile data
        profile_data = {
            "display_name": entity.profile.display_name,
            "bio": entity.profile.bio,
            "avatar_url": entity.profile.avatar_url,
            "date_of_birth": entity.profile.date_of_birth.isoformat() if entity.profile.date_of_birth else None,
            "location": entity.profile.location,
            "website": entity.profile.website,
            "social_links": entity.profile.social_links
        }
        
        # Prepare preferences data
        preferences_data = {
            "language": entity.preferences.language,
            "timezone": entity.preferences.timezone,
            "theme": entity.preferences.theme,
            "notifications": entity.preferences.notifications,
            "ai_interaction_style": entity.preferences.ai_interaction_style,
            "content_rating": entity.preferences.content_rating,
            "auto_save": entity.preferences.auto_save
        }
        
        # Model data
        model_data = {
            "username": entity.username,
            "email": entity.email,
            "password_hash": entity.password_hash,
            "email_verified": entity.email_verified,
            "phone": entity.phone,
            "phone_verified": entity.phone_verified,
            "role": entity.role.value,
            "is_active": entity.is_active,
            "last_login": entity.last_login,
            "login_count": entity.login_count,
            "subscription_tier": entity.subscription_tier.value,
            "subscription_expires_at": entity.subscription_expires_at,
            "trial_ends_at": entity.trial_ends_at,
            "profile_data": profile_data,
            "preferences": preferences_data,
            "total_conversations": entity.total_conversations,
            "total_messages": entity.total_messages,
            "last_activity": entity.last_activity,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
            "created_by": entity.created_by,
            "updated_by": entity.updated_by,
            "version": entity.version,
            "status": entity.status.value
        }
        
        if existing_model:
            # Update existing model
            for key, value in model_data.items():
                setattr(existing_model, key, value)
            existing_model.id = entity.id  # Ensure ID is set
            return existing_model
        else:
            # Create new model
            model_data["id"] = entity.id
            return UserModel(**model_data)
    
    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse datetime from string or return None."""
        if not value:
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return None
        
        return None