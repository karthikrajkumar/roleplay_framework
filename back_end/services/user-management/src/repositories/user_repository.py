"""
User repository implementation using SQLAlchemy.

This module provides concrete implementation of the user repository
interface for database operations using PostgreSQL.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import selectinload

from shared.interfaces.repository import IUserRepository
from shared.domain.entities import User, UserProfile, UserPreferences
from ..models.user_models import UserModel, UserPreferenceModel
from ..config.database import get_db_session
from ..mappers.user_mapper import UserMapper


class UserRepository(IUserRepository):
    """
    SQLAlchemy-based implementation of user repository.
    
    Provides database operations for user entities with proper
    mapping between domain entities and database models.
    """
    
    def __init__(self):
        self.mapper = UserMapper()
    
    async def get_by_id(self, entity_id: UUID) -> Optional[User]:
        """Get user by ID."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.id == entity_id)
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return await self.mapper.model_to_entity(user_model)
            return None
    
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.status != "deleted")
                .order_by(UserModel.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            user_models = result.scalars().all()
            
            return [await self.mapper.model_to_entity(model) for model in user_models]
    
    async def create(self, entity: User) -> User:
        """Create a new user."""
        async with get_db_session() as session:
            user_model = await self.mapper.entity_to_model(entity)
            session.add(user_model)
            await session.flush()
            await session.refresh(user_model)
            
            return await self.mapper.model_to_entity(user_model)
    
    async def update(self, entity: User) -> User:
        """Update an existing user."""
        async with get_db_session() as session:
            # Get existing model
            stmt = select(UserModel).where(UserModel.id == entity.id)
            result = await session.execute(stmt)
            existing_model = result.scalar_one_or_none()
            
            if not existing_model:
                raise ValueError(f"User with ID {entity.id} not found")
            
            # Update model with entity data
            updated_model = await self.mapper.entity_to_model(entity, existing_model)
            await session.flush()
            await session.refresh(updated_model)
            
            return await self.mapper.model_to_entity(updated_model)
    
    async def delete(self, entity_id: UUID) -> bool:
        """Soft delete user by ID."""
        async with get_db_session() as session:
            stmt = (
                update(UserModel)
                .where(UserModel.id == entity_id)
                .values(
                    status="deleted",
                    updated_at=datetime.utcnow()
                )
            )
            result = await session.execute(stmt)
            return result.rowcount > 0
    
    async def exists(self, entity_id: UUID) -> bool:
        """Check if user exists."""
        async with get_db_session() as session:
            stmt = (
                select(func.count(UserModel.id))
                .where(UserModel.id == entity_id)
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            count = result.scalar()
            return count > 0
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count users with optional filters."""
        async with get_db_session() as session:
            stmt = select(func.count(UserModel.id)).where(UserModel.status != "deleted")
            
            if filters:
                stmt = self._apply_filters(stmt, filters)
            
            result = await session.execute(stmt)
            return result.scalar()
    
    async def find_by_criteria(
        self, 
        filters: Dict[str, Any], 
        offset: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """Find users by criteria with pagination."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.status != "deleted")
                .offset(offset)
                .limit(limit)
            )
            
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(UserModel.created_at.desc())
            
            result = await session.execute(stmt)
            user_models = result.scalars().all()
            
            return [await self.mapper.model_to_entity(model) for model in user_models]
    
    # User-specific methods
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.email.ilike(email))
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return await self.mapper.model_to_entity(user_model)
            return None
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.username.ilike(username))
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            user_model = result.scalar_one_or_none()
            
            if user_model:
                return await self.mapper.model_to_entity(user_model)
            return None
    
    async def email_exists(self, email: str) -> bool:
        """Check if email is already registered."""
        async with get_db_session() as session:
            stmt = (
                select(func.count(UserModel.id))
                .where(UserModel.email.ilike(email))
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            count = result.scalar()
            return count > 0
    
    async def username_exists(self, username: str) -> bool:
        """Check if username is already taken."""
        async with get_db_session() as session:
            stmt = (
                select(func.count(UserModel.id))
                .where(UserModel.username.ilike(username))
                .where(UserModel.status != "deleted")
            )
            result = await session.execute(stmt)
            count = result.scalar()
            return count > 0
    
    async def get_by_subscription_tier(self, tier: str) -> List[User]:
        """Get users by subscription tier."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.subscription_tier == tier)
                .where(UserModel.status == "active")
                .order_by(UserModel.created_at.desc())
            )
            result = await session.execute(stmt)
            user_models = result.scalars().all()
            
            return [await self.mapper.model_to_entity(model) for model in user_models]
    
    async def get_active_users_since(self, since_date: datetime) -> List[User]:
        """Get users active since a specific date."""
        async with get_db_session() as session:
            stmt = (
                select(UserModel)
                .where(UserModel.last_activity >= since_date)
                .where(UserModel.status == "active")
                .order_by(UserModel.last_activity.desc())
            )
            result = await session.execute(stmt)
            user_models = result.scalars().all()
            
            return [await self.mapper.model_to_entity(model) for model in user_models]
    
    async def update_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp."""
        async with get_db_session() as session:
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(
                    last_login=datetime.utcnow(),
                    login_count=UserModel.login_count + 1,
                    updated_at=datetime.utcnow()
                )
            )
            await session.execute(stmt)
    
    async def update_activity(self, user_id: UUID) -> None:
        """Update user's last activity timestamp."""
        async with get_db_session() as session:
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(last_activity=datetime.utcnow())
            )
            await session.execute(stmt)
    
    async def increment_usage(self, user_id: UUID, conversations: int = 0, messages: int = 0) -> None:
        """Increment user usage counters."""
        async with get_db_session() as session:
            update_values = {"last_activity": datetime.utcnow()}
            
            if conversations > 0:
                update_values["total_conversations"] = UserModel.total_conversations + conversations
            
            if messages > 0:
                update_values["total_messages"] = UserModel.total_messages + messages
            
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(**update_values)
            )
            await session.execute(stmt)
    
    async def search_users(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 20
    ) -> List[User]:
        """Search users by username, email, or display name."""
        async with get_db_session() as session:
            search_term = f"%{query}%"
            
            stmt = (
                select(UserModel)
                .where(UserModel.status != "deleted")
                .where(
                    or_(
                        UserModel.username.ilike(search_term),
                        UserModel.email.ilike(search_term),
                        UserModel.profile_data["display_name"].astext.ilike(search_term)
                    )
                )
                .offset(offset)
                .limit(limit)
                .order_by(UserModel.username)
            )
            
            if filters:
                stmt = self._apply_filters(stmt, filters)
            
            result = await session.execute(stmt)
            user_models = result.scalars().all()
            
            return [await self.mapper.model_to_entity(model) for model in user_models]
    
    def _apply_filters(self, stmt, filters: Dict[str, Any]):
        """Apply filters to SQLAlchemy statement."""
        for key, value in filters.items():
            if key == "role":
                stmt = stmt.where(UserModel.role == value)
            elif key == "subscription_tier":
                stmt = stmt.where(UserModel.subscription_tier == value)
            elif key == "is_active":
                stmt = stmt.where(UserModel.is_active == value)
            elif key == "created_after":
                stmt = stmt.where(UserModel.created_at >= value)
            elif key == "created_before":
                stmt = stmt.where(UserModel.created_at <= value)
            elif key == "last_activity_after":
                stmt = stmt.where(UserModel.last_activity >= value)
            elif key == "email_verified":
                stmt = stmt.where(UserModel.email_verified == value)
        
        return stmt