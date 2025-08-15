"""
SQLAlchemy models for user management.

This module contains database models for users, profiles, and authentication
using SQLAlchemy ORM with PostgreSQL-specific features.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Boolean, DateTime, Integer, Float, Text, JSON, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from uuid import uuid4
import uuid

from ..config.database import Base


class UserModel(Base):
    """
    SQLAlchemy model for users table.
    
    Maps to the User domain entity for persistence.
    """
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4,
        index=True
    )
    
    # Authentication fields
    username: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    phone_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Role and permissions
    role: Mapped[str] = mapped_column(String(20), default="user", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Subscription and billing
    subscription_tier: Mapped[str] = mapped_column(String(20), default="free", index=True)
    subscription_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    trial_ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Profile data (JSON for flexibility)
    profile_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    
    # Usage tracking
    total_conversations: Mapped[int] = mapped_column(Integer, default=0)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)
    last_activity: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    
    # Relationships
    auth_tokens: Mapped[list["AuthTokenModel"]] = relationship(
        "AuthTokenModel", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    user_sessions: Mapped[list["UserSessionModel"]] = relationship(
        "UserSessionModel",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_users_email_active", email, is_active),
        Index("idx_users_subscription", subscription_tier, subscription_expires_at),
        Index("idx_users_activity", last_activity.desc()),
        CheckConstraint(
            "role IN ('user', 'premium', 'moderator', 'admin')",
            name="ck_users_role"
        ),
        CheckConstraint(
            "subscription_tier IN ('free', 'basic', 'premium', 'enterprise')",
            name="ck_users_subscription_tier"
        ),
        CheckConstraint(
            "status IN ('active', 'inactive', 'deleted', 'pending')",
            name="ck_users_status"
        )
    )


class AuthTokenModel(Base):
    """
    SQLAlchemy model for authentication tokens.
    
    Stores JWT refresh tokens and their metadata.
    """
    __tablename__ = "auth_tokens"
    
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True
    )
    
    # Token data
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    token_type: Mapped[str] = mapped_column(String(20), default="refresh")
    
    # Token metadata
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Client information
    client_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Relationships
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="auth_tokens")
    
    # Indexes
    __table_args__ = (
        Index("idx_auth_tokens_user_active", user_id, is_active),
        Index("idx_auth_tokens_expires", expires_at),
        CheckConstraint(
            "token_type IN ('access', 'refresh', 'reset', 'verification')",
            name="ck_auth_tokens_type"
        )
    )


class UserSessionModel(Base):
    """
    SQLAlchemy model for user sessions.
    
    Tracks active user sessions for security and analytics.
    """
    __tablename__ = "user_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True
    )
    
    # Session data
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    
    # Session metadata
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    
    # Client information
    client_ip: Mapped[str] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    location_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Usage tracking
    requests_count: Mapped[int] = mapped_column(Integer, default=0)
    data_transferred: Mapped[int] = mapped_column(Integer, default=0)  # bytes
    
    # Relationships
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="user_sessions")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_sessions_user_active", user_id, is_active),
        Index("idx_user_sessions_activity", last_activity.desc()),
        Index("idx_user_sessions_expires", expires_at),
        CheckConstraint(
            "end_reason IS NULL OR end_reason IN ('logout', 'timeout', 'expired', 'revoked')",
            name="ck_user_sessions_end_reason"
        )
    )


class UserPreferenceModel(Base):
    """
    SQLAlchemy model for user preferences.
    
    Stores user-specific settings and preferences.
    """
    __tablename__ = "user_preferences"
    
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True
    )
    
    # General preferences
    language: Mapped[str] = mapped_column(String(5), default="en")
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    theme: Mapped[str] = mapped_column(String(20), default="light")
    
    # Notification preferences
    email_notifications: Mapped[bool] = mapped_column(Boolean, default=True)
    push_notifications: Mapped[bool] = mapped_column(Boolean, default=True)
    sms_notifications: Mapped[bool] = mapped_column(Boolean, default=False)
    in_app_notifications: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # AI interaction preferences
    ai_interaction_style: Mapped[str] = mapped_column(String(20), default="balanced")
    content_rating: Mapped[str] = mapped_column(String(20), default="general")
    auto_save: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Privacy preferences
    profile_visibility: Mapped[str] = mapped_column(String(20), default="friends")
    share_analytics: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Custom preferences (flexible JSON)
    custom_preferences: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow
    )
    
    # Indexes
    __table_args__ = (
        CheckConstraint(
            "theme IN ('light', 'dark', 'auto')",
            name="ck_user_preferences_theme"
        ),
        CheckConstraint(
            "ai_interaction_style IN ('conservative', 'balanced', 'creative')",
            name="ck_user_preferences_ai_style"
        ),
        CheckConstraint(
            "content_rating IN ('general', 'teen', 'mature', 'adult')",
            name="ck_user_preferences_content_rating"
        ),
        CheckConstraint(
            "profile_visibility IN ('public', 'friends', 'private')",
            name="ck_user_preferences_visibility"
        )
    )