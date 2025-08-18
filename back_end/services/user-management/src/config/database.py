"""
Database configuration and connection management for User Management service.

This module handles database initialization, connection pooling,
and provides database session management for PostgreSQL.
"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
import logging

from .settings import get_settings

logger = logging.getLogger(__name__)

# Database engine and session
engine = None
async_session_maker = None


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )


async def init_database():
    """Initialize database connection and create tables."""
    global engine, async_session_maker
    
    settings = get_settings()
    
    # Construct database URL dynamically as fallback
    db_url = settings.database_url
    print(f"Original database URL: {db_url.replace(settings.db_password if settings.db_password in db_url else 'XXX', '***')}")
    
    # Always use constructed URL to ensure consistency
    db_url = f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    print(f"Using constructed database URL: {db_url.replace(settings.db_password, '***')}")
    
    # Test basic connection first
    try:
        import asyncpg
        test_conn = await asyncpg.connect(
            user=settings.db_user,
            password=settings.db_password, 
            database=settings.db_name,
            host=settings.db_host,
            port=settings.db_port
        )
        await test_conn.close()
        print("✓ Basic asyncpg connection test successful")
    except Exception as e:
        print(f"✗ Basic asyncpg connection failed: {e}")
        raise
    
    # Create async engine
    engine = create_async_engine(
        db_url,
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    # Create session maker
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Import all models to ensure they're registered
    from ..models import user_models
    
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database():
    """Close database connections."""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.
    
    Usage in FastAPI endpoints:
    async def endpoint(db: AsyncSession = Depends(get_db_session)):
        pass
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized")
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class DatabaseManager:
    """Database manager for handling connections and transactions."""
    
    @staticmethod
    async def execute_transaction(operation):
        """Execute operation in a transaction."""
        async with async_session_maker() as session:
            try:
                result = await operation(session)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise
    
    @staticmethod
    async def execute_query(query_func):
        """Execute read-only query."""
        async with async_session_maker() as session:
            return await query_func(session)
    
    @staticmethod
    async def health_check() -> bool:
        """Check database connectivity."""
        try:
            async with async_session_maker() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False