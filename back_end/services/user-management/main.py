"""
User Management service entry point.

This microservice handles user authentication, registration, profiles,
and user-related operations for the AI Roleplay Platform.
"""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging

from src.config.settings import get_settings
from src.config.database import init_database, close_database
from src.middleware.logging import LoggingMiddleware
from src.middleware.request_id import RequestIDMiddleware
from src.routers import health, users, auth, profiles
from src.services.user_service import UserService
from src.repositories.user_repository import UserRepository
from src.utils.dependency_injection import Container
from src.utils.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    
    # Startup
    logger.info("Starting User Management service...")
    
    # Initialize database
    await init_database()
    
    # Setup dependency injection
    await setup_dependencies()
    
    logger.info("User Management service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down User Management service...")
    
    # Close database connections
    await close_database()
    
    logger.info("User Management service shutdown complete")


async def setup_dependencies():
    """Setup dependency injection container."""
    from shared.interfaces.repository import IUserRepository
    from shared.interfaces.services import IUserService
    
    # Register repositories
    Container.register_singleton(IUserRepository, UserRepository)
    
    # Register services
    Container.register_singleton(IUserService, UserService)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="User Management Service",
        description="User authentication and profile management for AI Roleplay Platform",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(auth.router, prefix="/auth", tags=["authentication"])
    app.include_router(users.router, prefix="/users", tags=["users"])
    app.include_router(profiles.router, prefix="/profiles", tags=["profiles"])
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )
    
    return app


# Create the app
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )