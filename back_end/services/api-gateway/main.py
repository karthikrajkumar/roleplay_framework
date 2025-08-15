"""
API Gateway service entry point.

This is the main entry point for the API Gateway microservice,
which handles request routing, authentication, rate limiting,
and cross-cutting concerns for the AI Roleplay Platform.
"""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging

from src.config.settings import get_settings
from src.middleware.auth import AuthMiddleware
from src.middleware.rate_limiting import RateLimitMiddleware
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.logging import LoggingMiddleware
from src.routers import health, auth, users, roleplay, characters, scenarios, ai
from src.services.service_discovery import ServiceDiscovery
from src.utils.logging_config import setup_logging


# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    
    # Startup
    logger.info("Starting API Gateway...")
    
    # Initialize service discovery
    service_discovery = ServiceDiscovery(settings.consul_url)
    await service_discovery.register_service(
        name="api-gateway",
        port=settings.port,
        health_check_url=f"http://localhost:{settings.port}/health"
    )
    
    # Store in app state
    app.state.service_discovery = service_discovery
    
    logger.info("API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    
    # Deregister from service discovery
    await service_discovery.deregister_service("api-gateway")
    
    logger.info("API Gateway shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Roleplay Platform API Gateway",
        description="Central API gateway for the AI roleplay platform microservices",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters!)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    app.include_router(roleplay.router, prefix="/api/v1/roleplay", tags=["roleplay"])
    app.include_router(characters.router, prefix="/api/v1/characters", tags=["characters"])
    app.include_router(scenarios.router, prefix="/api/v1/scenarios", tags=["scenarios"])
    app.include_router(ai.router, prefix="/api/v1/ai", tags=["ai"])
    
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
        log_level="info",
        access_log=True
    )