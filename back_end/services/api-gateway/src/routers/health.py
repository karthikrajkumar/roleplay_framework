"""
Health check router for the API Gateway.

This router provides health check endpoints for monitoring
service availability and downstream service status.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from datetime import datetime
import httpx
import asyncio
import logging

from ..config.settings import get_settings
from ..services.service_discovery import ServiceDiscovery

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", summary="Basic health check")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "api-gateway",
        "version": "1.0.0"
    }


@router.get("/detailed", summary="Detailed health check")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including downstream services."""
    settings = get_settings()
    
    # Check downstream services
    services = {
        "user-service": settings.user_service_url,
        "roleplay-service": settings.roleplay_service_url,
        "ai-service": settings.ai_service_url,
        "notification-service": settings.notification_service_url,
        "analytics-service": settings.analytics_service_url
    }
    
    service_health = {}
    overall_healthy = True
    
    # Check each service health
    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = []
        for service_name, service_url in services.items():
            tasks.append(_check_service_health(client, service_name, f"{service_url}/health"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (service_name, _) in enumerate(services.items()):
            result = results[i]
            if isinstance(result, Exception):
                service_health[service_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
                overall_healthy = False
            else:
                service_health[service_name] = result
                if result["status"] != "healthy":
                    overall_healthy = False
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "api-gateway",
        "version": "1.0.0",
        "dependencies": service_health
    }


@router.get("/readiness", summary="Readiness probe")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    settings = get_settings()
    
    # Check critical dependencies
    try:
        # Check Redis connectivity
        from ..utils.cache_manager import CacheManager
        cache = CacheManager()
        await cache.ping()
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/liveness", summary="Liveness probe")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _check_service_health(client: httpx.AsyncClient, service_name: str, health_url: str) -> Dict[str, Any]:
    """Check health of a single service."""
    try:
        response = await client.get(health_url)
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
        else:
            return {
                "status": "unhealthy",
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
    
    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "error": "timeout"
        }
    except httpx.ConnectError:
        return {
            "status": "unhealthy",
            "error": "connection_failed"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }