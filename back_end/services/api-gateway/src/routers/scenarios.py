"""
Scenarios router for the API Gateway.
"""

from fastapi import APIRouter, HTTPException, status
import httpx
import logging
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def get_scenarios():
    """Get scenarios list."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ai_service_url}/api/v1/scenarios",
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Scenario service error"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to get scenarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scenario service unavailable"
        )