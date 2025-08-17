"""
Users router for the API Gateway.
Forwards user-related requests to the user management service.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import httpx
import logging
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Forward get user request to user management service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.user_service_url}/api/v1/users/{user_id}",
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json()
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to forward user request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service unavailable"
        )