"""
Roleplay router for the API Gateway.
Forwards roleplay-related requests to the appropriate services.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import httpx
import logging
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/sessions")
async def get_roleplay_sessions():
    """Get roleplay sessions from AI orchestration service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ai_service_url}/api/v1/roleplay/sessions",
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
        logger.error(f"Failed to forward roleplay request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Roleplay service unavailable"
        )