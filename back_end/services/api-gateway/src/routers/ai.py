"""
AI router for the API Gateway.
Forwards AI-related requests to the AI service.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import httpx
import logging
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate")
async def generate_ai_response(request_data: Dict[str, Any]):
    """Forward AI generation request to AI service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.ai_service_url}/api/v1/ai/generate",
                json=request_data,
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="AI service error"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to forward AI request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable"
        )


@router.post("/chat")
async def ai_chat(chat_data: Dict[str, Any]):
    """Forward AI chat request to AI service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.ai_service_url}/api/v1/ai/chat",
                json=chat_data,
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="AI chat service error"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to forward AI chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI chat service unavailable"
        )


@router.get("/models")
async def get_ai_models():
    """Get available AI models from AI service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ai_service_url}/api/v1/ai/models",
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="AI models service error"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to get AI models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI models service unavailable"
        )