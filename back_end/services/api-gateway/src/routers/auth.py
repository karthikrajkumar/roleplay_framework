"""
Authentication router for the API Gateway.

This module handles authentication-related endpoints including login,
logout, token refresh, and user registration forwarding.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Dict, Any
import httpx
import logging

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/login")
async def login(credentials: Dict[str, Any]):
    """Forward login request to user management service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.user_service_url}/api/v1/auth/login",
                json=credentials,
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
        logger.error(f"Failed to forward login request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )


@router.post("/logout")
async def logout(token: str = Depends(security)):
    """Forward logout request to user management service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.user_service_url}/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {token.credentials}"},
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
        logger.error(f"Failed to forward logout request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )


@router.post("/refresh")
async def refresh_token(token_data: Dict[str, Any]):
    """Forward token refresh request to user management service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.user_service_url}/api/v1/auth/refresh",
                json=token_data,
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
        logger.error(f"Failed to forward refresh request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )


@router.post("/register")
async def register(user_data: Dict[str, Any]):
    """Forward user registration request to user management service."""
    settings = get_settings()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.user_service_url}/api/v1/auth/register",
                json=user_data,
                timeout=settings.request_timeout_seconds
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json()
                )
                
    except httpx.RequestError as e:
        logger.error(f"Failed to forward registration request: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )