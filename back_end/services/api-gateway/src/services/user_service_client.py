"""
User Service Client for API Gateway.

This module provides a client for communicating with the User Management service
to validate users, check permissions, and manage authentication.
"""

import httpx
import logging
from typing import Optional, Dict, Any
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class UserServiceClient:
    """Client for User Management service communication."""
    
    def __init__(self):
        """Initialize the user service client."""
        self.settings = get_settings()
        self.base_url = self.settings.user_service_url
        self.timeout = self.settings.request_timeout_seconds
        
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token with the user service.
        
        Args:
            token: JWT token to validate
            
        Returns:
            User data if token is valid, None otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/validate",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"token": token}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Token validation failed: {response.status_code}")
                    return None
                    
        except httpx.RequestError as e:
            logger.error(f"Failed to validate token with user service: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by ID.
        
        Args:
            user_id: User ID to lookup
            
        Returns:
            User data if found, None otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v1/users/{user_id}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"User lookup failed: {response.status_code}")
                    return None
                    
        except httpx.RequestError as e:
            logger.error(f"Failed to get user from user service: {e}")
            return None
    
    async def check_permissions(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission for a specific resource and action.
        
        Args:
            user_id: User ID to check
            resource: Resource name
            action: Action name (read, write, delete, etc.)
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/permissions/check",
                    json={
                        "user_id": user_id,
                        "resource": resource,
                        "action": action
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("allowed", False)
                else:
                    logger.warning(f"Permission check failed: {response.status_code}")
                    return False
                    
        except httpx.RequestError as e:
            logger.error(f"Failed to check permissions with user service: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the user service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
                
        except httpx.RequestError as e:
            logger.error(f"User service health check failed: {e}")
            return False