"""
Authentication middleware for the API Gateway.

This middleware handles JWT token validation, user authentication,
and authorization for protected endpoints.
"""

import jwt
from typing import Optional, List, Callable
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from ..config.settings import get_settings
from ..services.user_service_client import UserServiceClient
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for validating JWT tokens and user permissions.
    
    This middleware:
    - Validates JWT tokens from Authorization header
    - Caches user information for performance
    - Sets user context in request state
    - Handles token expiration and refresh
    """
    
    def __init__(
        self,
        app,
        excluded_paths: Optional[List[str]] = None,
        optional_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.settings = get_settings()
        self.user_service = UserServiceClient()
        self.cache = CacheManager()
        
        # Paths that don't require authentication
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/metrics"
        ]
        
        # Paths where authentication is optional
        self.optional_paths = optional_paths or [
            "/api/v1/characters/public",
            "/api/v1/scenarios/public"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication middleware."""
        path = request.url.path
        
        # Skip authentication for excluded paths
        if any(path.startswith(excluded) for excluded in self.excluded_paths):
            return await call_next(request)
        
        # Check for authentication token
        authorization = request.headers.get("Authorization")
        token = None
        
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]  # Remove "Bearer " prefix
        
        # Handle optional authentication paths
        is_optional = any(path.startswith(optional) for optional in self.optional_paths)
        
        if not token:
            if is_optional:
                # Continue without user context
                request.state.user_id = None
                request.state.user = None
                return await call_next(request)
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authentication token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        try:
            # Validate and decode JWT token
            user_data = await self._validate_token(token)
            
            # Set user context in request state
            request.state.user_id = user_data["user_id"]
            request.state.user = user_data
            
            # Check if user is active
            if not user_data.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is inactive"
                )
            
            # Add user info to request headers for downstream services
            request.headers.__dict__["_list"].append(
                (b"x-user-id", str(user_data["user_id"]).encode())
            )
            request.headers.__dict__["_list"].append(
                (b"x-user-role", user_data.get("role", "user").encode())
            )
            
            return await call_next(request)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            
            if is_optional:
                # Continue without user context for optional auth paths
                request.state.user_id = None
                request.state.user = None
                return await call_next(request)
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token"
                )
    
    async def _validate_token(self, token: str) -> dict:
        """
        Validate JWT token and return user data.
        
        First checks cache for performance, then validates with user service.
        """
        try:
            # Decode token without verification first to get user_id for caching
            unverified_payload = jwt.decode(
                token, 
                options={"verify_signature": False}
            )
            user_id = unverified_payload.get("user_id")
            
            # Check cache for user data
            cache_key = f"auth:user:{user_id}:{hash(token)}"
            cached_user = await self.cache.get(cache_key)
            
            if cached_user:
                return cached_user
            
            # Verify token signature
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.jwt_algorithm]
            )
            
            # Get full user data from user service
            user_data = await self.user_service.get_user_by_id(user_id)
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Prepare user context data
            auth_data = {
                "user_id": user_id,
                "email": user_data.get("email"),
                "username": user_data.get("username"),
                "role": user_data.get("role", "user"),
                "is_active": user_data.get("is_active", True),
                "subscription_tier": user_data.get("subscription_tier", "free"),
                "token_exp": payload.get("exp"),
                "token_iat": payload.get("iat")
            }
            
            # Cache user data for 5 minutes
            await self.cache.set(cache_key, auth_data, ttl_seconds=300)
            
            return auth_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    def require_role(self, required_roles: List[str]):
        """
        Decorator to require specific user roles.
        
        Usage:
        @auth_middleware.require_role(["admin", "moderator"])
        async def admin_endpoint():
            pass
        """
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                user = getattr(request.state, "user", None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                user_role = user.get("role", "user")
                if user_role not in required_roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_subscription(self, required_tiers: List[str]):
        """
        Decorator to require specific subscription tiers.
        
        Usage:
        @auth_middleware.require_subscription(["premium", "enterprise"])
        async def premium_endpoint():
            pass
        """
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                user = getattr(request.state, "user", None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                subscription_tier = user.get("subscription_tier", "free")
                if subscription_tier not in required_tiers:
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail="Subscription upgrade required"
                    )
                
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator


# Helper functions for dependency injection
async def get_current_user(request: Request) -> Optional[dict]:
    """Get current authenticated user from request state."""
    return getattr(request.state, "user", None)


async def get_current_user_id(request: Request) -> Optional[str]:
    """Get current user ID from request state."""
    return getattr(request.state, "user_id", None)


def require_auth(request: Request) -> dict:
    """Require authentication and return user data."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user