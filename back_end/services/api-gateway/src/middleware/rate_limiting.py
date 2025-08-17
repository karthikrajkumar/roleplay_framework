"""
Rate limiting middleware for the API Gateway.

This middleware implements token bucket rate limiting per IP address.
"""

import time
import logging
from typing import Dict, Tuple
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        now = time.time()
        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.last_refill) * self.refill_rate
        )
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, app):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.settings = get_settings()
        self.buckets: Dict[str, TokenBucket] = {}
        self.cleanup_interval = 300  # Clean up old buckets every 5 minutes
        self.last_cleanup = time.time()
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def cleanup_old_buckets(self):
        """Remove old unused buckets to prevent memory leaks."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets that haven't been used in the last hour
        cutoff_time = now - 3600
        old_keys = [
            key for key, bucket in self.buckets.items()
            if bucket.last_refill < cutoff_time
        ]
        
        for key in old_keys:
            del self.buckets[key]
        
        self.last_cleanup = now
        
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old rate limit buckets")
    
    def get_bucket(self, client_ip: str) -> TokenBucket:
        """Get or create token bucket for client IP."""
        if client_ip not in self.buckets:
            self.buckets[client_ip] = TokenBucket(
                capacity=self.settings.rate_limit_burst,
                refill_rate=self.settings.rate_limit_per_minute / 60.0
            )
        
        return self.buckets[client_ip]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Periodic cleanup
        self.cleanup_old_buckets()
        
        # Get client IP and bucket
        client_ip = self.get_client_ip(request)
        bucket = self.get_bucket(client_ip)
        
        # Check rate limit
        if not bucket.consume():
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 60,
                    "limit": self.settings.rate_limit_per_minute,
                    "window": "1 minute"
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response