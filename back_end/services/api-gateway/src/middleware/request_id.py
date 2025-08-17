"""
Request ID middleware for the API Gateway.

This middleware adds unique request IDs to every request for tracing and logging.
"""

import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add unique request IDs for request tracing."""
    
    def __init__(self, app):
        """Initialize request ID middleware."""
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add request ID to request and response."""
        # Check if request ID already exists in headers
        request_id = request.headers.get("X-Request-ID")
        
        # Generate new request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in route handlers
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response