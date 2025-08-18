import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/Response logging middleware for the user management service."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not present
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "remote_addr": request.client.host if request.client else None,
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log request completion
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                }
            )
            
            # Add request ID to response headers
            response.headers["x-request-id"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate processing time for error cases
            process_time = time.time() - start_time
            
            # Log request error
            logger.error(
                "Request error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "process_time": round(process_time, 4),
                },
                exc_info=True
            )
            
            raise