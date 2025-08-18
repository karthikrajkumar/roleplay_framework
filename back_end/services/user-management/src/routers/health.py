from fastapi import APIRouter
from datetime import datetime
from src.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check():
    """Health check endpoint for user management service."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "user-management",
        "version": settings.version
    }