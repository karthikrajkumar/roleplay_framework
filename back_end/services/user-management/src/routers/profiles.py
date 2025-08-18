from fastapi import APIRouter, Depends, HTTPException
from src.config.settings import get_settings

router = APIRouter(prefix="/profiles", tags=["profiles"])
settings = get_settings()


@router.get("/{user_id}")
async def get_user_profile(user_id: int):
    """Get user profile by ID."""
    return {"message": f"User {user_id} profile endpoint - implementation coming soon"}


@router.put("/{user_id}")
async def update_user_profile(user_id: int):
    """Update user profile by ID."""
    return {"message": f"Update user {user_id} profile endpoint - implementation coming soon"}


@router.post("/{user_id}/avatar")
async def upload_avatar(user_id: int):
    """Upload user avatar."""
    return {"message": f"User {user_id} avatar upload endpoint - implementation coming soon"}


@router.delete("/{user_id}/avatar")
async def delete_avatar(user_id: int):
    """Delete user avatar."""
    return {"message": f"User {user_id} avatar delete endpoint - implementation coming soon"}