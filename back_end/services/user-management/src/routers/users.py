from fastapi import APIRouter, Depends, HTTPException
from typing import List
from src.config.settings import get_settings

router = APIRouter(prefix="/users", tags=["users"])
settings = get_settings()


@router.get("/")
async def get_users():
    """Get all users."""
    return {"message": "Users endpoint - implementation coming soon"}


@router.get("/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    return {"message": f"User {user_id} endpoint - implementation coming soon"}


@router.post("/")
async def create_user():
    """Create a new user."""
    return {"message": "Create user endpoint - implementation coming soon"}


@router.put("/{user_id}")
async def update_user(user_id: int):
    """Update user by ID."""
    return {"message": f"Update user {user_id} endpoint - implementation coming soon"}


@router.delete("/{user_id}")
async def delete_user(user_id: int):
    """Delete user by ID."""
    return {"message": f"Delete user {user_id} endpoint - implementation coming soon"}