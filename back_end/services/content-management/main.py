#!/usr/bin/env python3
"""
Content Management Service - Learning Content and Scenario Management
Part of the Advanced AI Roleplay Platform

This service handles learning content creation, scenario management,
educational material processing, and content versioning.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add shared library to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment configuration
SERVICE_NAME = "content-management"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8008"))
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting {SERVICE_NAME} service...")
    
    # Initialize content management systems
    try:
        # TODO: Initialize content database connections
        # TODO: Initialize file storage systems
        # TODO: Initialize content processing pipelines
        # TODO: Initialize search indexing
        logger.info("Content management systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize content systems: {e}")
        raise
    
    yield
    
    logger.info(f"Shutting down {SERVICE_NAME} service...")


# Create FastAPI application
app = FastAPI(
    title="Content Management Service",
    description="Learning Content and Scenario Management for AI Roleplay Platform",
    version="1.0.0",
    lifespan=lifespan,
    debug=DEBUG_MODE
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "port": SERVICE_PORT
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # TODO: Check database connections
    # TODO: Check file storage access
    # TODO: Check content processing pipelines
    # TODO: Check search index status
    return {
        "status": "ready",
        "service": SERVICE_NAME,
        "content_database": "operational",
        "file_storage": "operational"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": SERVICE_NAME,
        "message": "Content Management Service - Learning Content and Scenario Management",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs"
        }
    }


# TODO: Add content management endpoints
# @app.post("/api/v1/content/scenarios/create")
# @app.get("/api/v1/content/scenarios/{scenario_id}")
# @app.post("/api/v1/content/upload")
# @app.get("/api/v1/content/search")


if __name__ == "__main__":
    logger.info(f"Starting {SERVICE_NAME} on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        "main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=DEBUG_MODE,
        log_level="info" if not DEBUG_MODE else "debug"
    )