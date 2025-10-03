"""
FastAPI Main Application
Entry point for the AI Cost & Insights Copilot API
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .dependencies import initialize_app_dependencies
from .routes import router

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add default request_id for logs without one
class DefaultRequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'system'
        return True

for handler in logging.root.handlers:
    handler.addFilter(DefaultRequestIdFilter())

logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle:
    - Startup: Initialize all expensive resources (vector store, LLM, etc.)
    - Shutdown: Cleanup if needed
    """
    # Startup
    logger.info("=" * 80)
    logger.info("STARTING AI COST & INSIGHTS COPILOT API")
    logger.info("=" * 80)
    
    try:
        # Initialize all dependencies (Retriever, QACopilot, RecommendationEngine)
        initialize_app_dependencies()
        
        logger.info("=" * 80)
        logger.info("✓ API STARTUP COMPLETE - READY TO SERVE REQUESTS")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app WITH lifespan
app = FastAPI(
    title="AI Cost & Insights Copilot API",
    description="AI-powered cloud cost analytics and optimization recommendations",
    version="1.0.0",
    lifespan=lifespan  # THIS IS THE KEY FIX
)

# Get allowed origins from environment variable
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "").split(",")
CLIENT_ORIGINS = [origin.strip() for origin in CLIENT_ORIGINS if origin.strip()]

# If no origins specified, allow localhost for development
if not CLIENT_ORIGINS:
    CLIENT_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]
    logger.warning("No CLIENT_ORIGINS set, using default localhost origins for development")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include API routes with prefix
app.include_router(router, prefix="/api/v1", tags=["Cost Analytics"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "service": "AI Cost & Insights Copilot",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "kpi": "/api/v1/kpi",
            "ask": "/api/v1/ask",
            "recommendations": "/api/v1/recommendations",
            "health": "/api/v1/health"
        }
    }


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )