from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from config import get_settings
from routes import health, ask, ingest
from services.retrieval import ensure_collection_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="ERSE API",
    description="European Regulatory Source Engine - AI-powered EU regulation Q&A",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(ask.router, tags=["Ask"])
app.include_router(ingest.router, tags=["Ingest"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting ERSE API v{settings.app_version}")

    # Ensure Qdrant collection exists
    try:
        ensure_collection_exists()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning(f"Qdrant setup warning: {e}")

    logger.info("ERSE API started successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ERSE API",
        "version": settings.app_version,
        "description": "European Regulatory Source Engine",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
