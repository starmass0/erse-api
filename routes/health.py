from fastapi import APIRouter

from config import get_settings
from models import HealthResponse
from services.retrieval import check_qdrant_connection

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    qdrant_status = "connected" if check_qdrant_connection() else "disconnected"

    return HealthResponse(
        status="healthy",
        qdrant=qdrant_status,
        version=settings.app_version,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )
