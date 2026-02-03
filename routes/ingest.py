from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging

from models import IngestRequest, IngestResponse
from services.ingestion import (
    ingest_document,
    ingest_from_url,
    ingest_gdpr_batch,
    ingest_dsa_batch,
    ingest_nis2_batch,
    ingest_aiact_batch,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_regulation(request: IngestRequest):
    """Ingest a regulation document or article."""

    try:
        if request.content:
            # Direct content ingestion
            chunks_created = ingest_document(
                regulation=request.regulation.value,
                content=request.content,
                article_no=request.article_no,
                title=request.title or "",
                url=request.url,
            )
        else:
            # Scrape from URL
            chunks_created = ingest_from_url(
                regulation=request.regulation.value,
                url=request.url,
                article_no=request.article_no,
            )

        if chunks_created == 0:
            return IngestResponse(
                success=False,
                message="No content was extracted from the URL",
                chunks_created=0,
            )

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {chunks_created} chunks",
            chunks_created=chunks_created,
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/gdpr-batch", response_model=IngestResponse)
async def ingest_gdpr_articles(
    background_tasks: BackgroundTasks,
    articles: list[int] = None,
):
    """Ingest multiple GDPR articles in the background."""

    if articles is None:
        articles = list(range(1, 100))

    # Run in background
    background_tasks.add_task(ingest_gdpr_batch, articles)

    return IngestResponse(
        success=True,
        message=f"Started background ingestion of {len(articles)} GDPR articles",
        chunks_created=0,
    )


@router.post("/ingest/dsa-batch", response_model=IngestResponse)
async def ingest_dsa_articles(background_tasks: BackgroundTasks):
    """Ingest Digital Services Act (DSA) in the background."""

    background_tasks.add_task(ingest_dsa_batch)

    return IngestResponse(
        success=True,
        message="Started background ingestion of DSA",
        chunks_created=0,
    )


@router.post("/ingest/nis2-batch", response_model=IngestResponse)
async def ingest_nis2_articles(background_tasks: BackgroundTasks):
    """Ingest NIS2 Directive in the background."""

    background_tasks.add_task(ingest_nis2_batch)

    return IngestResponse(
        success=True,
        message="Started background ingestion of NIS2 Directive",
        chunks_created=0,
    )


@router.post("/ingest/aiact-batch", response_model=IngestResponse)
async def ingest_aiact_articles(background_tasks: BackgroundTasks):
    """Ingest AI Act in the background."""

    background_tasks.add_task(ingest_aiact_batch)

    return IngestResponse(
        success=True,
        message="Started background ingestion of AI Act",
        chunks_created=0,
    )
