from fastapi import APIRouter, HTTPException
import logging

from models import AskRequest, AskResponse, Citation
from services.retrieval import search_regulations
from services.synthesis import synthesize_answer
from services.analytics import track_query, get_analytics_summary

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question about EU regulations.

    - AI Act selected: Returns AI-synthesized answer with interpretation
    - Other regulations: Returns raw regulatory text without AI interpretation
    """

    # Check if AI Act is selected (enables AI synthesis mode)
    use_ai_synthesis = "aiact" in [r.lower() for r in request.regulations]

    # Search for relevant chunks
    chunks = search_regulations(
        query=request.question,
        regulations=request.regulations,
        k=request.k,
    )

    if not chunks:
        return AskResponse(
            answer="No relevant regulatory sources found. Please try rephrasing your question.",
            citations=[],
            confidence=0.0,
            sources=[],
        )

    # AI Act selected → Use LLM to synthesize answer
    if use_ai_synthesis:
        answer, citations, confidence = synthesize_answer(
            question=request.question,
            chunks=chunks,
            language=request.language,
            mode=request.mode,
        )
    else:
        # Other regulations → Return raw regulatory text (no AI interpretation)
        citations = []
        for chunk in chunks:
            citations.append(Citation(
                regulation=chunk["regulation"].upper(),
                article=str(chunk.get("article_no", "N/A")),
                title=chunk.get("title", ""),
                excerpt=chunk["content"],
                url=chunk.get("url", ""),
                relevance_score=chunk.get("score", 0.0),
            ))

        # Build answer from raw regulatory text
        answer_parts = []
        for chunk in chunks:
            article_info = f"Article {chunk.get('article_no')}" if chunk.get('article_no') else "Section"
            answer_parts.append(
                f"**{chunk['regulation'].upper()} - {article_info}**\n\n{chunk['content']}"
            )
        answer = "\n\n---\n\n".join(answer_parts)

        # Calculate confidence from relevance scores
        avg_score = sum(c.relevance_score for c in citations) / len(citations) if citations else 0
        confidence = min(avg_score, 1.0)

    # Build sources for backwards compatibility
    sources = [
        {
            "source_ref": c.url,
            "article_no": c.article,
            "source_type": c.regulation,
        }
        for c in citations
    ]

    # Track analytics
    track_query(
        question=request.question,
        regulations=request.regulations,
        language=request.language,
        confidence=confidence,
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        sources=sources,
    )


@router.get("/analytics")
async def get_analytics():
    """Get usage analytics summary."""
    return get_analytics_summary()
