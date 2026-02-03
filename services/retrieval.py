from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from typing import Optional
import logging
import re

from config import get_settings
from services.embeddings import get_embedding, get_embedding_dimension

settings = get_settings()
logger = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        if settings.qdrant_url and settings.qdrant_api_key:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            # Local in-memory for testing
            _client = QdrantClient(":memory:")
            logger.warning("Using in-memory Qdrant (no persistence)")
    return _client


def ensure_collection_exists():
    """Create collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if settings.qdrant_collection not in collection_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=get_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.qdrant_collection}")


def detect_article_number(query: str) -> Optional[int]:
    """Detect if user is asking about a specific article number."""
    patterns = [
        r'article\s*(\d+)',           # "article 6", "Article 6"
        r'art\.?\s*(\d+)',            # "art 6", "art. 6"
        r'gdpr\s*(\d+)',              # "GDPR 6", "gdpr 6"
        r'article\s*#?\s*(\d+)',      # "article #6"
    ]

    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))
    return None


def search_regulations(
    query: str,
    regulations: list[str] = None,
    k: int = 5,
) -> list[dict]:
    """Search for relevant regulation chunks."""
    client = get_qdrant_client()

    # Check if user is asking about a specific article
    article_num = detect_article_number(query)

    # Generate query embedding
    query_embedding = get_embedding(query)[0].tolist()

    # Build filter if regulations specified
    search_filter = None
    if regulations and len(regulations) > 0:
        # Filter by regulation field
        conditions = [
            FieldCondition(key="regulation", match=MatchValue(value=reg.lower()))
            for reg in regulations
        ]
        if len(conditions) == 1:
            search_filter = Filter(must=conditions)
        else:
            # Multiple regulations - use should (OR) logic
            search_filter = Filter(should=conditions)

    try:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=k,
            with_payload=True,
        )
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        return []

    # Format results
    chunks = []
    for result in results.points:
        payload = result.payload or {}
        chunks.append({
            "content": payload.get("content", ""),
            "regulation": payload.get("regulation", "unknown"),
            "article_no": payload.get("article_no"),
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
            "score": result.score,
        })

    # If user asked for specific article, prioritize those results
    if article_num:
        # Separate matching and non-matching articles
        matching = [c for c in chunks if c.get("article_no") == article_num]
        non_matching = [c for c in chunks if c.get("article_no") != article_num]

        # If we found matching articles, prioritize them
        if matching:
            # Boost score for matching articles
            for c in matching:
                c["score"] = min(c["score"] + 0.3, 1.0)
            chunks = matching + non_matching
        else:
            # Article not in results, search all data for it
            try:
                all_results = client.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=100,
                    with_payload=True,
                )
                for point in all_results[0]:
                    if point.payload.get("article_no") == article_num:
                        chunks.insert(0, {
                            "content": point.payload.get("content", ""),
                            "regulation": point.payload.get("regulation", "unknown"),
                            "article_no": point.payload.get("article_no"),
                            "title": point.payload.get("title", ""),
                            "url": point.payload.get("url", ""),
                            "score": 0.9,  # High score for exact match
                        })
            except Exception as e:
                logger.error(f"Article search error: {e}")

    return chunks[:k]


def check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        logger.error(f"Qdrant connection error: {e}")
        return False
