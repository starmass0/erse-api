"""Analytics service for ERSE - stores data in Qdrant Cloud for persistence."""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Analytics collection name
ANALYTICS_COLLECTION = "erse_analytics"
ANALYTICS_POINT_ID = 1  # Single point to store all analytics

_client: Optional[QdrantClient] = None


def _get_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        if settings.qdrant_url and settings.qdrant_api_key:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            _client = QdrantClient(":memory:")
            logger.warning("Using in-memory Qdrant for analytics")
    return _client


def _ensure_analytics_collection():
    """Create analytics collection if it doesn't exist."""
    client = _get_client()
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if ANALYTICS_COLLECTION not in collection_names:
            # Create collection with minimal vector (we only use payload)
            client.create_collection(
                collection_name=ANALYTICS_COLLECTION,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )
            logger.info(f"Created analytics collection: {ANALYTICS_COLLECTION}")

            # Initialize with empty analytics
            _save_analytics(_get_default_analytics())
    except Exception as e:
        logger.error(f"Error ensuring analytics collection: {e}")


def _get_default_analytics() -> dict:
    """Return default analytics structure."""
    return {
        "total_queries": 0,
        "queries_by_regulation": {},
        "queries_by_language": {},
        "queries_by_date": {},
        "recent_questions": [],
        "feedback_positive": 0,
        "feedback_negative": 0,
    }


def _load_analytics() -> dict:
    """Load analytics from Qdrant."""
    try:
        _ensure_analytics_collection()
        client = _get_client()

        # Try to get the analytics point
        results = client.retrieve(
            collection_name=ANALYTICS_COLLECTION,
            ids=[ANALYTICS_POINT_ID],
            with_payload=True,
        )

        if results and len(results) > 0:
            payload = results[0].payload or {}
            # Merge with defaults to handle missing fields
            defaults = _get_default_analytics()
            defaults.update(payload)
            return defaults

        return _get_default_analytics()

    except Exception as e:
        logger.error(f"Error loading analytics from Qdrant: {e}")
        return _get_default_analytics()


def _save_analytics(data: dict):
    """Save analytics to Qdrant."""
    try:
        _ensure_analytics_collection()
        client = _get_client()

        # Upsert the analytics point (dummy vector, real payload)
        client.upsert(
            collection_name=ANALYTICS_COLLECTION,
            points=[
                PointStruct(
                    id=ANALYTICS_POINT_ID,
                    vector=[0.0, 0.0, 0.0, 0.0],  # Dummy vector
                    payload=data,
                )
            ],
        )
    except Exception as e:
        logger.error(f"Error saving analytics to Qdrant: {e}")


def track_query(
    question: str,
    regulations: list[str],
    language: str = "en",
    confidence: float = 0.0,
):
    """Track a query for analytics."""
    try:
        data = _load_analytics()

        # Increment total
        data["total_queries"] += 1

        # Track by regulation
        for reg in regulations:
            reg_lower = reg.lower()
            data["queries_by_regulation"][reg_lower] = (
                data["queries_by_regulation"].get(reg_lower, 0) + 1
            )

        # Track by language
        data["queries_by_language"][language] = (
            data["queries_by_language"].get(language, 0) + 1
        )

        # Track by date
        today = datetime.now().strftime("%Y-%m-%d")
        data["queries_by_date"][today] = (
            data["queries_by_date"].get(today, 0) + 1
        )

        # Store recent questions (keep last 100)
        data["recent_questions"].insert(0, {
            "question": question[:200],  # Truncate long questions
            "regulations": regulations,
            "language": language,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })
        data["recent_questions"] = data["recent_questions"][:100]

        _save_analytics(data)

    except Exception as e:
        logger.error(f"Error tracking query: {e}")


def track_feedback(feedback_type: str):
    """Track user feedback (positive/negative)."""
    try:
        data = _load_analytics()

        # Initialize feedback counters if not present
        if "feedback_positive" not in data:
            data["feedback_positive"] = 0
        if "feedback_negative" not in data:
            data["feedback_negative"] = 0

        if feedback_type == "yes":
            data["feedback_positive"] += 1
        elif feedback_type == "no":
            data["feedback_negative"] += 1

        _save_analytics(data)

    except Exception as e:
        logger.error(f"Error tracking feedback: {e}")


def get_analytics_summary() -> dict:
    """Get analytics summary."""
    data = _load_analytics()

    # Calculate some derived stats
    total = data["total_queries"]
    by_reg = data["queries_by_regulation"]
    by_lang = data["queries_by_language"]
    by_date = data["queries_by_date"]

    # Most popular regulation
    most_popular_reg = max(by_reg, key=by_reg.get) if by_reg else None

    # Queries today
    today = datetime.now().strftime("%Y-%m-%d")
    queries_today = by_date.get(today, 0)

    # Queries this week (last 7 days)
    queries_week = 0
    for i in range(7):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        queries_week += by_date.get(day, 0)

    # Feedback stats
    feedback_positive = data.get("feedback_positive", 0)
    feedback_negative = data.get("feedback_negative", 0)
    total_feedback = feedback_positive + feedback_negative
    satisfaction_rate = round((feedback_positive / total_feedback * 100), 1) if total_feedback > 0 else 0

    return {
        "total_queries": total,
        "queries_today": queries_today,
        "queries_this_week": queries_week,
        "queries_by_regulation": by_reg,
        "queries_by_language": by_lang,
        "most_popular_regulation": most_popular_reg,
        "recent_questions_count": len(data["recent_questions"]),
        "feedback_positive": feedback_positive,
        "feedback_negative": feedback_negative,
        "total_feedback": total_feedback,
        "satisfaction_rate": satisfaction_rate,
    }
