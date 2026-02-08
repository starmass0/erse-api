"""Simple analytics service for ERSE."""
import json
import os
from datetime import datetime
from typing import Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# File-based storage for persistence across restarts
ANALYTICS_FILE = "/tmp/erse_analytics.json"


def _load_analytics() -> dict:
    """Load analytics from file."""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analytics: {e}")

    return {
        "total_queries": 0,
        "queries_by_regulation": {},
        "queries_by_language": {},
        "queries_by_date": {},
        "recent_questions": [],
    }


def _save_analytics(data: dict):
    """Save analytics to file."""
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving analytics: {e}")


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
    from datetime import timedelta
    queries_week = 0
    for i in range(7):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        queries_week += by_date.get(day, 0)

    return {
        "total_queries": total,
        "queries_today": queries_today,
        "queries_this_week": queries_week,
        "queries_by_regulation": by_reg,
        "queries_by_language": by_lang,
        "most_popular_regulation": most_popular_reg,
        "recent_questions_count": len(data["recent_questions"]),
    }
