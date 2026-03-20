"""
FraudShield data analyzer.

In real mode (DEMO_MODE=false, default): reads live statistics from SQLite DB.
In demo mode (DEMO_MODE=true): returns empty placeholder stats.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Demo / placeholder stats
# ---------------------------------------------------------------------------

_DEMO_STATS: dict = {
    "total": 0,
    "approved_count": 0,
    "flagged_count": 0,
    "blocked_count": 0,
    "fraud_rate": 0.0,
    "avg_risk_score": 0.0,
    "total_amount": 0.0,
    "total_blocked_amount": 0.0,
    "demo_mode": True,
    "message": (
        "No transactions yet. "
        "Use the eWallet to submit transactions via POST /api/wallet/transaction."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dashboard_stats() -> dict:
    """
    Return aggregate dashboard statistics.

    In real mode: reads from the SQLite database.
    In demo mode: returns zeroed placeholder stats.
    """
    if DEMO_MODE:
        return _DEMO_STATS.copy()

    try:
        from db.database import get_stats
        stats = get_stats()
        stats["demo_mode"] = False
        return stats
    except Exception as exc:
        logger.error("Failed to fetch dashboard stats from DB: %s", exc)
        return {**_DEMO_STATS, "error": str(exc)}


def get_recent_transactions(
    limit: int = 50,
    offset: int = 0,
    decision: Optional[str] = None,
) -> list[dict]:
    """
    Return recent scored transactions, newest first.

    In demo mode returns an empty list.
    """
    if DEMO_MODE:
        return []

    try:
        from db.database import get_transactions
        return get_transactions(limit=limit, offset=offset, decision_filter=decision)
    except Exception as exc:
        logger.error("Failed to fetch transactions from DB: %s", exc)
        return []


def get_chart_data() -> dict:
    """
    Return dashboard chart data: hourly transaction trend and risk distribution.

    In demo mode returns empty placeholders.
    """
    if DEMO_MODE:
        return {"hourly_trend": [], "risk_distribution": []}

    try:
        from db.database import get_hourly_trend, get_risk_distribution
        return {
            "hourly_trend":      get_hourly_trend(24),
            "risk_distribution": get_risk_distribution(),
        }
    except Exception as exc:
        logger.error("Failed to fetch chart data from DB: %s", exc)
        return {"hourly_trend": [], "risk_distribution": []}


def get_case_by_id(transaction_id: str) -> Optional[dict]:
    """
    Return a single scored transaction as a case dict, or None if not found.

    In demo mode always returns None.
    """
    if DEMO_MODE:
        return None

    try:
        from db.database import get_transaction_by_id
        return get_transaction_by_id(transaction_id)
    except Exception as exc:
        logger.error(
            "Failed to fetch transaction %s from DB: %s", transaction_id, exc
        )
        return None
