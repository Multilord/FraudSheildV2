"""
FraudShield SQLite database layer.
Thread-safe via threading.local (one connection per thread).
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "fraudshield.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local database connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_CREATE_SCORED_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    transaction_id      TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    amount              REAL NOT NULL,
    transaction_type    TEXT NOT NULL DEFAULT '',
    device_type         TEXT NOT NULL DEFAULT '',
    device_id           TEXT NOT NULL DEFAULT '',
    ip_address          TEXT NOT NULL DEFAULT '',
    location            TEXT NOT NULL DEFAULT '',
    merchant            TEXT NOT NULL DEFAULT '',
    merchant_category   TEXT NOT NULL DEFAULT '',
    timestamp           TEXT NOT NULL,
    risk_score          INTEGER NOT NULL,
    decision            TEXT NOT NULL,
    reasons             TEXT NOT NULL DEFAULT '[]',
    confidence          REAL NOT NULL DEFAULT 0.0,
    latency_ms          REAL NOT NULL DEFAULT 0.0,
    features            TEXT NOT NULL DEFAULT '{}',
    raw_payload         TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_USER_HISTORY = """
CREATE TABLE IF NOT EXISTS user_history (
    user_id                 TEXT PRIMARY KEY,
    transaction_count       INTEGER NOT NULL DEFAULT 0,
    total_amount            REAL NOT NULL DEFAULT 0.0,
    avg_amount              REAL NOT NULL DEFAULT 0.0,
    std_amount              REAL NOT NULL DEFAULT 0.0,
    max_amount              REAL NOT NULL DEFAULT 0.0,
    fraud_count             INTEGER NOT NULL DEFAULT 0,
    flag_count              INTEGER NOT NULL DEFAULT 0,
    last_transaction_time   TEXT,
    device_ids              TEXT NOT NULL DEFAULT '[]',
    locations               TEXT NOT NULL DEFAULT '[]',
    merchants               TEXT NOT NULL DEFAULT '[]',
    first_seen              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_st_user_id    ON scored_transactions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_st_decision    ON scored_transactions(decision);",
    "CREATE INDEX IF NOT EXISTS idx_st_timestamp   ON scored_transactions(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_st_risk_score  ON scored_transactions(risk_score);",
]


def init_db() -> None:
    """Create tables and indexes if they do not exist."""
    conn = _get_conn()
    conn.execute(_CREATE_SCORED_TRANSACTIONS)
    conn.execute(_CREATE_USER_HISTORY)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def insert_transaction(tx_dict: dict) -> None:
    """Insert a scored transaction record."""
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO scored_transactions (
            transaction_id, user_id, amount, transaction_type, device_type,
            device_id, ip_address, location, merchant, merchant_category,
            timestamp, risk_score, decision, reasons, confidence,
            latency_ms, features, raw_payload
        ) VALUES (
            :transaction_id, :user_id, :amount, :transaction_type, :device_type,
            :device_id, :ip_address, :location, :merchant, :merchant_category,
            :timestamp, :risk_score, :decision, :reasons, :confidence,
            :latency_ms, :features, :raw_payload
        )
        """,
        tx_dict,
    )
    conn.commit()


def get_transactions(
    limit: int = 50,
    offset: int = 0,
    decision_filter: Optional[str] = None,
) -> list[dict]:
    """Return a list of scored transactions, newest first."""
    conn = _get_conn()
    if decision_filter:
        rows = conn.execute(
            """
            SELECT * FROM scored_transactions
            WHERE decision = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (decision_filter.upper(), limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM scored_transactions
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_transaction_by_id(transaction_id: str) -> Optional[dict]:
    """Return a single transaction dict or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM scored_transactions WHERE transaction_id = ?",
        (transaction_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Deserialise JSON fields
    for field in ("reasons", "features", "raw_payload"):
        if isinstance(d.get(field), str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = d[field]
    return d


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    """Return aggregate dashboard statistics."""
    conn = _get_conn()

    row = conn.execute(
        """
        SELECT
            COUNT(*)                                         AS total,
            SUM(CASE WHEN decision='APPROVE' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN decision='FLAG'    THEN 1 ELSE 0 END) AS flagged_count,
            SUM(CASE WHEN decision='BLOCK'   THEN 1 ELSE 0 END) AS blocked_count,
            AVG(risk_score)                                  AS avg_risk_score,
            SUM(amount)                                      AS total_amount,
            SUM(CASE WHEN decision='BLOCK' THEN amount ELSE 0 END) AS total_blocked_amount
        FROM scored_transactions
        """
    ).fetchone()

    total = row["total"] or 0
    blocked_count = row["blocked_count"] or 0
    fraud_rate = round(blocked_count / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "approved_count": row["approved_count"] or 0,
        "flagged_count": row["flagged_count"] or 0,
        "blocked_count": blocked_count,
        "fraud_rate": fraud_rate,
        "avg_risk_score": round(float(row["avg_risk_score"] or 0), 2),
        "total_amount": round(float(row["total_amount"] or 0), 2),
        "total_blocked_amount": round(float(row["total_blocked_amount"] or 0), 2),
    }


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

def get_or_create_user_profile(user_id: str) -> dict:
    """Return existing user profile or create a blank one."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM user_history WHERE user_id = ?", (user_id,)
    ).fetchone()

    if row:
        profile = dict(row)
        for field in ("device_ids", "locations", "merchants"):
            if isinstance(profile.get(field), str):
                try:
                    profile[field] = json.loads(profile[field])
                except (json.JSONDecodeError, TypeError):
                    profile[field] = []
        return profile

    # Create blank profile
    now = datetime.now(timezone.utc).isoformat()
    blank = {
        "user_id": user_id,
        "transaction_count": 0,
        "total_amount": 0.0,
        "avg_amount": 0.0,
        "std_amount": 0.0,
        "max_amount": 0.0,
        "fraud_count": 0,
        "flag_count": 0,
        "last_transaction_time": None,
        "device_ids": [],
        "locations": [],
        "merchants": [],
        "first_seen": now,
        "updated_at": now,
    }
    conn.execute(
        """
        INSERT OR IGNORE INTO user_history (
            user_id, transaction_count, total_amount, avg_amount, std_amount,
            max_amount, fraud_count, flag_count, last_transaction_time,
            device_ids, locations, merchants, first_seen, updated_at
        ) VALUES (
            :user_id, :transaction_count, :total_amount, :avg_amount, :std_amount,
            :max_amount, :fraud_count, :flag_count, :last_transaction_time,
            :device_ids_json, :locations_json, :merchants_json, :first_seen, :updated_at
        )
        """,
        {
            **blank,
            "device_ids_json": "[]",
            "locations_json": "[]",
            "merchants_json": "[]",
        },
    )
    conn.commit()
    return blank


def update_user_profile(
    user_id: str,
    amount: float,
    decision: str,
    device_id: str,
    location: str,
    merchant: str,
) -> None:
    """Update user behavioural profile after a scored transaction."""
    profile = get_or_create_user_profile(user_id)

    n = profile["transaction_count"]
    old_avg = profile["avg_amount"]
    old_total = profile["total_amount"]

    # Welford online mean / variance (population std)
    new_n = n + 1
    new_total = old_total + amount
    new_avg = new_total / new_n

    # Running variance via Welford's method
    old_std = profile["std_amount"]
    old_m2 = (old_std ** 2) * n if n > 0 else 0.0
    delta = amount - old_avg
    delta2 = amount - new_avg
    new_m2 = old_m2 + delta * delta2
    new_std = (new_m2 / new_n) ** 0.5

    new_max = max(profile["max_amount"], amount)

    fraud_count = profile["fraud_count"] + (1 if decision == "BLOCK" else 0)
    flag_count = profile["flag_count"] + (1 if decision == "FLAG" else 0)

    # Update lists (keep last 20 unique values)
    def _update_list(existing: list, value: str) -> list:
        if not value:
            return existing
        updated = [v for v in existing if v != value]
        updated.insert(0, value)
        return updated[:20]

    device_ids = _update_list(profile["device_ids"], device_id)
    locations = _update_list(profile["locations"], location)
    merchants = _update_list(profile["merchants"], merchant)

    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    conn.execute(
        """
        UPDATE user_history SET
            transaction_count       = ?,
            total_amount            = ?,
            avg_amount              = ?,
            std_amount              = ?,
            max_amount              = ?,
            fraud_count             = ?,
            flag_count              = ?,
            last_transaction_time   = ?,
            device_ids              = ?,
            locations               = ?,
            merchants               = ?,
            updated_at              = ?
        WHERE user_id = ?
        """,
        (
            new_n,
            round(new_total, 4),
            round(new_avg, 4),
            round(new_std, 4),
            round(new_max, 4),
            fraud_count,
            flag_count,
            now,
            json.dumps(device_ids),
            json.dumps(locations),
            json.dumps(merchants),
            now,
            user_id,
        ),
    )
    conn.commit()
