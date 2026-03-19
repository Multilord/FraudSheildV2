"""
FraudShield API -- FastAPI backend
V HACK 2026

Endpoints
---------
POST   /api/wallet/transaction       -- submit + score a wallet transaction
GET    /api/dashboard/stats          -- aggregate fraud statistics
GET    /api/dashboard/transactions   -- recent scored transactions
GET    /api/dashboard/cases/{id}     -- transaction detail
GET    /api/dashboard/user/{user_id} -- user behavioural profile
GET    /api/metrics                  -- model training metrics
GET    /api/health                   -- service health
WS     /ws/alerts                    -- live transaction feed
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

from db.database import (
    init_db,
    insert_transaction,
    get_or_create_user_profile,
    update_user_profile,
)
from ml.engine import engine as fraud_engine
from data.analyzer import get_dashboard_stats, get_recent_transactions, get_case_by_id

app = FastAPI(
    title="FraudShield API",
    description="Real-time fraud detection for ASEAN digital wallets - V HACK 2026",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages active WebSocket connections for the live transaction feed."""

    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)
        logger.info("WebSocket client connected. Active: %d", len(self.connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)
        logger.info("WebSocket client disconnected. Active: %d", len(self.connections))

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


manager = ConnectionManager()


class WalletTransaction(BaseModel):
    """Payload for submitting a wallet transaction for fraud scoring."""

    user_id: str = Field(..., description="Wallet user identifier")
    amount: float = Field(..., gt=0, description="Transaction amount (positive)")
    transaction_type: str = Field(
        ...,
        description="transfer | payment | cashout | topup | merchant",
    )
    recipient_id: Optional[str] = Field(default=None, description="Recipient wallet ID")
    merchant: Optional[str] = Field(default=None, description="Merchant name")
    merchant_category: Optional[str] = Field(default=None, description="Merchant category")
    device_type: str = Field(default="mobile", description="mobile | desktop | pos")
    device_id: str = Field(..., description="Device fingerprint / identifier")
    ip_address: str = Field(default="0.0.0.0", description="Client IP address")
    location: str = Field(default="Unknown", description="City or region")
    is_new_device: bool = Field(default=False, description="True if device not seen before")
    note: Optional[str] = Field(default=None, description="Optional free-text note")


@app.on_event("startup")
async def startup() -> None:
    init_db()
    loaded = fraud_engine.load()
    if not loaded:
        logger.warning(
            "Model artifacts not found. "
            "The /api/wallet/transaction endpoint will return HTTP 503 until the model is trained. "
            "Run: cd backend && python training/train_engine.py --data-dir /path/to/ieee-cis-data"
        )


@app.get("/api/health", tags=["system"])
async def health() -> dict:
    """Service health check."""
    return {
        "status": "ok" if fraud_engine.is_loaded() else "degraded",
        "model_loaded": fraud_engine.is_loaded(),
        "artifact_version": fraud_engine.artifact_version,
        "demo_mode": os.getenv("DEMO_MODE", "false").lower() == "true",
        "database_connected": True,
        "thresholds": fraud_engine.thresholds if fraud_engine.is_loaded() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/wallet/transaction", tags=["wallet"])
async def submit_wallet_transaction(tx: WalletTransaction) -> dict:
    """
    Submit a wallet transaction for real-time fraud scoring.

    Pipeline:
      1. API receives real-time transaction payload.
      2. Behavioral metrics instantly calculated from user history.
      3. 4 ML models (XGBoost, LightGBM, Random Forest, Logistic Regression)
         create a weighted ensemble score.
      4. XAI (SHAP) returns the decision with feature-level explanations.
      5. Result persisted and broadcast live to the dashboard.
    """
    if not fraud_engine.is_loaded():
        raise HTTPException(
            status_code=503,
            detail=(
                "Fraud detection model is not loaded. "
                "Train the model first: cd backend && "
                "python training/train_engine.py --data-dir /path/to/ieee-cis-data"
            ),
        )

    now = datetime.now(timezone.utc)
    transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"

    user_profile = get_or_create_user_profile(tx.user_id)

    wallet_tx_dict = {
        "user_id": tx.user_id,
        "amount": tx.amount,
        "transaction_type": tx.transaction_type,
        "device_type": tx.device_type,
        "device_id": tx.device_id,
        "ip_address": tx.ip_address,
        "location": tx.location,
        "merchant": tx.merchant or "",
        "merchant_category": tx.merchant_category or "",
        "is_new_device": tx.is_new_device,
        "hour_of_day": now.hour,
    }

    result = fraud_engine.score(wallet_tx_dict, user_profile)

    tx_record = {
        "transaction_id": transaction_id,
        "user_id": tx.user_id,
        "amount": tx.amount,
        "transaction_type": tx.transaction_type,
        "device_type": tx.device_type,
        "device_id": tx.device_id,
        "ip_address": tx.ip_address,
        "location": tx.location,
        "merchant": tx.merchant or "",
        "merchant_category": tx.merchant_category or "",
        "timestamp": now.isoformat(),
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "reasons": json.dumps(result["reasons"]),
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "features": json.dumps({
            "model_breakdown": result["model_breakdown"],
            "xai_top_features": result.get("xai_top_features", []),
        }),
        "raw_payload": tx.model_dump_json(),
    }
    insert_transaction(tx_record)

    update_user_profile(
        tx.user_id,
        tx.amount,
        result["decision"],
        tx.device_id,
        tx.location,
        tx.merchant or "",
    )

    await manager.broadcast({
        "type": "new_transaction",
        "transaction_id": transaction_id,
        "user_id": tx.user_id,
        "amount": tx.amount,
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "reasons": result["reasons"],
        "timestamp": now.isoformat(),
    })

    return {
        "transaction_id": transaction_id,
        "user_id": tx.user_id,
        "amount": tx.amount,
        "transaction_type": tx.transaction_type,
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "reasons": result["reasons"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "action_required": result["action_required"],
        "timestamp": now.isoformat(),
        "model_breakdown": result["model_breakdown"],
        "xai_top_features": result.get("xai_top_features", []),
    }


@app.get("/api/dashboard/stats", tags=["dashboard"])
async def dashboard_stats() -> dict:
    """Aggregate fraud statistics for the fraud dashboard."""
    return get_dashboard_stats()


@app.get("/api/dashboard/transactions", tags=["dashboard"])
async def dashboard_transactions(
    limit: int = Query(default=50, le=200, description="Max transactions to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    decision: Optional[str] = Query(default=None, description="Filter: APPROVE/FLAG/BLOCK"),
) -> dict:
    """Return recent scored transactions, newest first."""
    txs = get_recent_transactions(limit=limit, offset=offset, decision=decision)
    return {"transactions": txs, "count": len(txs)}


@app.get("/api/dashboard/cases/{transaction_id}", tags=["dashboard"])
async def get_case(transaction_id: str) -> dict:
    """Return a single scored transaction as a case detail."""
    case = get_case_by_id(transaction_id)
    if not case:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{transaction_id}' not found",
        )
    return case


@app.get("/api/dashboard/user/{user_id}", tags=["dashboard"])
async def get_user_profile_endpoint(user_id: str) -> dict:
    """Return a user behavioural profile."""
    return get_or_create_user_profile(user_id)


@app.get("/api/metrics", tags=["model"])
async def get_metrics() -> dict:
    """Return model training metrics (ROC-AUC, PR-AUC, latency, etc.)."""
    if not fraud_engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    metrics = fraud_engine.get_metrics()
    if not metrics:
        return {"message": "No metrics available. Run training/train_engine.py first."}
    return metrics


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket) -> None:
    """WebSocket live feed endpoint for the fraud dashboard."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
