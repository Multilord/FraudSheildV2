2"""
MCP Scoring Server — Port 8001 (FastAPI implementation)
Tools: score_transaction, get_risk_flags, get_behavioral_profile, get_model_metrics
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ml.engine import FraudDetectionEngine

app = FastAPI(title="MCP Scoring Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = FraudDetectionEngine()


# ─── Models ───────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    amount: float
    card1: Optional[float] = None
    addr1: Optional[float] = None
    d1: Optional[float] = None
    extra_features: Optional[Dict[str, float]] = None


class RiskFlagsRequest(BaseModel):
    amount: float
    risk_score: float
    d1: Optional[float] = None


class BehavioralRequest(BaseModel):
    account_id: str
    transaction_count: int = 0
    avg_amount: float = 0.0
    vpn_rate: float = 0.0
    night_rate: float = 0.0
    geo_hops: int = 0


# ─── Tools endpoint ───────────────────────────────────────────────────────────

@app.get("/tools")
def list_tools():
    return {
        "server": "fraud-scoring-server",
        "port": 8001,
        "tools": ["score_transaction", "get_risk_flags", "get_behavioral_profile", "get_model_metrics"],
    }


@app.post("/tools/score_transaction")
def score_transaction(req: ScoreRequest) -> Dict[str, Any]:
    """Score a single transaction using the ML ensemble model."""
    features: Dict[str, float] = {"amount": req.amount}
    if req.card1 is not None:
        features["card1"] = req.card1
    if req.addr1 is not None:
        features["addr1"] = req.addr1
    if req.d1 is not None:
        features["d1"] = req.d1
    if req.extra_features:
        features.update(req.extra_features)
    return engine.score(features)


@app.post("/tools/get_risk_flags")
def get_risk_flags(req: RiskFlagsRequest) -> List[str]:
    """Get risk flags based on feature heuristics."""
    features: Dict[str, float] = {"amount": req.amount}
    if req.d1 is not None:
        features["d1"] = req.d1
    return engine._risk_flags(features, req.risk_score)


@app.post("/tools/get_behavioral_profile")
def get_behavioral_profile(req: BehavioralRequest) -> Dict[str, Any]:
    """Build a behavioral risk profile for an account."""
    risk_level = "LOW"
    risk_factors = []

    if req.vpn_rate > 0.5:
        risk_factors.append(f"High VPN usage: {req.vpn_rate:.0%}")
        risk_level = "HIGH"
    if req.night_rate > 0.6:
        risk_factors.append(f"Predominantly night activity: {req.night_rate:.0%}")
        risk_level = "HIGH" if risk_level != "HIGH" else risk_level
    if req.geo_hops > 4:
        risk_factors.append(f"Multiple geographic hops: {req.geo_hops}")
        risk_level = "HIGH"
    if req.avg_amount > 10000:
        risk_factors.append(f"High average transaction: ${req.avg_amount:,.0f}")
        risk_level = "MEDIUM" if risk_level == "LOW" else risk_level

    return {
        "account_id": req.account_id,
        "risk_level": risk_level,
        "transaction_count": req.transaction_count,
        "avg_amount": req.avg_amount,
        "vpn_rate": req.vpn_rate,
        "night_rate": req.night_rate,
        "geo_hops": req.geo_hops,
        "risk_factors": risk_factors,
        "summary": (
            f"Account {req.account_id} shows {len(risk_factors)} risk indicators. "
            f"Overall behavioral risk: {risk_level}."
        ),
    }


@app.get("/tools/get_model_metrics")
def get_model_metrics() -> Dict[str, Any]:
    """Return ML model performance metrics."""
    return engine.get_metrics()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
