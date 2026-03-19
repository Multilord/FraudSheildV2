"""
MCP Insights Server — Port 8004 (FastAPI implementation)
Tools: generate_hypothesis, suggest_actions, explain_for_user, get_pattern_matches
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from llm.insights import InsightsEngine

app = FastAPI(title="MCP Insights Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

insights = InsightsEngine()


# ─── Models ───────────────────────────────────────────────────────────────────

class HypothesisRequest(BaseModel):
    account_id: str
    fraud_type: str
    risk_score: float
    amount: float
    vpn_usage_rate: Optional[float] = None
    geo_hops: Optional[int] = None
    synthetic_id_prob: Optional[float] = None
    additional_context: Optional[str] = None


class ActionsRequest(BaseModel):
    risk_score: float
    fraud_type: str
    case_id: Optional[str] = None


class ExplainRequest(BaseModel):
    account_id: str
    fraud_type: str
    risk_score: float
    key_signals: Optional[str] = None


class PatternRequest(BaseModel):
    fraud_type: str
    risk_score: float
    top_n: int = 3


# ─── Tools ────────────────────────────────────────────────────────────────────

@app.get("/tools")
def list_tools():
    return {
        "server": "fraud-insights-server",
        "port": 8004,
        "tools": ["generate_hypothesis", "suggest_actions", "explain_for_user", "get_pattern_matches"],
    }


@app.post("/tools/generate_hypothesis")
async def generate_hypothesis(req: HypothesisRequest) -> Dict[str, Any]:
    transaction_data: Dict[str, Any] = {
        "account_id": req.account_id, "fraud_type": req.fraud_type,
        "risk_score": req.risk_score, "amount": req.amount,
    }
    if req.vpn_usage_rate is not None:
        transaction_data["vpn_usage_rate"] = req.vpn_usage_rate
    if req.geo_hops is not None:
        transaction_data["geo_hops"] = req.geo_hops
    if req.synthetic_id_prob is not None:
        transaction_data["synthetic_id_prob"] = req.synthetic_id_prob
    if req.additional_context:
        transaction_data["additional_context"] = req.additional_context

    hypothesis = await insights.generate_hypothesis(transaction_data)
    return {
        "account_id": req.account_id,
        "fraud_type": req.fraud_type,
        "hypothesis": hypothesis,
        "key_indicators": insights.extract_key_indicators(transaction_data),
        "detected_patterns": insights.detect_patterns(transaction_data),
        "confidence": req.risk_score,
    }


@app.post("/tools/suggest_actions")
async def suggest_actions(req: ActionsRequest) -> Dict[str, Any]:
    actions = await insights.suggest_actions(req.risk_score, req.fraud_type, req.case_id)
    urgency = "IMMEDIATE" if req.risk_score >= 0.85 else "HIGH" if req.risk_score >= 0.70 else "MEDIUM"
    return {
        "case_id": req.case_id, "fraud_type": req.fraud_type,
        "risk_score": req.risk_score, "urgency": urgency, "actions": actions,
    }


@app.post("/tools/explain_for_user")
def explain_for_user(req: ExplainRequest) -> Dict[str, Any]:
    signals = [s.strip() for s in req.key_signals.split(",")] if req.key_signals else []
    risk_pct = int(req.risk_score * 100)
    level = "very high" if risk_pct >= 85 else "high" if risk_pct >= 70 else "moderate"
    signal_text = (" Key signals include: " + "; ".join(signals[:3]) + ".") if signals else ""
    explanation = (
        f"Account {req.account_id} has been flagged with a {level} risk level "
        f"({risk_pct}% confidence) for suspected {req.fraud_type}.{signal_text} "
        f"Our AI system has analysed the account's transaction patterns, login behaviour, "
        f"and identity signals and found activity inconsistent with normal usage."
    )
    return {
        "account_id": req.account_id, "fraud_type": req.fraud_type,
        "risk_level": level.upper(), "risk_score_pct": risk_pct, "explanation": explanation,
        "next_steps": [
            "A fraud analyst has been assigned to your case.",
            "You may be contacted to verify your identity.",
            "Some account features may be temporarily restricted.",
        ],
    }


@app.post("/tools/get_pattern_matches")
def get_pattern_matches(req: PatternRequest) -> List[Dict[str, Any]]:
    pattern_db: Dict[str, List[Dict[str, Any]]] = {
        "Money Laundering": [
            {"id": "CS-2023-445", "match_pct": 94, "type": "Money Laundering",
             "outcome": "Confirmed Fraud", "exposure_usd": 52000},
            {"id": "CS-2023-112", "match_pct": 87, "type": "Money Laundering",
             "outcome": "Confirmed Fraud", "exposure_usd": 38500},
            {"id": "CS-2022-891", "match_pct": 79, "type": "Money Laundering",
             "outcome": "Confirmed Fraud", "exposure_usd": 29000},
        ],
        "Synthetic Identity": [
            {"id": "CS-2023-302", "match_pct": 91, "type": "Synthetic Identity",
             "outcome": "Confirmed Fraud", "exposure_usd": 18200},
            {"id": "CS-2023-188", "match_pct": 84, "type": "Synthetic Identity",
             "outcome": "Confirmed Fraud", "exposure_usd": 12400},
            {"id": "CS-2022-774", "match_pct": 76, "type": "Synthetic Identity",
             "outcome": "False Positive", "exposure_usd": 9800},
        ],
        "Account Takeover": [
            {"id": "CS-2023-509", "match_pct": 88, "type": "Account Takeover",
             "outcome": "Confirmed Fraud", "exposure_usd": 11500},
            {"id": "CS-2023-241", "match_pct": 82, "type": "Account Takeover",
             "outcome": "Confirmed Fraud", "exposure_usd": 7200},
            {"id": "CS-2022-630", "match_pct": 74, "type": "Account Takeover",
             "outcome": "Confirmed Fraud", "exposure_usd": 5600},
        ],
    }
    matches = pattern_db.get(req.fraud_type, [])
    return sorted(matches, key=lambda x: x["match_pct"], reverse=True)[:req.top_n]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="warning")
