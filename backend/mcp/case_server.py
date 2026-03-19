"""
MCP Case Server — Port 8002 (FastAPI implementation)
Tools: get_timeline, get_evidence, get_account_profile, list_cases
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="MCP Case Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Case data ────────────────────────────────────────────────────────────────

CASES: Dict[str, Dict[str, Any]] = {
    "CS-2024-001": {
        "id": "CS-2024-001", "name": "Marcus Volkov", "account": "ACC-78392",
        "type": "Money Laundering", "risk_score": 94, "exposure": 47500,
        "confidence": 92, "status": "Under Investigation", "level": "HIGH",
        "jurisdiction": "Romania / Moldova", "kyc_status": "Failed Verification",
        "registration_date": "2024-01-15",
        "details": {
            "financial": {"deposits": 47500, "withdrawals": 46800, "trades": 3,
                          "pnl": -320, "net_flow": -47180, "fund_recovery_rate": 0.985},
            "behavioral": {"vpn_usage_rate": 0.78, "night_activity_rate": 0.65,
                           "login_anomaly": 0.82, "device_consistency": 0.23,
                           "geo_hops": 7, "behavior_shift_score": 0.91},
            "identity": {"face_match": 0.34, "synthetic_id_prob": 0.91,
                         "document_quality": "Poor", "address_verification": "Failed",
                         "email_risk": "High", "kyc_score": 12},
            "network": {"shared_devices": 3, "shared_ips": 7, "correlated_accounts": 4,
                        "cluster_id": "FR-2024-089", "graph_density": 0.78,
                        "risk_propagation": "High"},
        },
    },
    "CS-2024-002": {
        "id": "CS-2024-002", "name": "Sarah Mitchell", "account": "ACC-82109",
        "type": "Synthetic Identity", "risk_score": 88, "exposure": 15200,
        "confidence": 86, "status": "Under Investigation", "level": "HIGH",
        "jurisdiction": "United States", "kyc_status": "Pending Re-verification",
        "registration_date": "2024-01-20",
        "details": {
            "financial": {"deposits": 15200, "withdrawals": 0, "trades": 12, "pnl": 420},
            "behavioral": {"vpn_usage_rate": 0.45, "night_activity_rate": 0.30,
                           "login_anomaly": 0.67, "device_consistency": 0.55,
                           "geo_hops": 2, "behavior_shift_score": 0.72},
            "identity": {"face_match": 0.34, "synthetic_id_prob": 0.91,
                         "document_quality": "Poor", "address_verification": "Failed",
                         "email_risk": "High", "kyc_score": 18},
            "network": {"shared_devices": 1, "shared_ips": 2, "correlated_accounts": 1,
                        "cluster_id": "FR-2024-102", "graph_density": 0.42,
                        "risk_propagation": "Medium"},
        },
    },
    "CS-2024-003": {
        "id": "CS-2024-003", "name": "James Patterson", "account": "ACC-65441",
        "type": "Account Takeover", "risk_score": 76, "exposure": 8900,
        "confidence": 81, "status": "Pending Review", "level": "MEDIUM",
        "jurisdiction": "United Kingdom", "kyc_status": "Verified",
        "registration_date": "2023-06-10",
        "details": {
            "financial": {"deposits": 8900, "withdrawals": 8900, "trades": 0, "pnl": 0},
            "behavioral": {"vpn_usage_rate": 0.20, "night_activity_rate": 0.85,
                           "login_anomaly": 0.90, "device_consistency": 0.10,
                           "geo_hops": 3, "behavior_shift_score": 0.88},
            "identity": {"face_match": 0.72, "synthetic_id_prob": 0.22,
                         "document_quality": "Good", "address_verification": "Passed",
                         "email_risk": "Medium", "kyc_score": 68},
            "network": {"shared_devices": 0, "shared_ips": 1, "correlated_accounts": 0,
                        "cluster_id": "N/A", "graph_density": 0.05,
                        "risk_propagation": "Low"},
        },
    },
}

TIMELINES: Dict[str, List[Dict[str, Any]]] = {
    "CS-2024-001": [
        {"id": 1, "date": "Jan 15, 2024", "time": "09:23 AM", "event": "Account Created",
         "detail": "New account registered via web portal", "ip": "185.220.101.45",
         "device": "Windows Chrome", "country": "Romania", "risk": "normal"},
        {"id": 2, "date": "Jan 15, 2024", "time": "10:45 AM", "event": "KYC Document Uploaded",
         "detail": "Passport submitted for identity verification (Moldova)",
         "ip": "185.220.101.45", "device": "Windows Chrome", "country": "Moldova", "risk": "medium"},
        {"id": 3, "date": "Jan 22, 2024", "time": "14:30 PM", "event": "First Login",
         "detail": "Account accessed after 7-day dormancy period",
         "ip": "185.220.101.45", "device": "Windows Chrome", "country": "Romania", "risk": "medium"},
        {"id": 4, "date": "Jan 28, 2024", "time": "16:45 PM", "event": "Large Deposit",
         "detail": "Bank transfer of $47,500 received", "ip": "91.108.56.77",
         "device": "Mobile App (Android)", "country": "Romania", "risk": "high"},
        {"id": 5, "date": "Jan 29, 2024", "time": "09:15 AM", "event": "Multiple Withdrawals",
         "detail": "$46,800 withdrawn to cryptocurrency wallet", "ip": "95.142.45.201",
         "device": "Windows Firefox", "country": "Romania", "risk": "critical"},
        {"id": 6, "date": "Jan 30, 2024", "time": "11:20 AM", "event": "Account Trade",
         "detail": "BTC/USDT trade executed — P/L: -$320", "ip": "94.23.148.90",
         "device": "Windows Chrome", "country": "Moldova", "risk": "high"},
    ],
    "CS-2024-002": [
        {"id": 1, "date": "Jan 20, 2024", "time": "11:02 AM", "event": "Account Created",
         "detail": "Registration via mobile app", "ip": "192.168.1.1",
         "device": "iPhone Safari", "country": "United States", "risk": "normal"},
        {"id": 2, "date": "Jan 20, 2024", "time": "11:30 AM", "event": "KYC Submission",
         "detail": "Driver's license submitted — quality issues detected",
         "ip": "192.168.1.1", "device": "iPhone Safari", "country": "United States", "risk": "medium"},
        {"id": 3, "date": "Jan 21, 2024", "time": "09:00 AM", "event": "Transaction Burst",
         "detail": "12 small transactions within 2 hours — velocity anomaly",
         "ip": "10.0.0.1", "device": "Android Chrome", "country": "United States", "risk": "high"},
        {"id": 4, "date": "Jan 29, 2024", "time": "02:15 AM", "event": "Large Transfer Attempt",
         "detail": "$15,200 wire transfer initiated at 2 AM", "ip": "172.16.0.5",
         "device": "Windows Chrome", "country": "United States", "risk": "critical"},
    ],
    "CS-2024-003": [
        {"id": 1, "date": "Jan 24, 2024", "time": "08:15 AM", "event": "Normal Login",
         "detail": "Regular login from New York, United States",
         "ip": "74.125.224.0", "device": "MacBook Safari", "country": "United States", "risk": "normal"},
        {"id": 2, "date": "Jan 25, 2024", "time": "02:47 AM", "event": "Suspicious Login",
         "detail": "Login from London UK at 2:47 AM — geographic anomaly",
         "ip": "185.199.108.0", "device": "Windows Chrome (new)", "country": "United Kingdom", "risk": "critical"},
        {"id": 3, "date": "Jan 25, 2024", "time": "02:51 AM", "event": "Password Changed",
         "detail": "Account password reset 4 minutes after suspicious login",
         "ip": "185.199.108.0", "device": "Windows Chrome (new)", "country": "United Kingdom", "risk": "high"},
        {"id": 4, "date": "Jan 25, 2024", "time": "03:02 AM", "event": "Withdrawal Attempt",
         "detail": "$8,900 withdrawal attempted to unrecognised bank account",
         "ip": "185.199.108.0", "device": "Windows Chrome (new)", "country": "United Kingdom", "risk": "critical"},
    ],
}


# ─── Models ───────────────────────────────────────────────────────────────────

class ListCasesRequest(BaseModel):
    min_risk: int = 0
    fraud_type: Optional[str] = None
    status: Optional[str] = None


# ─── Tools ────────────────────────────────────────────────────────────────────

@app.get("/tools")
def list_tools():
    return {
        "server": "fraud-case-server",
        "port": 8002,
        "tools": ["get_timeline", "get_evidence", "get_account_profile", "list_cases"],
    }


@app.get("/tools/get_timeline/{case_id}")
def get_timeline(case_id: str) -> Dict[str, Any]:
    events = TIMELINES.get(case_id, [])
    if not events:
        raise HTTPException(status_code=404, detail=f"No timeline for {case_id}")
    critical = [e for e in events if e["risk"] in ("critical", "high")]
    return {
        "case_id": case_id, "events": events,
        "total_events": len(events), "high_risk_events": len(critical),
    }


@app.get("/tools/get_evidence/{case_id}/{category}")
def get_evidence(case_id: str, category: str) -> Dict[str, Any]:
    valid = ("financial", "behavioral", "identity", "network")
    if category not in valid:
        raise HTTPException(status_code=400, detail=f"Category must be one of: {valid}")
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return {"case_id": case_id, "category": category, "data": case["details"].get(category, {})}


@app.get("/tools/get_account_profile/{case_id}")
def get_account_profile(case_id: str) -> Dict[str, Any]:
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return {
        "case_id": case["id"], "name": case["name"], "account": case["account"],
        "fraud_type": case["type"], "risk_score": case["risk_score"],
        "exposure_usd": case["exposure"], "confidence_pct": case["confidence"],
        "status": case["status"], "risk_level": case["level"],
        "jurisdiction": case["jurisdiction"], "kyc_status": case["kyc_status"],
        "registration_date": case["registration_date"],
        "details_available": list(case["details"].keys()),
    }


@app.post("/tools/list_cases")
def list_cases(req: ListCasesRequest) -> List[Dict[str, Any]]:
    results = []
    for case in CASES.values():
        if case["risk_score"] < req.min_risk:
            continue
        if req.fraud_type and case["type"].lower() != req.fraud_type.lower():
            continue
        if req.status and case["status"].lower() != req.status.lower():
            continue
        results.append({
            "id": case["id"], "name": case["name"], "account": case["account"],
            "type": case["type"], "risk_score": case["risk_score"],
            "exposure_usd": case["exposure"], "status": case["status"], "level": case["level"],
        })
    return sorted(results, key=lambda x: x["risk_score"], reverse=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
