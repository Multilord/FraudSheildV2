"""
MCP Network Server — Port 8003 (FastAPI implementation)
Tools: get_graph, detect_fraud_rings, get_connected_accounts, get_cluster_summary
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="MCP Network Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Fraud clusters ───────────────────────────────────────────────────────────

FRAUD_CLUSTERS: Dict[str, Dict[str, Any]] = {
    "FR-2024-089": {
        "cluster_id": "FR-2024-089",
        "member_accounts": ["ACC-78392", "ACC-44231", "ACC-55892", "ACC-12774", "ACC-98341"],
        "shared_ips": ["185.220.101.45", "91.108.56.77", "95.142.45.201",
                       "94.23.148.90", "185.220.101.90", "91.108.56.100", "95.142.45.250"],
        "shared_devices": ["device-alpha", "device-beta", "device-gamma"],
        "crypto_wallets": ["Wallet-BTC-A8f3", "Wallet-ETH-C2d1"],
        "total_exposure_usd": 94300, "graph_density": 0.78, "risk_propagation": "High",
        "fraud_type": "Money Laundering", "first_seen": "2024-01-15", "status": "Active Investigation",
    },
    "FR-2024-102": {
        "cluster_id": "FR-2024-102",
        "member_accounts": ["ACC-82109", "ACC-71205"],
        "shared_ips": ["172.16.0.5", "10.0.0.1"],
        "shared_devices": ["device-delta"],
        "crypto_wallets": [],
        "total_exposure_usd": 15200, "graph_density": 0.42, "risk_propagation": "Medium",
        "fraud_type": "Synthetic Identity", "first_seen": "2024-01-20", "status": "Monitoring",
    },
}

ACCOUNT_CLUSTERS: Dict[str, str] = {}
for _cluster in FRAUD_CLUSTERS.values():
    for _acc in _cluster["member_accounts"]:
        ACCOUNT_CLUSTERS[_acc] = _cluster["cluster_id"]


# ─── Models ───────────────────────────────────────────────────────────────────

class FraudRingsRequest(BaseModel):
    min_shared_accounts: int = 2
    min_graph_density: float = 0.4


class ConnectedRequest(BaseModel):
    account_id: str
    depth: int = 1


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_graph(subject: str, correlated: List[str], wallets: List[str],
                 ip_groups: List[str], subject_risk: int = 90) -> Dict[str, Any]:
    nodes = [{"id": subject, "label": subject, "risk": subject_risk,
               "type": "subject", "color": "#ef4444"}]
    edges = []
    rng = random.Random(hash(subject))
    for acc in correlated:
        nodes.append({"id": acc, "label": acc, "risk": rng.randint(60, 90),
                       "type": "correlated", "color": "#eab308"})
        edges.append({"from": subject, "to": acc, "weight": round(rng.uniform(0.6, 1.0), 2)})
    for wallet in wallets:
        nodes.append({"id": wallet, "label": wallet, "risk": 70,
                       "type": "wallet", "color": "#a855f7"})
        edges.append({"from": subject, "to": wallet, "weight": round(rng.uniform(0.7, 1.0), 2)})
    for i, ip in enumerate(ip_groups):
        gid = f"IP-Group-{i + 1}"
        nodes.append({"id": gid, "label": ip[:20], "risk": 50,
                       "type": "ip_group", "color": "#6366f1"})
        edges.append({"from": subject, "to": gid, "weight": round(rng.uniform(0.5, 0.9), 2)})
        for acc in correlated[:2]:
            edges.append({"from": acc, "to": gid, "weight": round(rng.uniform(0.4, 0.8), 2)})
    return {"nodes": nodes, "edges": edges}


# ─── Tools ────────────────────────────────────────────────────────────────────

@app.get("/tools")
def list_tools():
    return {
        "server": "fraud-network-server",
        "port": 8003,
        "tools": ["get_graph", "detect_fraud_rings", "get_connected_accounts", "get_cluster_summary"],
    }


@app.get("/tools/get_graph/{account_id}")
def get_graph(account_id: str) -> Dict[str, Any]:
    cluster_id = ACCOUNT_CLUSTERS.get(account_id)
    cluster = FRAUD_CLUSTERS.get(cluster_id) if cluster_id else None
    if cluster:
        correlated = [a for a in cluster["member_accounts"] if a != account_id]
        graph = _build_graph(account_id, correlated, cluster["crypto_wallets"],
                              cluster["shared_ips"][:3])
        return {**graph, "cluster_id": cluster_id,
                "graph_density": cluster["graph_density"],
                "risk_propagation": cluster["risk_propagation"]}
    rng = random.Random(hash(account_id))
    fake = [f"ACC-{rng.randint(10000, 99999)}" for _ in range(2)]
    graph = _build_graph(account_id, fake, [], [])
    return {**graph, "cluster_id": "N/A",
            "graph_density": round(rng.uniform(0.05, 0.25), 2), "risk_propagation": "Low"}


@app.post("/tools/detect_fraud_rings")
def detect_fraud_rings(req: FraudRingsRequest) -> List[Dict[str, Any]]:
    rings = []
    for c in FRAUD_CLUSTERS.values():
        if len(c["member_accounts"]) >= req.min_shared_accounts and \
                c["graph_density"] >= req.min_graph_density:
            rings.append({
                "cluster_id": c["cluster_id"],
                "member_count": len(c["member_accounts"]),
                "member_accounts": c["member_accounts"],
                "graph_density": c["graph_density"],
                "total_exposure_usd": c["total_exposure_usd"],
                "fraud_type": c["fraud_type"],
                "risk_propagation": c["risk_propagation"],
                "status": c["status"],
            })
    return sorted(rings, key=lambda x: x["graph_density"], reverse=True)


@app.post("/tools/get_connected_accounts")
def get_connected_accounts(req: ConnectedRequest) -> Dict[str, Any]:
    cluster_id = ACCOUNT_CLUSTERS.get(req.account_id)
    if not cluster_id:
        return {"account_id": req.account_id, "connected_accounts": [],
                "cluster_id": None, "message": "No known connections"}
    cluster = FRAUD_CLUSTERS[cluster_id]
    direct = [a for a in cluster["member_accounts"] if a != req.account_id]
    return {
        "account_id": req.account_id, "cluster_id": cluster_id,
        "direct_connections": direct, "total_connected": len(direct),
        "shared_ips": cluster["shared_ips"], "shared_devices": cluster["shared_devices"],
    }


@app.get("/tools/get_cluster_summary/{cluster_id}")
def get_cluster_summary(cluster_id: str) -> Dict[str, Any]:
    cluster = FRAUD_CLUSTERS.get(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    return cluster


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
