# FraudShield

Real-time fraud detection platform for ASEAN digital wallets. Built for V HACK 2026.

Scores every transaction in under 50 ms using a 4-model ML ensemble with explainable AI, behavioral profiling, and a live fraud intelligence dashboard.

---

## Features

- **Real-time scoring** — POST a wallet transaction, get a risk score + decision (APPROVE / FLAG / BLOCK) in < 50 ms
- **4-model ensemble** — XGBoost + LightGBM (ML core) + Isolation Forest + LOF (anomaly detection), combined via a meta-learner
- **Behavioral profiling** — Welford's online algorithm tracks per-user spending baseline; deviations raise risk
- **XAI explanations** — SHAP-based feature contributions identify _why_ a transaction was flagged
- **Live dashboard** — WebSocket-powered feed with hourly trend charts, risk distribution histogram, KPI cards, and case management
- **Multi-currency** — MYR default with full ASEAN currency support (SGD, THB, IDR, PHP, VND, and more) + EUR/GBP
- **Triage queue** — Analyst-facing page for reviewing FLAG/BLOCK cases
- **Case detail** — Per-transaction breakdown showing model scores, XAI features, and behavioral context

---

## Solution Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        FRAUD INVESTIGATION DASHBOARD                             │
│                         Next.js 14  ·  Tailwind CSS                              │
│         Live Feed · Triage Queue · Case Detail · XAI Report Viewer               │
└────────────────────────────┬─────────────────────────────────────────────────────┘
                             │  REST + WebSocket
                             ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND API                                         │
│                   FastAPI  ·  Pydantic  ·  Uvicorn  (Python)                     │
│              Real-time transaction processing  ·  XAI Report generation          │
└───────────────┬──────────────────────────────────────────────┬───────────────────┘
                │  score request                               │  natural language query
                ▼                                             ▼
┌───────────────────────────────────┐          ┌──────────────────────────────────┐
│          ML ENGINE                │          │          MCP SERVER              │
│                                   │          │     Model Context Protocol       │
│  ┌─────────────────────────────┐  │          │                                  │
│  │     SUPERVISED MODELS       │  │          │  ┌────────────────────────────┐  │
│  │  XGBoost  ·  LightGBM       │  │          │  │        AI AGENT            │  │
│  │                             │  │          │  │  "Why was this flagged?"   │  │
│  │  Trained on IEEE-CIS dataset│  │          │  │                            │  │
│  │  590,000+ labeled txns      │  │          │  │  Fetches data · Executes   │  │
│  │  Detects known patterns     │  │          │  │  scoring tools · Returns   │  │
│  └─────────────┬───────────────┘  │          │  │  plain-language answers    │  │
│                │                  │          │  └────────────────────────────┘  │
│                ├── Weighted ──────┼──────────┤                                  │
│                │   Ensemble       │          └──────────────────────────────────┘
│  ┌─────────────▼───────────────┐  │
│  │    UNSUPERVISED MODELS      │  │
│  │  Isolation Forest  ·  LOF   │  │
│  │  (PyOD library)             │  │
│  │                             │  │
│  │  No labels required         │  │
│  │  Detects novel fraud types  │  │
│  └─────────────────────────────┘  │
│                                   │
│  Final Score = weighted ensemble  │
│  → APPROVE / FLAG / BLOCK         │
└───────────────────────────────────┘
```

### How It Works

| Layer | Role |
|-------|------|
| **Frontend** | Next.js fraud investigation dashboard where analysts review transactions, inspect XAI reports, and manage the triage queue |
| **Backend API** | FastAPI + Pydantic validates every incoming transaction, orchestrates the scoring pipeline, and returns a full XAI report — not just a score, but a ranked breakdown of contributing features |
| **Supervised Models** | XGBoost and LightGBM trained on the IEEE-CIS dataset to detect known fraud patterns with high precision |
| **Unsupervised Models** | Isolation Forest and Local Outlier Factor (via PyOD) detect anomalies purely from transaction structure — catching brand-new fraud types the supervised models have never seen |
| **Ensemble** | All four model scores are combined via a weighted meta-learner into a single risk score per transaction |
| **MCP Server** | Model Context Protocol server exposes backend tools to an AI Agent — investigators query the system in plain English ("why was this flagged?") and receive a clear, readable answer without manual digging |

**Scoring pipeline:**

```
Transaction → Feature Engineering → Supervised:   XGBoost + LightGBM  ─┐
                                  → Unsupervised: Isolation Forest + LOF ┼─► Weighted Ensemble Score
                                  → Behavioral:   Velocity + Deviation  ─┘
                                  → Escalation Rules
                                  → Final Score = 0.60×ML + 0.15×anomaly + 0.20×behavioral + escalation
                                  → Decision: APPROVE / FLAG / BLOCK
                                  → XAI Report: per-feature SHAP contributions
```

---

## Project Structure

```
FraudShield/
├── backend/
│   ├── main.py                    # FastAPI app, API routes, WebSocket feed
│   ├── db/
│   │   └── database.py            # SQLite layer (transactions, user profiles, velocity)
│   ├── ml/
│   │   └── engine.py              # Fraud scoring engine (load, score, explain)
│   ├── training/
│   │   ├── train_engine.py        # Model training pipeline
│   │   ├── feature_engineering.py # Feature creation and preprocessing
│   │   ├── evaluate.py            # Metrics and latency benchmarks
│   │   ├── thresholds.py          # Precision-recall threshold optimisation
│   │   └── REPORT.md              # Training methodology and expected metrics
│   ├── data/
│   │   └── analyzer.py            # Dashboard stats, charts, case queries
│   ├── models/                    # Trained model artifacts (gitignored)
│   ├── seed_transactions.py       # Seed 107 realistic historical transactions
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # Fraud Intelligence Dashboard
│   │   ├── triage/page.tsx        # Analyst triage queue
│   │   ├── wallet/page.tsx        # Transaction submission (demo)
│   │   ├── case/[id]/page.tsx     # Case detail view
│   │   ├── CurrencyContext.tsx    # ASEAN multi-currency provider
│   │   └── layout.tsx             # Root layout + nav
│   └── lib/
│       └── api.ts                 # Typed API client + WebSocket hook
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) IEEE-CIS Fraud Detection dataset from Kaggle for training

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the environment file and configure:

```bash
cp .env.example .env
```

Start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server wipes and reinitialises the database on every startup (clean-slate design). Seed historical data after startup:

```bash
python seed_transactions.py
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev -- --port 3001
```

Open [http://localhost:3001](http://localhost:3001).

---


## API Reference

### POST `/api/wallet/transaction`

Score a transaction for fraud.

**Request body:**

```json
{
  "user_id": "ali_hassan",
  "amount": 250.00,
  "transaction_type": "transfer",
  "recipient_id": "siti_rahman",
  "device_type": "mobile",
  "device_id": "dev-ali-ph",
  "ip_address": "60.48.11.1",
  "location": "Kuala Lumpur, MY",
  "is_new_device": false,
  "merchant": "",
  "merchant_category": ""
}
```

**Response:**

```json
{
  "risk_score": 28,
  "decision": "APPROVE",
  "confidence": 0.91,
  "explanation": "Low-risk transfer consistent with user history.",
  "top_risk_factors": [],
  "transaction_id": "TXN-4A9F2B...",
  "latency_ms": 0.82,
  "model_breakdown": {
    "ml_ensemble": 0.17,
    "anomaly": 0.04,
    "behavioral": 0.05,
    "escalation": 0.02
  }
}
```

### GET `/api/dashboard/stats`

Aggregate fraud statistics (totals, fraud rate, amounts by decision).

### GET `/api/dashboard/transactions`

Paginated list of scored transactions. Supports `?decision=FLAG` filter.

### GET `/api/dashboard/charts`

Hourly trend (last 24h) and risk score distribution histogram data.

### GET `/api/dashboard/cases/{transaction_id}`

Full detail for a single transaction including XAI features.

### GET `/api/health`

Service health, model status, and active thresholds.

### WS `/ws/alerts`

WebSocket live feed. Broadcasts each scored transaction as a JSON message in real time.

---

## Decision Thresholds

| Decision | Threshold | Meaning |
|----------|-----------|---------|
| **BLOCK** | score ≥ 72 | High-confidence fraud — transaction stopped immediately |
| **FLAG** | 45 ≤ score < 72 | Suspicious — sent to analyst triage queue |
| **APPROVE** | score < 45 | Low risk — processed normally |

Thresholds are calibrated on the precision-recall curve targeting ≥ 85% precision at BLOCK and ≥ 60% at FLAG, minimising false blocks (blocking a legitimate transaction damages customer trust more than missing a low-value fraud event).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Uvicorn, Pydantic |
| ML Models | XGBoost, LightGBM, scikit-learn, PyOD |
| Database | SQLite (WAL mode, thread-local connections) |
| Frontend | Next.js 14, React 18, TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Real-time | WebSockets (native FastAPI) |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |
| `DEMO_MODE` | `false` | If `true`, stats endpoints return placeholder data |
| `ANTHROPIC_API_KEY` | — | For LLM-powered case insights (optional) |

---

*Built for V HACK 2026*
