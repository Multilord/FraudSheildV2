# FraudShield — 3-Minute Demo Script
### V HACK 2026 · Case Study 2: Real-Time Fraud Detection for ASEAN Digital Wallets

> **Setup before you start:**
> - Tab 1: Dashboard — `localhost:3001`
> - Tab 2: Wallet — `localhost:3001/wallet`
> - Backend running on port 8000

---

## [0:00–0:25] Opening — The Problem

**Say:**
> "Every second, millions of digital wallet transactions happen across ASEAN. Fraudsters exploit the speed — a compromised account can be drained in under a minute. FraudShield is a real-time fraud detection engine that scores every transaction before it clears, using a four-model ML ensemble with full explainability."

**Do:**
- Show **Tab 1 — Dashboard** (`localhost:3001`)
- Point to the top row: **Total Transactions, Blocked, Flagged, Fraud Rate**
- Point to **Model Active** badge and **Threat Level** indicator (top right)

**Say:**
> "This is the operations dashboard. [X] transactions processed, [X] blocked — every single one scored in under half a millisecond."

---

## [0:25–0:45] Model Performance — The Engine

**Do:**
- Scroll to the **Model Performance** strip at the bottom of the dashboard
- Point across all five cards left to right

**Say:**
> "Five models work together. XGBoost and LightGBM are our supervised learners — trained on the IEEE-CIS Fraud Detection dataset, 590,000 real transactions. They achieve 0.925 and 0.941 ROC-AUC respectively."

> "Isolation Forest and LOF are unsupervised anomaly detectors — trained only on legitimate transactions, so they catch fraud patterns the supervised models have never seen before. Zero-day detection."

> "The meta-ensemble stacks all four through a logistic regression meta-learner and reaches 0.958. Each model covers the blind spots of the others."

**Do:**
- Point to the **0.43ms latency** tile last

**Say:**
> "And all of that runs in 0.43 milliseconds per transaction on a standard CPU."

---

## [0:45–1:30] Fraud Transaction — Catching the Attack

**Do:**
- Switch to **Tab 2 — Wallet** (`localhost:3001/wallet`)
- Set User ID: `john_tan`
- Transaction type: **Transfer**
- Amount: `45000`
- Country: **Malaysia**, City: **Kuala Lumpur**
- Toggle **Advanced Options** — show new device is enabled

**Say:**
> "John Tan is a real user in our system — average transaction around RM 542. I'm going to simulate what an account takeover looks like: RM 45,000 transfer, 3 AM, brand new device, unknown recipient."

**Do:**
- Hit **Submit Transaction**

**Say (while result animates):**
> "XGBoost, LightGBM, Isolation Forest, LOF — all four score simultaneously, the meta-learner combines them."

**Do:**
- Point to **BLOCK decision** and **risk score near 100**
- Point to **Risk Signals** — new device, off-hours, first transaction to this recipient, amount 83× user average
- Point to **Model Contribution bars** — ML Ensemble and Behavioral both maxed out
- Open **XAI Feature Attribution**

**Say:**
> "This is what separates FraudShield from a rules engine. The XAI layer shows the SHAP contribution of every feature. The amount is at the 99.8th percentile of all transactions ever processed, and 83 standard deviations above John's own history. That's not a rule — the model learned it."

---

## [1:30–1:50] Normal Transaction — No False Positives

**Do:**
- Same user `john_tan`, change amount to `50`
- Type: **Purchase**, same device, same city, midday hour
- Hit **Submit Transaction**

**Say:**
> "Same user. Normal amount for lunch. Watch."

**Do:**
- Point to **APPROVE**, score near 0, ML Ensemble near 0%

**Say:**
> "Clean pass. The model doesn't cry wolf — it reads context. Same person, known device, proportionate spend, business hours. The behavioral profile confirms it's consistent with his history."

---

## [1:50–2:30] Triage — The Analyst Workflow

**Do:**
- Navigate to **Tab 1 → Triage** (`localhost:3001/triage`)
- Show the queue of BLOCK and FLAG cards
- Click **BLOCK** filter to narrow the list
- Click **Investigate** on the RM 45,000 transaction

**Say:**
> "Every flagged and blocked transaction lands here for the fraud ops team. They can filter by severity, search by user or transaction ID, and prioritise the highest-risk cases."

**Do:**
- On the **Case Detail page**, point to the hero: decision, amount, timestamp
- Point to **Risk Signals** on the left
- Point to **Model Contributions** on the right — breakdown showing points from ML, anomaly, behavioral, escalation
- Show **XAI Feature Attribution** — `amt_percentile`, `amt_z_card`, velocity features

**Say:**
> "Every case is fully documented. The analyst sees which signals fired, which models flagged it, and the exact feature weights that drove the score. This makes the block decision defensible — to the customer, to compliance, to regulators."

---

## [2:30–2:50] Live Feed — Real-Time Ops

**Do:**
- Navigate back to **Dashboard → Live Feed tab**
- Switch to Wallet tab — submit a quick transaction: `siti_rahman`, RM 200, payment, Malaysia
- Immediately switch back to Dashboard

**Say:**
> "The dashboard stays live via WebSocket — no polling, no refresh. The moment a transaction is scored anywhere in the system, it appears here. Every analyst on the team sees the same real-time picture simultaneously."

**Do:**
- Point to the new entry appearing in the live feed
- Point to the **stats updating** — total count increments, charts shift

---

## [2:50–3:00] Close

**Do:**
- Zoom out on the Dashboard to show the full view

**Say:**
> "FraudShield — five models, 25 behavioural features, full explainability, sub-millisecond inference, 11 ASEAN currencies. Built to stop fraud before the money moves."

---

## Key Numbers (memorise these)

| | |
|---|---|
| Dataset | IEEE-CIS Fraud Detection — 590,540 transactions, 3.28% fraud rate |
| XGBoost ROC-AUC | **0.9254** |
| LightGBM ROC-AUC | **0.9412** |
| Isolation Forest ROC-AUC | **0.8134** (unsupervised — expected lower) |
| LOF ROC-AUC | **0.8309** (unsupervised) |
| Meta-Ensemble ROC-AUC | **0.9583** (best — stacks all four) |
| Inference latency | **0.43 ms** average |
| Features | 25 wallet-native (no V-feature leakage) |
| Decision tiers | APPROVE · FLAG · BLOCK |
| FLAG threshold | 0.45 composite score |
| BLOCK threshold | 0.72 composite score |
| ASEAN coverage | 11 countries with live currency normalisation |

## Backup Transactions

If a scenario doesn't produce the expected result, use these:

```
High-risk (should BLOCK):   user=john_tan,    amount=45000, type=transfer,  location=Unknown City, MY, new_device=true
Borderline (may FLAG):      user=siti_rahman, amount=5000,  type=cashout,   location=Kuala Lumpur, MY, new_device=true
Normal (should APPROVE):    user=john_tan,    amount=50,    type=payment,   location=Kuala Lumpur, MY, new_device=false
Legit high-value:           user=somchai_k,   amount=2000,  type=transfer,  location=Bangkok, TH,      new_device=false
```

## Why Isolation Forest & LOF AUC Are Lower — If Asked

> "Isolation Forest and LOF are unsupervised — they were never shown a labelled fraud example during training. They learn what *normal* looks like, and flag deviations. A 0.81–0.83 AUC from a model that has never seen fraud is strong, and they catch distribution shifts that the supervised models miss because they've never encountered that pattern before. That's the architectural point — each model covers different failure modes."
