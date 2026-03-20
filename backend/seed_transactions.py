"""
FraudShield seed — 100 realistic transactions, direct DB insert.

Bypasses the live API so that:
  • Velocity signals do NOT contaminate scoring (no rapid-fire submission)
  • Risk scores are hand-tuned to realistic citizen fraud patterns
  • Decision distribution: ~82 APPROVE  ~11 FLAG  ~7 BLOCK

Amounts are realistic everyday citizen figures:
  MY users (MYR): coffee RM8-20, groceries RM40-180, bills RM50-250,
                   transfers RM100-800, cashout RM100-500
  SG user  (SGD): S$8-250  (displayed raw, similar scale to MYR)
  TH user  (THB): B80-800  (small THB amounts, not millions)

Anomalies (FLAG/BLOCK) are believable edge cases:
  FLAG  → large-ish amount for user, slightly odd hour, or borderline device
  BLOCK → new device + large cashout/transfer at odd hours

Run from the backend/ directory:
    python seed_transactions.py
"""

import json
import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

random.seed(7)

DB = Path(__file__).parent / "data" / "fraudshield.db"
now = datetime.now(timezone.utc)


# ─── helpers ──────────────────────────────────────────────────────────────────

def ts(days_ago: int, hour: int, minute: Optional[int] = None) -> str:
    m = minute if minute is not None else random.randint(0, 59)
    t = (now - timedelta(days=days_ago)).replace(
        hour=hour, minute=m, second=random.randint(5, 58), microsecond=0,
    )
    return t.isoformat()


def txn_id() -> str:
    return f"TXN-{uuid.uuid4().hex[:12].upper()}"


def decision_from_score(score: int) -> str:
    if score >= 72:
        return "BLOCK"
    if score >= 45:
        return "FLAG"
    return "APPROVE"


def reasons_for(score: int, tx: dict) -> list:
    base = []
    if score >= 72:
        if tx.get("is_new_device"):
            base.append("New device detected")
        if tx["transaction_type"] == "cashout":
            base.append("Large cashout outside normal pattern")
        elif tx["transaction_type"] == "transfer":
            base.append("High-value transfer to unknown recipient")
        base.append("Behavioural anomaly: amount z-score elevated")
    elif score >= 45:
        if tx["amount"] > 500:
            base.append("Amount higher than user baseline")
        if tx.get("is_new_device"):
            base.append("Unrecognised device")
        base.append("Moderate risk — flagged for review")
    else:
        base.append("Transaction within normal parameters")
    return base


def features_blob(score: int, new_device: bool) -> str:
    ml_p = round(score / 100 * 0.62, 4)
    beh  = round(score / 100 * 0.20, 4)
    anom = round(score / 100 * 0.12, 4)
    esc  = round(score / 100 * 0.06, 4)
    return json.dumps({
        "model_breakdown": {
            "ml_ensemble": ml_p,
            "behavioral":  beh,
            "anomaly":     anom,
            "escalation":  esc,
        },
        "model_raw_probabilities": {
            "xgboost": round(ml_p * 1.05, 4),
            "lightgbm": round(ml_p * 0.95, 4),
            "ml_ensemble": ml_p,
            "behavioral": beh,
            "final": round(score / 100, 4),
        },
        "xai_top_features": [],
        "explanation": "Seeded historical transaction",
        "top_risk_factors": [],
    })


# ─── transaction definitions ──────────────────────────────────────────────────
# Each row: dict of all fields + risk_score
# risk_score drives decision automatically.

def tx(uid, amt, ttype, ts_, risk,
       device, ip, location, new_device=False,
       merchant="", category="", recipient="", note=""):
    return {
        "transaction_id":   txn_id(),
        "user_id":          uid,
        "amount":           float(amt),
        "transaction_type": ttype,
        "device_type":      "mobile",
        "device_id":        device,
        "ip_address":       ip,
        "location":         location,
        "merchant":         merchant,
        "merchant_category": category,
        "timestamp":        ts_,
        "risk_score":       risk,
        "decision":         decision_from_score(risk),
        "reasons":          reasons_for(risk, {"amount": amt, "transaction_type": ttype,
                                               "is_new_device": new_device}),
        "confidence":       round(0.70 + random.uniform(0, 0.25), 3),
        "latency_ms":       round(random.uniform(0.4, 1.2), 3),
        "features":         features_blob(risk, new_device),
        "raw_payload":      json.dumps({"user_id": uid, "amount": amt,
                                        "transaction_type": ttype,
                                        "is_new_device": new_device,
                                        "location": location}),
    }


# Short aliases
ALI   = ("ali_hassan",  "dev-ali-ph",    "60.48.11.1",   "Kuala Lumpur, MY")
SITI  = ("siti_rahman", "dev-siti-ph",   "60.48.22.5",   "Kuala Lumpur, MY")
MEI   = ("mei_ling",    "dev-mei-ph",    "59.178.33.12",  "Penang, MY")
JOHN  = ("john_tan",    "dev-john-ph",   "211.25.44.8",  "Johor Bahru, MY")
NURUL = ("nurul_ain",   "dev-nurul-ph",  "60.48.55.6",   "Shah Alam, MY")
JAMES = ("james_lim",   "dev-james-ph",  "118.200.88.3", "Singapore, SG")
SOM   = ("somchai_k",   "dev-som-ph",    "171.96.12.8",  "Bangkok, TH")

def mk(user_tuple, amt, ttype, ts_, risk, merchant="", category="", recipient="",
        new_device=False, device_override=None, ip_override=None, note=""):
    uid, dev, ip, loc = user_tuple
    if device_override:
        dev = device_override
    if ip_override:
        ip = ip_override
    return tx(uid, amt, ttype, ts_, risk, dev, ip, loc,
              new_device=new_device, merchant=merchant,
              category=category, recipient=recipient, note=note)


ROWS = [

    # ══════════════════════════════════════════════════════════════════════════
    # ali_hassan  — KL — salaried professional — 25 transactions
    # Normal baseline: food RM10-40, groceries RM50-150, bills RM100-250
    # ══════════════════════════════════════════════════════════════════════════
    mk(ALI, 12.0,  "merchant", ts(30,8),  18, "Tealive",          "food"),
    mk(ALI, 38.5,  "merchant", ts(29,12), 14, "Grab Food",         "food"),
    mk(ALI, 175.0, "payment",  ts(28,10), 21, "Tenaga Nasional",   "utilities"),
    mk(ALI, 62.0,  "merchant", ts(27,17), 16, "Jaya Grocer",       "groceries"),
    mk(ALI, 18.0,  "merchant", ts(26,13), 12, "McDonald's",        "food"),
    mk(ALI, 35.0,  "payment",  ts(25,9),  19, "Touch n Go",        "transport"),
    mk(ALI, 500.0, "transfer", ts(24,19), 28, "","", "siti_rahman"),
    mk(ALI, 22.0,  "merchant", ts(23,8),  15, "Starbucks",         "food"),
    mk(ALI, 135.0, "merchant", ts(22,15), 17, "Giant Supermarket", "groceries"),
    mk(ALI, 240.0, "payment",  ts(21,11), 23, "Maxis",             "telecom"),
    mk(ALI, 28.0,  "merchant", ts(20,12), 13, "Grab Food",         "food"),
    mk(ALI, 300.0, "cashout",  ts(19,14), 25, "","",""),
    mk(ALI, 44.0,  "merchant", ts(18,16), 16, "Watsons",           "health"),
    mk(ALI, 750.0, "transfer", ts(17,10), 31, "","", "nurul_ain"),
    mk(ALI, 15.0,  "merchant", ts(16,8),  11, "Old Town White Coffee","food"),
    # ANOMALY — ali: large cashout at 2 am, unknown device → BLOCK
    mk(ALI, 4200.0,"cashout",  ts(15,2),  82,
       new_device=True, device_override="dev-ali-ROGUE-X9", ip_override="45.112.55.99"),
    mk(ALI, 68.0,  "merchant", ts(14,14), 20, "Parkson",          "retail"),
    mk(ALI, 170.0, "payment",  ts(13,11), 22, "Tenaga Nasional",  "utilities"),
    mk(ALI, 32.0,  "merchant", ts(12,19), 14, "KFC",              "food"),
    mk(ALI, 9.0,   "merchant", ts(11,7),  10, "Speed99",          "food"),
    mk(ALI, 115.0, "merchant", ts(10,17), 18, "Aeon",             "groceries"),
    mk(ALI, 400.0, "transfer", ts(8,11),  29, "","", "mei_ling"),
    mk(ALI, 24.0,  "merchant", ts(6,13),  13, "Grab Food",        "food"),
    mk(ALI, 200.0, "cashout",  ts(4,15),  24, "","",""),
    mk(ALI, 42.0,  "merchant", ts(2,18),  15, "Jaya Grocer",      "groceries"),

    # ══════════════════════════════════════════════════════════════════════════
    # siti_rahman — KL — housewife / small business — 17 transactions
    # ══════════════════════════════════════════════════════════════════════════
    mk(SITI, 90.0,  "merchant", ts(30,10), 17, "Tesco Extra",       "groceries"),
    mk(SITI, 48.0,  "merchant", ts(27,12), 15, "Grab Food",         "food"),
    mk(SITI, 160.0, "payment",  ts(25,9),  20, "Indah Water",       "utilities"),
    mk(SITI, 200.0, "topup",    ts(23,11), 14, "","",""),
    mk(SITI, 33.0,  "merchant", ts(21,14), 13, "Shopee",            "retail"),
    mk(SITI, 75.0,  "merchant", ts(19,10), 18, "Caring Pharmacy",   "health"),
    mk(SITI, 125.0, "merchant", ts(17,15), 17, "Giant Supermarket", "groceries"),
    mk(SITI, 350.0, "transfer", ts(15,16), 26, "","", "ali_hassan"),
    mk(SITI, 22.0,  "merchant", ts(13,9),  12, "Tealive",           "food"),
    mk(SITI, 88.0,  "merchant", ts(11,14), 19, "Parkson",           "retail"),
    # ANOMALY — siti: new device, large night transfer → BLOCK
    mk(SITI, 2800.0,"transfer", ts(8,23),  79,
       new_device=True, device_override="dev-siti-STRANGE-77", ip_override="103.28.89.55",
       recipient="acct_unknown_88"),
    mk(SITI, 55.0,  "merchant", ts(6,10),  16, "Aeon Big",          "groceries"),
    mk(SITI, 180.0, "payment",  ts(4,9),   21, "Tenaga Nasional",   "utilities"),
    mk(SITI, 40.0,  "merchant", ts(2,12),  14, "Grab Food",         "food"),
    mk(SITI, 65.0,  "merchant", ts(1,11),  17, "Cold Storage",      "groceries"),
    # FLAG — siti: late-evening higher-than-usual cashout
    mk(SITI, 950.0, "cashout",  ts(9,22),  52),
    mk(SITI, 150.0, "topup",    ts(3,10),  14),

    # ══════════════════════════════════════════════════════════════════════════
    # mei_ling — Penang — young professional / student — 17 transactions
    # ══════════════════════════════════════════════════════════════════════════
    mk(MEI, 8.5,   "merchant", ts(30,15), 11, "Chatime",             "food"),
    mk(MEI, 15.90, "payment",  ts(28,10),  9, "Netflix",             "entertainment"),
    mk(MEI, 75.0,  "topup",    ts(26,9),  13, "","",""),
    mk(MEI, 29.0,  "merchant", ts(24,16), 14, "Shopee",              "retail"),
    mk(MEI, 12.0,  "merchant", ts(22,13), 10, "Tealive",             "food"),
    mk(MEI, 85.0,  "merchant", ts(20,15), 18, "Gurney Plaza",        "retail"),
    mk(MEI, 100.0, "transfer", ts(18,11), 22, "","", "ali_hassan"),
    mk(MEI, 9.90,  "payment",  ts(17,10),  9, "Spotify",             "entertainment"),
    mk(MEI, 45.0,  "merchant", ts(15,17), 15, "Guardian",            "health"),
    # FLAG — mei: late-night larger shopping, new tablet
    mk(MEI, 1650.0,"payment",  ts(12,1),  58,
       new_device=True, device_override="dev-mei-TABLET-NEW", ip_override="59.178.99.22",
       merchant="Lazada", category="retail"),
    mk(MEI, 18.0,  "merchant", ts(10,12), 12, "Grab Food",           "food"),
    mk(MEI, 100.0, "topup",    ts(8,9),   13, "","",""),
    mk(MEI, 55.0,  "merchant", ts(6,14),  17, "Aeon Mall Penang",    "retail"),
    mk(MEI, 22.0,  "merchant", ts(4,13),  12, "McDonald's",          "food"),
    mk(MEI, 30.0,  "payment",  ts(2,10),  13, "Digi",                "telecom"),
    mk(MEI, 250.0, "transfer", ts(14,15), 26, "","", "john_tan"),
    mk(MEI, 38.0,  "merchant", ts(5,16),  15, "Watsons",             "health"),

    # ══════════════════════════════════════════════════════════════════════════
    # john_tan — Johor Bahru — small business owner — 16 transactions
    # ══════════════════════════════════════════════════════════════════════════
    mk(JOHN, 65.0,  "merchant", ts(29,7),  18, "Petronas",           "fuel"),
    mk(JOHN, 220.0, "merchant", ts(27,14), 22, "Sutera Mall JB",     "retail"),
    mk(JOHN, 450.0, "transfer", ts(25,11), 28, "","", "siti_rahman"),
    mk(JOHN, 35.0,  "merchant", ts(23,13), 14, "Nasi Kandar",        "food"),
    mk(JOHN, 180.0, "payment",  ts(21,10), 21, "Telekom Malaysia",   "telecom"),
    mk(JOHN, 500.0, "cashout",  ts(19,15), 25, "","",""),
    mk(JOHN, 90.0,  "merchant", ts(17,17), 17, "Jusco Tebrau",       "groceries"),
    mk(JOHN, 600.0, "transfer", ts(15,9),  30, "","", "mei_ling"),
    mk(JOHN, 28.0,  "merchant", ts(13,12), 13, "Grab Food",          "food"),
    # FLAG — john: rapid-succession transfers at 3am (velocity pattern)
    mk(JOHN, 750.0, "transfer", ts(5,3,8),  55, "","", "acct_x1"),
    mk(JOHN, 750.0, "transfer", ts(5,3,19), 60, "","", "acct_x2"),
    mk(JOHN, 700.0, "transfer", ts(5,3,34), 63, "","", "acct_x3"),
    # BLOCK — john: large cashout, new POS device, 4am
    mk(JOHN, 3800.0,"cashout",  ts(3,4),   77,
       new_device=True, device_override="dev-john-POS-UNKNOWN", ip_override="43.229.88.99"),
    mk(JOHN, 75.0,  "merchant", ts(1,8),   18, "Petronas",           "fuel"),
    mk(JOHN, 110.0, "merchant", ts(9,16),  19, "AEON Tebrau",        "groceries"),
    mk(JOHN, 320.0, "transfer", ts(20,15), 27, "","", "ali_hassan"),

    # ══════════════════════════════════════════════════════════════════════════
    # nurul_ain — Shah Alam — teacher — 12 transactions
    # ══════════════════════════════════════════════════════════════════════════
    mk(NURUL, 150.0, "topup",    ts(28,9),  14, "","",""),
    mk(NURUL, 42.0,  "merchant", ts(25,16), 15, "Mydin",             "groceries"),
    mk(NURUL, 195.0, "payment",  ts(22,10), 21, "Tenaga Nasional",   "utilities"),
    mk(NURUL, 25.0,  "merchant", ts(19,12), 12, "Secret Recipe",     "food"),
    mk(NURUL, 300.0, "transfer", ts(16,14), 26, "","", "ali_hassan"),
    mk(NURUL, 80.0,  "merchant", ts(13,11), 18, "Popular Bookstore", "education"),
    mk(NURUL, 55.0,  "merchant", ts(10,17), 16, "Tesco",             "groceries"),
    mk(NURUL, 18.0,  "merchant", ts(7,13),  11, "Grab Food",         "food"),
    mk(NURUL, 145.0, "payment",  ts(4,10),  20, "Astro",             "entertainment"),
    mk(NURUL, 60.0,  "merchant", ts(2,15),  16, "Parkson",           "retail"),
    mk(NURUL, 35.0,  "merchant", ts(1,12),  13, "Grab Food",         "food"),
    # FLAG — nurul: unusually large payment (above her normal range)
    mk(NURUL, 820.0, "payment",  ts(6,21),  51, "Online Merchant",   "retail"),

    # ══════════════════════════════════════════════════════════════════════════
    # james_lim — Singapore — office worker — SGD amounts (S$8-250 normal)
    # ══════════════════════════════════════════════════════════════════════════
    mk(JAMES, 8.5,   "merchant", ts(29,8),  14, "Kopitiam",           "food"),
    mk(JAMES, 45.0,  "merchant", ts(26,17), 17, "FairPrice Finest",  "groceries"),
    mk(JAMES, 120.0, "payment",  ts(23,11), 22, "Singtel",            "telecom"),
    mk(JAMES, 18.0,  "merchant", ts(20,9),  13, "Grab",               "transport"),
    mk(JAMES, 250.0, "transfer", ts(17,14), 27, "","", "ali_hassan"),
    mk(JAMES, 35.0,  "merchant", ts(14,12), 15, "Hawker Centre",      "food"),
    mk(JAMES, 95.0,  "merchant", ts(11,17), 18, "NTUC FairPrice",     "groceries"),
    mk(JAMES, 55.0,  "payment",  ts(8,10),  19, "SP Group",           "utilities"),
    # BLOCK — james: large transfer to unknown, new desktop, 2am
    mk(JAMES, 1200.0,"transfer", ts(5,2),   76,
       new_device=True, device_override="dev-james-DESKTOP-NEW", ip_override="45.77.100.22",
       recipient="acct_offshore_X"),
    mk(JAMES, 22.0,  "merchant", ts(2,8),   13, "Toast Box",          "food"),

    # ══════════════════════════════════════════════════════════════════════════
    # somchai_k — Bangkok — freelancer — THB amounts (B80-800 normal)
    # ══════════════════════════════════════════════════════════════════════════
    mk(SOM, 85.0,  "merchant", ts(28,7),  14, "7-Eleven",           "food"),
    mk(SOM, 450.0, "merchant", ts(24,15), 17, "Big C",              "groceries"),
    mk(SOM, 199.0, "payment",  ts(20,10), 16, "TrueMove",           "telecom"),
    mk(SOM, 120.0, "merchant", ts(16,14), 15, "Grab Thailand",      "transport"),
    mk(SOM, 800.0, "transfer", ts(12,11), 28, "","", "mei_ling"),
    mk(SOM, 350.0, "merchant", ts(8,16),  18, "Robinson Department","retail"),
    mk(SOM, 65.0,  "merchant", ts(5,19),  13, "MK Restaurant",      "food"),
    # BLOCK — somchai: large cashout, rogue POS device, 3am
    mk(SOM, 8500.0,"cashout",  ts(6,3),   80,
       new_device=True, device_override="dev-som-POS-ROGUE", ip_override="45.64.200.11"),
    mk(SOM, 180.0, "merchant", ts(3,13),  17, "Makro",              "groceries"),
    mk(SOM, 400.0, "topup",    ts(1,9),   15),

]

assert len(ROWS) == 107, f"Got {len(ROWS)} rows, expected 107"


# ─── Ensure DB schema exists ──────────────────────────────────────────────────

def ensure_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
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
        recipients              TEXT NOT NULL DEFAULT '[]',
        first_seen              TEXT NOT NULL,
        updated_at              TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_st_user_id   ON scored_transactions(user_id);
    CREATE INDEX IF NOT EXISTS idx_st_decision   ON scored_transactions(decision);
    CREATE INDEX IF NOT EXISTS idx_st_timestamp  ON scored_transactions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_st_risk_score ON scored_transactions(risk_score);
    CREATE INDEX IF NOT EXISTS idx_st_user_ts   ON scored_transactions(user_id, timestamp);
    """)
    conn.commit()
    return conn


# ─── Insert transactions ──────────────────────────────────────────────────────

conn = ensure_db()

print(f"Inserting {len(ROWS)} transactions into {DB} ...\n")
print(f"  {'#':>3}  {'Decision':<7}  {'Risk':>4}  {'User':<18}  {'Type':<12}  {'Amount':>10}")
print("  " + "-" * 70)

counts = {"APPROVE": 0, "FLAG": 0, "BLOCK": 0}

for i, row in enumerate(ROWS, 1):
    row["reasons"] = json.dumps(row["reasons"])
    conn.execute("""
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
    """, row)
    counts[row["decision"]] += 1
    d = row["decision"]
    marker = "[BLOCK]" if d == "BLOCK" else "[FLAG] " if d == "FLAG" else "[OK]   "
    print(f"  {i:>3}  {marker}  {row['risk_score']:>4}  {row['user_id']:<18}  "
          f"{row['transaction_type']:<12}  {row['amount']:>10,.1f}")

conn.commit()
print()


# ─── Build user_history from inserted data ────────────────────────────────────

print("Building user_history profiles ...")

users = conn.execute(
    "SELECT DISTINCT user_id FROM scored_transactions"
).fetchall()

for (uid,) in users:
    rows = conn.execute(
        "SELECT amount, decision, device_id, location, merchant, timestamp "
        "FROM scored_transactions WHERE user_id = ? ORDER BY timestamp",
        (uid,),
    ).fetchall()

    amounts   = [r[0] for r in rows]
    n         = len(amounts)
    total     = sum(amounts)
    avg       = total / n
    variance  = sum((a - avg) ** 2 for a in amounts) / n
    std       = variance ** 0.5
    max_amt   = max(amounts)
    fraud_cnt = sum(1 for r in rows if r[1] == "BLOCK")
    flag_cnt  = sum(1 for r in rows if r[1] == "FLAG")
    first_ts  = rows[0][5]
    last_ts   = rows[-1][5]

    # Keep last 20 unique values
    def top20(vals):
        seen, out = set(), []
        for v in reversed(vals):
            if v and v not in seen:
                seen.add(v); out.append(v)
        return out[:20]

    device_ids = top20([r[2] for r in rows])
    locations  = top20([r[3] for r in rows])
    merchants  = top20([r[4] for r in rows])

    conn.execute("""
        INSERT OR REPLACE INTO user_history (
            user_id, transaction_count, total_amount, avg_amount, std_amount,
            max_amount, fraud_count, flag_count, last_transaction_time,
            device_ids, locations, merchants, recipients, first_seen, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'[]',?,?)
    """, (
        uid, n, round(total, 4), round(avg, 4), round(std, 4), round(max_amt, 4),
        fraud_cnt, flag_cnt, last_ts,
        json.dumps(device_ids), json.dumps(locations), json.dumps(merchants),
        first_ts, last_ts,
    ))

conn.commit()
conn.close()


# ─── Summary ─────────────────────────────────────────────────────────────────

print()
print("=" * 50)
print(f"  Total inserted  : {len(ROWS)}")
print(f"  APPROVE         : {counts['APPROVE']}  ({counts['APPROVE']}%)")
print(f"  FLAG            : {counts['FLAG']}")
print(f"  BLOCK           : {counts['BLOCK']}")
print(f"  Fraud rate      : {round((counts['FLAG']+counts['BLOCK'])/len(ROWS)*100,1)}%")
print("=" * 50)
print()

for (uid,) in [("ali_hassan",), ("siti_rahman",), ("mei_ling",),
               ("john_tan",), ("nurul_ain",), ("james_lim",), ("somchai_k",)]:
    conn2 = sqlite3.connect(str(DB))
    conn2.row_factory = sqlite3.Row
    r = conn2.execute("SELECT * FROM user_history WHERE user_id=?", (uid,)).fetchone()
    if r:
        print(f"  {uid:<18}  txns={r['transaction_count']:>2}  "
              f"avg={r['avg_amount']:>8,.1f}  std={r['std_amount']:>8,.1f}  "
              f"blocks={r['fraud_count']}  flags={r['flag_count']}")
    conn2.close()

print()
print("Done. Start the backend, then refresh the dashboard.")
