"""
FraudShield — Synthetic Wallet Transaction Dataset Generator
=============================================================

Generates a realistic wallet fraud dataset using ONLY wallet-native features
(features available at real-time inference), stored locally at:

    backend/data/synthetic_wallet_fraud.parquet

This eliminates the train/inference mismatch that caused weak ML signals when
training on IEEE-CIS V-features that are median-imputed at wallet inference.

Fraud patterns encoded (maps directly to wallet inference signals):
  P1 – Large cashout + new device + off-hours   (D10=9999, id_01=-5, hour 0-4)
  P2 – Velocity burst                            (C2 high, C3 high)
  P3 – New location + new recipient + elevated   (D3=9999, C9=1)
  P4 – Extreme amount vs user baseline           (very high amt_z_card)
  P5 – High absolute amount, minimal other noise (is_high_amount, amt_percentile)

Usage:
    cd backend
    python training/generate_synthetic_data.py
    python training/generate_synthetic_data.py --rows 250000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR     = _BACKEND_DIR / "data"
OUTPUT_PATH  = DATA_DIR / "synthetic_wallet_fraud.parquet"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    n_total:    int   = 250_000,
    fraud_rate: float = 0.035,
    seed:       int   = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic wallet fraud DataFrame.

    Columns produced (match get_wallet_feature_vector keys exactly):
        TransactionAmt, TransactionDT, card1,
        C1, C2, C3, C9, C11, C14,
        D1, D3, D4, D10, D15,
        id_01, M5, ProductCD, DeviceType, isFraud

    engineer_features() will add:
        amt_log, hour_of_day, day_of_week, amt_to_dist_ratio,
        is_high_amount, is_night_transaction, amt_vs_card_mean,
        amt_z_card, amt_percentile
    """
    rng = np.random.RandomState(seed)

    n_fraud  = int(n_total * fraud_rate)
    n_normal = n_total - n_fraud

    # ── User population ────────────────────────────────────────────────────
    n_users = 10_000
    # Log-normal user average spending (USD): median ~$25, range $2-$800
    user_avg = np.exp(rng.normal(3.2, 1.2, n_users)).clip(2.0, 800.0)
    user_std = user_avg * rng.uniform(0.2, 0.8, n_users)
    user_tx_counts = rng.randint(5, 300, n_users).astype(float)

    # ── NORMAL transactions (vectorized) ──────────────────────────────────
    uid_n    = rng.randint(0, n_users, n_normal)
    avg_n    = user_avg[uid_n]
    std_n    = user_std[uid_n]

    # Amount: log-normal around user average
    amt_n = (avg_n * np.exp(rng.normal(0, 0.35, n_normal))).clip(0.5, 4999.0)

    # Time: 6-month window, biased toward 8am-9pm business hours
    day_n  = rng.randint(0, 180, n_normal) * 86400
    hour_n = rng.choice(
        np.concatenate([
            np.arange(8, 21),   # 13 daytime hours, weight 3×
            np.arange(8, 21),
            np.arange(8, 21),
            np.arange(6, 8),    # early morning
            np.arange(21, 24),  # late evening
            np.arange(0, 6),    # night (rare)
        ]),
        n_normal,
    )
    dt_n = day_n + hour_n * 3600 + rng.randint(0, 3600, n_normal)

    # C/D features for normal transactions
    c1_n  = user_tx_counts[uid_n]
    c2_n  = rng.choice([0, 0, 0, 0, 1, 1, 2], n_normal).astype(float)
    c3_n  = rng.choice([0, 0, 1, 1, 2, 3, 4, 5], n_normal).astype(float)
    c9_n  = rng.choice([0, 0, 0, 0, 0, 1], n_normal).astype(float)
    d1_n  = rng.uniform(0, 1000, n_normal)
    # Most normal tx: known device (D10=0), known location (D3=0)
    d10_n = np.where(rng.random(n_normal) < 0.04, 9999.0, 0.0)
    d3_n  = np.where(rng.random(n_normal) < 0.05, 9999.0, 0.0)
    d15_n = d3_n.copy()
    id01_n = np.where(d10_n > 0, -5.0, 0.0)
    m5_n   = np.where(d10_n > 0, 'F', 'T')
    prod_n = rng.choice(['W', 'W', 'W', 'W', 'C', 'R'], n_normal)
    dev_n  = rng.choice(['mobile', 'mobile', 'mobile', 'desktop'], n_normal)

    normal_df = pd.DataFrame({
        'TransactionAmt': amt_n,
        'TransactionDT':  dt_n,
        'card1':          uid_n,
        'C1':  c1_n, 'C2': c2_n,  'C3': c3_n,   'C9': c9_n,
        'C11': c1_n, 'C14': c1_n,
        'D1':  d1_n, 'D3': d3_n,  'D4': d1_n.copy(),
        'D10': d10_n, 'D15': d15_n,
        'id_01': id01_n, 'M5': m5_n,
        'ProductCD': prod_n, 'DeviceType': dev_n,
        'isFraud': 0,
    })

    # ── FRAUD transactions ─────────────────────────────────────────────────
    fraud_rows: list[dict] = []

    # ── Pattern 1: Large cashout + new device + off-hours (30%) ───────────
    n_p1 = int(n_fraud * 0.30)
    for _ in range(n_p1):
        uid  = int(rng.randint(0, n_users))
        amt  = float(rng.uniform(3_000, 50_000))
        hour = int(rng.randint(0, 5))      # midnight to 4am
        day  = int(rng.randint(0, 180))
        fraud_rows.append({
            'TransactionAmt': amt,
            'TransactionDT':  day * 86400 + hour * 3600 + int(rng.randint(0, 3600)),
            'card1': uid,
            'C1': float(user_tx_counts[uid]), 'C2': float(rng.randint(2, 7)),
            'C3': float(rng.randint(5, 20)),  'C9': 1.0,
            'C11': float(user_tx_counts[uid]), 'C14': float(user_tx_counts[uid]),
            'D1': float(rng.uniform(0, 500)),  'D3': 9999.0,
            'D4': float(rng.uniform(0, 500)),  'D10': 9999.0, 'D15': 9999.0,
            'id_01': -5.0, 'M5': 'F',
            'ProductCD': 'C',
            'DeviceType': str(rng.choice(['mobile', 'desktop'])),
            'isFraud': 1,
        })

    # ── Pattern 2: Velocity burst (20%) ──────────────────────────────────
    n_p2 = int(n_fraud * 0.20)
    for _ in range(n_p2):
        uid  = int(rng.randint(0, n_users))
        amt  = float(rng.uniform(50, 600))
        hour = int(rng.randint(7, 22))
        day  = int(rng.randint(0, 180))
        fraud_rows.append({
            'TransactionAmt': amt,
            'TransactionDT':  day * 86400 + hour * 3600 + int(rng.randint(0, 3600)),
            'card1': uid,
            'C1': float(user_tx_counts[uid]), 'C2': float(rng.randint(6, 18)),
            'C3': float(rng.randint(25, 60)),  'C9': 0.0,
            'C11': float(user_tx_counts[uid]), 'C14': float(user_tx_counts[uid]),
            'D1': float(rng.uniform(0, 1000)), 'D3': 0.0,
            'D4': float(rng.uniform(0, 1000)), 'D10': 0.0, 'D15': 0.0,
            'id_01': 0.0, 'M5': 'T',
            'ProductCD': str(rng.choice(['W', 'C'])),
            'DeviceType': 'mobile',
            'isFraud': 1,
        })

    # ── Pattern 3: New location + new recipient + elevated amount (25%) ───
    n_p3 = int(n_fraud * 0.25)
    for _ in range(n_p3):
        uid  = int(rng.randint(0, n_users))
        amt  = float(rng.uniform(500, 15_000))
        hour = int(rng.randint(6, 23))
        day  = int(rng.randint(0, 180))
        new_dev = bool(rng.random() < 0.65)
        fraud_rows.append({
            'TransactionAmt': amt,
            'TransactionDT':  day * 86400 + hour * 3600 + int(rng.randint(0, 3600)),
            'card1': uid,
            'C1': float(user_tx_counts[uid]), 'C2': float(rng.randint(1, 5)),
            'C3': float(rng.randint(2, 12)),   'C9': 1.0,
            'C11': float(user_tx_counts[uid]), 'C14': float(user_tx_counts[uid]),
            'D1': float(rng.uniform(0, 600)),  'D3': 9999.0,
            'D4': float(rng.uniform(0, 600)),  'D10': 9999.0 if new_dev else 0.0,
            'D15': 9999.0,
            'id_01': -5.0 if new_dev else 0.0,
            'M5': 'F' if new_dev else 'T',
            'ProductCD': str(rng.choice(['W', 'W', 'C'])),
            'DeviceType': str(rng.choice(['mobile', 'desktop'])),
            'isFraud': 1,
        })

    # ── Pattern 4: Extreme amount vs user baseline (20%) ─────────────────
    n_p4 = int(n_fraud * 0.20)
    for _ in range(n_p4):
        uid  = int(rng.randint(0, n_users))
        avg  = float(user_avg[uid])
        amt  = min(float(avg * rng.uniform(15, 80)), 100_000.0)
        hour = int(rng.randint(6, 22))
        day  = int(rng.randint(0, 180))
        new_dev = bool(rng.random() < 0.40)
        fraud_rows.append({
            'TransactionAmt': amt,
            'TransactionDT':  day * 86400 + hour * 3600 + int(rng.randint(0, 3600)),
            'card1': uid,
            'C1': float(user_tx_counts[uid]), 'C2': float(rng.randint(0, 3)),
            'C3': float(rng.randint(0, 6)),   'C9': float(rng.choice([0, 1])),
            'C11': float(user_tx_counts[uid]), 'C14': float(user_tx_counts[uid]),
            'D1': float(rng.uniform(0, 1000)), 'D3': float(rng.choice([0, 9999])),
            'D4': float(rng.uniform(0, 1000)), 'D10': 9999.0 if new_dev else 0.0,
            'D15': float(rng.choice([0, 9999])),
            'id_01': -5.0 if new_dev else 0.0,
            'M5': 'F' if new_dev else 'T',
            'ProductCD': str(rng.choice(['W', 'W', 'C', 'R'])),
            'DeviceType': str(rng.choice(['mobile', 'desktop'])),
            'isFraud': 1,
        })

    # ── Pattern 5: Remaining — multi-signal mixed (fills to n_fraud) ──────
    n_p5 = n_fraud - n_p1 - n_p2 - n_p3 - n_p4
    for _ in range(max(n_p5, 0)):
        uid  = int(rng.randint(0, n_users))
        # Combine signals: moderate-to-high amount + 2 novelty signals
        amt  = float(rng.uniform(1_000, 30_000))
        hour = int(rng.randint(0, 24))
        day  = int(rng.randint(0, 180))
        new_dev = bool(rng.random() < 0.7)
        fraud_rows.append({
            'TransactionAmt': amt,
            'TransactionDT':  day * 86400 + hour * 3600 + int(rng.randint(0, 3600)),
            'card1': uid,
            'C1': float(user_tx_counts[uid]), 'C2': float(rng.randint(0, 4)),
            'C3': float(rng.randint(0, 10)),  'C9': float(rng.choice([0, 1])),
            'C11': float(user_tx_counts[uid]), 'C14': float(user_tx_counts[uid]),
            'D1': float(rng.uniform(0, 600)), 'D3': 9999.0,
            'D4': float(rng.uniform(0, 600)), 'D10': 9999.0 if new_dev else 0.0,
            'D15': float(rng.choice([0, 9999])),
            'id_01': -5.0 if new_dev else 0.0,
            'M5': 'F' if new_dev else 'T',
            'ProductCD': str(rng.choice(['W', 'C'])),
            'DeviceType': str(rng.choice(['mobile', 'desktop'])),
            'isFraud': 1,
        })

    fraud_df = pd.DataFrame(fraud_rows)

    # ── Combine, shuffle, cast types ─────────────────────────────────────
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Explicit dtypes for consistency
    float_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 'C9', 'C11', 'C14',
                  'D1', 'D3', 'D4', 'D10', 'D15', 'id_01']
    int_cols   = ['TransactionDT', 'card1', 'isFraud']
    str_cols   = ['M5', 'ProductCD', 'DeviceType']

    for c in float_cols:
        df[c] = df[c].astype(float)
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in str_cols:
        df[c] = df[c].astype(str)

    return df


# ---------------------------------------------------------------------------
# Main — save to disk
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic wallet fraud dataset for FraudShield training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows",   type=int,   default=250_000, help="Total rows to generate")
    parser.add_argument("--fraud",  type=float, default=0.035,   help="Fraud rate (e.g. 0.035 = 3.5%%)")
    parser.add_argument("--seed",   type=int,   default=42,      help="Random seed")
    parser.add_argument("--output", type=str,   default=str(OUTPUT_PATH), help="Output file path (.parquet or .csv)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows:,} rows (fraud rate: {args.fraud:.1%}, seed: {args.seed}) ...")
    t0 = time.perf_counter()
    df = generate(n_total=args.rows, fraud_rate=args.fraud, seed=args.seed)
    elapsed = time.perf_counter() - t0

    n_fraud  = int(df['isFraud'].sum())
    n_normal = len(df) - n_fraud
    print(f"  Generated in {elapsed:.1f}s: {len(df):,} rows | "
          f"{n_normal:,} normal | {n_fraud:,} fraud | rate={df['isFraud'].mean():.4%}")
    print(f"  TransactionAmt: p1=${df['TransactionAmt'].quantile(0.01):.2f}  "
          f"p50=${df['TransactionAmt'].quantile(0.5):.2f}  "
          f"p95=${df['TransactionAmt'].quantile(0.95):.2f}  "
          f"p99=${df['TransactionAmt'].quantile(0.99):.2f}")

    if str(output_path).endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}  ({output_path.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
