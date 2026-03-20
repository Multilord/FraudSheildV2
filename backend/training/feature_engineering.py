"""
FraudShield feature engineering.

Two feature modes:
  wallet_only=False  — full IEEE-CIS feature set (155+ features, original behaviour)
  wallet_only=True   — wallet-native features only (~25 features)

WALLET-NATIVE MODE (default for training)
-----------------------------------------
Trains ONLY on the features that get_wallet_feature_vector() can populate at
real-time inference, eliminating the train/inference mismatch that caused weak
ML signals:

  Training used 155 features → inference populates ~25 (rest median-imputed)
  → XGBoost/LightGBM see a near-median V-feature vector regardless of how
    suspicious the transaction is → ml_prob collapses to ~3-5% background rate

With wallet_only=True ALL training features are populated at inference → no
signal collapse.

Wallet-native features (~25 total):
  Numerical (13): TransactionAmt, C1, C2, C3, C9, C11, C14,
                  D1, D3, D4, D10, D15, id_01
  Categorical (3): ProductCD, DeviceType, M5
  Derived (9):     amt_log, hour_of_day, day_of_week, amt_to_dist_ratio,
                   is_high_amount, is_night_transaction,
                   amt_vs_card_mean, amt_z_card, amt_percentile
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------------------------
# Full IEEE-CIS feature definitions (kept for backward compat / full training)
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES: list[str] = [
    "TransactionAmt",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53", "V54", "V55",
    "V95", "V96", "V97", "V98", "V99", "V100",
    "V126", "V127", "V128", "V129", "V130",
    "V258", "V259", "V260", "V261", "V262", "V263", "V264", "V265", "V266", "V267",
    "V279", "V280", "V281", "V282", "V283", "V284", "V285", "V286", "V287", "V288", "V289",
    "V291", "V292", "V293", "V294", "V295", "V296", "V297", "V298", "V299", "V300",
    "V306", "V307", "V308", "V309", "V310", "V311", "V312", "V313", "V314", "V315",
    "V316", "V317",
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
]

CATEGORICAL_FEATURES: list[str] = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38",
]

DERIVED_FEATURES: list[str] = [
    "amt_log",
    "hour_of_day",
    "day_of_week",
    "amt_to_dist_ratio",
    "is_high_amount",
    "is_night_transaction",
    "amt_vs_card_mean",
    "amt_z_card",
    "amt_percentile",
]


# ---------------------------------------------------------------------------
# Wallet-native feature definitions (train on these to match inference exactly)
# ---------------------------------------------------------------------------

# Features that get_wallet_feature_vector() explicitly populates from the
# wallet transaction + user_profile. Training on ONLY these features means
# every training feature is fully available at inference — no median collapse.

WALLET_NATIVE_NUMERICAL: list[str] = [
    "TransactionAmt",
    "C1", "C2", "C3", "C9", "C11", "C14",
    "D1", "D3", "D4", "D10", "D15",
    "id_01",
]  # 13 features

WALLET_NATIVE_CATEGORICAL: list[str] = [
    "ProductCD", "DeviceType", "M5",
]  # 3 features

WALLET_NATIVE_DERIVED: list[str] = [
    "amt_log",
    "hour_of_day",
    "day_of_week",
    "amt_to_dist_ratio",
    "is_high_amount",
    "is_night_transaction",
    "amt_vs_card_mean",
    "amt_z_card",
    "amt_percentile",
]  # 9 features  →  total 25 wallet-native features


# ---------------------------------------------------------------------------
# Population quantiles
# ---------------------------------------------------------------------------

def compute_pop_quantiles(df: pd.DataFrame, n_quantiles: int = 1000) -> np.ndarray:
    """
    Compute n_quantiles evenly-spaced quantile values of TransactionAmt.

    Saved during training so wallet inference can compute amt_percentile as
    np.searchsorted(pop_quantiles, amount) / len(pop_quantiles).
    """
    vals = df["TransactionAmt"].dropna().values
    return np.percentile(vals, np.linspace(0, 100, n_quantiles))


# ---------------------------------------------------------------------------
# Feature engineering (derived features from raw columns)
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    pop_quantiles: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Add derived features to the dataframe (returns a copy).

    Works with both the full IEEE-CIS dataset and the synthetic wallet dataset.
    All derived features are in DERIVED_FEATURES / WALLET_NATIVE_DERIVED.
    """
    df = df.copy()
    amt = df["TransactionAmt"].fillna(0)

    df["amt_log"] = np.log1p(amt)

    if "TransactionDT" in df.columns:
        df["hour_of_day"] = (df["TransactionDT"] % 86400 // 3600).astype(int)
        df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(int)
    else:
        df["hour_of_day"] = 12
        df["day_of_week"] = 0

    dist1 = df["dist1"].fillna(0) if "dist1" in df.columns else pd.Series(0, index=df.index)
    df["amt_to_dist_ratio"] = amt / (dist1 + 1)

    df["is_high_amount"]      = (amt > 1000).astype(int)
    df["is_night_transaction"] = df["hour_of_day"].isin(range(0, 6)).astype(int)

    # ── Population-relative amount features ──────────────────────────────
    # card1 is the per-user identifier in both IEEE-CIS and the synthetic set.
    if "card1" in df.columns:
        card_stats = df.groupby("card1")["TransactionAmt"].agg(
            _crd_mean="mean", _crd_std="std"
        )
        df = df.join(card_stats, on="card1")
        crd_mean = df["_crd_mean"].fillna(float(amt.mean()) if len(amt) > 0 else 1.0)
        crd_std  = df["_crd_std"].fillna(1.0)
        df["amt_vs_card_mean"] = amt / (crd_mean + 1e-6)
        df["amt_z_card"]       = (amt - crd_mean) / (crd_std + 1e-6)
        df.drop(columns=["_crd_mean", "_crd_std"], inplace=True)
    else:
        global_mean = float(amt.mean()) if len(amt) > 0 else 1.0
        global_std  = float(amt.std())  if len(amt) > 1 else 1.0
        df["amt_vs_card_mean"] = amt / (global_mean + 1e-6)
        df["amt_z_card"]       = (amt - global_mean) / (global_std + 1e-6)

    if pop_quantiles is not None:
        pct = np.searchsorted(pop_quantiles, amt.values, side="right") / len(pop_quantiles)
        df["amt_percentile"] = pct.clip(0.0, 1.0)
    else:
        ranks = np.argsort(np.argsort(amt.values)).astype(float)
        df["amt_percentile"] = ranks / max(len(ranks) - 1, 1)

    return df


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------

def get_feature_list(
    df: Optional[pd.DataFrame] = None,
    wallet_only: bool = True,
) -> list[str]:
    """
    Return the list of features to use for training.

    Parameters
    ----------
    df          : If supplied, only returns features present in df.
    wallet_only : If True (default), return wallet-native features only.
                  These are the ~25 features that get_wallet_feature_vector()
                  populates at inference — no median collapse.
                  If False, use the full IEEE-CIS feature set (155+).
    """
    if wallet_only:
        all_features = (WALLET_NATIVE_NUMERICAL
                        + WALLET_NATIVE_CATEGORICAL
                        + WALLET_NATIVE_DERIVED)
    else:
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES

    if df is None:
        return all_features
    return [f for f in all_features if f in df.columns]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(
    df: pd.DataFrame,
    wallet_only: bool = True,
) -> "tuple[Pipeline, list[str]]":
    """
    Build and fit a sklearn preprocessing pipeline.

    Parameters
    ----------
    df          : Training DataFrame (engineer_features already applied).
    wallet_only : If True (default), build pipeline for wallet-native features.
                  If False, use the full IEEE-CIS feature set.

    Returns
    -------
    (fitted_pipeline, final_feature_names_list)
    """
    if wallet_only:
        base_numerical   = WALLET_NATIVE_NUMERICAL
        base_categorical = WALLET_NATIVE_CATEGORICAL
        base_derived     = WALLET_NATIVE_DERIVED
    else:
        base_numerical   = NUMERICAL_FEATURES
        base_categorical = CATEGORICAL_FEATURES
        base_derived     = DERIVED_FEATURES

    available_numerical   = [f for f in base_numerical   if f in df.columns]
    available_categorical = [f for f in base_categorical if f in df.columns]
    available_derived     = [f for f in base_derived     if f in df.columns]

    all_numerical = available_numerical + available_derived

    df = df.copy()
    for col in available_categorical:
        df[col] = df[col].fillna("missing").astype(str)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )),
    ])

    transformers = []
    if all_numerical:
        transformers.append(("numerical", numerical_pipeline, all_numerical))
    if available_categorical:
        transformers.append(("categorical", categorical_pipeline, available_categorical))

    ct       = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline([("preprocessor", ct)])
    pipeline.fit(df)

    final_feature_names = all_numerical + available_categorical
    return pipeline, final_feature_names


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform_features(
    df: pd.DataFrame,
    preprocessor: Pipeline,
    feature_names: list[str],
) -> np.ndarray:
    """
    Apply a fitted preprocessor pipeline to a dataframe.

    Missing columns are filled with NaN (numerical) or 'missing' (categorical)
    before transforming.
    """
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            if col in CATEGORICAL_FEATURES or col in WALLET_NATIVE_CATEGORICAL:
                df[col] = "missing"
            else:
                df[col] = np.nan
    return preprocessor.transform(df)


# ---------------------------------------------------------------------------
# Wallet feature vector (inference — unchanged from original)
# ---------------------------------------------------------------------------

_TX_TYPE_MAP: dict[str, str] = {
    "transfer": "W",
    "payment":  "W",
    "merchant": "W",
    "cashout":  "C",
    "topup":    "R",
    "hotel":    "H",
    "services": "S",
}


def get_wallet_feature_vector(
    wallet_tx: dict,
    user_profile: dict,
    preprocessor: Pipeline,
    feature_names: list[str],
    feature_medians: dict,
    pop_quantiles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map a wallet transaction dict to the model feature vector.

    Explicitly sets all wallet-native features from the transaction and
    user_profile. Remaining features (if any, e.g. when using full IEEE-CIS
    model) are filled from feature_medians.

    With wallet_only training all features are explicitly set → no median
    collapse, and the ml_ensemble contribution is genuine.
    """
    amount      = float(wallet_tx.get("amount", 0.0))
    amount_usd  = float(wallet_tx.get("amount_usd", amount))
    tx_type     = wallet_tx.get("transaction_type", "payment").lower()
    device_type = wallet_tx.get("device_type", "mobile")
    hour_of_day = int(wallet_tx.get("hour_of_day", datetime.now().hour))
    day_of_week = datetime.now().weekday()
    is_new_dev  = bool(wallet_tx.get("is_new_device", False))
    device_id   = wallet_tx.get("device_id", "")
    location    = wallet_tx.get("location", "")

    avg_amount  = float(user_profile.get("avg_amount",  0.0) or 0.0)
    std_amount  = float(user_profile.get("std_amount",  0.0) or 0.0)
    tx_count    = int(user_profile.get("transaction_count", 0) or 0)
    device_ids  = user_profile.get("device_ids",  []) or []
    locations   = user_profile.get("locations",   []) or []

    usd_rate       = amount_usd / amount if amount > 0 else 1.0
    avg_amount_usd = avg_amount * usd_rate
    std_amount_usd = std_amount * usd_rate

    first_seen_str = user_profile.get("first_seen")
    d1_days = 0.0
    if first_seen_str:
        try:
            first_seen = datetime.fromisoformat(first_seen_str.replace("Z", "+00:00"))
            d1_days    = float((datetime.now(first_seen.tzinfo) - first_seen).days)
        except (ValueError, TypeError):
            d1_days = 0.0

    # Seed row with medians for any features not explicitly set below
    row: dict = {}
    for feat in feature_names:
        row[feat] = feature_medians.get(feat, 0.0)

    # ── Amount features ────────────────────────────────────────────────────
    row["TransactionAmt"]       = amount_usd
    row["amt_log"]              = math.log1p(amount_usd)
    row["hour_of_day"]          = hour_of_day
    row["day_of_week"]          = day_of_week
    row["amt_to_dist_ratio"]    = amount_usd   # no dist at wallet inference
    row["is_high_amount"]       = 1 if amount_usd > 1_000 else 0
    row["is_night_transaction"] = 1 if (hour_of_day < 6 or hour_of_day >= 22) else 0

    # ── Population-relative amount features ────────────────────────────────
    # Falls back to population median when user has no history (new account).
    pop_median = float(pop_quantiles[len(pop_quantiles) // 2]) if (
        pop_quantiles is not None and len(pop_quantiles) > 0
    ) else float(feature_medians.get("TransactionAmt") or 25.0)

    card_mean = avg_amount_usd if avg_amount_usd > 0 else max(pop_median, 1.0)
    card_std  = std_amount_usd if std_amount_usd > 0 else max(card_mean * 0.5, 1.0)

    row["amt_vs_card_mean"] = amount_usd / (card_mean + 1e-6)
    row["amt_z_card"]       = (amount_usd - card_mean) / (card_std + 1e-6)

    if pop_quantiles is not None and len(pop_quantiles) > 0:
        pct = float(np.searchsorted(pop_quantiles, amount_usd, side="right")) / len(pop_quantiles)
        row["amt_percentile"] = float(np.clip(pct, 0.0, 1.0))
    else:
        row["amt_percentile"] = min(1.0, math.log1p(amount_usd) / math.log1p(50_000))

    # ── Categorical ────────────────────────────────────────────────────────
    row["ProductCD"]  = _TX_TYPE_MAP.get(tx_type, "W")
    row["DeviceType"] = device_type

    # ── Velocity / count proxies ───────────────────────────────────────────
    velocity_1h  = int(user_profile.get("velocity_1h",  0) or 0)
    velocity_24h = int(user_profile.get("velocity_24h", 0) or 0)
    recipient_id = wallet_tx.get("recipient_id", "")
    recipients   = user_profile.get("recipients", []) or []

    row["C1"]  = float(tx_count)
    row["C2"]  = float(velocity_1h)
    row["C3"]  = float(velocity_24h)
    row["C11"] = float(tx_count)
    row["C14"] = float(tx_count)
    row["D1"]  = d1_days
    row["D4"]  = d1_days

    # ── Recipient novelty ──────────────────────────────────────────────────
    recipient_is_new = bool(recipient_id and recipients and recipient_id not in recipients)
    row["C9"] = 1.0 if recipient_is_new else 0.0

    # ── Device novelty ─────────────────────────────────────────────────────
    known_device  = bool(device_id and device_ids and device_id in device_ids)
    device_is_new = is_new_dev or (bool(device_id) and not known_device)
    if device_is_new:
        row["D10"]   = 9_999.0
        row["id_01"] = -5.0
        row["M5"]    = "F"
    else:
        row["D10"]   = 0.0
        row["id_01"] = 0.0
        row["M5"]    = "T"

    # ── Location novelty ───────────────────────────────────────────────────
    location_is_new = bool(location and locations and location not in locations)
    if location_is_new:
        row["D3"]  = 9_999.0
        row["D15"] = 9_999.0
    else:
        row["D3"]  = 0.0
        row["D15"] = 0.0

    # ── Build row DataFrame and transform ─────────────────────────────────
    df_row = pd.DataFrame([row])

    # Ensure derived feature columns exist in the row
    all_derived = list(set(DERIVED_FEATURES + WALLET_NATIVE_DERIVED))
    for col in all_derived:
        if col not in df_row.columns:
            df_row[col] = row.get(col, 0.0)

    X = transform_features(df_row, preprocessor, feature_names)
    return X.ravel()
