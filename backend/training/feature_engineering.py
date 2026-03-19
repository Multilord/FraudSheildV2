"""
FraudShield feature engineering for IEEE-CIS dataset.

Provides functions for:
  - Defining the feature set (numerical, categorical, derived)
  - Engineering derived features from raw IEEE-CIS columns
  - Building and applying a sklearn preprocessing pipeline
  - Mapping a wallet transaction dict to the model feature vector
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
# Feature definitions
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
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the dataframe in-place (returns the same df).

    Derived features added:
      amt_log              — log1p of TransactionAmt
      hour_of_day          — transaction hour (0-23) from TransactionDT
      day_of_week          — day of week (0=Monday…6=Sunday)
      amt_to_dist_ratio    — amount / (dist1 + 1)
      is_high_amount       — 1 if TransactionAmt > 1000
      is_night_transaction — 1 if hour_of_day in [0, 5]
    """
    df = df.copy()

    df["amt_log"] = np.log1p(df["TransactionAmt"].fillna(0))

    if "TransactionDT" in df.columns:
        df["hour_of_day"] = (df["TransactionDT"] % 86400 // 3600).astype(int)
        df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(int)
    else:
        df["hour_of_day"] = 12
        df["day_of_week"] = 0

    dist1 = df["dist1"].fillna(0) if "dist1" in df.columns else pd.Series(0, index=df.index)
    df["amt_to_dist_ratio"] = df["TransactionAmt"].fillna(0) / (dist1 + 1)

    df["is_high_amount"] = (df["TransactionAmt"].fillna(0) > 1000).astype(int)

    df["is_night_transaction"] = (
        df["hour_of_day"].isin(range(0, 6))
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature list (filtered to columns present in df)
# ---------------------------------------------------------------------------

def get_feature_list(df: Optional[pd.DataFrame] = None) -> list[str]:
    """
    Return the complete list of features (numerical + categorical + derived).

    If a dataframe is supplied, only returns features that actually exist in df.
    """
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES
    if df is None:
        return all_features
    return [f for f in all_features if f in df.columns]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """
    Build and fit a preprocessing pipeline.

    Strategy:
      - Numerical  : SimpleImputer(median)
      - Categorical: fillna('missing') + OrdinalEncoder(handle_unknown='use_encoded_value')

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe (already engineer_features applied).

    Returns
    -------
    (fitted_pipeline, final_feature_names_list)
    """
    available_numerical = [f for f in NUMERICAL_FEATURES if f in df.columns]
    available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    available_derived = [f for f in DERIVED_FEATURES if f in df.columns]

    # All numerical including derived
    all_numerical = available_numerical + available_derived

    # Fill categoricals before building
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

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
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

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw + derived features.
    preprocessor : fitted Pipeline
        Returned by build_preprocessor.
    feature_names : list[str]
        The feature name list returned by build_preprocessor (used to ensure
        column alignment).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    # Ensure all expected columns are present (fill missing with NaN)
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            if col in CATEGORICAL_FEATURES:
                df[col] = "missing"
            else:
                df[col] = np.nan

    return preprocessor.transform(df)


# ---------------------------------------------------------------------------
# Wallet feature vector
# ---------------------------------------------------------------------------

_TX_TYPE_MAP: dict[str, str] = {
    "transfer":  "W",
    "payment":   "W",
    "merchant":  "W",
    "cashout":   "C",
    "topup":     "R",
    "hotel":     "H",
    "services":  "S",
}


def get_wallet_feature_vector(
    wallet_tx: dict,
    user_profile: dict,
    preprocessor: Pipeline,
    feature_names: list[str],
    feature_medians: dict,
) -> np.ndarray:
    """
    Map a wallet transaction to the IEEE-CIS feature vector.

    Parameters
    ----------
    wallet_tx : dict
        Fields: user_id, amount, transaction_type, device_type, device_id,
                ip_address, location, merchant, merchant_category,
                is_new_device, hour_of_day (0-23).
    user_profile : dict
        Fields: avg_amount, std_amount, transaction_count, device_ids (list),
                locations (list), last_transaction_time, first_seen.
    preprocessor : fitted Pipeline
    feature_names : list[str]
    feature_medians : dict
        Median value per feature, used for unknown/unmappable features.

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    amount = float(wallet_tx.get("amount", 0.0))
    tx_type = wallet_tx.get("transaction_type", "payment").lower()
    device_type = wallet_tx.get("device_type", "mobile")
    hour_of_day = int(wallet_tx.get("hour_of_day", datetime.now().hour))
    day_of_week = datetime.now().weekday()

    avg_amount = float(user_profile.get("avg_amount", 0.0) or 0.0)
    std_amount = float(user_profile.get("std_amount", 0.0) or 0.0)
    tx_count = int(user_profile.get("transaction_count", 0) or 0)

    # Estimate days since first transaction
    first_seen_str = user_profile.get("first_seen")
    d1_days = 0.0
    if first_seen_str:
        try:
            first_seen = datetime.fromisoformat(first_seen_str.replace("Z", "+00:00"))
            d1_days = float((datetime.now(first_seen.tzinfo) - first_seen).days)
        except (ValueError, TypeError):
            d1_days = 0.0

    # Build a dict seeded with median values for all features
    row: dict = {}
    for feat in feature_names:
        row[feat] = feature_medians.get(feat, 0.0)

    # Override with known mappings
    row["TransactionAmt"] = amount
    row["amt_log"] = math.log1p(amount)
    row["hour_of_day"] = hour_of_day
    row["day_of_week"] = day_of_week
    row["amt_to_dist_ratio"] = amount  # no dist data available at wallet time
    row["is_high_amount"] = 1 if amount > 1000 else 0
    row["is_night_transaction"] = 1 if hour_of_day < 6 else 0

    # Categorical mappings
    row["ProductCD"] = _TX_TYPE_MAP.get(tx_type, "W")
    row["DeviceType"] = device_type

    # Proxy count/time features from user profile
    row["C1"] = float(tx_count)    # proxy for card count feature
    row["D1"] = d1_days             # days since account was first seen

    # Build a single-row DataFrame and transform
    df_row = pd.DataFrame([row])

    # Apply engineer_features-style derived columns that may be needed
    # (they are already in `row` above, but ensure they exist in the df)
    for col in DERIVED_FEATURES:
        if col not in df_row.columns:
            df_row[col] = row.get(col, 0.0)

    X = transform_features(df_row, preprocessor, feature_names)
    return X.ravel()
