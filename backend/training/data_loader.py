"""
FraudShield data loader for IEEE-CIS Fraud Detection dataset.

Expected files (download from Kaggle):
  - train_transaction.csv   (required)
  - train_identity.csv      (optional, merged when present)
"""

from pathlib import Path

import pandas as pd


def load_ieee_cis(data_dir: str) -> pd.DataFrame:
    """
    Load the IEEE-CIS Fraud Detection dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing train_transaction.csv and optionally
        train_identity.csv.

    Returns
    -------
    pd.DataFrame
        Merged dataframe indexed by integer position (TransactionID dropped).
    """
    data_path = Path(data_dir)
    tx_path = data_path / "train_transaction.csv"
    id_path = data_path / "train_identity.csv"

    if not tx_path.exists():
        raise FileNotFoundError(
            f"train_transaction.csv not found in '{data_dir}'.\n"
            "Download the IEEE-CIS Fraud Detection dataset from:\n"
            "  https://www.kaggle.com/c/ieee-fraud-detection/data"
        )

    print(f"[DataLoader] Loading {tx_path} ...")
    df_tx = pd.read_csv(tx_path)
    print(f"[DataLoader] Transactions loaded: {len(df_tx):,} rows, {df_tx.shape[1]} columns")

    if id_path.exists():
        print(f"[DataLoader] Loading identity data from {id_path} ...")
        df_id = pd.read_csv(id_path)
        print(f"[DataLoader] Identity data: {len(df_id):,} rows, {df_id.shape[1]} columns")
        df = df_tx.merge(df_id, on="TransactionID", how="left")
        print(f"[DataLoader] Merged shape: {df.shape}")
    else:
        print("[DataLoader] train_identity.csv not found — proceeding without identity features.")
        df = df_tx.copy()

    # Drop TransactionID — use integer index for training
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    df = df.reset_index(drop=True)

    # Summary
    fraud_rate = df["isFraud"].mean() if "isFraud" in df.columns else float("nan")
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    print(f"[DataLoader] Final shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[DataLoader] Fraud rate   : {fraud_rate:.4%}")
    print(f"[DataLoader] Memory usage : {mem_mb:.1f} MB")

    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame looks like the IEEE-CIS dataset.

    Checks
    ------
    - Minimum required columns exist.
    - Fraud rate is in a plausible range (1–10 %).

    Returns
    -------
    bool
        True if the dataset passes all checks, False otherwise.
    """
    required_columns = {"TransactionAmt", "isFraud", "TransactionDT"}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"[DataLoader] Validation FAILED — missing columns: {missing}")
        return False

    fraud_rate = df["isFraud"].mean()
    if not (0.01 <= fraud_rate <= 0.10):
        print(
            f"[DataLoader] Validation FAILED — fraud rate {fraud_rate:.4%} is outside "
            "the expected 1–10% range. Check that isFraud column is correct."
        )
        return False

    print(
        f"[DataLoader] Validation PASSED — "
        f"{len(df):,} rows, fraud rate {fraud_rate:.4%}"
    )
    return True
