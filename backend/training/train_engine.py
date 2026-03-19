"""
FraudShield Training Engine
============================
Trains XGBoost + LightGBM on the IEEE-CIS Fraud Detection dataset and saves
all artifacts required by the inference engine.

Usage
-----
  cd backend
  python training/train_engine.py --data-dir /path/to/ieee-cis-data

  # Fast test with 50 000 rows:
  python training/train_engine.py --data-dir /path/to/ieee-cis-data --sample 50000

  # Skip LightGBM (XGBoost only):
  python training/train_engine.py --data-dir /path/to/ieee-cis-data --no-lgbm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── sys.path fix: allow imports from sibling directories when run directly ──
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import data_loader
import evaluate
import feature_engineering
import thresholds as thresholds_module


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudShield — Train XGBoost + LightGBM on IEEE-CIS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing train_transaction.csv (and optionally train_identity.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BACKEND_DIR / "models"),
        help="Directory where trained artifacts are saved",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If > 0, use only this many rows (stratified sample) for quick testing",
    )
    parser.add_argument(
        "--no-lgbm",
        action="store_true",
        default=False,
        help="Skip LightGBM training (XGBoost only)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    print("=" * 70)
    print("  FraudShield Training Engine")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Sample    : {args.sample if args.sample > 0 else 'all rows'}")
    print(f"  LightGBM  : {'disabled' if args.no_lgbm else 'enabled'}")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/9] Loading dataset ...")
    df = data_loader.load_ieee_cis(args.data_dir)

    if not data_loader.validate_dataset(df):
        print("[ERROR] Dataset validation failed. Aborting.")
        sys.exit(1)

    if args.sample > 0 and args.sample < len(df):
        print(f"[1/9] Sampling {args.sample:,} rows (stratified) for quick test ...")
        df, _ = train_test_split(
            df,
            train_size=args.sample,
            stratify=df["isFraud"],
            random_state=42,
        )
        df = df.reset_index(drop=True)
        print(f"[1/9] Sample shape: {df.shape}, fraud rate: {df['isFraud'].mean():.4%}")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/9] Engineering features ...")
    df = feature_engineering.engineer_features(df)

    # ── 3. Feature list ──────────────────────────────────────────────────────
    print("\n[3/9] Selecting features ...")
    feature_names = feature_engineering.get_feature_list(df)
    print(f"[3/9] Total features : {len(feature_names)}")

    # ── 4. Build preprocessor ────────────────────────────────────────────────
    print("\n[4/9] Building preprocessor ...")
    # Work on a copy with target column removed
    df_features = df[feature_names + ["isFraud"]].copy()
    preprocessor, final_feature_names = feature_engineering.build_preprocessor(
        df_features.drop(columns=["isFraud"])
    )
    print(f"[4/9] Final feature count after preprocessing: {len(final_feature_names)}")

    # ── 5. Prepare X, y ──────────────────────────────────────────────────────
    print("\n[5/9] Preparing train/validation split ...")
    y = df["isFraud"].values.astype(int)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df_features.drop(columns=["isFraud"]),
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    print(f"[5/9] Train: {len(X_train_raw):,}  Val: {len(X_val_raw):,}")
    print(f"[5/9] Train fraud rate: {y_train.mean():.4%}  Val fraud rate: {y_val.mean():.4%}")

    # ── 6. Transform ─────────────────────────────────────────────────────────
    print("\n[6/9] Transforming features ...")
    X_train = feature_engineering.transform_features(X_train_raw, preprocessor, final_feature_names)
    X_val = feature_engineering.transform_features(X_val_raw, preprocessor, final_feature_names)
    print(f"[6/9] X_train shape: {X_train.shape}  X_val shape: {X_val.shape}")

    # Compute per-feature medians for wallet inference imputation
    feature_medians: dict[str, float] = {
        name: float(np.nanmedian(X_train[:, i]))
        for i, name in enumerate(final_feature_names)
    }

    # ── 7. Train XGBoost ─────────────────────────────────────────────────────
    print("\n[7/9] Training XGBoost ...")
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"[7/9] scale_pos_weight = {scale_pos_weight:.2f}  "
          f"(neg={neg_count:,} / pos={pos_count:,})")

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_start = time.perf_counter()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"[7/9] XGBoost trained in {time.perf_counter() - xgb_start:.1f}s "
          f"(best iteration: {xgb_model.best_iteration})")

    # ── 8. Train LightGBM ────────────────────────────────────────────────────
    lgbm_model = None
    has_lgbm = False

    if not args.no_lgbm:
        try:
            import lightgbm as lgb
            print("\n[8/9] Training LightGBM ...")
            lgbm_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            lgbm_start = time.perf_counter()
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            has_lgbm = True
            print(f"[8/9] LightGBM trained in {time.perf_counter() - lgbm_start:.1f}s")
        except ImportError:
            print("[8/9] LightGBM not installed — skipping.")
        except Exception as exc:
            print(f"[8/9] LightGBM training failed ({exc}) — skipping.")
    else:
        print("\n[8/9] LightGBM skipped (--no-lgbm flag).")

    # ── Step 8b. Train Random Forest ─────────────────────────────────────────
    print("\n[RF] Training Random Forest (model 3/4) ...")
    rf_start = time.perf_counter()
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    print(f"[RF] Random Forest trained in {time.perf_counter() - rf_start:.1f}s")

    # ── Step 8c. Train Logistic Regression ───────────────────────────────────
    print("\n[LR] Training Logistic Regression (model 4/4) ...")
    lr_start = time.perf_counter()
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
    )
    lr_model.fit(X_train, y_train)
    print(f"[LR] Logistic Regression trained in {time.perf_counter() - lr_start:.1f}s")

    # ── 9. Predict & 4-model ensemble ────────────────────────────────────────
    print("\n[9/9] Evaluating 4-model ensemble ...")
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    rf_prob = rf_model.predict_proba(X_val)[:, 1]
    lr_prob = lr_model.predict_proba(X_val)[:, 1]

    prob_arrays = [xgb_prob, rf_prob, lr_prob]
    lgbm_prob = None
    if has_lgbm and lgbm_model is not None:
        lgbm_prob = lgbm_model.predict_proba(X_val)[:, 1]
        prob_arrays.insert(1, lgbm_prob)

    ensemble_prob = np.mean(prob_arrays, axis=0)

    # Threshold tuning on ensemble probabilities
    thresh_info = thresholds_module.compute_decision_thresholds(y_val, ensemble_prob)

    # Evaluate all models
    xgb_metrics = evaluate.evaluate_model(y_val, xgb_prob, model_name="xgboost")
    evaluate.print_evaluation_summary(xgb_metrics)

    lgbm_metrics = None
    if has_lgbm and lgbm_model is not None:
        lgbm_metrics = evaluate.evaluate_model(y_val, lgbm_prob, model_name="lightgbm")
        evaluate.print_evaluation_summary(lgbm_metrics)

    rf_metrics = evaluate.evaluate_model(y_val, rf_prob, model_name="random_forest")
    evaluate.print_evaluation_summary(rf_metrics)

    lr_metrics = evaluate.evaluate_model(y_val, lr_prob, model_name="logistic_regression")
    evaluate.print_evaluation_summary(lr_metrics)

    ensemble_metrics = evaluate.evaluate_model(
        y_val, ensemble_prob,
        threshold=thresh_info["flag"],
        model_name="ensemble",
    )
    evaluate.print_evaluation_summary(ensemble_metrics)

    # Latency benchmark
    print("[9/9] Running latency benchmark ...")
    xgb_latency = evaluate.compute_latency_benchmark(xgb_model, X_val)
    print(f"[9/9] XGBoost latency — mean: {xgb_latency['mean_ms']:.2f}ms  "
          f"p95: {xgb_latency['p95_ms']:.2f}ms  p99: {xgb_latency['p99_ms']:.2f}ms")

    # Feature importance (top 20)
    importances = xgb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [
        {
            "name": final_feature_names[i],
            "importance": round(float(importances[i]), 6),
        }
        for i in top_indices
    ]

    print("\n  Top 20 features by XGBoost importance:")
    for rank, feat in enumerate(top_features, start=1):
        print(f"  {rank:>2}. {feat['name']:<40} {feat['importance']:.6f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    print(f"\n[Saving] Writing artifacts to {output_dir} ...")

    joblib.dump(xgb_model, output_dir / "xgb_model.joblib")
    print(f"  OK xgb_model.joblib")

    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    print(f"  OK preprocessor.joblib")

    if has_lgbm and lgbm_model is not None:
        joblib.dump(lgbm_model, output_dir / "lgbm_model.joblib")
        print(f"  OK lgbm_model.joblib")

    joblib.dump(rf_model, output_dir / "rf_model.joblib")
    print(f"  OK rf_model.joblib")

    joblib.dump(lr_model, output_dir / "lr_model.joblib")
    print(f"  OK lr_model.joblib")

    # feature_metadata.json
    metadata = {
        "feature_names": final_feature_names,
        "feature_medians": feature_medians,
        "has_lgbm": has_lgbm,
        "has_rf": True,
        "has_lr": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "top_features": top_features,
        "xgb_best_iteration": int(xgb_model.best_iteration),
    }
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  OK feature_metadata.json")

    # thresholds.json
    thresh_out = {"flag": thresh_info["flag"], "block": thresh_info["block"]}
    (output_dir / "thresholds.json").write_text(
        json.dumps(thresh_out, indent=2), encoding="utf-8"
    )
    print(f"  OK thresholds.json  (flag={thresh_out['flag']}, block={thresh_out['block']})")

    # metrics.json
    metrics_out = {
        "xgboost": {k: v for k, v in xgb_metrics.items() if k != "classification_report"},
        "lightgbm": (
            {k: v for k, v in lgbm_metrics.items() if k != "classification_report"}
            if lgbm_metrics else None
        ),
        "random_forest": {k: v for k, v in rf_metrics.items() if k != "classification_report"},
        "logistic_regression": {k: v for k, v in lr_metrics.items() if k != "classification_report"},
        "ensemble": {k: v for k, v in ensemble_metrics.items() if k != "classification_report"},
        "latency": xgb_latency,
        "threshold_description": thresh_info["description"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )
    print(f"  OK metrics.json")

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Total time       : {elapsed:.1f}s")
    print(f"  Ensemble ROC-AUC : {ensemble_metrics['roc_auc']:.4f}")
    print(f"  Ensemble PR-AUC  : {ensemble_metrics['pr_auc']:.4f}")
    print(f"  Flag threshold   : {thresh_out['flag']}")
    print(f"  Block threshold  : {thresh_out['block']}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start the API server:")
    print("       cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("  2. Submit a wallet transaction:")
    print("       POST http://localhost:8000/api/wallet/transaction")
    print("  3. View the fraud dashboard:")
    print("       GET  http://localhost:8000/api/dashboard/stats")
    print()


if __name__ == "__main__":
    main()
