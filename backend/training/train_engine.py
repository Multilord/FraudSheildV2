"""
FraudShield Training Engine
============================
Trains a calibrated 4-model ensemble on the IEEE-CIS Fraud Detection dataset.

Models
------
  1. XGBoost          — supervised gradient boosting (primary classifier)
  2. LightGBM         — supervised gradient boosting (complementary)
  3. Isolation Forest — unsupervised anomaly detection (O(n) inference)
  4. LOF              — Local Outlier Factor (novelty=True, subsampled to 50k normals)

Ensemble
--------
  A meta-learner (LogisticRegression) is trained on a held-out meta-train split
  (8% of data) using predictions from all 4 base models as input features.
  This calibrated stacking replaces naive equal-weight averaging.

Splits
------
  base_train : 72% — train XGBoost, LightGBM, Isolation Forest, LOF
  meta_train :  8% — train meta-learner (OOF predictions prevent data leakage)
  val        : 20% — final evaluation and threshold selection

Usage
-----
  cd backend
  python training/train_engine.py --data-dir /path/to/ieee-cis-data

  # Fast test with 50 000 rows:
  python training/train_engine.py --data-dir /path/to/ieee-cis-data --sample 50000

  # Skip LightGBM (XGBoost + IF + LOF only):
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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
        description="FraudShield — Train calibrated IF+LOF+XGB+LGB ensemble on IEEE-CIS",
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
        help="Skip LightGBM training (XGBoost + IF + LOF only)",
    )
    parser.add_argument(
        "--lof-max-samples",
        type=int,
        default=50_000,
        help="Maximum normal-class samples for LOF training (avoids O(n²) memory/time)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sigmoid normalization for anomaly scores
# ---------------------------------------------------------------------------

def sigmoid_normalize(
    raw_scores: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] via a sigmoid centred at the training mean.

    Parameters
    ----------
    raw_scores : raw anomaly scores (higher = more anomalous)
    mean       : mean of raw scores on base_train (stored in metadata)
    std        : std  of raw scores on base_train (stored in metadata)

    Returns
    -------
    np.ndarray in [0, 1], higher = more likely fraud
    """
    z = (raw_scores - mean) / max(std, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    print("=" * 70)
    print("  FraudShield Training Engine  (IF + LOF + XGB + LGB + Meta-Learner)")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Sample    : {args.sample if args.sample > 0 else 'all rows'}")
    print(f"  LightGBM  : {'disabled' if args.no_lgbm else 'enabled'}")
    print(f"  LOF max   : {args.lof_max_samples:,} normal samples")
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

    # ── 2. Population quantiles (before split to maximise stability) ──────────
    print("\n[2/9] Computing population quantiles ...")
    pop_quantiles = feature_engineering.compute_pop_quantiles(df, n_quantiles=1000)
    print(f"[2/9] TransactionAmt: p1={np.percentile(df['TransactionAmt'].dropna(), 1):.1f}  "
          f"p50={np.percentile(df['TransactionAmt'].dropna(), 50):.1f}  "
          f"p99={np.percentile(df['TransactionAmt'].dropna(), 99):.1f}")

    # ── 3. Feature engineering ───────────────────────────────────────────────
    print("\n[3/9] Engineering features ...")
    df = feature_engineering.engineer_features(df, pop_quantiles=pop_quantiles)

    # ── 4. Feature list & preprocessor ───────────────────────────────────────
    print("\n[4/9] Selecting features and building preprocessor ...")
    feature_names = feature_engineering.get_feature_list(df)
    print(f"[4/9] Total features : {len(feature_names)}")

    df_features = df[feature_names + ["isFraud"]].copy()
    preprocessor, final_feature_names = feature_engineering.build_preprocessor(
        df_features.drop(columns=["isFraud"])
    )
    print(f"[4/9] Final feature count after preprocessing: {len(final_feature_names)}")

    # ── 5. 3-way split: base_train (72%) / meta_train (8%) / val (20%) ───────
    print("\n[5/9] Preparing 3-way split (72% base / 8% meta / 20% val) ...")
    y = df["isFraud"].values.astype(int)
    X_df = df_features.drop(columns=["isFraud"])

    # First cut: 80% train_val / 20% val
    X_tv_raw, X_val_raw, y_tv, y_val = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=42,
    )
    # Second cut: 90% base / 10% meta of the 80% (= 72% / 8% of total)
    X_base_raw, X_meta_raw, y_base, y_meta = train_test_split(
        X_tv_raw, y_tv, test_size=0.10, stratify=y_tv, random_state=42,
    )

    print(f"[5/9] base_train: {len(X_base_raw):,}  "
          f"meta_train: {len(X_meta_raw):,}  "
          f"val: {len(X_val_raw):,}")
    print(f"[5/9] Fraud rates — base: {y_base.mean():.4%}  "
          f"meta: {y_meta.mean():.4%}  val: {y_val.mean():.4%}")

    # ── 6. Transform features ─────────────────────────────────────────────────
    print("\n[6/9] Transforming features ...")
    X_base = feature_engineering.transform_features(X_base_raw, preprocessor, final_feature_names)
    X_meta = feature_engineering.transform_features(X_meta_raw, preprocessor, final_feature_names)
    X_val  = feature_engineering.transform_features(X_val_raw,  preprocessor, final_feature_names)
    print(f"[6/9] X_base: {X_base.shape}  X_meta: {X_meta.shape}  X_val: {X_val.shape}")

    # Per-feature medians for wallet inference imputation
    feature_medians: dict[str, float] = {
        name: float(np.nanmedian(X_base[:, i]))
        for i, name in enumerate(final_feature_names)
    }

    # ── 7. Train XGBoost ─────────────────────────────────────────────────────
    print("\n[7/9] Training XGBoost ...")
    neg_count = int((y_base == 0).sum())
    pos_count = int((y_base == 1).sum())
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
        X_base, y_base,
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
                X_base, y_base,
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

    # ── 8b. Train Isolation Forest ────────────────────────────────────────────
    print("\n[IF] Training Isolation Forest (model 3/4, normal-only base_train) ...")
    X_base_normal = X_base[y_base == 0]
    print(f"[IF] Training on {len(X_base_normal):,} normal transactions "
          f"(200 trees, max_samples=256) ...")
    if_start = time.perf_counter()
    iforest = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_base_normal)
    print(f"[IF] Isolation Forest trained in {time.perf_counter() - if_start:.1f}s")

    # Normalization stats: compute on full base_train (including fraud) for stability
    if_raw_base = -iforest.score_samples(X_base)  # positive = more anomalous
    if_score_mean = float(if_raw_base.mean())
    if_score_std  = float(if_raw_base.std())
    print(f"[IF] Raw score stats: mean={if_score_mean:.4f}  std={if_score_std:.4f}")

    # ── 8c. Train LOF ─────────────────────────────────────────────────────────
    print(f"\n[LOF] Training Local Outlier Factor (model 4/4, novelty=True) ...")
    lof_n = min(len(X_base_normal), args.lof_max_samples)
    if lof_n < len(X_base_normal):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_base_normal), lof_n, replace=False)
        X_lof_train = X_base_normal[idx]
        print(f"[LOF] Subsampled to {lof_n:,} normal transactions (from {len(X_base_normal):,})")
    else:
        X_lof_train = X_base_normal
        print(f"[LOF] Using all {lof_n:,} normal transactions")

    lof_start = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    lof.fit(X_lof_train)
    print(f"[LOF] LOF trained in {time.perf_counter() - lof_start:.1f}s")

    # Normalization stats on base_train
    lof_raw_base = -lof.decision_function(X_base)  # positive = more anomalous
    lof_score_mean = float(lof_raw_base.mean())
    lof_score_std  = float(lof_raw_base.std())
    print(f"[LOF] Raw score stats: mean={lof_score_mean:.4f}  std={lof_score_std:.4f}")

    # ── 9. Get predictions on meta_train & val ────────────────────────────────
    print("\n[9/9] Building meta-learner stack ...")

    # meta_train predictions
    xgb_meta  = xgb_model.predict_proba(X_meta)[:, 1]
    if_meta   = sigmoid_normalize(-iforest.score_samples(X_meta), if_score_mean, if_score_std)
    lof_meta  = sigmoid_normalize(-lof.decision_function(X_meta), lof_score_mean, lof_score_std)

    # val predictions
    xgb_val_prob  = xgb_model.predict_proba(X_val)[:, 1]
    if_val_prob   = sigmoid_normalize(-iforest.score_samples(X_val), if_score_mean, if_score_std)
    lof_val_prob  = sigmoid_normalize(-lof.decision_function(X_val), lof_score_mean, lof_score_std)

    meta_feature_names: list[str] = ["xgboost", "isolation_forest", "lof"]
    meta_cols_meta = [xgb_meta, if_meta, lof_meta]
    meta_cols_val  = [xgb_val_prob, if_val_prob, lof_val_prob]

    if has_lgbm and lgbm_model is not None:
        lgb_meta     = lgbm_model.predict_proba(X_meta)[:, 1]
        lgb_val_prob = lgbm_model.predict_proba(X_val)[:, 1]
        meta_feature_names.insert(1, "lightgbm")
        meta_cols_meta.insert(1, lgb_meta)
        meta_cols_val.insert(1, lgb_val_prob)

    X_meta_stacked = np.column_stack(meta_cols_meta)
    X_val_stacked  = np.column_stack(meta_cols_val)

    print(f"[9/9] Meta features: {meta_feature_names}")
    print(f"[9/9] X_meta_stacked: {X_meta_stacked.shape}  "
          f"fraud rate: {y_meta.mean():.4%}")

    # ── 9b. Train meta-learner ────────────────────────────────────────────────
    print("[9/9] Training meta-learner (LogisticRegression on stacked predictions) ...")
    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )
    meta_model.fit(X_meta_stacked, y_meta)

    # ── 9c. Ensemble probability on val ───────────────────────────────────────
    ensemble_prob_val = meta_model.predict_proba(X_val_stacked)[:, 1]

    # ── 9d. Cost-sensitive threshold selection ────────────────────────────────
    thresh_info = thresholds_module.compute_decision_thresholds(
        y_val, ensemble_prob_val,
        min_block_precision=0.85,
        min_flag_block_recall=0.75,
    )

    # ── 9e. Evaluate ──────────────────────────────────────────────────────────
    print("\n[Evaluate] Binary metrics per model:")
    xgb_metrics = evaluate.evaluate_model(y_val, xgb_val_prob, model_name="xgboost")
    evaluate.print_evaluation_summary(xgb_metrics)

    lgbm_metrics = None
    if has_lgbm and lgbm_model is not None:
        lgbm_metrics = evaluate.evaluate_model(y_val, lgb_val_prob, model_name="lightgbm")
        evaluate.print_evaluation_summary(lgbm_metrics)

    if_metrics  = evaluate.evaluate_model(y_val, if_val_prob,  model_name="isolation_forest")
    lof_metrics = evaluate.evaluate_model(y_val, lof_val_prob, model_name="lof")
    evaluate.print_evaluation_summary(if_metrics)
    evaluate.print_evaluation_summary(lof_metrics)

    ensemble_metrics = evaluate.evaluate_model(
        y_val, ensemble_prob_val,
        threshold=thresh_info["flag"],
        model_name="meta_ensemble",
    )
    evaluate.print_evaluation_summary(ensemble_metrics)

    # 3-way decision evaluation
    three_way = evaluate.evaluate_3way_decisions(
        y_val, ensemble_prob_val,
        flag_thresh=thresh_info["flag"],
        block_thresh=thresh_info["block"],
        model_name="meta_ensemble",
    )

    # Ablation comparison
    ablation_scores: dict[str, np.ndarray] = {
        "xgboost":          xgb_val_prob,
        "isolation_forest": if_val_prob,
        "lof":              lof_val_prob,
        "meta_ensemble":    ensemble_prob_val,
    }
    if has_lgbm and lgbm_model is not None:
        ablation_scores["lightgbm"] = lgb_val_prob
    evaluate.print_ablation_comparison(
        y_val, ablation_scores, thresh_info["flag"], thresh_info["block"]
    )

    # Latency benchmark on XGBoost (representative fast path)
    print("[Evaluate] Running latency benchmark ...")
    xgb_latency = evaluate.compute_latency_benchmark(xgb_model, X_val)
    print(f"[Evaluate] XGBoost latency — mean: {xgb_latency['mean_ms']:.2f}ms  "
          f"p95: {xgb_latency['p95_ms']:.2f}ms  p99: {xgb_latency['p99_ms']:.2f}ms")

    # Feature importance (top 20 from XGBoost)
    importances = xgb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [
        {"name": final_feature_names[i], "importance": round(float(importances[i]), 6)}
        for i in top_indices
    ]
    print("\n  Top 20 features by XGBoost importance:")
    for rank, feat in enumerate(top_features, start=1):
        print(f"  {rank:>2}. {feat['name']:<40} {feat['importance']:.6f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    print(f"\n[Saving] Writing artifacts to {output_dir} ...")

    joblib.dump(xgb_model,    output_dir / "xgb_model.joblib");    print("  OK xgb_model.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib"); print("  OK preprocessor.joblib")
    joblib.dump(iforest,      output_dir / "iforest_model.joblib"); print("  OK iforest_model.joblib")
    joblib.dump(lof,          output_dir / "lof_model.joblib");     print("  OK lof_model.joblib")
    joblib.dump(meta_model,   output_dir / "meta_model.joblib");    print("  OK meta_model.joblib")

    if has_lgbm and lgbm_model is not None:
        joblib.dump(lgbm_model, output_dir / "lgbm_model.joblib")
        print("  OK lgbm_model.joblib")

    # feature_metadata.json
    metadata = {
        "feature_names":    final_feature_names,
        "feature_medians":  feature_medians,
        "pop_quantiles":    pop_quantiles.tolist(),
        "meta_feature_names": meta_feature_names,
        "anomaly_score_stats": {
            "iforest_mean": if_score_mean,
            "iforest_std":  if_score_std,
            "lof_mean":     lof_score_mean,
            "lof_std":      lof_score_std,
        },
        "has_lgbm":         has_lgbm,
        "has_iforest":      True,
        "has_lof":          True,
        "has_meta":         True,
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "n_base_train":     int(len(X_base)),
        "n_meta_train":     int(len(X_meta)),
        "n_val":            int(len(X_val)),
        "top_features":     top_features,
        "xgb_best_iteration": int(xgb_model.best_iteration),
    }
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("  OK feature_metadata.json")

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
        "isolation_forest": {k: v for k, v in if_metrics.items() if k != "classification_report"},
        "lof":              {k: v for k, v in lof_metrics.items() if k != "classification_report"},
        "meta_ensemble": {k: v for k, v in ensemble_metrics.items() if k != "classification_report"},
        "three_way_decisions": three_way,
        "latency":  xgb_latency,
        "threshold_description": thresh_info["description"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )
    print("  OK metrics.json")

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Ensemble ROC-AUC   : {ensemble_metrics['roc_auc']:.4f}")
    print(f"  Ensemble PR-AUC    : {ensemble_metrics['pr_auc']:.4f}")
    print(f"  Block precision    : {three_way['block_precision']:.4f}")
    print(f"  Flag+Block recall  : {three_way['flag_block_recall']:.4f}")
    print(f"  Flag threshold     : {thresh_out['flag']}")
    print(f"  Block threshold    : {thresh_out['block']}")
    print(f"  Meta features      : {meta_feature_names}")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Start the API server:")
    print("       cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("  2. Submit a wallet transaction:")
    print("       POST http://localhost:8000/api/wallet/transaction")
    print()


if __name__ == "__main__":
    main()
