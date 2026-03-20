"""
FraudShield Training Engine
============================
Trains a calibrated 4-model ensemble for real-time wallet fraud detection.

Models
------
  1. XGBoost          — supervised gradient boosting (primary)
  2. LightGBM         — supervised gradient boosting (complementary)
  3. Isolation Forest — unsupervised anomaly (normal-only training)
  4. LOF              — Local Outlier Factor (novelty=True)

Ensemble
--------
  Meta-learner (LogisticRegression) stacks all 4 base models on a held-out
  meta-train split (8%).  This calibrated stacking replaces naive averaging
  and ensures strong ml_ensemble signals at inference.

Feature mode (CRITICAL FIX)
----------------------------
  wallet_only=True (DEFAULT):
    Trains on ONLY the ~25 features that get_wallet_feature_vector() can
    populate from a wallet transaction + user_profile at inference.

    Root cause of the previous weak-ML-signal bug:
      Training used 155+ IEEE-CIS features (incl. V1-V317).
      At inference, only ~25 wallet features were set; the rest were
      median-imputed.  XGBoost/LightGBM saw a near-median V-feature
      vector regardless of transaction risk → ml_prob ≈ 3-5% always.

    Fix: train on wallet-native features only → all training features are
    explicitly populated at inference → ml_prob reflects genuine risk.

  wallet_only=False:
    Uses the full IEEE-CIS feature set.  Only meaningful if you have
    train_transaction.csv AND are willing to accept median-imputed inference.

Dataset
-------
  1. IEEE-CIS (train_transaction.csv) — if present in --data-dir
  2. Synthetic wallet dataset          — auto-generated if IEEE-CIS missing
     (stored at backend/data/synthetic_wallet_fraud.parquet)

Splits
------
  base_train : 72% — train XGBoost, LightGBM, IF, LOF
  meta_train :  8% — train meta-learner (OOF prevents data leakage)
  val        : 20% — evaluation and threshold selection

Usage
-----
  # Quick start — generates synthetic data and trains everything:
  cd backend
  python training/train_engine.py

  # With real IEEE-CIS data:
  python training/train_engine.py --data-dir data/

  # Fast test (50k rows):
  python training/train_engine.py --sample 50000

  # Skip LightGBM:
  python training/train_engine.py --no-lgbm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR  = Path(__file__).resolve().parent.parent
_TRAINING_DIR = Path(__file__).resolve().parent
for _p in (_BACKEND_DIR, _TRAINING_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudShield — Train calibrated 4-model ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(_BACKEND_DIR / "data"),
        help=(
            "Directory containing train_transaction.csv (IEEE-CIS) or "
            "synthetic_wallet_fraud.parquet.  If neither found, the synthetic "
            "dataset is auto-generated here."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BACKEND_DIR / "models"),
        help="Directory where trained artifacts are saved",
    )
    parser.add_argument(
        "--sample",
        type=int, default=0,
        help="If > 0, use only this many rows (stratified) for quick testing",
    )
    parser.add_argument(
        "--no-lgbm",
        action="store_true", default=False,
        help="Skip LightGBM (XGBoost + IF + LOF only)",
    )
    parser.add_argument(
        "--lof-max-samples",
        type=int, default=50_000,
        help="Max normal-class samples for LOF training (avoids O(n²) overhead)",
    )
    parser.add_argument(
        "--wallet-features-only",
        action="store_true", default=True,
        help=(
            "Train on wallet-native features only (~25 features available at "
            "inference).  Eliminates train/inference mismatch.  Highly recommended."
        ),
    )
    parser.add_argument(
        "--full-features",
        action="store_true", default=False,
        help="Override --wallet-features-only and use all IEEE-CIS features (legacy)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sigmoid normalisation for anomaly scores
# ---------------------------------------------------------------------------

def sigmoid_normalize(
    raw_scores: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    z = (raw_scores - mean) / max(std, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    wallet_only = (not args.full_features)   # default True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    print("=" * 72)
    print("  FraudShield Training Engine  (XGB + LGBM + IF + LOF + Meta-Learner)")
    print(f"  Data dir      : {args.data_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Sample        : {args.sample if args.sample > 0 else 'all rows'}")
    print(f"  LightGBM      : {'disabled' if args.no_lgbm else 'enabled'}")
    print(f"  Wallet-only   : {wallet_only}  "
          f"({'~25 wallet-native features' if wallet_only else 'full IEEE-CIS 155+ features'})")
    print("=" * 72)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/10] Loading dataset ...")
    df = data_loader.load_dataset(args.data_dir)

    if not data_loader.validate_dataset(df):
        print("[ERROR] Dataset validation failed. Aborting.")
        sys.exit(1)

    if args.sample > 0 and args.sample < len(df):
        print(f"[1/10] Sampling {args.sample:,} rows (stratified) ...")
        df, _ = train_test_split(
            df, train_size=args.sample,
            stratify=df["isFraud"], random_state=42,
        )
        df = df.reset_index(drop=True)
        print(f"[1/10] Sample: {df.shape}, fraud rate: {df['isFraud'].mean():.4%}")

    # ── 2. Population quantiles ───────────────────────────────────────────────
    print("\n[2/10] Computing population quantiles ...")
    pop_quantiles = feature_engineering.compute_pop_quantiles(df, n_quantiles=1000)
    amt_arr = df["TransactionAmt"].dropna()
    print(f"[2/10] TransactionAmt: p1=${np.percentile(amt_arr, 1):.1f}  "
          f"p50=${np.percentile(amt_arr, 50):.1f}  "
          f"p99=${np.percentile(amt_arr, 99):.1f}")

    # ── 3. Feature engineering ───────────────────────────────────────────────
    print("\n[3/10] Engineering features ...")
    df = feature_engineering.engineer_features(df, pop_quantiles=pop_quantiles)

    # ── 4. Feature selection ─────────────────────────────────────────────────
    print("\n[4/10] Selecting features ...")
    feature_names_raw = feature_engineering.get_feature_list(df, wallet_only=wallet_only)
    print(f"[4/10] Feature candidates: {len(feature_names_raw)}  "
          f"(wallet_only={wallet_only})")
    print(f"[4/10] Features: {feature_names_raw}")

    df_features = df[feature_names_raw + ["isFraud"]].copy()

    # ── 5. 3-way split ───────────────────────────────────────────────────────
    print("\n[5/10] Preparing 3-way split (72% base / 8% meta / 20% val) ...")
    y    = df["isFraud"].values.astype(int)
    X_df = df_features.drop(columns=["isFraud"])

    X_tv_raw, X_val_raw, y_tv, y_val = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=42,
    )
    X_base_raw, X_meta_raw, y_base, y_meta = train_test_split(
        X_tv_raw, y_tv, test_size=0.10, stratify=y_tv, random_state=42,
    )
    print(f"[5/10] base: {len(X_base_raw):,}  meta: {len(X_meta_raw):,}  val: {len(X_val_raw):,}")
    print(f"[5/10] Fraud rates — base: {y_base.mean():.4%}  "
          f"meta: {y_meta.mean():.4%}  val: {y_val.mean():.4%}")

    # ── 6a. Preprocessor — fit on base_train ONLY ────────────────────────────
    print("\n[6a/10] Building preprocessor (fit on base_train only) ...")
    preprocessor, final_feature_names = feature_engineering.build_preprocessor(
        X_base_raw, wallet_only=wallet_only,
    )
    print(f"[6a/10] Final feature count: {len(final_feature_names)}")

    # ── 6b. Transform ────────────────────────────────────────────────────────
    print("\n[6b/10] Transforming features ...")
    X_base = feature_engineering.transform_features(X_base_raw, preprocessor, final_feature_names)
    X_meta = feature_engineering.transform_features(X_meta_raw, preprocessor, final_feature_names)
    X_val  = feature_engineering.transform_features(X_val_raw,  preprocessor, final_feature_names)
    print(f"[6b/10] X_base: {X_base.shape}  X_meta: {X_meta.shape}  X_val: {X_val.shape}")

    feature_medians: dict[str, float] = {
        name: float(np.nanmedian(X_base[:, i]))
        for i, name in enumerate(final_feature_names)
    }

    # ── 6c. StandardScaler for anomaly models ────────────────────────────────
    print("\n[6c/10] Building StandardScaler for anomaly models ...")
    X_base_normal = X_base[y_base == 0]
    anomaly_scaler = StandardScaler()
    anomaly_scaler.fit(X_base_normal)

    X_base_normal_scaled = anomaly_scaler.transform(X_base_normal)
    X_base_scaled        = anomaly_scaler.transform(X_base)
    X_meta_scaled        = anomaly_scaler.transform(X_meta)
    X_val_scaled         = anomaly_scaler.transform(X_val)

    print(f"[6c/10] Scaler fit on {len(X_base_normal):,} normal transactions")

    # ── 7. XGBoost ───────────────────────────────────────────────────────────
    print("\n[7/10] Training XGBoost ...")
    neg_count = int((y_base == 0).sum())
    pos_count = int((y_base == 1).sum())
    scale_pw  = neg_count / max(pos_count, 1)
    print(f"[7/10] scale_pos_weight = {scale_pw:.2f}  (neg={neg_count:,} / pos={pos_count:,})")

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    t0 = time.perf_counter()
    xgb_model.fit(X_base, y_base, eval_set=[(X_val, y_val)], verbose=100)
    print(f"[7/10] XGBoost trained in {time.perf_counter() - t0:.1f}s  "
          f"(best iter: {xgb_model.best_iteration})")

    # ── 8. LightGBM ──────────────────────────────────────────────────────────
    lgbm_model = None
    has_lgbm   = False

    if not args.no_lgbm:
        try:
            import lightgbm as lgb
            print("\n[8/10] Training LightGBM ...")
            lgbm_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                metric="average_precision",
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            t0 = time.perf_counter()
            lgbm_model.fit(
                X_base, y_base,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            has_lgbm = True
            print(f"[8/10] LightGBM trained in {time.perf_counter() - t0:.1f}s")
        except ImportError:
            print("[8/10] LightGBM not installed — skipping.")
        except Exception as exc:
            print(f"[8/10] LightGBM failed ({exc}) — skipping.")
    else:
        print("\n[8/10] LightGBM skipped (--no-lgbm).")

    # ── 8b. Isolation Forest ─────────────────────────────────────────────────
    print(f"\n[IF] Training Isolation Forest on {len(X_base_normal_scaled):,} "
          "scaled normal transactions ...")
    t0 = time.perf_counter()
    iforest = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_base_normal_scaled)
    print(f"[IF] Trained in {time.perf_counter() - t0:.1f}s")

    if_raw_base   = -iforest.score_samples(X_base_scaled)
    if_score_mean = float(if_raw_base.mean())
    if_score_std  = float(if_raw_base.std())
    print(f"[IF] Score stats: mean={if_score_mean:.4f}  std={if_score_std:.4f}")

    # ── 8c. LOF ───────────────────────────────────────────────────────────────
    lof_n = min(len(X_base_normal_scaled), args.lof_max_samples)
    print(f"\n[LOF] Training LOF (novelty=True) on {lof_n:,} scaled normal transactions ...")
    if lof_n < len(X_base_normal_scaled):
        idx         = np.random.RandomState(42).choice(
            len(X_base_normal_scaled), lof_n, replace=False,
        )
        X_lof_train = X_base_normal_scaled[idx]
    else:
        X_lof_train = X_base_normal_scaled

    t0  = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    lof.fit(X_lof_train)
    print(f"[LOF] Trained in {time.perf_counter() - t0:.1f}s")

    lof_raw_base   = -lof.decision_function(X_base_scaled)
    lof_score_mean = float(lof_raw_base.mean())
    lof_score_std  = float(lof_raw_base.std())
    print(f"[LOF] Score stats: mean={lof_score_mean:.4f}  std={lof_score_std:.4f}")

    # ── 9. Meta-learner ───────────────────────────────────────────────────────
    print("\n[9/10] Building meta-learner stack ...")

    xgb_meta      = xgb_model.predict_proba(X_meta)[:, 1]
    if_meta       = sigmoid_normalize(-iforest.score_samples(X_meta_scaled), if_score_mean, if_score_std)
    lof_meta      = sigmoid_normalize(-lof.decision_function(X_meta_scaled), lof_score_mean, lof_score_std)

    xgb_val_prob  = xgb_model.predict_proba(X_val)[:, 1]
    if_val_prob   = sigmoid_normalize(-iforest.score_samples(X_val_scaled), if_score_mean, if_score_std)
    lof_val_prob  = sigmoid_normalize(-lof.decision_function(X_val_scaled), lof_score_mean, lof_score_std)

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

    print(f"[9/10] Meta features: {meta_feature_names}")
    print(f"[9/10] X_meta_stacked: {X_meta_stacked.shape}  fraud={y_meta.mean():.4%}")

    print("[9/10] Training meta-learner ...")
    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    )
    meta_model.fit(X_meta_stacked, y_meta)

    ensemble_prob_val = meta_model.predict_proba(X_val_stacked)[:, 1]

    # ── 9b. Thresholds ────────────────────────────────────────────────────────
    thresh_info = thresholds_module.compute_decision_thresholds(
        y_val, ensemble_prob_val,
        min_block_precision=0.85,
        min_flag_block_recall=0.75,
    )

    # ── 9c. Evaluation ────────────────────────────────────────────────────────
    print("\n[Evaluate] Per-model metrics:")
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

    three_way = evaluate.evaluate_3way_decisions(
        y_val, ensemble_prob_val,
        flag_thresh=thresh_info["flag"],
        block_thresh=thresh_info["block"],
        model_name="meta_ensemble",
    )

    ablation: dict[str, np.ndarray] = {
        "xgboost":          xgb_val_prob,
        "isolation_forest": if_val_prob,
        "lof":              lof_val_prob,
        "meta_ensemble":    ensemble_prob_val,
    }
    if has_lgbm and lgbm_model is not None:
        ablation["lightgbm"] = lgb_val_prob
    evaluate.print_ablation_comparison(y_val, ablation, thresh_info["flag"], thresh_info["block"])

    print("[Evaluate] Running latency benchmark ...")
    xgb_latency = evaluate.compute_latency_benchmark(xgb_model, X_val)
    print(f"[Evaluate] XGBoost latency — mean: {xgb_latency['mean_ms']:.2f}ms  "
          f"p95: {xgb_latency['p95_ms']:.2f}ms")

    # Feature importance
    importances = xgb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [
        {"name": final_feature_names[i], "importance": round(float(importances[i]), 6)}
        for i in top_indices
    ]
    print("\n  Top 20 features by XGBoost importance:")
    for rank, feat in enumerate(top_features, start=1):
        print(f"  {rank:>2}. {feat['name']:<40} {feat['importance']:.6f}")

    # ── Sanity check: score a few synthetic transactions ─────────────────────
    print("\n[Sanity] Spot-checking ensemble scores on val set ...")
    fraud_probs   = ensemble_prob_val[y_val == 1]
    legit_probs   = ensemble_prob_val[y_val == 0]
    print(f"  Fraud transactions  — mean: {fraud_probs.mean():.3f}  "
          f"p50: {np.median(fraud_probs):.3f}  p90: {np.percentile(fraud_probs, 90):.3f}")
    print(f"  Legit transactions  — mean: {legit_probs.mean():.3f}  "
          f"p50: {np.median(legit_probs):.3f}  p90: {np.percentile(legit_probs, 90):.3f}")
    if fraud_probs.mean() > legit_probs.mean() * 3:
        print("  [OK] Fraud mean score is >3x legit mean — ensemble is discriminative.")
    else:
        print("  [WARN] Fraud/legit separation may be weak. "
              "Check feature engineering and fraud patterns.")

    # ── 10. Save artifacts ───────────────────────────────────────────────────
    print(f"\n[10/10] Writing artifacts to {output_dir} ...")

    joblib.dump(xgb_model,      output_dir / "xgb_model.joblib");      print("  OK xgb_model.joblib")
    joblib.dump(preprocessor,   output_dir / "preprocessor.joblib");   print("  OK preprocessor.joblib")
    joblib.dump(anomaly_scaler, output_dir / "anomaly_scaler.joblib"); print("  OK anomaly_scaler.joblib")
    joblib.dump(iforest,        output_dir / "iforest_model.joblib");  print("  OK iforest_model.joblib")
    joblib.dump(lof,            output_dir / "lof_model.joblib");      print("  OK lof_model.joblib")
    joblib.dump(meta_model,     output_dir / "meta_model.joblib");     print("  OK meta_model.joblib")

    if has_lgbm and lgbm_model is not None:
        joblib.dump(lgbm_model, output_dir / "lgbm_model.joblib")
        print("  OK lgbm_model.joblib")

    metadata = {
        "feature_names":    final_feature_names,
        "feature_medians":  feature_medians,
        "pop_quantiles":    pop_quantiles.tolist(),
        "meta_feature_names": meta_feature_names,
        "anomaly_score_stats": {
            "iforest_mean": if_score_mean, "iforest_std": if_score_std,
            "lof_mean":     lof_score_mean, "lof_std":   lof_score_std,
        },
        "has_lgbm":           has_lgbm,
        "has_iforest":        True,
        "has_lof":            True,
        "has_meta":           True,
        "has_anomaly_scaler": True,
        "wallet_only":        wallet_only,
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "n_base_train":       int(len(X_base)),
        "n_meta_train":       int(len(X_meta)),
        "n_val":              int(len(X_val)),
        "top_features":       top_features,
        "xgb_best_iteration": int(xgb_model.best_iteration),
        "note": (
            "wallet_only=True: model trained exclusively on wallet-native features. "
            "All training features are populated at inference — no median-imputation "
            "collapse.  ml_ensemble contribution is genuine."
        ),
    }
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("  OK feature_metadata.json")

    thresh_out = {"flag": thresh_info["flag"], "block": thresh_info["block"]}
    (output_dir / "thresholds.json").write_text(
        json.dumps(thresh_out, indent=2), encoding="utf-8"
    )
    print(f"  OK thresholds.json  (flag={thresh_out['flag']}, block={thresh_out['block']})")

    metrics_out = {
        "xgboost":           {k: v for k, v in xgb_metrics.items() if k != "classification_report"},
        "lightgbm":          ({k: v for k, v in lgbm_metrics.items() if k != "classification_report"}
                              if lgbm_metrics else None),
        "isolation_forest":  {k: v for k, v in if_metrics.items()   if k != "classification_report"},
        "lof":               {k: v for k, v in lof_metrics.items()   if k != "classification_report"},
        "meta_ensemble":     {k: v for k, v in ensemble_metrics.items() if k != "classification_report"},
        "three_way_decisions": three_way,
        "latency":           xgb_latency,
        "threshold_description": thresh_info["description"],
        "training_note": (
            f"wallet_only={wallet_only}: trained on {len(final_feature_names)} features. "
            "All features populated at inference — ml_ensemble is genuine, not median-collapsed."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )
    print("  OK metrics.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Features trained   : {len(final_feature_names)}  (wallet_only={wallet_only})")
    print(f"  Ensemble ROC-AUC   : {ensemble_metrics['roc_auc']:.4f}")
    print(f"  Ensemble PR-AUC    : {ensemble_metrics['pr_auc']:.4f}")
    print(f"  Block precision    : {three_way['block_precision']:.4f}")
    print(f"  Flag+Block recall  : {three_way['flag_block_recall']:.4f}")
    print(f"  Flag threshold     : {thresh_out['flag']}")
    print(f"  Block threshold    : {thresh_out['block']}")
    print(f"  Meta features      : {meta_feature_names}")
    print(f"  Fraud val sample   : mean={fraud_probs.mean():.3f}  p90={np.percentile(fraud_probs,90):.3f}")
    print(f"  Legit val sample   : mean={legit_probs.mean():.3f}  p90={np.percentile(legit_probs,90):.3f}")
    print("=" * 72)
    print("\nNext steps:")
    print("  1. Restart the API (it wipes and re-initialises the DB on startup):")
    print("       cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("  2. Re-seed historical data:")
    print("       python seed_transactions.py")
    print()


if __name__ == "__main__":
    main()
