"""
FraudShield model evaluation utilities.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core evaluation (binary)
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true    : ground-truth binary labels (0/1)
    y_prob    : predicted fraud probabilities [0, 1]
    threshold : classification threshold (default 0.5)
    model_name: identifier used in the returned dict

    Returns
    -------
    dict with keys: model_name, threshold, roc_auc, pr_auc, precision,
                    recall, f1, confusion_matrix, classification_report,
                    n_positives, n_negatives, positive_rate
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0)

    n_positives = int(y_true.sum())
    n_negatives = int(len(y_true) - n_positives)
    positive_rate = float(n_positives / len(y_true)) if len(y_true) > 0 else 0.0

    return {
        "model_name": model_name,
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "positive_rate": round(positive_rate, 6),
    }


# ---------------------------------------------------------------------------
# 3-way decision evaluation (APPROVE / FLAG / BLOCK)
# ---------------------------------------------------------------------------

def evaluate_3way_decisions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    flag_thresh: float,
    block_thresh: float,
    model_name: str = "ensemble",
) -> dict[str, Any]:
    """
    Evaluate 3-way APPROVE / FLAG / BLOCK decisions.

    Computes per-tier metrics:
      - Block precision  : P(fraud | BLOCK decision)
      - Block recall     : fraction of fraud that gets BLOCK
      - Flag+block recall: fraction of fraud that gets FLAG or BLOCK
      - False block rate : fraction of legitimate tx that gets BLOCK
      - False flag rate  : fraction of legitimate tx that gets FLAG or BLOCK

    Parameters
    ----------
    y_true       : ground-truth binary labels
    y_prob       : predicted probabilities
    flag_thresh  : probability threshold for FLAG tier
    block_thresh : probability threshold for BLOCK tier
    model_name   : label for output

    Returns
    -------
    dict with 3-way metrics
    """
    decisions = np.where(
        y_prob >= block_thresh, 2,          # BLOCK
        np.where(y_prob >= flag_thresh, 1,  # FLAG
                 0)                          # APPROVE
    )

    fraud = y_true == 1
    legit = y_true == 0

    n_fraud = int(fraud.sum())
    n_legit = int(legit.sum())

    blocked       = decisions == 2
    flagged       = decisions == 1
    flag_or_block = decisions >= 1

    block_tp = int((blocked & fraud).sum())
    block_fp = int((blocked & legit).sum())

    flag_tp  = int((flagged & fraud).sum())
    flag_fp  = int((flagged & legit).sum())

    block_precision  = block_tp / max(blocked.sum(), 1)
    block_recall     = block_tp / max(n_fraud, 1)
    flag_block_recall = int((flag_or_block & fraud).sum()) / max(n_fraud, 1)
    false_block_rate  = block_fp / max(n_legit, 1)
    false_flag_rate   = int((flag_or_block & legit).sum()) / max(n_legit, 1)

    n_blocked = int(blocked.sum())
    n_flagged = int(flagged.sum())
    n_approved = int((decisions == 0).sum())

    print(f"\n{'-' * 60}")
    print(f"  3-Way Decision Evaluation: {model_name}")
    print(f"{'-' * 60}")
    print(f"  Thresholds      : FLAG >= {flag_thresh:.4f} | BLOCK >= {block_thresh:.4f}")
    print(f"  Decisions       : APPROVE={n_approved:,}  FLAG={n_flagged:,}  BLOCK={n_blocked:,}")
    print(f"  Block precision : {block_precision:.4f}  ({block_tp}/{n_blocked} blocked are fraud)")
    print(f"  Block recall    : {block_recall:.4f}  ({block_tp}/{n_fraud} fraud get BLOCK)")
    print(f"  Flag+Block recall: {flag_block_recall:.4f}  (fraction of fraud caught)")
    print(f"  False block rate: {false_block_rate:.4f}  (legit tx wrongly blocked)")
    print(f"  False flag rate : {false_flag_rate:.4f}  (legit tx flagged or blocked)")
    print(f"{'-' * 60}\n")

    return {
        "model_name": model_name,
        "flag_thresh": flag_thresh,
        "block_thresh": block_thresh,
        "n_approved": n_approved,
        "n_flagged": n_flagged,
        "n_blocked": n_blocked,
        "block_precision": round(block_precision, 4),
        "block_recall": round(block_recall, 4),
        "flag_block_recall": round(flag_block_recall, 4),
        "false_block_rate": round(false_block_rate, 4),
        "false_flag_rate": round(false_flag_rate, 4),
        "block_tp": block_tp,
        "block_fp": block_fp,
        "flag_tp": flag_tp,
        "flag_fp": flag_fp,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
    }


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------

def print_ablation_comparison(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    flag_thresh: float,
    block_thresh: float,
) -> None:
    """
    Print a side-by-side ablation table comparing individual models and ensemble.

    Parameters
    ----------
    y_true       : ground-truth binary labels
    model_scores : dict mapping model name → probability/score array
    flag_thresh  : FLAG threshold
    block_thresh : BLOCK threshold
    """
    print("\n" + "=" * 80)
    print("  ABLATION COMPARISON")
    print("=" * 80)
    header = f"{'Model':<22} {'ROC-AUC':>8} {'PR-AUC':>8} {'BlkPrec':>8} {'BlkRec':>8} {'F+BRec':>8}"
    print(header)
    print("-" * 80)

    for name, probs in model_scores.items():
        try:
            roc = roc_auc_score(y_true, probs)
            pr = average_precision_score(y_true, probs)
        except Exception:
            roc, pr = 0.0, 0.0

        decisions = np.where(probs >= block_thresh, 2, np.where(probs >= flag_thresh, 1, 0))
        fraud = y_true == 1
        legit = y_true == 0
        blocked = decisions == 2
        n_blocked = int(blocked.sum())
        block_tp = int((blocked & fraud).sum())
        blk_prec = block_tp / max(n_blocked, 1)
        blk_rec = block_tp / max(int(fraud.sum()), 1)
        fb_rec = int(((decisions >= 1) & fraud).sum()) / max(int(fraud.sum()), 1)

        print(f"  {name:<20} {roc:>8.4f} {pr:>8.4f} {blk_prec:>8.4f} {blk_rec:>8.4f} {fb_rec:>8.4f}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 0.5,
) -> float:
    """
    Find the probability threshold that maximises F-beta score.

    beta < 1 weights precision more than recall — appropriate for fraud
    detection where false blocks (false positives) are costly.

    Parameters
    ----------
    y_true : ground-truth binary labels
    y_prob : predicted probabilities
    beta   : F-beta weighting (0.5 = precision-weighted)

    Returns
    -------
    float : optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_thresh = 0.5
    best_fbeta = 0.0
    beta_sq = beta ** 2

    for i, thresh in enumerate(thresholds):
        p = precisions[i]
        r = recalls[i]
        denom = beta_sq * p + r
        if denom > 0:
            fbeta = (1 + beta_sq) * p * r / denom
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_thresh = float(thresh)

    return round(best_thresh, 4)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def compute_latency_benchmark(
    model: Any,
    X_sample: np.ndarray,
    n_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark single-sample inference latency.

    Parameters
    ----------
    model    : fitted sklearn-compatible model with predict_proba
    X_sample : feature matrix — only the first row is used
    n_runs   : number of timing iterations

    Returns
    -------
    dict with mean_ms, p50_ms, p95_ms, p99_ms
    """
    sample = X_sample[:1]
    latencies: list[float] = []

    # Warm-up (avoid JIT / cache cold-start skewing results)
    for _ in range(min(5, n_runs)):
        model.predict_proba(sample)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_evaluation_summary(metrics: dict) -> None:
    """Print a formatted evaluation summary to stdout."""
    name = metrics.get("model_name", "model")
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Model        : {name}")
    print(f"  Threshold    : {metrics.get('threshold', 0.5):.4f}")
    print(f"  ROC-AUC      : {metrics.get('roc_auc', 0):.4f}")
    print(f"  PR-AUC       : {metrics.get('pr_auc', 0):.4f}")
    print(f"  Precision    : {metrics.get('precision', 0):.4f}")
    print(f"  Recall       : {metrics.get('recall', 0):.4f}")
    print(f"  F1           : {metrics.get('f1', 0):.4f}")
    print(f"  Positives    : {metrics.get('n_positives', 0):,}  "
          f"({metrics.get('positive_rate', 0):.2%})")
    print(f"  Negatives    : {metrics.get('n_negatives', 0):,}")
    print(f"{sep}")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    if cm:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(f"  Confusion Matrix:")
        print(f"    TN={tn:>8,}  FP={fp:>8,}")
        print(f"    FN={fn:>8,}  TP={tp:>8,}")
    print(f"{sep}\n")
    report = metrics.get("classification_report", "")
    if report:
        print(report)
