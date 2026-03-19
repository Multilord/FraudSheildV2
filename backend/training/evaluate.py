"""
FraudShield model evaluation utilities.
"""

from __future__ import annotations

import json
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
# Core evaluation
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
    sep = "─" * 60
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
