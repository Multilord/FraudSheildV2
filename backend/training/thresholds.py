"""
FraudShield threshold analysis for three-tier (APPROVE / FLAG / BLOCK) decisions.

Uses cost-sensitive threshold selection:
  - BLOCK  : highest threshold where block precision >= 85% (minimize false blocks)
  - FLAG   : highest threshold below BLOCK where flag+block recall >= 75%
             (ensure most fraud gets at least flagged for review)

Falls back to F-beta optimization if the cost constraints cannot be met.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Analyse precision, recall and F1 at thresholds from 0.0 to 1.0 in
    steps of 0.10.

    Returns
    -------
    dict mapping threshold string keys to metric dicts, e.g.
    {
      "0.10": {"threshold": 0.10, "precision": ..., "recall": ..., "f1": ...,
               "n_predicted_positive": ...},
      ...
    }
    """
    precisions, recalls, thresh_vals = precision_recall_curve(y_true, y_prob)

    analysis: dict = {}
    steps = [round(t, 2) for t in np.arange(0.0, 1.01, 0.10)]

    for target in steps:
        if len(thresh_vals) == 0:
            p, r = 0.0, 0.0
        else:
            idx = int(np.argmin(np.abs(thresh_vals - target)))
            p = float(precisions[idx])
            r = float(recalls[idx])

        denom = p + r
        f1 = 2 * p * r / denom if denom > 0 else 0.0
        n_pred_pos = int((y_prob >= target).sum())

        key = f"{target:.2f}"
        analysis[key] = {
            "threshold": target,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "n_predicted_positive": n_pred_pos,
        }

    return analysis


# ---------------------------------------------------------------------------
# Decision threshold computation (cost-sensitive)
# ---------------------------------------------------------------------------

def _fbeta(p: np.ndarray, r: np.ndarray, b: float) -> np.ndarray:
    b2 = b ** 2
    denom = b2 * p + r
    return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)


def compute_decision_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_block_precision: float = 0.85,
    min_flag_block_recall: float = 0.75,
) -> dict:
    """
    Find BLOCK and FLAG thresholds using cost-sensitive objectives.

    Strategy
    --------
    BLOCK threshold:
      Among all candidate thresholds where precision >= min_block_precision (85%),
      select the lowest one (maximises recall at block tier while keeping
      false-block rate acceptable).
      Falls back to F0.5-maximising threshold if constraint cannot be met.

    FLAG threshold:
      Among thresholds below BLOCK, select the highest one where combined
      flag+block recall >= min_flag_block_recall (75%).  This ensures that
      the overwhelming majority of fraud cases get at least flagged.
      Falls back to F1-maximising threshold below BLOCK if constraint cannot be met.

    APPROVE: below FLAG threshold.

    Parameters
    ----------
    y_true                : ground-truth binary labels
    y_prob                : predicted fraud probabilities
    min_block_precision   : minimum precision required to BLOCK (default 0.85)
    min_flag_block_recall : minimum combined recall required at FLAG tier (default 0.75)

    Returns
    -------
    dict with keys: flag (float), block (float), description (str)
    """
    precisions, recalls, thresh_vals = precision_recall_curve(y_true, y_prob)

    # Drop the trailing sentinel entry (precision=1, recall=0, no threshold)
    n = len(thresh_vals)
    precisions = precisions[:n]
    recalls = recalls[:n]

    # ── BLOCK threshold ───────────────────────────────────────────────────────
    high_prec_mask = precisions >= min_block_precision
    if high_prec_mask.any():
        # Lowest threshold meeting the precision constraint = highest recall
        block_thresh = float(thresh_vals[high_prec_mask].min())
        block_idx = int(np.argmin(np.abs(thresh_vals - block_thresh)))
        block_method = f"cost-sensitive (precision >= {min_block_precision:.0%})"
    else:
        # Fallback: maximise F0.5 (precision-weighted)
        fb_block = _fbeta(precisions, recalls, 0.5)
        block_idx = int(np.argmax(fb_block))
        block_thresh = round(float(thresh_vals[block_idx]), 4)
        block_method = "F0.5-optimised (precision constraint not achievable)"

    block_thresh = round(block_thresh, 4)
    print(
        f"[Thresholds] BLOCK: threshold={block_thresh:.4f}  "
        f"precision={precisions[block_idx]:.4f}  recall={recalls[block_idx]:.4f}  "
        f"method={block_method}"
    )

    # ── FLAG threshold ────────────────────────────────────────────────────────
    below_block = thresh_vals < block_thresh

    if below_block.any():
        # Find thresholds below BLOCK where recall >= min_flag_block_recall
        high_recall_mask = below_block & (recalls >= min_flag_block_recall)
        if high_recall_mask.any():
            # Highest such threshold (most selective FLAG that meets recall goal)
            flag_thresh = float(thresh_vals[high_recall_mask].max())
            flag_method = f"cost-sensitive (recall >= {min_flag_block_recall:.0%})"
        else:
            # Fallback: maximise F1 below block
            fb_flag = _fbeta(precisions, recalls, 1.0)
            fb_flag_masked = np.where(below_block, fb_flag, -1.0)
            flag_idx = int(np.argmax(fb_flag_masked))
            flag_thresh = float(thresh_vals[flag_idx])
            flag_method = "F1-optimised (recall constraint not achievable below BLOCK)"
    else:
        flag_thresh = block_thresh * 0.6
        flag_method = "fallback midpoint"

    flag_thresh = round(flag_thresh, 4)
    flag_idx_approx = int(np.argmin(np.abs(thresh_vals - flag_thresh)))
    print(
        f"[Thresholds] FLAG:  threshold={flag_thresh:.4f}  "
        f"precision={precisions[flag_idx_approx]:.4f}  recall={recalls[flag_idx_approx]:.4f}  "
        f"method={flag_method}"
    )

    description = (
        f"BLOCK >= {block_thresh} ({block_method}), "
        f"FLAG >= {flag_thresh} ({flag_method}), "
        f"APPROVE < {flag_thresh}"
    )
    print(f"[Thresholds] {description}")

    return {
        "flag": flag_thresh,
        "block": block_thresh,
        "description": description,
    }
