"""
FraudShield threshold analysis for three-tier (APPROVE / FLAG / BLOCK) decisions.
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
        # Find the index in thresh_vals closest to the target
        if len(thresh_vals) == 0:
            p, r = 0.0, 0.0
        else:
            idx = int(np.argmin(np.abs(thresh_vals - target)))
            # precision_recall_curve returns n+1 precision/recall values
            # for n thresholds; index i corresponds to threshold thresh_vals[i]
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
# Decision threshold computation
# ---------------------------------------------------------------------------

def compute_decision_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta_block: float = 0.5,
    beta_flag: float = 1.0,
) -> dict:
    """
    Find BLOCK and FLAG thresholds using F-beta score optimisation.

    Strategy
    --------
    - BLOCK threshold : threshold maximising F(beta=0.5), weights precision
                        4x more than recall — minimises false blocks.
    - FLAG threshold  : threshold maximising F(beta=1.0, F1) below the BLOCK
                        threshold — balanced precision/recall for review queue.
    - APPROVE         : below FLAG threshold.

    Parameters
    ----------
    y_true      : ground-truth binary labels
    y_prob      : predicted fraud probabilities
    beta_block  : F-beta weight for BLOCK tier (default 0.5, precision-focused)
    beta_flag   : F-beta weight for FLAG tier (default 1.0, balanced)

    Returns
    -------
    dict with keys: flag (float), block (float), description (str)
    """
    precisions, recalls, thresh_vals = precision_recall_curve(y_true, y_prob)

    # Drop the trailing sentinel entry (precision=1, recall=0, no threshold)
    n = len(thresh_vals)
    precisions = precisions[:n]
    recalls = recalls[:n]

    def fbeta(p: np.ndarray, r: np.ndarray, b: float) -> np.ndarray:
        b2 = b ** 2
        denom = b2 * p + r
        return np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)

    # ── BLOCK threshold (maximise F-beta_block) ──────────────────────────────
    fb_block = fbeta(precisions, recalls, beta_block)
    block_idx = int(np.argmax(fb_block))
    block_thresh = round(float(thresh_vals[block_idx]), 4)

    print(
        f"[Thresholds] BLOCK: threshold={block_thresh:.4f}  "
        f"precision={precisions[block_idx]:.4f}  recall={recalls[block_idx]:.4f}  "
        f"F{beta_block}={fb_block[block_idx]:.4f}"
    )

    # ── FLAG threshold (maximise F-beta_flag below block_thresh) ─────────────
    below_block = thresh_vals < block_thresh
    if below_block.any():
        fb_flag = fbeta(precisions, recalls, beta_flag)
        fb_flag_masked = np.where(below_block, fb_flag, -1.0)
        flag_idx = int(np.argmax(fb_flag_masked))
        flag_thresh = round(float(thresh_vals[flag_idx]), 4)
        print(
            f"[Thresholds] FLAG:  threshold={flag_thresh:.4f}  "
            f"precision={precisions[flag_idx]:.4f}  recall={recalls[flag_idx]:.4f}  "
            f"F{beta_flag}={fb_flag[flag_idx]:.4f}"
        )
    else:
        flag_thresh = round(block_thresh * 0.6, 4)
        print(f"[Thresholds] FLAG: fallback midpoint={flag_thresh:.4f}")

    description = (
        f"BLOCK >= {block_thresh} (F{beta_block}-optimised), "
        f"FLAG >= {flag_thresh} (F{beta_flag}-optimised below block), "
        f"APPROVE < {flag_thresh}"
    )
    print(f"[Thresholds] {description}")

    return {
        "flag": flag_thresh,
        "block": block_thresh,
        "description": description,
    }
