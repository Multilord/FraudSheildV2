"""
FraudShield Runtime Inference Engine
=====================================
Loads trained artifacts and scores wallet transactions in real-time.

No synthetic fallback — if artifacts are missing, raise a clear error
directing the user to run the training pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# XAI feature labels — maps IEEE-CIS / wallet feature names to human-readable text
# ---------------------------------------------------------------------------
_FEATURE_HUMAN_LABELS: dict[str, str] = {
    "TransactionAmt":      "Transaction amount",
    "amt_log":             "Transaction amount magnitude",
    "amt_to_dist_ratio":   "Amount-to-distance ratio",
    "is_high_amount":      "High-value transaction flag",
    "is_night_transaction":"Off-hours activity flag",
    "hour_of_day":         "Transaction hour",
    "day_of_week":         "Day of week",
    "C1":  "Card-linked account count",
    "C2":  "Transaction velocity",
    "C3":  "Transaction velocity",
    "C4":  "Transaction velocity",
    "C5":  "Transaction velocity",
    "C6":  "Transaction network pattern",
    "C7":  "Transaction pattern",
    "C8":  "Transaction count",
    "C9":  "Transaction count",
    "C10": "Address match count",
    "C11": "Historical transaction count",
    "C12": "Transaction frequency",
    "C13": "Card usage count",
    "C14": "Transaction count",
    "D1":  "Account age (days)",
    "D2":  "Days since last transaction",
    "D3":  "Days since last address change",
    "D4":  "Days since card first use",
    "D10": "Days since last device use",
    "D15": "Days since last address update",
    "ProductCD":      "Transaction product type",
    "DeviceType":     "Device type",
    "card4":          "Card network",
    "card6":          "Card type (credit/debit)",
    "P_emaildomain":  "Payer email domain",
    "R_emaildomain":  "Recipient email domain",
    "M4": "Card match status",
    "M5": "Card match status",
    "M6": "Card match status",
    "id_01": "Device identity signal",
    "id_02": "Device identity signal",
    "id_05": "Device identity signal",
    "id_06": "Device identity signal",
}


# ---------------------------------------------------------------------------
# Category-amount mismatch thresholds (local currency units).
# Catches "19 000 grocery" as suspicious regardless of user history.
# Naturally scales with currency: IDR amounts are large, SGD are small —
# a 19 000 SGD grocery triggers this; a 19 000 IDR grocery does not.
# ---------------------------------------------------------------------------
_CATEGORY_TYPICAL_MAX: dict[str, float] = {
    "grocery":       2_000,
    "food":          1_000,
    "utility":       5_000,
    "transport":     2_000,
    "entertainment": 5_000,
    "other":        15_000,
}


class FraudEngine:
    """Real-time fraud scoring engine backed by a 4-model ML ensemble with XAI."""

    def __init__(self) -> None:
        self.xgb_model = None
        self.lgbm_model = None
        self.rf_model = None
        self.lr_model = None
        self.preprocessor = None
        self.feature_names: list[str] = []
        self.feature_medians: dict[str, float] = {}
        self.thresholds: dict[str, float] = {"flag": 0.4, "block": 0.7}
        self.has_lgbm: bool = False
        self.has_rf: bool = False
        self.has_lr: bool = False
        self.loaded: bool = False
        self.artifact_version: Optional[str] = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load all model artifacts from MODELS_DIR.

        Returns True on success, False if artifacts are missing or corrupt.
        """
        try:
            meta_path = MODELS_DIR / "feature_metadata.json"
            if not meta_path.exists():
                logger.warning(
                    "No model artifacts found at '%s'. "
                    "Run: cd backend && python training/train_engine.py --data-dir /path/to/ieee-cis-data",
                    MODELS_DIR,
                )
                return False

            import joblib

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_names = metadata["feature_names"]
            self.feature_medians = metadata["feature_medians"]
            self.has_lgbm = metadata.get("has_lgbm", False)
            self.has_rf = metadata.get("has_rf", False)
            self.has_lr = metadata.get("has_lr", False)
            self.artifact_version = metadata.get("trained_at", "unknown")

            self.xgb_model = joblib.load(MODELS_DIR / "xgb_model.joblib")
            self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

            if self.has_lgbm and (MODELS_DIR / "lgbm_model.joblib").exists():
                self.lgbm_model = joblib.load(MODELS_DIR / "lgbm_model.joblib")

            if self.has_rf and (MODELS_DIR / "rf_model.joblib").exists():
                self.rf_model = joblib.load(MODELS_DIR / "rf_model.joblib")

            if self.has_lr and (MODELS_DIR / "lr_model.joblib").exists():
                self.lr_model = joblib.load(MODELS_DIR / "lr_model.joblib")

            thresh_path = MODELS_DIR / "thresholds.json"
            if thresh_path.exists():
                self.thresholds = json.loads(thresh_path.read_text(encoding="utf-8"))

            metrics_path = MODELS_DIR / "metrics.json"
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.loaded = True
            n_models = 1 + self.has_lgbm + self.has_rf + self.has_lr
            logger.info(
                "FraudEngine loaded successfully. Version: %s  "
                "Models: %d/4  Thresholds: flag=%.3f block=%.3f",
                self.artifact_version,
                n_models,
                self.thresholds["flag"],
                self.thresholds["block"],
            )
            return True

        except Exception as exc:
            logger.error("Failed to load model artifacts: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, wallet_tx: dict, user_profile: dict) -> dict:
        """
        Score a wallet transaction.

        Parameters
        ----------
        wallet_tx : dict
            Keys: user_id, amount, transaction_type, device_type, device_id,
                  ip_address, location, merchant, merchant_category,
                  is_new_device (bool), hour_of_day (int 0-23).
        user_profile : dict
            Keys: avg_amount, std_amount, transaction_count, device_ids (list),
                  locations (list), last_transaction_time, first_seen.

        Returns
        -------
        dict with keys:
            risk_score (0-100 int), decision (APPROVE/FLAG/BLOCK), reasons (list),
            confidence (float 0-1), latency_ms, action_required, model_breakdown.
        """
        if not self.loaded:
            raise RuntimeError(
                "Fraud detection model is not loaded. "
                "Run: cd backend && python training/train_engine.py --data-dir /path/to/ieee-cis-data"
            )

        start = time.perf_counter()

        # Import here to avoid circular imports at module load time
        from training.feature_engineering import get_wallet_feature_vector

        X = get_wallet_feature_vector(
            wallet_tx,
            user_profile,
            self.preprocessor,
            self.feature_names,
            self.feature_medians,
        )
        X = X.reshape(1, -1)

        # ── Step 2: Behavioral metrics instantly calculated ───────────────────
        behavioral_prob = self._behavioral_risk_score(wallet_tx, user_profile)

        # ── Step 3: 4 ML models create an ensemble score ──────────────────────
        xgb_prob = float(self.xgb_model.predict_proba(X)[0, 1])
        ml_probs: dict[str, float] = {"xgboost": xgb_prob}

        if self.lgbm_model is not None:
            ml_probs["lightgbm"] = float(self.lgbm_model.predict_proba(X)[0, 1])

        if self.rf_model is not None:
            ml_probs["random_forest"] = float(self.rf_model.predict_proba(X)[0, 1])

        if self.lr_model is not None:
            ml_probs["logistic_regression"] = float(self.lr_model.predict_proba(X)[0, 1])

        # Equal-weight average across all available ML models
        ml_prob = sum(ml_probs.values()) / len(ml_probs)

        # Behavioral overlay: wallet transactions lack IEEE-CIS V-series signals,
        # so V-features are median-imputed, weakening the ML signal (~0.25-0.35
        # baseline). Behavioral signals carry more weight to compensate.
        ensemble_prob = 0.30 * ml_prob + 0.70 * behavioral_prob

        model_breakdown: dict = {
            model: round(p * 100, 1) for model, p in ml_probs.items()
        }
        model_breakdown["behavioral"] = round(behavioral_prob * 100, 1)
        model_breakdown["ensemble"] = round(ensemble_prob * 100, 1)

        # ── Step 4: XAI — SHAP feature attribution ────────────────────────────
        xai_top_features = self._xai_top_features(X)

        risk_score = int(round(ensemble_prob * 100))

        flag_thresh = self.thresholds["flag"]
        block_thresh = self.thresholds["block"]

        if ensemble_prob >= block_thresh:
            decision = "BLOCK"
            action_required: Optional[str] = "Transaction blocked due to high fraud risk"
        elif ensemble_prob >= flag_thresh:
            decision = "FLAG"
            action_required = "Additional verification required"
        else:
            decision = "APPROVE"
            action_required = None

        reasons = self._generate_reasons(wallet_tx, user_profile, ensemble_prob, risk_score, xai_top_features)

        # Confidence: normalised distance from nearest decision boundary
        if decision == "BLOCK":
            span = max(1.0 - block_thresh, 1e-6)
            confidence = min(1.0, 0.6 + (ensemble_prob - block_thresh) / span * 0.4)
        elif decision == "FLAG":
            midpoint = (flag_thresh + block_thresh) / 2
            half_span = max((block_thresh - flag_thresh) / 2, 1e-6)
            confidence = 0.5 + (1 - abs(ensemble_prob - midpoint) / half_span) * 0.4
            confidence = max(0.5, min(0.9, confidence))
        else:
            span = max(flag_thresh, 1e-6)
            confidence = min(1.0, 0.6 + (flag_thresh - ensemble_prob) / span * 0.4)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "risk_score": risk_score,
            "decision": decision,
            "reasons": reasons,
            "confidence": round(float(confidence), 3),
            "latency_ms": round(latency_ms, 2),
            "action_required": action_required,
            "model_breakdown": model_breakdown,
            "xai_top_features": xai_top_features,
        }

    # ------------------------------------------------------------------
    # Reason generation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # XAI — SHAP feature attribution
    # ------------------------------------------------------------------

    def _xai_top_features(self, X: np.ndarray) -> list[dict]:
        """
        Return the top-5 SHAP feature contributions for this transaction
        using XGBoost's built-in SHAP support (no extra dependency needed).

        Each entry: {feature, label, contribution, direction}
        """
        try:
            import xgboost as xgb  # already a required dependency

            booster = self.xgb_model.get_booster()
            dm = xgb.DMatrix(X)
            # pred_contribs returns shape (n_samples, n_features + 1);
            # last column is the bias term — drop it.
            contribs = booster.predict(dm, pred_contribs=True)[0][:-1]
            top_idx = np.argsort(np.abs(contribs))[::-1][:5]
            result = []
            for i in top_idx:
                if i >= len(self.feature_names):
                    continue
                fname = self.feature_names[i]
                label = _FEATURE_HUMAN_LABELS.get(fname)
                # V-series features have obfuscated names — use a generic label
                if label is None:
                    if fname.startswith("V"):
                        label = "Card network risk signal"
                    elif fname.startswith("id_"):
                        label = "Device identity signal"
                    else:
                        label = f"Model signal ({fname})"
                contrib = float(contribs[i])
                result.append({
                    "feature": fname,
                    "label": label,
                    "contribution": round(contrib, 4),
                    "direction": "increases_risk" if contrib > 0 else "reduces_risk",
                })
            return result
        except Exception as exc:
            logger.debug("SHAP extraction failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Reason generation
    # ------------------------------------------------------------------

    def _generate_reasons(
        self,
        wallet_tx: dict,
        user_profile: dict,
        prob: float,
        risk_score: int,
        xai_top_features: Optional[list] = None,
    ) -> list[str]:
        """
        Generate human-readable risk explanations combining behavioral signals
        and SHAP-based XAI feature attributions.
        Returns at most 5 reasons.
        """
        reasons: list[str] = []

        amount = float(wallet_tx.get("amount", 0))
        merchant_cat = (wallet_tx.get("merchant_category") or "").lower()
        avg_amount = float(user_profile.get("avg_amount", 0) or 0)
        std_amount = float(user_profile.get("std_amount", 0) or 0)
        tx_count = int(user_profile.get("transaction_count", 0) or 0)
        hour = int(wallet_tx.get("hour_of_day", 12))
        is_new_device = bool(wallet_tx.get("is_new_device", False))
        device_id = wallet_tx.get("device_id", "")
        device_ids: list = user_profile.get("device_ids", []) or []
        location = wallet_tx.get("location", "")
        locations: list = user_profile.get("locations", []) or []
        tx_type = wallet_tx.get("transaction_type", "").lower()

        # Category-amount mismatch (most impactful signal — surface first)
        typical_max = _CATEGORY_TYPICAL_MAX.get(merchant_cat, 15_000)
        if merchant_cat and amount > typical_max * 2:
            multiple = amount / typical_max
            reasons.append(
                f"Amount {amount:,.0f} is {multiple:.1f}× the typical maximum for "
                f"a {merchant_cat} transaction — possible fraud"
            )

        # Amount anomaly vs personal history
        if avg_amount > 0 and std_amount > 0:
            z_score = (amount - avg_amount) / max(std_amount, 1.0)
            if z_score > 3.0:
                reasons.append(
                    f"Amount {amount:,.0f} is {z_score:.1f}σ above your account average "
                    f"({avg_amount:,.0f})"
                )
            elif z_score > 1.5:
                reasons.append("Amount is significantly higher than your usual spending")
        elif avg_amount > 0 and amount > avg_amount * 3:
            reasons.append(
                f"Amount {amount:,.0f} is {amount/avg_amount:.1f}× your account average "
                f"({avg_amount:,.0f})"
            )

        # High-value threshold
        if amount >= 10_000:
            reasons.append(f"High-value transaction: {amount:,.0f}")

        # Unrecognized device
        known_device = device_id and device_ids and device_id in device_ids
        if is_new_device or (device_id and device_ids and not known_device):
            reasons.append("Transaction initiated from an unrecognized device")

        # Off-hours activity
        if hour < 5 or hour >= 23:
            reasons.append(f"Unusual transaction time ({hour:02d}:00 — off-hours activity)")

        # New location
        if location and locations and location not in locations:
            reasons.append(f"New location detected: {location}")

        # Account age / history
        if tx_count == 0:
            reasons.append("First transaction from this account")
        elif tx_count < 3:
            reasons.append("New account with limited transaction history")

        # Large cash-out
        if tx_type == "cashout" and amount >= 1_000:
            reasons.append(f"Large cash-out transaction: {amount:,.0f}")

        # XAI: append top model-driven signals for high/medium risk scores
        if xai_top_features and risk_score >= 30:
            for feat in xai_top_features:
                if feat["direction"] == "increases_risk" and len(reasons) < 5:
                    reasons.append(f"Model flagged: {feat['label']} (XAI signal)")

        # Generic reasons when none of the above triggered
        if not reasons:
            if risk_score < 30:
                reasons.append("Transaction consistent with normal account activity")
                reasons.append("Known device and location")
                reasons.append("Amount within typical spending range")
            elif risk_score < 60:
                reasons.append("Ensemble models detected unusual patterns")
            else:
                reasons.append("Multiple fraud risk signals detected across all models")

        return reasons[:5]

    # ------------------------------------------------------------------
    # Behavioral risk scoring
    # ------------------------------------------------------------------

    def _behavioral_risk_score(self, wallet_tx: dict, user_profile: dict) -> float:
        """
        Compute a [0, 1] behavioral risk score from wallet context signals.

        Two distinct regimes:
          - With history : amount z-score vs personal baseline (primary signal).
          - No history   : category-relative and absolute thresholds.

        Signal max contributions:
          - Amount z-score vs history   : up to 0.45  (was 0.20)
          - Category-amount mismatch    : up to 0.40  (NEW — catches 19k grocery)
          - Absolute high-value         : up to 0.20  (was 0.10)
          - Unrecognized device         : 0.15
          - Off-hours activity          : 0.08
          - New location                : 0.12
          - Thin / new account          : 0.10
          - Large cash-out              : 0.20        (was 0.10)
        """
        score = 0.0

        amount = float(wallet_tx.get("amount", 0))
        merchant_cat = (wallet_tx.get("merchant_category") or "other").lower()
        avg_amount = float(user_profile.get("avg_amount", 0) or 0)
        std_amount = float(user_profile.get("std_amount", 0) or 0)
        tx_count = int(user_profile.get("transaction_count", 0) or 0)
        hour = int(wallet_tx.get("hour_of_day", 12))
        is_new_device = bool(wallet_tx.get("is_new_device", False))
        device_id = wallet_tx.get("device_id", "")
        device_ids: list = user_profile.get("device_ids", []) or []
        location = wallet_tx.get("location", "")
        locations: list = user_profile.get("locations", []) or []
        tx_type = wallet_tx.get("transaction_type", "").lower()

        # ── 1. Amount vs personal history (z-score) ───────────────────────────
        if avg_amount > 0 and std_amount > 0:
            z = (amount - avg_amount) / max(std_amount, 1.0)
            if z > 10.0:
                score += 0.45
            elif z > 5.0:
                score += 0.35
            elif z > 3.0:
                score += 0.22
            elif z > 2.0:
                score += 0.12
            elif z > 1.5:
                score += 0.06
        elif avg_amount > 0:
            # Only average available (< 2 transactions) — use simple ratio
            ratio = amount / max(avg_amount, 1.0)
            if ratio > 10:
                score += 0.40
            elif ratio > 5:
                score += 0.25
            elif ratio > 3:
                score += 0.12

        # ── 2. Category-amount mismatch ───────────────────────────────────────
        # Core fix: "19 000 grocery" is suspicious regardless of user history.
        typical_max = _CATEGORY_TYPICAL_MAX.get(merchant_cat, 15_000)
        if amount > typical_max * 10:
            score += 0.40
        elif amount > typical_max * 5:
            score += 0.30
        elif amount > typical_max * 2:
            score += 0.20
        elif amount > typical_max:
            score += 0.08

        # ── 3. Absolute high-value threshold ──────────────────────────────────
        if amount >= 50_000:
            score += 0.20
        elif amount >= 10_000:
            score += 0.15
        elif amount >= 5_000:
            score += 0.08
        elif amount >= 2_000:
            score += 0.04

        # ── 4. Unrecognized device ────────────────────────────────────────────
        known_device = device_id and device_ids and device_id in device_ids
        if is_new_device or (device_id and not known_device):
            score += 0.15

        # ── 5. Off-hours (midnight–4 AM or 11 PM+) ───────────────────────────
        if hour < 4 or hour >= 23:
            score += 0.08
        elif hour < 6:
            score += 0.04

        # ── 6. New location ───────────────────────────────────────────────────
        if location and locations and location not in locations:
            score += 0.12
        elif location and not locations:
            score += 0.06

        # ── 7. Account history depth ──────────────────────────────────────────
        if tx_count == 0:
            score += 0.10
        elif tx_count < 3:
            score += 0.07
        elif tx_count < 10:
            score += 0.03

        # ── 8. Large cash-out ─────────────────────────────────────────────────
        if tx_type == "cashout" and amount >= 5_000:
            score += 0.20
        elif tx_type == "cashout" and amount >= 1_000:
            score += 0.12
        elif tx_type == "cashout" and amount >= 500:
            score += 0.07

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return training metrics loaded from metrics.json."""
        return self.metrics

    def is_loaded(self) -> bool:
        """Return True if model artifacts are loaded and ready."""
        return self.loaded


# ---------------------------------------------------------------------------
# Global singleton used by the FastAPI application
# ---------------------------------------------------------------------------
engine = FraudEngine()
