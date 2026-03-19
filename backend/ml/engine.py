"""
FraudShield Runtime Inference Engine
=====================================
Loads trained artifacts and scores wallet transactions in real-time.

Pipeline (mirrors the 4-step solution architecture):
  1. API receives real-time transaction payload
  2. Behavioral metrics instantly calculated from user history
  3. 4 ML models create a calibrated ensemble score:
       XGBoost + LightGBM (supervised)
       Isolation Forest + LOF (unsupervised anomaly detection)
       → Meta-learner (LogisticRegression) produces final probability
  4. XAI (SHAP) returns decision with feature-level explanations

No synthetic fallback — if artifacts are missing, a clear error directs
the user to run the training pipeline.
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
    "amt_vs_card_mean":    "Amount vs card-level average",
    "amt_z_card":          "Amount z-score vs card history",
    "amt_percentile":      "Amount population percentile",
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


def _sigmoid_normalize(raw_scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (raw_scores - mean) / max(std, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Currency normalization — approximate USD equivalent per ASEAN country.
# Makes behavioral thresholds currency-aware: 100k VND ($4) ≠ 100k SGD ($74k).
# Rates are stable enough for risk-scoring purposes; update yearly if needed.
# ---------------------------------------------------------------------------
_COUNTRY_TO_USD: dict[str, float] = {
    "BN": 0.74,       # BND  — Brunei Dollar
    "KH": 0.00025,    # KHR  — Cambodian Riel
    "ID": 0.000063,   # IDR  — Indonesian Rupiah
    "LA": 0.000047,   # LAK  — Lao Kip
    "MY": 0.21,       # MYR  — Malaysian Ringgit
    "MM": 0.00048,    # MMK  — Myanmar Kyat
    "PH": 0.018,      # PHP  — Philippine Peso
    "SG": 0.74,       # SGD  — Singapore Dollar
    "TH": 0.028,      # THB  — Thai Baht
    "TL": 1.0,        # USD  — Timor-Leste uses USD
    "VN": 0.000040,   # VND  — Vietnamese Dong
}

# Merchant categories where high amounts are inherently suspicious
# (these are typically small-ticket purchases).
_LOW_AMOUNT_CATEGORIES: frozenset[str] = frozenset({
    "grocery", "food", "transport", "coffee", "pharmacy",
    "convenience", "fast_food", "snack", "beverage",
})


def _usd_equivalent(amount: float, location: str) -> float:
    """Convert local-currency amount to approximate USD using the country code
    embedded in the location string (e.g., 'Manila, PH' → code 'PH')."""
    code = location.split(", ")[-1].strip().upper() if location else ""
    rate = _COUNTRY_TO_USD.get(code, 1.0)   # default 1:1 (USD) for unknowns
    return amount * rate


class FraudEngine:
    """Real-time fraud scoring engine backed by a calibrated 4-model ML ensemble with XAI."""

    def __init__(self) -> None:
        self.xgb_model      = None
        self.lgbm_model     = None
        self.iforest_model  = None
        self.lof_model      = None
        self.meta_model     = None
        self.preprocessor   = None
        self.feature_names: list[str] = []
        self.feature_medians: dict[str, float] = {}
        self.pop_quantiles: Optional[np.ndarray] = None
        self.anomaly_score_stats: dict[str, float] = {}
        self.meta_feature_names: list[str] = []
        self.thresholds: dict[str, float] = {"flag": 0.4, "block": 0.7}
        self.has_lgbm:    bool = False
        self.has_iforest: bool = False
        self.has_lof:     bool = False
        self.has_meta:    bool = False
        self.loaded:      bool = False
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
            self.feature_names       = metadata["feature_names"]
            self.feature_medians     = metadata["feature_medians"]
            self.has_lgbm            = metadata.get("has_lgbm", False)
            self.has_iforest         = metadata.get("has_iforest", False)
            self.has_lof             = metadata.get("has_lof", False)
            self.has_meta            = metadata.get("has_meta", False)
            self.artifact_version    = metadata.get("trained_at", "unknown")
            self.meta_feature_names  = metadata.get("meta_feature_names", [])
            self.anomaly_score_stats = metadata.get("anomaly_score_stats", {})

            pq = metadata.get("pop_quantiles")
            if pq:
                self.pop_quantiles = np.array(pq, dtype=np.float64)

            self.xgb_model    = joblib.load(MODELS_DIR / "xgb_model.joblib")
            self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

            if self.has_lgbm and (MODELS_DIR / "lgbm_model.joblib").exists():
                self.lgbm_model = joblib.load(MODELS_DIR / "lgbm_model.joblib")

            if self.has_iforest and (MODELS_DIR / "iforest_model.joblib").exists():
                self.iforest_model = joblib.load(MODELS_DIR / "iforest_model.joblib")

            if self.has_lof and (MODELS_DIR / "lof_model.joblib").exists():
                self.lof_model = joblib.load(MODELS_DIR / "lof_model.joblib")

            if self.has_meta and (MODELS_DIR / "meta_model.joblib").exists():
                self.meta_model = joblib.load(MODELS_DIR / "meta_model.joblib")

            thresh_path = MODELS_DIR / "thresholds.json"
            if thresh_path.exists():
                self.thresholds = json.loads(thresh_path.read_text(encoding="utf-8"))

            metrics_path = MODELS_DIR / "metrics.json"
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self.loaded = True
            n_models = 1 + self.has_lgbm + self.has_iforest + self.has_lof
            logger.info(
                "FraudEngine loaded. Version: %s  Models: %d/4  "
                "Meta-learner: %s  Thresholds: flag=%.3f block=%.3f",
                self.artifact_version,
                n_models,
                "yes" if self.has_meta else "no",
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

        from training.feature_engineering import get_wallet_feature_vector

        X = get_wallet_feature_vector(
            wallet_tx,
            user_profile,
            self.preprocessor,
            self.feature_names,
            self.feature_medians,
            pop_quantiles=self.pop_quantiles,
        )
        X = X.reshape(1, -1)

        # ── Step 2: Behavioral metrics instantly calculated ───────────────────
        behavioral_prob = self._behavioral_risk_score(wallet_tx, user_profile)

        # ── Step 3: 4 ML models create a calibrated ensemble score ────────────

        # Supervised models
        xgb_prob = float(self.xgb_model.predict_proba(X)[0, 1])
        ml_scores: dict[str, float] = {"xgboost": xgb_prob}

        if self.lgbm_model is not None:
            ml_scores["lightgbm"] = float(self.lgbm_model.predict_proba(X)[0, 1])

        # Unsupervised anomaly models
        if self.iforest_model is not None:
            if_raw = float(-self.iforest_model.score_samples(X)[0])
            if_mean = self.anomaly_score_stats.get("iforest_mean", if_raw)
            if_std  = self.anomaly_score_stats.get("iforest_std", 1.0)
            if_prob = float(_sigmoid_normalize(np.array([if_raw]), if_mean, if_std)[0])
            ml_scores["isolation_forest"] = if_prob
        else:
            if_raw, if_prob = 0.0, 0.5

        if self.lof_model is not None:
            lof_raw = float(-self.lof_model.decision_function(X)[0])
            lof_mean = self.anomaly_score_stats.get("lof_mean", lof_raw)
            lof_std  = self.anomaly_score_stats.get("lof_std", 1.0)
            lof_prob = float(_sigmoid_normalize(np.array([lof_raw]), lof_mean, lof_std)[0])
            ml_scores["lof"] = lof_prob
        else:
            lof_raw, lof_prob = 0.0, 0.5

        # ── Meta-learner ensemble ─────────────────────────────────────────────
        if self.meta_model is not None and self.meta_feature_names:
            # Build stack in the exact order the meta-learner was trained on
            stack = []
            for feat_name in self.meta_feature_names:
                stack.append(ml_scores.get(feat_name, 0.5))
            X_stack = np.array([stack], dtype=np.float64)
            ml_prob = float(self.meta_model.predict_proba(X_stack)[0, 1])
        else:
            # Fallback: equal-weight average if meta-learner not available
            ml_prob = sum(ml_scores.values()) / max(len(ml_scores), 1)

        # Behavioral overlay — wallet transactions lack IEEE-CIS V-series signals
        # so V-features are median-imputed. Behavioral signals carry more weight.
        # 25/75 split: behavioral dominates; ML adds calibration from training data.
        ensemble_prob = 0.25 * ml_prob + 0.75 * behavioral_prob

        model_breakdown: dict = {
            model: round(p * 100, 1) for model, p in ml_scores.items()
        }
        model_breakdown["behavioral"] = round(behavioral_prob * 100, 1)
        model_breakdown["ensemble"]   = round(ensemble_prob * 100, 1)

        # ── Step 4: XAI — SHAP feature attribution ────────────────────────────
        xai_top_features = self._xai_top_features(X)

        risk_score = int(round(ensemble_prob * 100))

        flag_thresh  = self.thresholds["flag"]
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
            "risk_score":      risk_score,
            "decision":        decision,
            "reasons":         reasons,
            "confidence":      round(float(confidence), 3),
            "latency_ms":      round(latency_ms, 2),
            "action_required": action_required,
            "model_breakdown": model_breakdown,
            "xai_top_features": xai_top_features,
        }

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
            import xgboost as xgb

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
                if label is None:
                    if fname.startswith("V"):
                        label = "Card network risk signal"
                    elif fname.startswith("id_"):
                        label = "Device identity signal"
                    else:
                        label = f"Model signal ({fname})"
                contrib = float(contribs[i])
                result.append({
                    "feature":      fname,
                    "label":        label,
                    "contribution": round(contrib, 4),
                    "direction":    "increases_risk" if contrib > 0 else "reduces_risk",
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

        amount       = float(wallet_tx.get("amount", 0))
        location     = wallet_tx.get("location", "")
        merchant_cat = (wallet_tx.get("merchant_category") or "").lower().strip()
        avg_amount   = float(user_profile.get("avg_amount", 0) or 0)
        std_amount   = float(user_profile.get("std_amount", 0) or 0)
        tx_count     = int(user_profile.get("transaction_count", 0) or 0)
        hour         = int(wallet_tx.get("hour_of_day", 12))
        is_new_dev   = bool(wallet_tx.get("is_new_device", False))
        device_id    = wallet_tx.get("device_id", "")
        device_ids   = user_profile.get("device_ids", []) or []
        locations    = user_profile.get("locations", []) or []
        tx_type      = wallet_tx.get("transaction_type", "").lower()
        usd          = _usd_equivalent(amount, location)

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

        # High-value (USD-aware)
        if usd >= 10_000:
            reasons.append(f"Very high-value transaction (~${usd:,.0f} USD equivalent)")
        elif usd >= 1_000:
            reasons.append(f"High-value transaction (~${usd:,.0f} USD equivalent)")
        elif amount >= 10_000:
            reasons.append(f"High-value transaction: {amount:,.0f}")

        # Small-category but large amount
        if merchant_cat in _LOW_AMOUNT_CATEGORIES and usd >= 100:
            reasons.append(
                f"Unusually large {merchant_cat} transaction "
                f"(~${usd:,.0f} USD) — this category typically involves small amounts"
            )

        # Unrecognized device
        known_device = device_id and device_ids and device_id in device_ids
        if is_new_dev or (device_id and device_ids and not known_device):
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

        # Generic fallback
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
    # Behavioral risk scoring (generalized signal-convergence)
    # ------------------------------------------------------------------

    def _behavioral_risk_score(self, wallet_tx: dict, user_profile: dict) -> float:
        """
        Compute a [0, 1] behavioral risk score from wallet context signals.

        Design principles:
          - Currency-aware: amounts are converted to USD equivalent so that
            100k VND ($4) and 100k SGD ($74k) are scored differently.
          - Signal convergence: more independent anomaly signals → higher risk
            via a convergence multiplier.
          - No per-category hardcoded amount caps; instead a lightweight
            "small-ticket category + high USD" compound signal covers the
            grocery/food/transport pattern without category-specific magic numbers.

        Signal contributions (approximate maximums):
          - Amount z-score vs personal history   : up to 0.42
          - Absolute USD threshold               : up to 0.55
          - Small-category + high USD            : up to 0.22
          - Compound (new account + high USD)    : up to 0.22
          - Unrecognized device                  : 0.10
          - Off-hours activity                   : 0.08
          - New/unknown location                 : 0.10
          - Thin / new account                   : 0.08
          - Large cash-out (USD-aware)            : up to 0.22
        """
        amount      = float(wallet_tx.get("amount", 0))
        location    = wallet_tx.get("location", "")
        merchant_cat = (wallet_tx.get("merchant_category") or "").lower().strip()
        avg_amount  = float(user_profile.get("avg_amount", 0) or 0)
        std_amount  = float(user_profile.get("std_amount", 0) or 0)
        tx_count    = int(user_profile.get("transaction_count", 0) or 0)
        hour        = int(wallet_tx.get("hour_of_day", 12))
        is_new_dev  = bool(wallet_tx.get("is_new_device", False))
        device_id   = wallet_tx.get("device_id", "")
        device_ids  = user_profile.get("device_ids", []) or []
        locations   = user_profile.get("locations", []) or []
        tx_type     = wallet_tx.get("transaction_type", "").lower()

        # Normalize amount to USD for currency-agnostic absolute thresholds
        usd = _usd_equivalent(amount, location)

        signals: list[float] = []

        # ── 1. Amount vs personal history (z-score) ───────────────────────────
        if avg_amount > 0 and std_amount > 0:
            z = (amount - avg_amount) / max(std_amount, 1.0)
            if z > 10.0:   signals.append(0.42)
            elif z > 5.0:  signals.append(0.34)
            elif z > 3.0:  signals.append(0.22)
            elif z > 2.0:  signals.append(0.12)
            elif z > 1.5:  signals.append(0.06)
        elif avg_amount > 0:
            # Only average available (< 2 txns) — ratio-based
            ratio = amount / max(avg_amount, 1.0)
            if ratio > 10:  signals.append(0.35)
            elif ratio > 5: signals.append(0.22)
            elif ratio > 3: signals.append(0.10)

        # ── 2. Absolute USD threshold (currency-aware) ────────────────────────
        # Thresholds represent USD equivalents; works correctly across all ASEAN
        # currencies — 100k VND ($4) won't trigger, 100k SGD ($74k) will hit max.
        if usd >= 50_000:    signals.append(0.55)
        elif usd >= 10_000:  signals.append(0.45)
        elif usd >= 5_000:   signals.append(0.35)
        elif usd >= 2_000:   signals.append(0.28)
        elif usd >= 1_500:   signals.append(0.22)
        elif usd >= 1_000:   signals.append(0.18)
        elif usd >= 500:     signals.append(0.10)
        elif usd >= 200:     signals.append(0.05)

        # ── 3. Small-category + large USD (generalised pattern) ──────────────
        # Covers "grocery for 100k" without needing per-category amount caps.
        # Only triggered when the category is a typically small-ticket one.
        if merchant_cat in _LOW_AMOUNT_CATEGORIES:
            if usd >= 300:    signals.append(0.22)
            elif usd >= 100:  signals.append(0.10)

        # ── 4. Compound: new/thin account + high USD ──────────────────────────
        # First-party fraud pattern: new account making large first transaction.
        if tx_count < 3 and usd >= 500:   signals.append(0.22)
        elif tx_count < 3 and usd >= 100:  signals.append(0.10)

        # ── 5. Unrecognized device ────────────────────────────────────────────
        known_device = device_id and device_ids and device_id in device_ids
        if is_new_dev or (device_id and not known_device):
            signals.append(0.10)

        # ── 6. Off-hours (midnight–4 AM or 11 PM+) ───────────────────────────
        if hour < 4 or hour >= 23:    signals.append(0.08)
        elif hour < 6:                signals.append(0.04)

        # ── 7. New / unknown location ─────────────────────────────────────────
        if location and locations and location not in locations:
            signals.append(0.10)
        elif location and not locations:
            signals.append(0.05)

        # ── 8. Account depth ──────────────────────────────────────────────────
        if tx_count == 0:    signals.append(0.08)
        elif tx_count < 3:   signals.append(0.05)
        elif tx_count < 10:  signals.append(0.02)

        # ── 9. Large cash-out (USD-aware) ─────────────────────────────────────
        if tx_type == "cashout":
            if usd >= 1_000:   signals.append(0.22)
            elif usd >= 200:   signals.append(0.14)
            elif usd >= 50:    signals.append(0.08)

        # ── Convergence escalation ────────────────────────────────────────────
        # Multiple independent signals co-occurring = amplified risk.
        base = sum(signals)
        n    = len(signals)
        if n >= 5:    base *= 1.50
        elif n >= 4:  base *= 1.35
        elif n >= 3:  base *= 1.20
        elif n >= 2:  base *= 1.10

        return min(1.0, base)

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
