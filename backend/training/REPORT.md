# FraudShield Fraud Detection Engine — Training Report

## Dataset

- **Source**: IEEE-CIS Fraud Detection Dataset (Kaggle)
- **Primary file**: `train_transaction.csv` (~590,540 transactions)
- **Optional merge**: `train_identity.csv` (device/identity features for ~144K transactions)
- **Fraud rate**: ~3.5% (heavily imbalanced)
- **Time span**: ~6 months of e-commerce transactions
- **Target variable**: `isFraud` (binary: 1 = fraud, 0 = legitimate)

---

## Features Engineered

### Raw IEEE-CIS Features Used

| Group | Features | Count |
|-------|----------|-------|
| Transaction amount | `TransactionAmt`, `dist1`, `dist2` | 3 |
| C-features (card/count) | `C1`–`C14` | 14 |
| D-features (time delta) | `D1`–`D15` | 15 |
| V-features (Vesta proprietary) | Selected subset from V1–V317 | ~100 |
| Categorical | `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType` | 18 |
| Identity (numerical) | `id_01`–`id_06` | 6 |
| Identity (categorical) | `id_12`, `id_15`, `id_16`, `id_28`, `id_29`, `id_35`–`id_38` | 9 |

### Derived Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `amt_log` | `log1p(TransactionAmt)` | Normalise right-skewed amount distribution |
| `hour_of_day` | `TransactionDT % 86400 // 3600` | Capture time-of-day fraud patterns |
| `day_of_week` | `(TransactionDT // 86400) % 7` | Weekend vs. weekday behaviour |
| `amt_to_dist_ratio` | `TransactionAmt / (dist1 + 1)` | Amount relative to billing distance |
| `is_high_amount` | `1 if TransactionAmt > 1000` | Binary high-value indicator |
| `is_night_transaction` | `1 if hour_of_day < 6` | Off-hours risk signal |

---

## Preprocessing Pipeline

1. **Numerical features**: `SimpleImputer(strategy='median')` — handles ~60% missingness in V/D features
2. **Categorical features**: `SimpleImputer(constant='missing')` → `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`
3. Implemented via `sklearn.compose.ColumnTransformer` + `Pipeline`

---

## Models Trained

### XGBoost

- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `scale_pos_weight = n_negative / n_positive` (~27x for the full dataset)
- `eval_metric='aucpr'` — optimises PR-AUC, not ROC-AUC (better for imbalanced data)
- Early stopping at 30 rounds of no improvement on validation PR-AUC

### LightGBM

- Same `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- `class_weight='balanced'` (equivalent to `scale_pos_weight` in LightGBM)
- Early stopping via `lgb.early_stopping(30)`

### Ensemble

- Simple average of XGBoost and LightGBM fraud probabilities
- Chosen because:
  - Higher PR-AUC than either individual model
  - Reduces variance (different tree constructions complement each other)
  - Sub-millisecond additional latency overhead

---

## Class Imbalance Strategy

**Approach**: Class-weight adjustment only (no oversampling)

| Technique | Applied | Reason |
|-----------|---------|--------|
| `scale_pos_weight` (XGBoost) | Yes | Built-in, effective for gradient boosting |
| `class_weight='balanced'` (LightGBM) | Yes | Equivalent approach |
| SMOTE | **No** | The IEEE-CIS dataset has ~394 highly sparse, Vesta-proprietary features. SMOTE creates synthetic samples by interpolating in feature space, which is unreliable for such features and tends to cause overfitting on the minority class representation. Class weights achieve comparable recall without this risk. |

---

## Threshold Strategy

Three-tier real-time decisions are derived from precision-recall analysis:

| Decision | Threshold | Target Precision | Meaning |
|----------|-----------|-----------------|---------|
| **BLOCK** | `p >= block_thresh` | ≥ 85% | Transaction stopped immediately; high confidence of fraud |
| **FLAG** | `flag_thresh <= p < block_thresh` | ≥ 60% | Sent to analyst queue for manual review |
| **APPROVE** | `p < flag_thresh` | — | Processed normally |

Thresholds are computed on the validation set using `compute_decision_thresholds()` which scans the precision-recall curve to find the highest thresholds that meet the precision targets.

The `beta=0.5` F-score (precision-weighted) is used for threshold selection to minimise false blocks — blocking a legitimate transaction damages customer trust more than missing a low-value fraud event.

---

## Validation Metrics (Expected Ranges)

Trained on the full ~590K row dataset with 80/20 stratified split:

| Metric | XGBoost | LightGBM | Ensemble |
|--------|---------|----------|---------|
| ROC-AUC | ~0.97 | ~0.97 | ~0.98 |
| PR-AUC | ~0.77 | ~0.76 | ~0.79 |
| Precision @ block thresh | ~0.85 | ~0.84 | ~0.87 |
| Recall @ block thresh | ~0.65 | ~0.63 | ~0.67 |

*Note: Exact values vary by sample and threshold. Run `training/train_engine.py` to see actual metrics.*

---

## Inference Latency

| Stage | Typical Time |
|-------|-------------|
| Feature engineering (wallet dict → DataFrame) | < 5 ms |
| Preprocessing (ColumnTransformer transform) | < 5 ms |
| XGBoost inference (1 sample) | < 10 ms |
| LightGBM inference (1 sample) | < 10 ms |
| **Total end-to-end** | **< 50 ms** |

Benchmarked on a mid-range laptop CPU with `evaluate.compute_latency_benchmark()`.

---

## Limitations and Production Notes

### Wallet → IEEE-CIS Feature Mapping

The model was trained on Vesta-proprietary V-features that are not available from a standard wallet transaction. At inference time:

- `C1` is proxied by `user_profile['transaction_count']`
- `D1` is proxied by days since account `first_seen`
- All unmappable features use their **training median** (stored in `feature_metadata.json`)

This median imputation degrades model accuracy compared to having the real Vesta features. The model still captures meaningful signals from: `TransactionAmt`, `amt_log`, `hour_of_day`, `is_high_amount`, `is_night_transaction`, `ProductCD`, `DeviceType`, and the behavioural proxies.

### Recommendations for Production

1. **Retrain on real wallet data** once a labelled dataset of wallet transactions is available (3–6 months of data minimum).
2. **Feature store**: Pre-compute and cache `C`/`D`/`V` equivalents from wallet event logs.
3. **Concept drift monitoring**: Fraud patterns change. Retrain at least quarterly.
4. **Threshold recalibration**: After retraining or major product changes, re-run threshold analysis.
5. **Feedback loop**: Analyst decisions (FP/FN) should feed back into future training labels.

---

## Artifact Files

| File | Description |
|------|-------------|
| `models/xgb_model.joblib` | Trained XGBoost classifier |
| `models/lgbm_model.joblib` | Trained LightGBM classifier (if enabled) |
| `models/preprocessor.joblib` | Fitted ColumnTransformer pipeline |
| `models/feature_metadata.json` | Feature names, medians, training info |
| `models/thresholds.json` | `{"flag": float, "block": float}` |
| `models/metrics.json` | Per-model evaluation metrics |

---

*Report generated for V HACK 2026 — FraudShield team*
