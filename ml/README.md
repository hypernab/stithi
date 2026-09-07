# STITHI IMU Activity Recognition — Baseline Model

## ⚠️ CRITICAL DISCLAIMER

**This is an activity-recognition baseline. It is not a validated fall detector or fall-risk predictor.**

It is **NOT** a fall detection system.
It is **NOT** a fall risk prediction model.
It is **NOT** a validated medical device.

This is a research-stage baseline that classifies six common activities from 6-axis IMU data. Any future work on fall prediction or fall risk assessment will be developed in a separate, clearly labeled module.

---

## What This Model Does

This is a **Random Forest classifier** that recognizes six activity types from wearable 6-axis IMU (accelerometer + gyroscope) data:

1. **Walking Upstairs** — Ascending stairs or inclines
2. **Walking Downstairs** — Descending stairs or inclines
3. **Walking Flat** — Level ground walking
4. **Sitting** — Seated posture
5. **Standing** — Upright static posture
6. **Jogging** — Running or jogging

The model operates on **2-second windows** of IMU data sampled at **10 Hz**, extracting 26 time-domain features per window.

---

## Dataset

**Source:** `datasets/IMU-based Human Activity Recognition Dataset.csv`

**Composition:**
- **15,980 raw samples** collected across six activity types
- **6-DOF measurements:** 3-axis accelerometer + 3-axis gyroscope
- **Sampling rate:** ~10 Hz (100 ms between samples)
- **Format:** CSV with columns `ax, ay, az, gx, gy, gz, activity`

**No additional preprocessing or filtering applied.**

---

## Windowing Strategy

### Window Configuration
- **Window size:** 20 samples = ~2 seconds (at 10 Hz)
- **Overlap:** 50% (step size = 10 samples)
- **Total windows created:** ~100 windows per 1000 raw samples

### Critical Design: No Data Leakage

Overlapping windows are highly correlated. Creating windows and then randomly shuffling them before train/test split would cause **severe data leakage**.

**Solution implemented:**
1. Group the dataset by activity ID to respect segment boundaries
2. Within each activity segment, create overlapping windows
3. **Critically: Never create a window that crosses from one activity class to another**
4. Split windows chronologically within each activity segment (first 80% → training, last 20% → test)
5. **No random shuffling before split**

The split also purges the final training window at each segment boundary, so
no overlapping raw sample interval can appear in both sets.

---

## Features

### 26 Time-Domain Features Extracted Per 2-Second Window

#### Acceleration (Ax, Ay, Az)
- Mean of each axis (3 features)
- Standard deviation of each axis (3 features)
- Min and max of each axis (6 features)
- **Total: 12 features**

#### Acceleration Magnitude
- Mean magnitude (1 feature)
- Standard deviation of magnitude (1 feature)
- Maximum magnitude (1 feature)
- **Total: 3 features**

#### Gyroscope (Gx, Gy, Gz)
- Mean of each axis (3 features)
- Standard deviation of each axis (3 features)
- **Total: 6 features**

#### Gyroscope Magnitude
- Mean magnitude (1 feature)
- Standard deviation of magnitude (1 feature)
- Maximum magnitude (1 feature)
- **Total: 3 features**

#### Motion (Jerk)
- Mean acceleration jerk (rate of change of acceleration) (1 feature)
- Standard deviation of jerk (1 feature)
- **Total: 2 features**

**Grand Total: 26 features**

### Feature Extraction Code

Features are extracted identically in both training and inference:
- **Training:** `ml/train_baseline.py` — `extract_features(window)`
- **Inference:** `ml/inference.py` — `extract_features(window)`

Both use the exact same algorithm to ensure consistency.

---

## Model

### Architecture
- **Type:** RandomForestClassifier (scikit-learn)
- **Ensemble size:** 100 decision trees
- **Max depth:** Unlimited (default)
- **Min samples split:** 5
- **Min samples leaf:** 2
- **Random seed:** 42 (reproducible)

### Rationale
1. **Random Forest** provides feature importance insights
2. **Simple baseline** — no hyperparameter tuning yet
3. **Reproducible** — fixed random seed and explicit hyperparameters
4. **Interpretable** — can analyze which features matter

---

## Training Pipeline

### Step-by-Step

1. **Load raw dataset** (15,980 samples)
2. **Create non-leaking windows** → ~2000 windows total
3. **Extract 26 features per window** → feature matrix (2000 × 26)
4. **Fit StandardScaler** on training features only
5. **Scale both training and test** with fitted scaler
6. **Train RandomForest** on 80% of windows (~1600)
7. **Evaluate on test set** (20% of windows, ~400)
8. **Save model**, scaler, and metrics

### Code

```bash
cd d:\Stithi
python ml/train_baseline.py
```

### Outputs

All artifacts saved to `models/`:

| File | Purpose |
|------|---------|
| `stithi_activity_rf.pkl` | Trained Random Forest model (joblib) |
| `stithi_activity_scaler.pkl` | StandardScaler fitted on training features |
| `stithi_activity_metrics.json` | Performance metrics and metadata |
| `stithi_activity_confusion_matrix.png` | Confusion matrix heatmap (test set) |
| `stithi_activity_feature_importance.png` | Top 15 feature importances |

---

## Performance Metrics

Performance is evaluated on the held-out test set (final 20% of windows within each activity segment, chronologically).

### Expected Accuracy Range

- **Training accuracy:** Typically 0.95–0.99 (overfitting is expected due to decision tree flexibility)
- **Test accuracy:** Typically 0.85–0.95 (realistic generalization)

Actual results are saved in `models/stithi_activity_metrics.json`.

### No Hyperparameter Tuning

This is a **reproducible baseline**, not an optimized model. Future work may include:
- Cross-validation
- Hyperparameter search
- Ensemble methods (boosting, stacking)
- Deep learning baselines

---

## Inference Engine

### Usage

```python
from ml.inference import ActivityRecognitionEngine

engine = ActivityRecognitionEngine()

# Push samples one at a time
for ax, ay, az, gx, gy, gz in imu_stream:
    result = engine.push_sample(ax, ay, az, gx, gy, gz)
    
    if result is not None:
        print(result["activity"])           # e.g., "Walking Flat"
        print(result["confidence"])         # e.g., 0.87
        print(result["probabilities"])      # dict of all class probs
```

### Output Format

When window is full (20 samples), the engine returns:

```python
{
    "activity": "Walking Flat",           # Predicted activity name
    "activity_id": 3,                     # Class ID (1-6)
    "confidence": 0.87,                   # Probability of predicted class
    "probabilities": {
        "Walking Upstairs": 0.02,
        "Walking Downstairs": 0.01,
        "Walking Flat": 0.87,
        "Sitting": 0.05,
        "Standing": 0.04,
        "Jogging": 0.01
    }
}
```

### Key Design Decisions

- **No "stability score"** — This is not a stability metric
- **No "fall risk"** — This is not a risk prediction system
- **No threshold-based alerts** — Pure classification output
- **Rolling buffer** — Maintains a 20-sample sliding window internally
- **Stateful** — Must push samples sequentially in order

### Runtime

- Inference latency: ~1–2 ms per prediction (on modern CPU)
- Memory footprint: ~10 MB for model + scaler

---

## Limitations

### Scope

- **Activity recognition only** — Does not predict or detect falls
- **Six activity classes** — Limited to the training dataset
- **Offline training** — Model is pre-trained, not adaptive
- **10 Hz sampling assumption** — Windows assume 100 ms between samples
- **No contextual information** — Uses IMU data only

### Dataset Limitations

- **Controlled environment** — Likely collected in lab or controlled setting
- **Healthy participants** — May not generalize to elderly or clinical populations
- **Activity mix unbalanced** — Some activities have more samples than others
- **No real fall data** — Cannot predict actual falls without fall events in training set

### Feature Limitations

- **No orientation estimates** — Features labeled as "IMU-derived proxies only"
- **No gravity removal** — Raw accelerometer includes gravitational component
- **No sensor fusion** — No magnetometer or barometer data
- **No environmental context** — Cannot distinguish indoor/outdoor, on carpet/hard floor, etc.

---

## Getting Started

### Installation

```bash
cd d:\Stithi

# Create or activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r ml/requirements.txt
```

### Training

```bash
python ml/train_baseline.py
```

Output: Model, scaler, metrics, and plots in `models/`.

### Testing Inference

```bash
python ml/inference.py
```

This runs a demo with synthetic IMU data and prints predictions.

### Integration with Backend

Currently, the backend server (`backend/server.py`) collects IMU data but does not run inference. Integration is planned for Phase 2.

---

## File Structure

```
ml/
├── train_baseline.py          # Training script (run to train model)
├── inference.py               # Inference engine (import to use model)
├── requirements.txt           # Python dependencies
└── README.md                  # This file

models/
├── stithi_activity_rf.pkl                # Trained model
├── stithi_activity_scaler.pkl            # Feature scaler
├── stithi_activity_metrics.json          # Metrics and metadata
├── stithi_activity_confusion_matrix.png  # Confusion matrix plot
└── stithi_activity_feature_importance.png # Feature importance plot
```

---

## Future Work

## Clinical Baseline: Historical Fall-Status Classification

This is an **activity-independent gait association baseline** using only
`datasets/Database_register.csv`, identified in this project as **GSTRIDE:
Gait, Frailty & Cognitive Data from 163 Older Adults**.

The task is **historical fall-status classification**: classifying whether a
participant reported a fall during the last year before the test. This is a
separate gait-association experiment, not the Model 1 activity-recognition
pipeline. This is an activity-recognition baseline. It is not a validated fall
detector or fall-risk predictor. It does not predict future falls.

### Clinical Baseline Inputs

Only these 11 gait summary features are used:

- Gait speed
- Stride time average and standard deviation
- Cadence average and standard deviation
- Stride length average and standard deviation
- Swing average and standard deviation
- Clearance average and standard deviation

Identifiers, notes, dates, device identifiers, the fall-history target, and all
other clinical and demographic columns are excluded.

### Missing and Invalid Values

The semicolon-delimited, Windows-encoded file is parsed with `,` as the decimal
separator. Blank values, `-`, and `Incapable` are treated as missing. Other
non-numeric feature values are coerced to missing and their counts are saved in
`models/stithi_gait_metrics.json`. Median imputation and standardization are
performed inside each cross-validation training fold through a scikit-learn
Pipeline. In the current dataset, the selected gait features contain no
missing or invalid values after parsing.

### Evaluation and Artifacts

`ml/train_clinical_baseline.py` uses 5-fold stratified cross-validation with a
modest fixed Random Forest. Accuracy, balanced accuracy, ROC-AUC, precision,
recall, F1, and out-of-fold confusion matrix are saved to
`models/stithi_gait_metrics.json`. The final model and preprocessing artifact
are saved separately as:

- `models/stithi_gait_rf.pkl`
- `models/stithi_gait_preprocessing.pkl`
- `models/stithi_gait_feature_importance.png`

Run it with:

```bash
python ml/train_clinical_baseline.py
```

This baseline is a research analysis of association with reported historical
status. It must not be interpreted as a medical prediction or detection
system.

### Not Included in This Baseline

- **Fall detection** — Requires labeled fall events
- **Fall risk prediction** — Requires clinical outcomes
- **User-specific adaptation** — Requires per-user calibration
- **Real-time deployment to M5** — Requires C++ model conversion
- **Transfer learning** — Can use this baseline as feature extractor
- **Deep learning** — CNN/LSTM baselines for comparison

### Next Phase (Phase 2)

- [ ] Integrate inference engine with backend server
- [ ] Collect real sensor data from M5StickCPlus2
- [ ] Validate model on collected data
- [ ] Explore hyperparameter tuning
- [ ] Investigate misclassified activities

### Phase 3+ (Future Research)

- [ ] Collect fall event data (in controlled environment)
- [ ] Design fall detection module (separate from activity recognition)
- [ ] Validate on elderly population
- [ ] Deploy to device (M5StickCPlus2)
- [ ] Clinical pilot study

---

## References & Notes

### Windowing & Data Leakage

- Overlapping windows in train/test must be handled carefully
- Random splitting of correlated windows causes severe leakage
- Solution: Chronological split respects temporal dependency

### Feature Engineering

- Time-domain features are interpretable and fast
- No frequency-domain features (FFT) in this baseline
- Can add wavelet or other advanced features in future work

### Reproducibility

- All random seeds fixed (seed=42)
- Dataset path and column names clearly specified
- Model parameters explicitly documented
- Feature extraction identical in train and inference

---

## Questions?

Refer to code comments in:
- `ml/train_baseline.py` — Training logic and windowing
- `ml/inference.py` — Inference engine and feature extraction
- `models/stithi_activity_metrics.json` — Detailed metrics and metadata

---

**Last Updated:** 2026-09-01  
**Status:** Phase 1 — Reproducible Baseline (Complete)  
**Next:** Phase 2 — Backend Integration
