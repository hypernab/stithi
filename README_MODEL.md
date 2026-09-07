# Legacy model notes

> This document contains historical notes and claims from an earlier prototype. It is not the current system specification. The current architecture and truthful scope are documented in [README.md](README.md): Model 1 performs activity recognition only; Model 2 is a separate historical-gait research layer; and the current laptop-side indicators do not calculate clinical fall risk or actual LOG/BOS.

# STITHI Baseline Fall Detection Model - Complete Guide

## 🎯 Project Overview

**STITHI** (Seniors' Technical Health & Injury Prevention Initiative) is a wearable-based fall prevention system for elderly people using the M5StickCPlus2 IMU sensor and machine learning.

**Problem**: One-third of community-dwelling adults aged 65+ experience falls annually, leading to injury, disability, and psychological trauma.

**Solution**: Real-time fall risk detection via wearable technology that provides tactile/visual/auditory feedback to users during high-risk activities, enabling preventive action.

---

## ✅ Training Results Summary

### Model Performance
- **Training Accuracy**: 97.97%
- **Testing Accuracy**: 95.40%
- **Model Type**: Random Forest Classifier (100 trees, max_depth=15)
- **Training Samples**: 12,784 (80% of 15,980 total)
- **Test Samples**: 3,196 (20% of total)

### Per-Activity Performance (Test Set)

| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Activity 1 | 0.94 | 0.94 | 0.94 | 417 |
| Activity 2 | 0.91 | 0.92 | 0.92 | 407 |
| Activity 3 | 0.89 | 0.95 | 0.92 | 572 |
| Activity 4 | 1.00 | 0.98 | 0.99 | 724 |
| Activity 5 | 0.98 | 0.97 | 0.97 | 517 |
| Activity 6 | 0.99 | 0.94 | 0.97 | 559 |

### Top Features for Fall Detection

1. **Acceleration Magnitude Std** (9.7%) - Variability in movement intensity
2. **Accelerometer Y Std** (5.9%) - Lateral movement variability  
3. **Gyro Magnitude Max** (5.4%) - Peak rotation rate (critical for detecting loss of balance)
4. **Vertical Acceleration Change** (5.4%) - Rapid vertical motion (key for fall detection)
5. **Roll Angle** (5.3%) - Left/right body lean
6. **Accelerometer X Mean** (5.0%) - Average forward/backward motion

**Key Insight**: The model detects falls by monitoring Line of Gravity (LOG) vs Base of Support (BOS) changes through:
- Pitch and roll angles (body lean)
- Acceleration magnitude (movement intensity)
- Vertical acceleration changes (loss of support)
- Rotation rates (inability to recover balance)

---

## 📁 Project Structure

```
d:\Stithi/
├── src/
│   ├── main.cpp              # M5StickCPlus2 firmware
│   └── model_integration.py   # Integration guide for C++
├── ml/
│   ├── train_baseline.py      # Training script (COMPLETED ✓)
│   ├── inference.py           # Real-time inference module
│   └── [convert_to_c.py]      # TODO: Convert model to C/C++
├── backend/
│   └── server.py              # FastAPI server for data collection
├── datasets/
│   ├── Database_register.csv  # Clinical fall data
│   └── IMU-based Human Activity Recognition Dataset.csv  # Training data
├── data/                       # Collected IMU data
├── models/                     # Trained models
│   ├── baseline_model.pkl      # Random Forest model ✓
│   ├── scaler.pkl              # Feature scaler ✓
│   └── baseline_model_analysis.png  # Performance plots ✓
└── platformio.ini              # PlatformIO configuration
```

---

## 🚀 Quick Start

### Step 1: Train the Model (Already Done ✓)
```bash
python ml/train_baseline.py
```
This generates:
- `models/baseline_model.pkl` - Trained Random Forest
- `models/scaler.pkl` - Feature scaler
- `models/baseline_model_analysis.png` - Performance visualization

### Step 2: Test Inference on Desktop
```bash
python ml/inference.py
```
Tests the model with simulated IMU data and prints predictions.

### Step 3: Integrate with Backend Server
```bash
python backend/server.py
```
Starts FastAPI server to receive IMU data from M5StickCPlus2 and store it.

### Step 4: Deploy to M5StickCPlus2 (Next Steps)
See [Model Deployment to Device](#model-deployment-to-device) below.

---

## 📊 Feature Engineering

The baseline model extracts **29 features** from raw IMU data:

### Accelerometer Features (9)
- Mean and Std: `ax_mean, ay_mean, az_mean, ax_std, ay_std, az_std`
- Magnitude: `acc_magnitude, acc_mag_mean, acc_mag_std, acc_mag_max`

### Gyroscope Features (9)
- Mean and Std: `gx_mean, gy_mean, gz_mean, gx_std, gy_std, gz_std`
- Magnitude: `gyro_mag_mean, gyro_mag_std, gyro_mag_max`

### Body Orientation (6)
- Pitch (forward/backward): `pitch, pitch_mean, pitch_std, pitch_max_deviation`
- Roll (left/right): `roll, roll_mean, roll_std`

### Vertical Motion (2)
- Z-axis mean absolute value, rapid change detection

**Window Size**: 10 samples (100ms at 10 Hz sampling rate)

---

## 🧠 Model Architecture

**Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Max Depth**: 15 levels
- **Min Samples Split**: 10
- **Min Samples Leaf**: 5
- **Class Weights**: Balanced (handles imbalanced activities)

**Why Random Forest?**
1. Naturally handles non-linear relationships in IMU data
2. Feature importance reveals which movements are fall-indicative
3. Fast inference (~1-5ms per prediction)
4. Robust to noisy sensor data
5. No scaling needed for predictions (though we scale for better convergence)

---

## 🔄 Model Deployment to Device

### Option A: C/C++ Model Export (Recommended for M5StickCPlus2)

Create `ml/convert_to_c.py`:
```python
import joblib
import json
from pathlib import Path

# Load model
model = joblib.load("models/baseline_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Extract tree structure
def export_tree_to_c(tree, tree_index):
    # Generate C code for decision tree
    # Each node becomes an if-else statement
    pass

# Export forest
def export_forest_to_c(model, output_file):
    # Combine all trees into single C function
    # Minimal memory footprint (~50-100 KB)
    pass

export_forest_to_c(model, "include/model_coefficients.h")
```

Then in `main.cpp`:
```cpp
#include "model_coefficients.h"

int predict_activity(float* features) {
    // Compiled tree decision logic
    return model_predict(features);
}
```

### Option B: TensorFlow Lite Model (For future optimization)
- Convert RandomForest to TFLite format
- Deploy on device with ~5-10ms latency
- Requires TFLite runtime (~200KB)

### Option C: Python MicroPython (Simplest but slower)
- Port `inference.py` to MicroPython
- Requires more memory (device has ~320KB RAM)
- Latency: ~50-100ms per prediction

**Recommended**: Option A (C/C++ tree export) - fast, lightweight, accurate

---

## 🎯 Next Steps for Improvement

### 1. **Collect More Data**
   - Current: 15,980 samples from 6 activities
   - Target: 100,000+ samples with explicit "fall" events
   - Add: User age, weight, height, health conditions

### 2. **Binary Fall Classification**
   - Current: 6-class activity classification
   - Better: Binary (fall/non-fall) with confidence scoring
   - Requires: Labeled fall event data

### 3. **Real-time Calibration**
   - Baseline model assumes standard device orientation
   - Add calibration routine to account for individual wearing patterns
   - Allow thresholds to adapt per user

### 4. **Temporal Analysis**
   - Current: Per-window predictions
   - Better: Track prediction history for 1-2 seconds
   - Detect sustained "loss of balance" state

### 5. **Multi-modal Feedback**
   - Visual: LED color changes (green → yellow → red)
   - Tactile: Vibration alerts on high-risk detection
   - Audio: Gentle alert sound if available

### 6. **Cloud Integration**
   - Upload high-risk events to backend
   - Enable emergency contact notifications
   - Track fall statistics per user

---

## 📝 Usage Examples

### Python Desktop Inference
```python
from ml.inference import FallDetectionModel

# Load model
model = FallDetectionModel(window_size=10)

# Simulate IMU stream
for i in range(100):
    ax, ay, az = get_accelerometer_data()
    gx, gy, gz = get_gyroscope_data()
    
    result = model.add_imu_sample(ax, ay, az, gx, gy, gz)
    
    if result:
        print(f"Activity: {result['activity_name']}")
        print(f"Risk: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
```

### Backend Data Collection
```python
# backend/server.py handles POST requests from M5StickCPlus2
# Example:
curl -X POST http://localhost:8000/imu \
  -H "Content-Type: application/json" \
  -d '{"ax": -1248, "ay": 14736, "az": -6780, "gx": 1057, "gy": 1422, "gz": 2038}'
```

---

## 📚 References

### Fall Prevention Programs Referenced
- **Otago Exercise Program**: 35% fall reduction
- **Tai Ji Quan: Moving for Better Balance**: 55% fall reduction
- **Short Physical Performance Battery (SPPB)**: Balance, gait, strength assessment

### IMU-Based Fall Detection Research
- Combines pitch/roll angles with acceleration magnitude
- Key metric: Line of Gravity (LOG) outside Base of Support (BOS)
- Typical detection latency: 200-500ms from fall initiation

### Wearable Device Capabilities (M5StickCPlus2)
- **Sampling Rate**: Up to 100 Hz (currently 10 Hz)
- **Axes**: 6-DOF (3 accelerometer + 3 gyroscope)
- **Range**: ±16g acceleration, ±2000°/s rotation
- **Power**: Battery life ~10 hours at 10 Hz

---

## 🐛 Troubleshooting

### Model Not Training
**Error**: `FileNotFoundError: IMU dataset not found`
- Ensure `datasets/IMU-based Human Activity Recignition Dataset.csv` exists
- Check file path spelling (note: typo in original filename "Recignition")

### Inference Accuracy Low on Device
- Verify M5StickCPlus2 accelerometer is calibrated
- Check sampling rate matches training (10 Hz)
- Ensure device is worn consistently (same position/orientation)
- Collect user-specific calibration data

### Backend Not Receiving Data
- Verify WiFi connection: `http://172.16.169.86:8000/imu`
- Check firewall allows port 8000
- Monitor `data/` folder for CSV files

---

## 📞 Contact & Support

For questions about:
- **Model Performance**: Review `models/baseline_model_analysis.png`
- **Integration**: See `src/model_integration.py`
- **Real-World Deployment**: Collect more fall event data and retrain

---

## 🏆 Key Achievements (Hackathon)

✅ Trained 95.4% accurate activity recognition model  
✅ Identified fall-risk features (acceleration variance, body angles)  
✅ Created inference engine for real-time predictions  
✅ Designed backend data collection pipeline  
✅ Documented deployment strategy for embedded devices  

**Ready for Phase 2**: Device integration, real-world testing, and user feedback collection.
