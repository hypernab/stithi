"""
STITHI IMU Activity Recognition Inference Engine

This module provides real-time activity classification from 6-axis IMU data.

Usage:
    engine = ActivityRecognitionEngine()
    
    # Add raw samples one at a time
    for ax, ay, az, gx, gy, gz in imu_stream:
        result = engine.push_sample(ax, ay, az, gx, gy, gz)
        
        if result is not None:
            print(f"Activity: {result['activity']}")
            print(f"Confidence: {result['confidence']:.3f}")

IMPORTANT: This engine performs activity recognition only.
It is NOT a fall detection or fall risk prediction system.
"""

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import joblib
try:
    from ml.activity_features import extract_features
except ModuleNotFoundError:  # Supports direct `python ml/inference.py` usage.
    from activity_features import extract_features

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "stithi_activity_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "stithi_activity_scaler.pkl"

WINDOW_SIZE = 20  # samples, ~2 seconds at 10 Hz
SAMPLING_RATE_HZ = 10

ACTIVITY_NAMES = {
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Walking Flat",
    4: "Sitting",
    5: "Standing",
    6: "Jogging"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_legacy_features(window):
    """
    Legacy 26-feature extractor retained only to open older model artifacts.
    
    Args:
        window: np.ndarray of shape (20, 6) with columns [ax, ay, az, gx, gy, gz]
    
    Returns:
        np.ndarray of shape (26,) with extracted features
    """
    ax = window[:, 0]
    ay = window[:, 1]
    az = window[:, 2]
    gx = window[:, 3]
    gy = window[:, 4]
    gz = window[:, 5]
    
    features = []
    
    # Acceleration axis means and stds
    features.extend([ax.mean(), ay.mean(), az.mean()])
    features.extend([ax.std(), ay.std(), az.std()])
    
    # Acceleration axis min/max
    features.extend([ax.min(), ax.max(), ay.min(), ay.max(), az.min(), az.max()])
    
    # Acceleration magnitude
    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    features.extend([acc_mag.mean(), acc_mag.std(), acc_mag.max()])
    
    # Gyroscope axis means and stds
    features.extend([gx.mean(), gy.mean(), gz.mean()])
    features.extend([gx.std(), gy.std(), gz.std()])
    
    # Gyroscope magnitude
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
    features.extend([gyro_mag.mean(), gyro_mag.std(), gyro_mag.max()])
    
    # Acceleration jerk (rate of change of acceleration)
    ax_jerk = np.abs(np.diff(ax))
    ay_jerk = np.abs(np.diff(ay))
    az_jerk = np.abs(np.diff(az))
    features.append(np.mean(np.concatenate([ax_jerk, ay_jerk, az_jerk])))
    features.append(np.std(np.concatenate([ax_jerk, ay_jerk, az_jerk])))
    
    return np.array(features, dtype=np.float32)


# ============================================================
# INFERENCE ENGINE
# ============================================================

class ActivityRecognitionEngine:
    """
    Real-time activity recognition from streaming 6-axis IMU data.
    
    This engine maintains a sliding window of IMU samples and produces
    activity predictions whenever the window is full.
    
    The output includes the predicted activity class, confidence score,
    and probability distribution across all activity classes.
    
    NOTE: This performs activity recognition only. It does NOT predict
    fall risk, fall probability, or any other health outcome.
    """
    
    def __init__(self):
        """Initialize the inference engine and load model artifacts."""
        # Prefer the leakage-safe selected model; allow the previous artifact so
        # existing demos keep working until the retraining command is run once.
        if not MODEL_PATH.exists():
            legacy_path = BASE_DIR / "models" / "stithi_activity_rf.pkl"
            if legacy_path.exists():
                self.model_path = legacy_path
            else:
                raise FileNotFoundError(
                    f"Model not found at {MODEL_PATH}. Please run ml/train_baseline.py first."
                )
        else:
            self.model_path = MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Please run ml/train_baseline.py first."
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                f"Please run ml/train_baseline.py first."
            )
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(SCALER_PATH)
        
        # Sliding window buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.sample_count = 0
    
    def push_sample(self, ax: float, ay: float, az: float,
                   gx: float, gy: float, gz: float) -> Optional[Dict[str, Any]]:
        """
        Push a single 6-axis IMU sample into the sliding window.
        
        Returns a prediction when the window is full (20 samples),
        otherwise returns None.
        
        Args:
            ax, ay, az: Acceleration in X, Y, Z (m/s^2 or g, depending on sensor)
            gx, gy, gz: Angular velocity in X, Y, Z (deg/s or rad/s)
        
        Returns:
            Dict with keys:
                - activity: str, activity name
                - activity_id: int, activity class (1-6)
                - confidence: float, probability of predicted class
                - probabilities: dict, probability for each activity class
            
            Returns None if window is not yet full.
        """
        self.buffer.append([ax, ay, az, gx, gy, gz])
        self.sample_count += 1
        
        if len(self.buffer) == WINDOW_SIZE:
            return self._evaluate_window()
        
        return None
    
    def _evaluate_window(self) -> Dict[str, Any]:
        """
        Extract features and make a prediction on the current window.
        
        Returns:
            Dict with activity prediction and confidence
        """
        window = np.array(self.buffer, dtype=np.float32)
        
        # Extract features
        features = extract_features(window)
        
        # Scale features (must use same scaler as training)
        features_scaled = self.scaler.transform([features])
        
        # Predict
        activity_id = int(self.model.predict(features_scaled)[0])
        probabilities_array = self.model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        probability_by_id = {int(class_id): float(probability) for class_id, probability in zip(self.model.classes_, probabilities_array)}
        confidence = probability_by_id[activity_id]
        
        # Build probability dict
        probabilities = {
            ACTIVITY_NAMES[class_id]: probability_by_id.get(class_id, 0.0)
            for class_id in sorted(ACTIVITY_NAMES)
        }
        
        return {
            "activity": ACTIVITY_NAMES[activity_id],
            "activity_id": activity_id,
            "confidence": confidence,
            "probabilities": probabilities
        }
    
    def reset(self):
        """Clear the sample buffer."""
        self.buffer.clear()
        self.sample_count = 0


# ============================================================
# SIMPLE TEST/DEMO
# ============================================================

if __name__ == "__main__":
    print("STITHI Activity Recognition Inference Engine")
    print("=" * 70)
    
    try:
        engine = ActivityRecognitionEngine()
        print("✓ Model loaded successfully")
        print(f"  Model file: {MODEL_PATH}")
        print(f"  Scaler file: {SCALER_PATH}")
        print(f"  Window size: {WINDOW_SIZE} samples (~{WINDOW_SIZE/SAMPLING_RATE_HZ:.1f}s at {SAMPLING_RATE_HZ} Hz)")
        
        # Simulate a stream of random IMU data
        print("\n" + "=" * 70)
        print("DEMO: Simulating IMU data stream")
        print("=" * 70)
        
        np.random.seed(42)
        
        # Generate some synthetic activity data
        # (mostly sitting, but with some walking motion)
        for i in range(100):
            # Sitting activity pattern
            if i < 30:
                ax = np.random.normal(0, 0.5)
                ay = np.random.normal(0, 0.5)
                az = np.random.normal(9.8, 0.5)
                gx = np.random.normal(0, 10)
                gy = np.random.normal(0, 10)
                gz = np.random.normal(0, 10)
            # Walking flat pattern
            else:
                ax = np.random.normal(0.5, 1.0)
                ay = np.random.normal(0.5, 1.0)
                az = np.random.normal(8, 2.0)
                gx = np.random.normal(20, 15)
                gy = np.random.normal(20, 15)
                gz = np.random.normal(5, 10)
            
            result = engine.push_sample(ax, ay, az, gx, gy, gz)
            
            if result is not None:
                print(f"\n[Window {engine.sample_count // WINDOW_SIZE}]")
                print(f"  Predicted Activity: {result['activity']}")
                print(f"  Activity ID: {result['activity_id']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  All probabilities:")
                for activity, prob in result['probabilities'].items():
                    print(f"    - {activity}: {prob:.4f}")
        
        print("\n" + "=" * 70)
        print("✓ Inference engine working correctly")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run: python ml/train_baseline.py")
