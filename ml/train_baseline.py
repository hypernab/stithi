"""
STITHI IMU Activity Recognition Baseline Training

This script trains a Random Forest classifier to recognize six activity types
from 6-axis IMU data collected at approximately 10 Hz.

IMPORTANT: This is NOT a fall detection or fall risk prediction model.
This model performs activity recognition only.

Key Design Decisions:
- Windowing: 20 samples per window (~2 seconds at 10 Hz) with 50% overlap
- No Data Leakage: Windows are created within activity segments only.
  Overlapping windows never cross activity boundaries.
- Train/Test Split: Chronological 80/20 split within each contiguous activity
    segment. Windows are never shuffled or shared between train and test.
- Features: 26 IMU-derived features computed per window
- Model: Random Forest (100 trees, reproducible random_state=42)

Dataset:
  datasets/IMU-based Human Activity Recognition Dataset.csv
  Expected columns: ax, ay, az, gx, gy, gz, activity

Activities:
  1 = Walking Upstairs
  2 = Walking Downstairs
  3 = Walking Flat
  4 = Sitting
  5 = Standing
  6 = Jogging
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib
import json
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 20  # samples, ~2 seconds at 10 Hz
WINDOW_OVERLAP = 0.5  # 50% overlap
SAMPLING_RATE_HZ = 10

TRAIN_RATIO = 0.8  # 80% for training

RANDOM_SEED = 42

RF_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

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

def extract_features(window):
    """
    Extract 26 IMU features from a 20-sample window.
    
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
# WINDOWING WITH NO LEAKAGE
# ============================================================

def create_non_leaking_windows(data):
    """
    Create overlapping windows within activity segments only.
    
    CRITICAL: Never create a window that crosses from one activity to another.
    
    Args:
        data: pd.DataFrame with columns [ax, ay, az, gx, gy, gz, activity]
    
    Returns:
        (windows, labels, window_ids): 
        - windows: np.ndarray of shape (N, 20, 6)
        - labels: np.ndarray of activity IDs
        - window_ids: (segment ID, start, end) identifier for each window
    """
    windows = []
    labels = []
    window_ids = []
    
    # Detect contiguous runs so repeated activity labels remain separate.
    segment_ids = data["activity"].ne(data["activity"].shift()).cumsum()
    
    # Create windows within each contiguous activity segment only.
    for segment_id, group_data in data.groupby(segment_ids, sort=False):
        activity_id = group_data["activity"].iloc[0]
        values = group_data[["ax", "ay", "az", "gx", "gy", "gz"]].values

        n_samples = len(values)
        step = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))  # Step size for 50% overlap
        
        for start_idx in range(0, n_samples - WINDOW_SIZE + 1, step):
            end_idx = start_idx + WINDOW_SIZE
            
            window = values[start_idx:end_idx]
            windows.append(window)
            labels.append(activity_id)
            window_ids.append((int(segment_id), start_idx, end_idx))
    
    return np.array(windows), np.array(labels), window_ids


def chronological_split(window_ids):
    """Return chronological train/test positions with a boundary purge."""
    train_positions = []
    test_positions = []
    segment_positions = {}

    for position, (segment_id, _, _) in enumerate(window_ids):
        segment_positions.setdefault(segment_id, []).append(position)

    for positions in segment_positions.values():
        split_position = int(len(positions) * TRAIN_RATIO)
        split_position = max(1, min(split_position, len(positions) - 1))
        # Remove the final training window so it cannot overlap the first
        # testing window when adjacent windows share 50% of their samples.
        train_positions.extend(positions[:split_position - 1])
        test_positions.extend(positions[split_position:])

    return np.array(train_positions), np.array(test_positions)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("STITHI IMU ACTIVITY RECOGNITION BASELINE TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/8] Loading dataset...")
    dataset_path = Path(__file__).parent.parent / "datasets" / "IMU-based Human Activity Recignition Dataset.csv"
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    data = pd.read_csv(dataset_path)
    print(f"  ✓ Loaded {len(data)} raw samples")
    print(f"  ✓ Columns: {list(data.columns)}")
    print(f"  ✓ Activities: {sorted(data['activity'].unique())}")
    
    # Create windows
    print("\n[2/8] Creating non-leaking windows...")
    windows, labels, window_ids = create_non_leaking_windows(data)
    print(f"  ✓ Created {len(windows)} windows")
    print(f"  ✓ Window size: {WINDOW_SIZE} samples (~{WINDOW_SIZE / SAMPLING_RATE_HZ:.1f} seconds)")
    print(f"  ✓ Activity distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for activity_id, count in zip(unique_labels, counts):
        print(f"    - {ACTIVITY_NAMES[activity_id]}: {count} windows")
    
    # Extract features
    print("\n[3/8] Extracting features...")
    features = np.array([extract_features(w) for w in windows], dtype=np.float32)
    print(f"  ✓ Extracted {features.shape[1]} features per window")
    print(f"  ✓ Feature matrix shape: {features.shape}")
    
    # Train/test split (chronological within each contiguous segment)
    print("\n[4/8] Splitting train/test (chronological within segments)...")
    train_positions, test_positions = chronological_split(window_ids)
    train_ids = {window_ids[position] for position in train_positions}
    test_ids = {window_ids[position] for position in test_positions}
    shared_ids = train_ids.intersection(test_ids)
    if shared_ids:
        raise RuntimeError(
            f"Window leakage detected: {len(shared_ids)} identifiers appear in both sets"
        )
    train_intervals = {
        (segment_id, sample_start, sample_end)
        for segment_id, sample_start, sample_end in train_ids
    }
    test_intervals = {
        (segment_id, sample_start, sample_end)
        for segment_id, sample_start, sample_end in test_ids
    }
    for train_segment, train_start, train_end in train_intervals:
        for test_segment, test_start, test_end in test_intervals:
            if train_segment == test_segment and train_start < test_end and test_start < train_end:
                raise RuntimeError("Raw sample overlap detected between train and test windows")
    print("  ✓ Leakage check passed: identifiers and raw sample intervals are disjoint")

    X_train, X_test = features[train_positions], features[test_positions]
    y_train, y_test = labels[train_positions], labels[test_positions]
    
    print(f"  ✓ Training windows: {len(X_train)}")
    print(f"  ✓ Testing windows: {len(X_test)}")
    print(f"  ✓ Split methodology: Chronological 80/20 within activity segments")
    print(f"  ✓ Boundary purge: one overlapping training window removed per segment")
    print(f"  ✓ Windows shuffled: False")
    print(f"  ℹ Train activity distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for activity_id, count in zip(unique_train, counts_train):
        print(f"    - {ACTIVITY_NAMES[activity_id]}: {count} windows")
    print(f"  ℹ Test activity distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for activity_id, count in zip(unique_test, counts_test):
        print(f"    - {ACTIVITY_NAMES[activity_id]}: {count} windows")
    
    # Feature scaling
    print("\n[5/8] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  ✓ Scaler fitted on training data")
    print(f"  ✓ Feature means (first 5): {scaler.mean_[:5]}")
    print(f"  ✓ Feature stds (first 5): {scaler.scale_[:5]}")
    
    # Train model
    print("\n[6/8] Training Random Forest...")
    print(f"  ✓ Model parameters: {RF_PARAMS}")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_scaled, y_train)
    print(f"  ✓ Training complete")
    
    # Evaluate
    print("\n[7/8] Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"  ✓ Training Accuracy: {train_acc:.4f}")
    print(f"  ✓ Testing Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\n  Classification Report (Test Set):")
    print("  " + "-" * 70)
    class_report = classification_report(
        y_test, y_pred_test,
        target_names=[ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES.keys())],
        digits=4
    )
    for line in class_report.split("\n"):
        print(f"  {line}")
    
    # Save model and scaler
    print("\n[8/8] Saving artifacts...")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "stithi_activity_rf.pkl"
    scaler_path = models_dir / "stithi_activity_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Model saved to {model_path}")
    print(f"  ✓ Scaler saved to {scaler_path}")
    
    # Save metrics
    metrics = {
        "model_name": "STITHI IMU Activity Recognition Baseline",
        "model_type": "RandomForestClassifier",
        "dataset": "IMU-based Human Activity Recognition Dataset.csv",
        "n_raw_samples": len(data),
        "n_windows": len(windows),
        "window_size": WINDOW_SIZE,
        "window_overlap": WINDOW_OVERLAP,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "n_features": features.shape[1],
        "feature_names": [
            "ax_mean", "ay_mean", "az_mean",
            "ax_std", "ay_std", "az_std",
            "ax_min", "ax_max", "ay_min", "ay_max", "az_min", "az_max",
            "acc_mag_mean", "acc_mag_std", "acc_mag_max",
            "gx_mean", "gy_mean", "gz_mean",
            "gx_std", "gy_std", "gz_std",
            "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max",
            "jerk_mean", "jerk_std"
        ],
        "n_classes": len(ACTIVITY_NAMES),
        "class_names": ACTIVITY_NAMES,
        "n_train_windows": len(X_train),
        "n_test_windows": len(X_test),
        "train_test_split": TRAIN_RATIO,
        "split_method": "chronological 80/20 within activity segments",
        "shuffle": False,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "random_seed": RANDOM_SEED,
        "model_parameters": RF_PARAMS,
        "classification_report": class_report
    }
    
    metrics_path = models_dir / "stithi_activity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")
    
    # Plot confusion matrix
    print("\n  Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred_test)
    activity_names_list = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES.keys())]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(activity_names_list)), activity_names_list, rotation=45, ha="right")
    plt.yticks(range(len(activity_names_list)), activity_names_list)
    threshold = cm.max() / 2.0
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            plt.text(
                column_index,
                row_index,
                cm[row_index, column_index],
                ha="center",
                va="center",
                color="white" if cm[row_index, column_index] > threshold else "black"
            )
    plt.title("STITHI Activity Recognition - Confusion Matrix (Test Set)")
    plt.ylabel("True Activity")
    plt.xlabel("Predicted Activity")
    plt.tight_layout()
    
    cm_path = models_dir / "stithi_activity_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Confusion matrix saved to {cm_path}")
    
    # Plot feature importance
    print("  Generating feature importance plot...")
    feature_names = metrics["feature_names"]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    plt.figure(figsize=(12, 6))
    plt.title("STITHI Activity Recognition - Top 15 Feature Importances")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    
    fi_path = models_dir / "stithi_activity_feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Feature importance saved to {fi_path}")
    
    plt.close("all")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"\nArtifacts:")
    print(f"  - {model_path}")
    print(f"  - {scaler_path}")
    print(f"  - {metrics_path}")
    print(f"  - {cm_path}")
    print(f"  - {fi_path}")
    print("\n⚠ IMPORTANT: This model performs activity recognition only.")
    print("It is NOT a fall detection or fall risk prediction model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
