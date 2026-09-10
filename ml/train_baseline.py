"""Leakage-safe training for STITHI's six-class IMU activity recognizer.

Activity recognition only: not fall prediction, prevention, or clinical validation.
"""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
try:  # Supports both `python ml/train_baseline.py` and `python -m ml.train_baseline`.
    from ml.activity_features import FEATURE_NAMES, REFERENCE_FEATURE_NAMES, extract_features, extract_reference_features
except ModuleNotFoundError:
    from activity_features import FEATURE_NAMES, REFERENCE_FEATURE_NAMES, extract_features, extract_reference_features

WINDOW_SIZE, WINDOW_OVERLAP, TRAIN_RATIO, RANDOM_SEED = 20, 0.5, 0.8, 42
ACTIVITY_NAMES = {1: "Walking Upstairs", 2: "Walking Downstairs", 3: "Walking Flat", 4: "Sitting", 5: "Standing", 6: "Jogging"}


def create_non_leaking_windows(data):
    """Create windows only within contiguous activity runs."""
    windows, labels, ids = [], [], []
    segments = data.activity.ne(data.activity.shift()).cumsum()
    for segment_id, group in data.groupby(segments, sort=False):
        values = group[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - WINDOW_SIZE + 1, int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))):
            windows.append(values[start:start + WINDOW_SIZE])
            labels.append(group.activity.iloc[0])
            ids.append((int(segment_id), start, start + WINDOW_SIZE))
    return np.asarray(windows), np.asarray(labels), ids


def purged_chronological_split(positions, ids, ratio):
    """Time-order split each run, removing the boundary window that would overlap."""
    grouped = {}
    for position in positions:
        grouped.setdefault(ids[position][0], []).append(position)
    left, right = [], []
    for values in grouped.values():
        split = max(1, min(int(len(values) * ratio), len(values) - 1))
        left.extend(values[:max(0, split - 1)])
        right.extend(values[split:])
    return np.asarray(left), np.asarray(right)


def assert_no_raw_overlap(left, right, ids):
    for a in left:
        sa, start_a, end_a = ids[a]
        for b in right:
            sb, start_b, end_b = ids[b]
            if sa == sb and start_a < end_b and start_b < end_a:
                raise RuntimeError("raw-sample overlap between partitions")


def candidates():
    return {
        "random_forest_26_feature_reference": ("reference", RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        "extra_trees_130_feature": ("expanded", ExtraTreesClassifier(n_estimators=200, max_features=0.8, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        "hist_gradient_boosting_130_feature": ("expanded", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_SEED)),
    }


def score(model, x_train, y_train, x_eval, y_eval):
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval)
    return {"accuracy": float(accuracy_score(y_eval, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y_eval, prediction)), "macro_f1": float(f1_score(y_eval, prediction, average="macro"))}, prediction


def main():
    root = Path(__file__).resolve().parent.parent
    data = pd.read_csv(root / "datasets" / "IMU-based Human Activity Recignition Dataset.csv")
    windows, labels, ids = create_non_leaking_windows(data)
    outer_train, test = purged_chronological_split(np.arange(len(windows)), ids, TRAIN_RATIO)
    fit, validation = purged_chronological_split(outer_train, ids, 0.75)
    assert_no_raw_overlap(outer_train, test, ids)
    assert_no_raw_overlap(fit, validation, ids)

    expanded = np.asarray([extract_features(window) for window in windows])
    # Kept solely for an apples-to-apples reference candidate.
    features = {"reference": np.asarray([extract_reference_features(window) for window in windows]), "expanded": expanded}
    validation_results = {}
    for name, (feature_set, model) in candidates().items():
        x = features[feature_set]
        scaler = StandardScaler().fit(x[fit])  # fit on the training partition only
        validation_results[name], _ = score(model, scaler.transform(x[fit]), labels[fit], scaler.transform(x[validation]), labels[validation])

    # The test partition is untouched until this validation-only selection is complete.
    # Accuracy is the primary requested comparison metric; balanced accuracy and
    # macro-F1 are retained to expose uneven class performance.
    selected = max(validation_results, key=lambda name: validation_results[name]["accuracy"])
    feature_set, model = candidates()[selected]
    x = features[feature_set]
    scaler = StandardScaler().fit(x[outer_train])
    final_metrics, predicted = score(model, scaler.transform(x[outer_train]), labels[outer_train], scaler.transform(x[test]), labels[test])
    report = classification_report(labels[test], predicted, labels=sorted(ACTIVITY_NAMES), target_names=[ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES)], output_dict=True, zero_division=0)

    models = root / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(model, models / "stithi_activity_model.pkl")
    joblib.dump(scaler, models / "stithi_activity_scaler.pkl")
    metrics = {
        "model_name": "STITHI IMU Activity Recognition", "model_type": type(model).__name__, "selected_candidate": selected,
        "selection_rule": "highest inner chronological validation accuracy; final test held out until selection",
        "n_raw_samples": int(len(data)), "n_windows": int(len(windows)), "window_size": WINDOW_SIZE, "window_overlap": WINDOW_OVERLAP,
        "n_features": int(x.shape[1]), "feature_names": FEATURE_NAMES if feature_set == "expanded" else REFERENCE_FEATURE_NAMES,
        "split_method": "purged chronological 80/20 within contiguous activity runs; no raw-sample overlap",
        "n_train_windows": int(len(outer_train)), "n_test_windows": int(len(test)), "validation_results": validation_results,
        "final_test_metrics": final_metrics, "classification_report": report,
        "limitations": ["One contiguous recording exists per activity class, so this is not subject-independent evaluation.", "Different people, sensor placements, and sampling conditions require a new external test.", "Activity recognition only; not a fall prediction or clinical model."],
    }
    with open(models / "stithi_activity_metrics.json", "w") as output:
        json.dump(metrics, output, indent=2)
    matrix = confusion_matrix(labels[test], predicted, labels=sorted(ACTIVITY_NAMES))
    plt.figure(figsize=(9, 7)); plt.imshow(matrix, cmap="Blues"); plt.colorbar()
    names = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES)]
    plt.xticks(range(6), names, rotation=35, ha="right"); plt.yticks(range(6), names)
    for row in range(6):
        for column in range(6): plt.text(column, row, matrix[row, column], ha="center", va="center")
    plt.xlabel("Predicted activity"); plt.ylabel("True activity"); plt.tight_layout(); plt.savefig(models / "stithi_activity_confusion_matrix.png", dpi=150); plt.close()
    print(json.dumps({"selected_candidate": selected, "validation": validation_results[selected], "test": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
