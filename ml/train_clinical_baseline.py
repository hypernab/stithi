"""
STITHI historical fall-status classification baseline.

This research baseline uses gait summary variables to classify whether a
participant reported a fall during the last year before the test. It is an
association/classification experiment on historical status, not future fall
prediction and not a validated fall detector.

The dataset has 163 participant-level records. The target and feature names
are mapped to the exact column names in Database_register.csv. Missing and
non-numeric values such as '-', 'Incapable', and blank cells are coerced to
NaN. Median imputation and standardization are fitted inside each CV fold by a
Pipeline, preventing preprocessing leakage. A final preprocessing artifact is
then fitted on all rows for later inference.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "Database_register.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLUMN = "Falls during the last year - prior to test (YES/NO)"
FEATURE_COLUMNS = [
    "Gait speed (m/s)",
    "Stride time - Avg. (s)",
    "Stride time - STD (s)",
    "Cadence - Avg. (strides/min)",
    "Cadence - STD (strides/min)",
    "Stride Length - Avg. (m)",
    "Stride Length - STD (m)",
    "Swing - Avg. (%)",
    "Swing - STD (%)",
    "Clearance - Avg. (m)",
    "Clearance - STD (m)",
]
CLASS_NAMES = ["NO", "YES"]
RANDOM_SEED = 42
N_SPLITS = 5
MODEL_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def load_dataset():
    """Load the semicolon-delimited, cp1252-encoded clinical register."""
    data = pd.read_csv(
        DATASET_PATH,
        sep=";",
        encoding="cp1252",
        skiprows=[0, 2],
        na_values=["", "-", "Incapable"],
        keep_default_na=True,
    )
    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_values = {}
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip().str.upper()
    data = data[data[TARGET_COLUMN].isin(CLASS_NAMES)].copy()
    for column in FEATURE_COLUMNS:
        raw_values = data[column].copy()
        numeric_values = pd.to_numeric(
            raw_values.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        invalid_values[column] = int(
            (raw_values.notna() & numeric_values.isna()).sum()
        )
        data[column] = numeric_values
    return data, invalid_values


def build_pipeline():
    """Build fold-local imputation, scaling, and classification steps."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**MODEL_PARAMETERS)),
        ]
    )


def main():
    data, invalid_values = load_dataset()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    print("STITHI historical fall-status classification baseline")
    print(f"Raw participant rows: {len(data)}")
    print(f"Class counts: {y.value_counts().reindex(CLASS_NAMES).to_dict()}")
    print("Missing values before imputation:")
    print(X.isna().sum().to_string())
    print(f"Invalid non-numeric values coerced to NaN: {invalid_values}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    all_true = []
    all_pred = []
    all_probability = []

    for fold_number, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        pipeline = build_pipeline()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index("YES")]
        fold_metrics.append({
            "fold": fold_number,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score((y_test == "YES").astype(int), probabilities),
            "precision": precision_score(y_test, predictions, pos_label="YES", zero_division=0),
            "recall": recall_score(y_test, predictions, pos_label="YES", zero_division=0),
            "f1": f1_score(y_test, predictions, pos_label="YES", zero_division=0),
        })
        all_true.extend(y_test.tolist())
        all_pred.extend(predictions.tolist())
        all_probability.extend(probabilities.tolist())

    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "precision", "recall", "f1"]
    mean_metrics = {
        f"mean_cv_{name}": float(np.mean([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }
    labels = CLASS_NAMES
    matrix = confusion_matrix(all_true, all_pred, labels=labels).tolist()
    report = classification_report(
        all_true,
        all_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # Fit the final model and preprocessing together on all available rows.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    classifier = final_pipeline.named_steps["classifier"]
    preprocessing = Pipeline(
        steps=[
            ("imputer", final_pipeline.named_steps["imputer"]),
            ("scaler", final_pipeline.named_steps["scaler"]),
        ]
    )
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "stithi_gait_rf.pkl"
    preprocessing_path = MODELS_DIR / "stithi_gait_preprocessing.pkl"
    metrics_path = MODELS_DIR / "stithi_gait_metrics.json"
    plot_path = MODELS_DIR / "stithi_gait_feature_importance.png"
    joblib.dump(classifier, model_path)
    joblib.dump(preprocessing, preprocessing_path)

    metrics = {
        "model_name": "STITHI historical fall-status classification baseline",
        "dataset": "Database_register.csv",
        "dataset_description": "GSTRIDE: Gait, Frailty & Cognitive Data from 163 Older Adults (as provided)",
        "target": TARGET_COLUMN,
        "task": "historical fall-status classification / gait association",
        "n_subjects": int(len(data)),
        "class_counts": {name: int((y == name).sum()) for name in CLASS_NAMES},
        "missing_values_before_imputation": {
            column: int(value) for column, value in X.isna().sum().items()
        },
        "invalid_values_before_coercion": invalid_values,
        "invalid_values_handling": "Non-numeric values such as '-', 'Incapable', and blanks were coerced to missing; median imputation was fitted within each CV fold.",
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "cv": "5-fold stratified cross-validation",
        "preprocessing_inside_cv": True,
        "imputation": "median, fit within each CV training fold",
        "scaling": "StandardScaler, fit within each CV training fold",
        "random_seed": RANDOM_SEED,
        "model_parameters": MODEL_PARAMETERS,
        "fold_metrics": fold_metrics,
        **mean_metrics,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix,
        "classification_report": report,
        "artifacts": {
            "model": model_path.name,
            "preprocessing": preprocessing_path.name,
            "feature_importance_plot": plot_path.name,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    importances = classifier.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh([FEATURE_COLUMNS[index] for index in order][::-1], importances[order][::-1])
    plt.xlabel("Random Forest importance")
    plt.title("STITHI Gait Features: Historical Fall-Status Classification")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Mean CV accuracy: {mean_metrics['mean_cv_accuracy']:.4f}")
    print(f"Mean CV balanced accuracy: {mean_metrics['mean_cv_balanced_accuracy']:.4f}")
    print(f"Mean CV ROC-AUC: {mean_metrics['mean_cv_roc_auc']:.4f}")
    print(f"Mean CV precision: {mean_metrics['mean_cv_precision']:.4f}")
    print(f"Mean CV recall: {mean_metrics['mean_cv_recall']:.4f}")
    print(f"Mean CV F1: {mean_metrics['mean_cv_f1']:.4f}")
    print(f"Confusion matrix (OOF predictions, labels {labels}): {matrix}")
    print(f"Saved: {model_path}")
    print(f"Saved: {preprocessing_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
