import os
import sys
import json
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    load_data,
    clean_data,
    build_preprocessor,
    get_feature_names,
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.evaluate_model import evaluate_predictions

DEFAULT_DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = "models"
RANDOM_STATE = 42


def train_and_evaluate_all_models(data_path: str = DEFAULT_DATA_PATH) -> Dict[str, Any]:
    """
    Executes the complete reproducible training workflow:
    1. Loads raw CSV data
    2. Cleans & validates records
    3. Splits into train & test sets (stratified, 80/20)
    4. Builds ColumnTransformer preprocessor
    5. Trains Logistic Regression, Random Forest, and Decision Tree
    6. Evaluates predictions on test set
    7. Selects best model (F1 / ROC-AUC criteria)
    8. Persists models and metrics to 'models/' directory
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 65)
    print(f"Loading dataset from: {data_path}")

    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df, is_training=True)

    print(f"Dataset shape after cleaning: {cleaned_df.shape}")

    # Separate features and target
    X = cleaned_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = cleaned_df[TARGET_COLUMN].copy()

    # Stratified Train/Test Split (ensures churn ratio is identical in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Class distribution (train): {y_train.value_counts(normalize=True).to_dict()}")

    # Define Candidate Classifiers
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    all_metrics: Dict[str, Any] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    for name, clf in candidate_models.items():
        print(f"\nTraining [{name}]...")
        # Fresh preprocessor to avoid any data leakage
        preprocessor = build_preprocessor()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Fit exclusively on training data
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Predictions on held-out test data
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test.values, y_pred, y_proba)
        all_metrics[name] = metrics

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    # Model Selection: Prioritizing ROC-AUC and F1-Score for imbalanced churn detection
    best_model_name = max(
        all_metrics.keys(),
        key=lambda m: (all_metrics[m]["roc_auc"] + all_metrics[m]["f1_score"]) / 2
    )
    print("\n" + "=" * 65)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Based on combined ROC-AUC ({all_metrics[best_model_name]['roc_auc']}) and F1 ({all_metrics[best_model_name]['f1_score']})")
    print("=" * 65)

    # Extract Feature Importances from Random Forest & Logistic Regression
    feature_importances: Dict[str, Any] = {}
    fitted_preprocessor = trained_pipelines[best_model_name].named_steps["preprocessor"]
    transformed_feature_names = get_feature_names(fitted_preprocessor)

    # Random Forest Importance
    rf_pipe = trained_pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps["classifier"]
    rf_importances = pd.Series(rf_clf.feature_importances_, index=transformed_feature_names)
    feature_importances["Random Forest"] = (
        rf_importances.sort_values(ascending=False).head(15).to_dict()
    )

    # Logistic Regression Coefficients (Magnitude reflects log-odds impact)
    lr_pipe = trained_pipelines["Logistic Regression"]
    lr_clf = lr_pipe.named_steps["classifier"]
    lr_coeffs = pd.Series(lr_clf.coef_[0], index=transformed_feature_names)
    feature_importances["Logistic Regression"] = (
        lr_coeffs.reindex(lr_coeffs.abs().sort_values(ascending=False).index).head(15).to_dict()
    )

    # Save artifacts
    best_pipe = trained_pipelines[best_model_name]
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))
    joblib.dump(trained_pipelines["Random Forest"], os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"))
    joblib.dump(trained_pipelines["Logistic Regression"], os.path.join(MODELS_DIR, "logistic_regression_pipeline.joblib"))
    joblib.dump(trained_pipelines["Decision Tree"], os.path.join(MODELS_DIR, "decision_tree_pipeline.joblib"))

    # Save metadata and evaluation results
    output_metadata = {
        "best_model_name": best_model_name,
        "models_evaluated": list(candidate_models.keys()),
        "metrics": all_metrics,
        "feature_importances": feature_importances,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "feature_columns": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE
    }

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=4)

    print(f"\nArtifacts successfully persisted to '{MODELS_DIR}/'")
    print(f"  - Best Pipeline: {os.path.join(MODELS_DIR, 'best_model_pipeline.joblib')}")
    print(f"  - Metrics & Insights: {metrics_path}")

    return output_metadata


if __name__ == "__main__":
    train_and_evaluate_all_models()
