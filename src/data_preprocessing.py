"""
src/data_preprocessing.py
Data loading, cleaning, validation, and preprocessing pipeline definitions
for the Customer Churn Prediction System.
"""

from typing import Tuple, List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Canonical feature lists
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

ALL_EXPECTED_COLUMNS: List[str] = [ID_COLUMN] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads customer churn data from a CSV file.
    Raises FileNotFoundError with clear instructions if file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is placed inside the 'data/' directory."
        )
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans the raw Telco dataset:
    - Removes whitespace from string columns
    - Handles blank spaces in 'TotalCharges' (common in 0-tenure customers)
    - Converts 'TotalCharges' to float
    - Ensures 'SeniorCitizen' is formatted consistently
    - Encodes target 'Churn' as 0/1 if present
    """
    cleaned = df.copy()

    # Strip whitespace from object/string columns
    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        cleaned[c] = cleaned[c].astype(str).str.strip()

    # TotalCharges can contain spaces " " when tenure is 0
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        # Impute missing TotalCharges with MonthlyCharges * tenure or 0 for tenure 0
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["MonthlyCharges"] * cleaned["tenure"]).fillna(0.0)

    # Ensure SeniorCitizen is treated as int or categorical representation
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)

    # Encode target if present and is_training
    if TARGET_COLUMN in cleaned.columns and is_training:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].replace({"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0}).astype(int)

    return cleaned


def validate_input_columns(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates that the required features exist in the given DataFrame.
    Returns (is_valid, missing_columns).
    """
    required = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    if require_target:
        required = required + [TARGET_COLUMN]

    missing = [col for col in required if col not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a reproducible Scikit-Learn ColumnTransformer:
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    # Numerical
    feature_names.extend(NUMERICAL_FEATURES)
    # Categorical
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(encoded_cat_names)
    return feature_names
