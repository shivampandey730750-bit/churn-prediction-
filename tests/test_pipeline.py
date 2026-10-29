"""
tests/test_pipeline.py
Automated test suite verifying data loading, cleaning, validation,
model training, persistence, and single & batch prediction pipelines.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    load_data,
    clean_data,
    validate_input_columns,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.predict import (
    load_model_pipeline,
    predict_single_customer,
    predict_batch_customers,
    determine_risk_tier
)


@pytest.fixture
def sample_dataset_path():
    path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(path):
        pytest.skip(f"Dataset not present at {path}")
    return path


@pytest.fixture
def sample_customer_dict():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }


def test_data_loading_and_shape(sample_dataset_path):
    """Test that data loads properly and has expected dimensions."""
    df = load_data(sample_dataset_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "Churn" in df.columns


def test_missing_values_and_blank_handling():
    """Test cleaning logic handles whitespace in TotalCharges correctly."""
    dirty_data = pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [0, 5],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "No"],
        "OnlineBackup": ["No", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [25.0, 70.0],
        "TotalCharges": [" ", "350.0"],
        "Churn": ["No", "Yes"]
    })

    cleaned = clean_data(dirty_data, is_training=True)
    # Check that blank was converted to float without raising error
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert not cleaned["TotalCharges"].isnull().any()
    # Check churn was converted to integer (0 and 1)
    assert cleaned["Churn"].tolist() == [0, 1]


def test_column_validation():
    """Test input validation detects missing features."""
    valid_df = pd.DataFrame(columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    is_valid, missing = validate_input_columns(valid_df)
    assert is_valid
    assert len(missing) == 0

    invalid_df = pd.DataFrame(columns=["tenure", "MonthlyCharges"])
    is_valid, missing = validate_input_columns(invalid_df)
    assert not is_valid
    assert len(missing) > 0


def test_model_loading_and_single_prediction(sample_customer_dict):
    """Test that the trained pipeline loads and produces valid single predictions."""
    result = predict_single_customer(sample_customer_dict)

    assert result["prediction_class"] in ["Churn", "No Churn"]
    assert result["prediction_binary"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert 0.0 <= result["stay_probability"] <= 1.0
    assert np.isclose(result["churn_probability"] + result["stay_probability"], 1.0, atol=0.01)
    assert result["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(result["recommendations"]) > 0


def test_risk_tier_thresholds():
    """Test risk tier classification logic."""
    assert determine_risk_tier(0.15)[0] == "Low Risk"
    assert determine_risk_tier(0.45)[0] == "Medium Risk"
    assert determine_risk_tier(0.75)[0] == "High Risk"


def test_batch_prediction_validation(sample_customer_dict):
    """Test batch prediction produces correct output columns and non-null values."""
    batch_records = pd.DataFrame([sample_customer_dict, sample_customer_dict])
    results = predict_batch_customers(batch_records)

    assert "Predicted_Churn" in results.columns
    assert "Churn_Probability" in results.columns
    assert "Retention_Probability" in results.columns
    assert "Risk_Category" in results.columns
    assert len(results) == 2
    assert results["Churn_Probability"].between(0.0, 1.0).all()
