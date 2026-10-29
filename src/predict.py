"""
src/predict.py
Inference module for single customer and batch CSV churn predictions.
Loads the saved pipeline, calculates calibrated churn probabilities, assigns risk tiers,
and produces actionable, non-discriminatory customer retention suggestions.
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    clean_data,
    validate_input_columns,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN
)

DEFAULT_MODEL_PATH = os.path.join("models", "best_model_pipeline.joblib")


def load_model_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads the serialized Scikit-learn pipeline from disk.
    Raises FileNotFoundError with friendly instructions if not trained yet.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please train the model first by running: python src/train_model.py"
        )
    return joblib.load(model_path)


def determine_risk_tier(churn_probability: float) -> Tuple[str, str, str]:
    """
    Maps estimated churn probability into business-friendly risk categories.
    Returns: (Risk Level, Color Code, Description)
    """
    if churn_probability >= 0.60:
        return (
            "High Risk",
            "#EF4444",
            "This customer shows multiple strong behavioral churn indicators. Immediate retention intervention is recommended."
        )
    elif churn_probability >= 0.35:
        return (
            "Medium Risk",
            "#F59E0B",
            "This customer exhibits moderate churn vulnerability. Proactive engagement and value-reinforcing offers are advised."
        )
    else:
        return (
            "Low Risk",
            "#10B981",
            "This customer appears well-engaged with low churn probability. Maintain standard service excellence and regular check-ins."
        )


def generate_retention_recommendations(customer_dict: Dict[str, Any], churn_prob: float) -> List[str]:
    """
    Produces actionable, ethical, and service-based retention strategies.
    Never relies on sensitive personal attributes (gender, age) for decisions.
    """
    recommendations = []

    contract = customer_dict.get("Contract", "")
    payment = customer_dict.get("PaymentMethod", "")
    tech_support = customer_dict.get("TechSupport", "")
    online_sec = customer_dict.get("OnlineSecurity", "")
    internet = customer_dict.get("InternetService", "")
    tenure = float(customer_dict.get("tenure", 0))
    monthly = float(customer_dict.get("MonthlyCharges", 0))

    if contract == "Month-to-month":
        recommendations.append(
            "📋 Contract Upgrade: Present an annual contract offer featuring a 10% monthly discount or loyalty credits to improve contract commitment."
        )

    if "Electronic check" in payment:
        recommendations.append(
            "💳 Automated Billing Incentive: Encourage switching to Automated Credit Card or Bank Transfer by offering a one-time $10 statement credit."
        )

    if tech_support == "No" and internet != "No":
        recommendations.append(
            "🛠️ Complimentary Tech Support Trial: Grant 60 days of complimentary premium technical assistance to resolve onboarding friction."
        )

    if online_sec == "No" and internet != "No":
        recommendations.append(
            "🛡️ Security Package Add-On: Offer a 3-month trial of Online Security & Antivirus protection to enhance service stickiness."
        )

    if tenure <= 6:
        recommendations.append(
            "🤝 Early Lifecycle Check-In: Schedule a proactive Customer Success call within the first 90 days to verify satisfaction and resolve queries."
        )

    if internet == "Fiber optic" and monthly > 85.0:
        recommendations.append(
            "⚡ Plan Optimization Review: Evaluate plan usage and propose a right-sized broadband package or stream bundle to prevent bill fatigue."
        )

    if not recommendations:
        recommendations.append(
            "⭐ Customer Appreciation: Customer profile is well-optimized. Send a loyalty thank-you communication or periodic satisfaction survey."
        )

    return recommendations


def predict_single_customer(
    customer_data: Dict[str, Any],
    pipeline=None,
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Generates prediction and probability for a single customer record.
    """
    if pipeline is None:
        pipeline = load_model_pipeline(model_path)

    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    cleaned_df = clean_data(df, is_training=False)

    is_valid, missing = validate_input_columns(cleaned_df)
    if not is_valid:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Select exact feature list
    X = cleaned_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    pred_class_numeric = int(pipeline.predict(X)[0])
    probabilities = pipeline.predict_proba(X)[0]

    churn_prob = float(probabilities[1])
    stay_prob = float(probabilities[0])
    pred_label = "Churn" if pred_class_numeric == 1 else "No Churn"

    risk_tier, risk_color, risk_desc = determine_risk_tier(churn_prob)
    recommendations = generate_retention_recommendations(customer_data, churn_prob)

    return {
        "prediction_class": pred_label,
        "prediction_binary": pred_class_numeric,
        "churn_probability": round(churn_prob, 4),
        "stay_probability": round(stay_prob, 4),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "risk_description": risk_desc,
        "recommendations": recommendations,
        "disclaimer": "Predictions are statistical estimates based on historical patterns and do not guarantee individual customer outcomes."
    }


def predict_batch_customers(
    batch_df: pd.DataFrame,
    pipeline=None,
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """
    Performs batch predictions over multiple customer records.
    Returns the original DataFrame enriched with predictions, probabilities, and risk tiers.
    """
    if pipeline is None:
        pipeline = load_model_pipeline(model_path)

    cleaned_df = clean_data(batch_df, is_training=False)
    is_valid, missing = validate_input_columns(cleaned_df)
    if not is_valid:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    X = cleaned_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    result_df = batch_df.copy()
    result_df["Predicted_Churn"] = np.where(preds == 1, "Yes", "No")
    result_df["Churn_Probability"] = np.round(probs, 4)
    result_df["Retention_Probability"] = np.round(1 - probs, 4)
    result_df["Risk_Category"] = [determine_risk_tier(p)[0] for p in probs]

    return result_df
