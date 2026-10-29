"""
src/evaluate_model.py
Evaluation metrics calculation, confusion matrix generation, ROC curve computation,
and plain-English metric explanations for the Customer Churn Prediction System.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import plotly.graph_objects as go


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics from actual model predictions.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "classification_report": clf_report
    }


def plot_confusion_matrix_heatmap(cm_dict: Dict[str, int], model_name: str = "Classifier") -> go.Figure:
    """
    Generates a modern styled Confusion Matrix heatmap using Plotly.
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted: Retained", "Predicted: Churn"]
    y_labels = ["Actual: Retained", "Actual: Churn"]

    annotations = [
        dict(
            x=x_labels[0], y=y_labels[0],
            text=f"<b>TN</b><br>{tn}<br><span style='font-size:11px;color:#94a3b8;'>Correctly Retained</span>",
            showarrow=False, font=dict(color="#ffffff", size=14)
        ),
        dict(
            x=x_labels[1], y=y_labels[0],
            text=f"<b>FP</b><br>{fp}<br><span style='font-size:11px;color:#94a3b8;'>False Alarm</span>",
            showarrow=False, font=dict(color="#ffffff", size=14)
        ),
        dict(
            x=x_labels[0], y=y_labels[1],
            text=f"<b>FN</b><br>{fn}<br><span style='font-size:11px;color:#94a3b8;'>Missed Churn!</span>",
            showarrow=False, font=dict(color="#ffffff", size=14)
        ),
        dict(
            x=x_labels[1], y=y_labels[1],
            text=f"<b>TP</b><br>{tp}<br><span style='font-size:11px;color:#94a3b8;'>Detected Churn</span>",
            showarrow=False, font=dict(color="#ffffff", size=14)
        )
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#1e293b"], [0.5, "#2563eb"], [1, "#0284c7"]],
            showscale=False
        )
    )
    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=16, color="#f8fafc")),
        annotations=annotations,
        xaxis=dict(tickfont=dict(color="#cbd5e1", size=13), side="bottom"),
        yaxis=dict(tickfont=dict(color="#cbd5e1", size=13), autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def plot_roc_curves_comparison(models_metrics: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Overlays ROC curves for multiple models to show discriminative power.
    """
    fig = go.Figure()

    # Diagonal baseline (random guess)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#64748b", width=2),
        name="Random Chance (AUC = 0.50)"
    ))

    colors = ["#38bdf8", "#34d399", "#f43f5e", "#fbbf24"]

    for idx, (m_name, metrics) in enumerate(models_metrics.items()):
        roc = metrics.get("roc_curve", {})
        auc_score = metrics.get("roc_auc", 0.5)
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        if fpr and tpr:
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=2.5),
                name=f"{m_name} (AUC = {auc_score:.3f})"
            ))

    fig.update_layout(
        title=dict(text="Receiver Operating Characteristic (ROC) Comparison", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", color="#94a3b8", gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def explain_evaluation_metrics() -> Dict[str, str]:
    """
    Returns beginner-friendly explanations of each evaluation metric in customer churn context.
    """
    return {
        "Accuracy": (
            "The percentage of total correct predictions (both churners and stayers). "
            "However, if 73% of customers do not churn, a naive model that predicts everyone stays has 73% accuracy "
            "while missing 100% of churners! Hence accuracy alone is NOT sufficient."
        ),
        "Recall (Sensitivity)": (
            "Out of all customers who REALLY churned, what percentage did the model catch? "
            "High Recall is crucial for customer retention: missing an at-risk customer (False Negative) means losing recurring revenue forever."
        ),
        "Precision": (
            "When the model predicts a customer will churn, how often is it right? "
            "Higher precision minimizes wasted retention discounts on customers who were going to stay anyway (False Positives)."
        ),
        "F1-Score": (
            "The harmonic mean between Precision and Recall. It provides a balanced score especially when classes are imbalanced."
        ),
        "ROC-AUC": (
            "Area Under the ROC Curve. Measures how effectively the model ranks a randomly chosen churner higher than a non-churner. "
            "1.0 represents a perfect model, 0.5 is no better than a coin flip."
        ),
        "False Positive (FP)": (
            "The model predicted the customer would churn, but they actually stayed. "
            "Business cost: sending an unnecessary discount or retention offer, but retaining customer loyalty."
        ),
        "False Negative (FN)": (
            "The model predicted the customer would stay, but they silently cancelled their subscription. "
            "Business cost: CRITICAL - permanent loss of customer lifetime value (LTV) and recurring revenue."
        )
    }
