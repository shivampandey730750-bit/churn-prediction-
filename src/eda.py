"""
src/eda.py
Exploratory Data Analysis utilities and interactive Plotly visualization helpers
for the Customer Churn Prediction System.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes high-level dataset metrics and health checks.
    """
    total_customers = len(df)
    churn_col = "Churn" if "Churn" in df.columns else None

    if churn_col is not None:
        churn_counts = df[churn_col].value_counts()
        churn_yes = int(churn_counts.get("Yes", churn_counts.get(1, 0)))
        churn_no = int(churn_counts.get("No", churn_counts.get(0, 0)))
        churn_rate = (churn_yes / total_customers * 100) if total_customers > 0 else 0.0
    else:
        churn_yes, churn_no, churn_rate = 0, 0, 0.0

    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "total_customers": total_customers,
        "churn_yes": churn_yes,
        "churn_no": churn_no,
        "churn_rate": churn_rate,
        "total_columns": df.shape[1],
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows
    }


def compute_churn_rate_by_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Calculates churn rate and counts grouped by a specified categorical feature.
    """
    if feature not in df.columns or "Churn" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    # Safely convert Yes/No or 1/0 churn representations to integer 0/1
    temp["Churn_Numeric"] = temp["Churn"].replace({"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0})
    temp["Churn_Numeric"] = pd.to_numeric(temp["Churn_Numeric"], errors="coerce").fillna(0).astype(int)

    grouped = temp.groupby(feature).agg(
        Total_Customers=("Churn_Numeric", "count"),
        Churned_Customers=("Churn_Numeric", "sum")
    ).reset_index()

    grouped["Retained_Customers"] = grouped["Total_Customers"] - grouped["Churned_Customers"]
    grouped["Churn_Rate_%"] = (grouped["Churned_Customers"] / grouped["Total_Customers"] * 100).round(2)
    grouped = grouped.sort_values(by="Churn_Rate_%", ascending=False)
    return grouped


def plot_churn_pie_chart(df: pd.DataFrame) -> go.Figure:
    """
    Returns a modern donut chart showing Churn vs Non-Churn distribution.
    """
    summary = get_dataset_summary(df)
    labels = ["Retained (No Churn)", "Churned"]
    values = [summary["churn_no"], summary["churn_yes"]]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=14, color="#ffffff")
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Overall Customer Churn Ratio", font=dict(size=18, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8"), orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=30, l=20, r=20)
    )
    return fig


def plot_churn_by_categorical(df: pd.DataFrame, feature: str, title: str) -> go.Figure:
    """
    Generates a grouped bar chart displaying Retained vs Churned counts by category.
    """
    table = compute_churn_rate_by_feature(df, feature)
    if table.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=table[feature],
        y=table["Retained_Customers"],
        name="Retained",
        marker_color="#10B981",
        text=table["Retained_Customers"],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=table[feature],
        y=table["Churned_Customers"],
        name="Churned",
        marker_color="#EF4444",
        text=table["Churned_Customers"],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title=feature, color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="Number of Customers", color="#94a3b8", gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#cbd5e1")),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def plot_tenure_cohorts(df: pd.DataFrame) -> go.Figure:
    """
    Categorizes tenure into cohorts (0-12m, 13-24m, 25-48m, 49-72m) and visualizes churn rate.
    """
    temp = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["1st Year (0-12m)", "2nd Year (13-24m)", "Mid-term (25-48m)", "Long-term (49-72m)"]
    temp["Tenure_Group"] = pd.cut(temp["tenure"], bins=bins, labels=labels, include_lowest=True)

    grouped = compute_churn_rate_by_feature(temp, "Tenure_Group")
    fig = px.bar(
        grouped,
        x="Tenure_Group",
        y="Churn_Rate_%",
        text="Churn_Rate_%",
        title="Churn Rate by Customer Tenure Cohort (%)",
        color="Churn_Rate_%",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"]
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(title="Tenure Group", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(title="Churn Rate (%)", color="#94a3b8", gridcolor="#334155", range=[0, max(grouped["Churn_Rate_%"]) * 1.25]),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def plot_monthly_vs_total_charges(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of Monthly Charges vs Total Charges colored by Churn status.
    """
    fig = px.scatter(
        df,
        x="MonthlyCharges",
        y="TotalCharges",
        color="Churn",
        color_discrete_map={"No": "#10B981", "Yes": "#EF4444", 0: "#10B981", 1: "#EF4444"},
        opacity=0.6,
        title="Monthly Charges vs. Total Charges by Churn Status",
        labels={"MonthlyCharges": "Monthly Charges ($)", "TotalCharges": "Total Charges ($)"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#94a3b8", gridcolor="#334155"),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def get_key_eda_insights() -> List[Dict[str, str]]:
    """
    Returns curated business and analytical observations from the IBM Telco dataset.
    Highlights associations without asserting causation.
    """
    return [
        {
            "category": "Contract Type",
            "observation": "Customers on Month-to-Month contracts exhibit an exceptionally high churn rate (~42%) compared to One-Year (~11%) and Two-Year (~2.8%) contracts.",
            "business_implication": "Incentivizing longer-term commitment through introductory discounts or loyalty rewards could stabilize recurring revenue."
        },
        {
            "category": "Tenure Duration",
            "observation": "Over 47% of first-year customers (0-12 months) churn, whereas customer churn drops below 7% for accounts older than 4 years.",
            "business_implication": "The customer onboarding window (first 90-180 days) is the critical retention milestone where proactive engagement creates the most value."
        },
        {
            "category": "Internet Service Type",
            "observation": "Fiber Optic internet subscribers show significantly higher churn (~41.9%) than DSL subscribers (~19.0%) despite higher monthly charges.",
            "business_implication": "Investigate technical service stability, price-to-value satisfaction, and competitor fiber offerings in key service zones."
        },
        {
            "category": "Payment Method",
            "observation": "Customers paying via Electronic Check churn at roughly 45%, compared to 15-18% for automatic bank transfers and credit card payments.",
            "business_implication": "Offering a small one-time billing discount to transition users onto automated payment methods directly correlates with longer retention."
        },
        {
            "category": "Value-Add Security & Tech Support",
            "observation": "Subscribers with Online Security and Tech Support features have churn rates nearly 3x lower than those without these add-ons.",
            "business_implication": "Bundling technical support and security into base plans improves user perceived utility and switches cost."
        }
    ]
