"""
app.py
Customer Churn Prediction System — Modern SaaS Dashboard
A complete, interactive Streamlit application with Dark Navy SaaS styling,
KPI cards, single prediction with presets, batch CSV inference,
interactive Plotly visualizations, model performance comparison, and retention strategies.
"""

import os
import sys
import json
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    load_data,
    clean_data,
    validate_input_columns,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from src.eda import (
    get_dataset_summary,
    plot_churn_pie_chart,
    plot_churn_by_categorical,
    plot_tenure_cohorts,
    plot_monthly_vs_total_charges,
    get_key_eda_insights
)
from src.predict import (
    load_model_pipeline,
    predict_single_customer,
    predict_batch_customers,
    determine_risk_tier
)
from src.evaluate_model import (
    plot_confusion_matrix_heatmap,
    plot_roc_curves_comparison,
    explain_evaluation_metrics
)

# -----------------------------------------------------------------------------
# STREAMLIT APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# MODERN SAAS DARK THEME CUSTOM CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background and Layout */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    /* Metric / KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-subtext {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 6px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #0369a1 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(2, 132, 199, 0.3);
    }

    /* Result Box */
    .prediction-card {
        border-radius: 14px;
        padding: 24px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #1e293b;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Form Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: scale(1.01);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        raw_df = load_data(data_path)
        return clean_data(raw_df, is_training=False)
    return None


@st.cache_resource
def get_cached_pipeline():
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib")
    if os.path.exists(model_path):
        return load_model_pipeline(model_path)
    return None


@st.cache_data
def get_cached_metrics():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Load resources
dataset = get_cached_dataset()
pipeline = get_cached_pipeline()
metrics_data = get_cached_metrics()


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #0284c7, #2563eb); border-radius: 10px; padding: 10px; display: inline-flex;">
                <span style="font-size: 24px;">🛡️</span>
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">RetainIQ AI</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #38bdf8;">Churn Intelligence Suite</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page_selection = st.radio(
        "NAVIGATION",
        [
            "🏠 Home Dashboard",
            "🔮 Single Customer Prediction",
            "📁 Batch Prediction",
            "📊 Analytics & EDA",
            "⚡ Model Performance",
            "💡 Strategic Retention Insights"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if pipeline is not None:
        best_name = metrics_data.get("best_model_name", "Random Forest") if metrics_data else "Random Forest"
        st.success(f"● Model Active: **{best_name}**")
    else:
        st.error("● Model Not Found. Run training script.")

    if dataset is not None:
        st.info(f"● Dataset: **{len(dataset):,} Records**")
    else:
        st.warning("● Dataset CSV Missing")

    st.markdown("""
        <div style="margin-top: 30px; padding: 12px; border-radius: 8px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); font-size: 0.75rem; color: #94a3b8;">
            <b>B.Tech AI/ML Capstone Project</b><br>
            Tech: Python, Scikit-learn, Streamlit, Plotly.<br>
            <i>All predictions are probability estimates.</i>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# -----------------------------------------------------------------------------
if page_selection == "🏠 Home Dashboard":
    st.markdown("""
        <div class="hero-banner">
            <h1 style="color: #ffffff; margin: 0 0 8px 0; font-size: 2.2rem; font-weight: 800;">
                Customer Churn Prediction System
            </h1>
            <p style="color: #cbd5e1; font-size: 1.05rem; margin: 0; max-width: 850px; line-height: 1.6;">
                An end-to-end Machine Learning intelligence platform designed to detect at-risk customers, 
                quantify churn probabilities, diagnose critical drivers of attrition, and deliver targeted 
                retention strategies before customer departures occur.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if dataset is None:
        st.error("⚠️ Dataset not found! Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' inside the 'data/' directory.")
    else:
        summary = get_dataset_summary(dataset)

        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Monitored Customers</div>
                    <div class="kpi-value">{summary['total_customers']:,}</div>
                    <div class="kpi-subtext">Active and historical accounts</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Retained Accounts</div>
                    <div class="kpi-value" style="color: #10B981;">{summary['churn_no']:,}</div>
                    <div class="kpi-subtext">{100 - summary['churn_rate']:.1f}% customer retention rate</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Churned Accounts</div>
                    <div class="kpi-value" style="color: #EF4444;">{summary['churn_yes']:,}</div>
                    <div class="kpi-subtext">Discontinued subscriptions</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Overall Churn Rate</div>
                    <div class="kpi-value" style="color: #F59E0B;">{summary['churn_rate']:.2f}%</div>
                    <div class="kpi-subtext">Baseline customer attrition</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Main Charts
        chart_col1, chart_col2 = st.columns([1, 1.3])
        with chart_col1:
            st.plotly_chart(plot_churn_pie_chart(dataset), use_container_width=True)

        with chart_col2:
            st.plotly_chart(
                plot_churn_by_categorical(dataset, "Contract", "Churn Distribution by Contract Type"),
                use_container_width=True
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Secondary Row
        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            st.plotly_chart(plot_tenure_cohorts(dataset), use_container_width=True)
        with sec_col2:
            st.plotly_chart(
                plot_churn_by_categorical(dataset, "InternetService", "Churn by Internet Service Architecture"),
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# PAGE 2: SINGLE CUSTOMER PREDICTION
# -----------------------------------------------------------------------------
elif page_selection == "🔮 Single Customer Prediction":
    st.markdown("""
        <div>
            <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 4px;">Single Customer Churn Risk Assessment</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                Enter a customer's contract, account, and service parameters to compute real-time churn probability and obtain tailored retention actions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("⚠️ Trained model not detected! Please run `python src/train_model.py` in your terminal first.")
    else:
        # Quick Presets for Demo
        st.markdown("##### ⚡ Quick Demonstration Presets")
        p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.2, 1.2, 2.4])

        if "preset" not in st.session_state:
            st.session_state.preset = "default"

        if p_col1.button("🔴 High-Risk Profile"):
            st.session_state.preset = "high_risk"
        if p_col2.button("🟢 Low-Risk Profile"):
            st.session_state.preset = "low_risk"
        if p_col3.button("🟡 Moderate-Risk Profile"):
            st.session_state.preset = "moderate_risk"

        # Initialize defaults based on preset
        if st.session_state.preset == "high_risk":
            def_tenure = 2
            def_contract = "Month-to-month"
            def_internet = "Fiber optic"
            def_tech = "No"
            def_sec = "No"
            def_backup = "No"
            def_device = "No"
            def_payment = "Electronic check"
            def_paperless = "Yes"
            def_monthly = 98.5
            def_total = 197.0
        elif st.session_state.preset == "low_risk":
            def_tenure = 64
            def_contract = "Two year"
            def_internet = "DSL"
            def_tech = "Yes"
            def_sec = "Yes"
            def_backup = "Yes"
            def_device = "Yes"
            def_payment = "Credit card (automatic)"
            def_paperless = "No"
            def_monthly = 64.0
            def_total = 4096.0
        elif st.session_state.preset == "moderate_risk":
            def_tenure = 18
            def_contract = "One year"
            def_internet = "Fiber optic"
            def_tech = "No"
            def_sec = "Yes"
            def_backup = "No"
            def_device = "Yes"
            def_payment = "Bank transfer (automatic)"
            def_paperless = "Yes"
            def_monthly = 84.5
            def_total = 1521.0
        else:
            def_tenure = 12
            def_contract = "Month-to-month"
            def_internet = "Fiber optic"
            def_tech = "No"
            def_sec = "No"
            def_backup = "Yes"
            def_device = "No"
            def_payment = "Electronic check"
            def_paperless = "Yes"
            def_monthly = 75.0
            def_total = 900.0

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        with st.form("customer_input_form"):
            st.markdown("#### 1. Customer Demographics & Account Tenure")
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            with col_d2:
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No", index=0)
            with col_d3:
                partner = st.selectbox("Partner", ["No", "Yes"], index=1 if st.session_state.preset == "low_risk" else 0)
            with col_d4:
                dependents = st.selectbox("Dependents", ["No", "Yes"], index=1 if st.session_state.preset == "low_risk" else 0)

            tenure_val = st.slider("Account Tenure (Months)", min_value=1, max_value=72, value=def_tenure, help="Number of months customer has subscribed.")

            st.markdown("---")
            st.markdown("#### 2. Contract, Billing & Payment")
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                contract_val = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_contract))
            with col_b2:
                payment_val = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment)
                )
            with col_b3:
                paperless_val = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(def_paperless))
            with col_b4:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                monthly_val = st.number_input("Monthly Charges ($ USD)", min_value=18.0, max_value=150.0, value=float(def_monthly), step=0.5)
            with col_c2:
                auto_total = round(monthly_val * tenure_val, 2)
                total_val = st.number_input("Total Charges ($ USD)", min_value=0.0, max_value=10000.0, value=float(auto_total), step=10.0, help="Total amount billed across tenure.")

            st.markdown("---")
            st.markdown("#### 3. Internet & Value-Added Services")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                internet_val = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_internet))
            with col_s2:
                multiline_val = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
            with col_s3:
                tech_val = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_tech))
            with col_s4:
                sec_val = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_sec))

            col_s5, col_s6, col_s7, col_s8 = st.columns(4)
            with col_s5:
                backup_val = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_backup))
            with col_s6:
                device_val = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(def_device))
            with col_s7:
                tv_val = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
            with col_s8:
                movies_val = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔮 Predict Churn Risk & Generate Insights", use_container_width=True)

        if submit_btn:
            input_customer = {
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure_val),
                "PhoneService": phone_service,
                "MultipleLines": multiline_val,
                "InternetService": internet_val,
                "OnlineSecurity": sec_val,
                "OnlineBackup": backup_val,
                "DeviceProtection": device_val,
                "TechSupport": tech_val,
                "StreamingTV": tv_val,
                "StreamingMovies": movies_val,
                "Contract": contract_val,
                "PaperlessBilling": paperless_val,
                "PaymentMethod": payment_val,
                "MonthlyCharges": float(monthly_val),
                "TotalCharges": float(total_val)
            }

            with st.spinner("Analyzing customer risk profile through machine learning pipeline..."):
                prediction_result = predict_single_customer(input_customer, pipeline=pipeline)

            st.markdown("---")
            st.markdown("### 📊 Churn Prediction Assessment")

            res_col1, res_col2 = st.columns([1.2, 1.8])

            churn_prob = prediction_result["churn_probability"]
            stay_prob = prediction_result["stay_probability"]
            risk_tier = prediction_result["risk_tier"]
            risk_color = prediction_result["risk_color"]

            with res_col1:
                # Gauge chart
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={"suffix": "%", "font": {"color": "#ffffff", "size": 36}},
                    title={"text": "<b>Estimated Churn Probability</b>", "font": {"color": "#cbd5e1", "size": 15}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": risk_color, "thickness": 0.3},
                        "bgcolor": "#1e293b",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                            {"range": [35, 60], "color": "rgba(245, 158, 11, 0.25)"},
                            {"range": [60, 100], "color": "rgba(239, 68, 68, 0.25)"}
                        ]
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=40, b=20, l=30, r=30),
                    height=240
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            with res_col2:
                st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border-left: 6px solid {risk_color}; padding: 20px; border-radius: 10px;">
                        <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Prediction Output</div>
                        <h2 style="color: {risk_color}; margin: 4px 0 10px 0; font-size: 1.8rem;">
                            {prediction_result['prediction_class'].upper()} ({risk_tier.upper()})
                        </h2>
                        <p style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 12px;">
                            {prediction_result['risk_description']}
                        </p>
                        <div style="display: flex; gap: 20px; margin-top: 10px;">
                            <div>
                                <span style="color: #94a3b8; font-size: 0.8rem;">Non-Churn Probability:</span><br>
                                <b style="color: #10B981; font-size: 1.15rem;">{stay_prob * 100:.1f}%</b>
                            </div>
                            <div>
                                <span style="color: #94a3b8; font-size: 0.8rem;">Churn Probability:</span><br>
                                <b style="color: #EF4444; font-size: 1.15rem;">{churn_prob * 100:.1f}%</b>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 🎯 Prescriptive Retention Recommendations")
            for rec in prediction_result["recommendations"]:
                st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); padding: 12px 18px; border-radius: 8px; margin-bottom: 8px; color: #cbd5e1;">
                        {rec}
                    </div>
                """, unsafe_allow_html=True)

            st.caption(f"ℹ️ **Disclaimer**: {prediction_result['disclaimer']}")


# -----------------------------------------------------------------------------
# PAGE 3: BATCH PREDICTION
# -----------------------------------------------------------------------------
elif page_selection == "📁 Batch Prediction":
    st.markdown("""
        <div>
            <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 4px;">Batch Customer Churn Prediction</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                Upload a CSV containing customer records to generate bulk churn predictions, probability scores, and risk classifications for proactive campaign outreach.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("⚠️ Model pipeline not found! Train the model first.")
    else:
        sample_batch_file = os.path.join(PROJECT_ROOT, "data", "sample_batch_test.csv")

        b_action_col1, b_action_col2 = st.columns([1.5, 2.5])
        with b_action_col1:
            st.markdown("##### ⚡ Quick Demonstration")
            use_sample = st.button(" Load Sample Customer Batch (25 Accounts)", use_container_width=True)

        uploaded_file = st.file_uploader(
            "Or upload your customer CSV file",
            type=["csv"],
            help="Upload a CSV with standard Telco customer features."
        )

        df_to_predict = None
        if use_sample and os.path.exists(sample_batch_file):
            df_to_predict = pd.read_csv(sample_batch_file)
            st.success("Loaded 25 customer records from sample test batch!")
        elif uploaded_file is not None:
            try:
                df_to_predict = pd.read_csv(uploaded_file)
                st.success(f"Uploaded CSV loaded successfully ({len(df_to_predict)} records).")
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

        if df_to_predict is not None:
            is_valid, missing = validate_input_columns(df_to_predict)
            if not is_valid:
                st.error(f" Uploaded CSV is missing the following required columns: **{', '.join(missing)}**")
                st.info("Expected columns include: " + ", ".join(NUMERICAL_FEATURES + CATEGORICAL_FEATURES))
            else:
                with st.spinner("Processing batch predictions with machine learning pipeline..."):
                    results_df = predict_batch_customers(df_to_predict, pipeline=pipeline)

                # Batch KPI Summary
                total_batch = len(results_df)
                predicted_churners = (results_df["Predicted_Churn"] == "Yes").sum()
                predicted_stayers = total_batch - predicted_churners
                batch_churn_rate = (predicted_churners / total_batch * 100) if total_batch > 0 else 0
                high_risk_count = (results_df["Risk_Category"] == "High Risk").sum()

                st.markdown("---")
                st.markdown("#### 📈 Batch Inference Overview")
                bk1, bk2, bk3, bk4 = st.columns(4)
                bk1.metric("Batch Records Processed", f"{total_batch:,}")
                bk2.metric("Predicted At-Risk (Churn)", f"{predicted_churners:,}", delta=f"{batch_churn_rate:.1f}% Churn Rate", delta_color="inverse")
                bk3.metric("Predicted Retained", f"{predicted_stayers:,}")
                bk4.metric("High-Risk Urgency", f"{high_risk_count:,}", delta="Requires Outreach", delta_color="inverse")

                # Filter Controls
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                f_col1, f_col2 = st.columns([1.5, 2.5])
                with f_col1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Category",
                        ["High Risk", "Medium Risk", "Low Risk"],
                        default=["High Risk", "Medium Risk", "Low Risk"]
                    )
                with f_col2:
                    search_id = st.text_input("Search Customer ID (if available)", "")

                filtered_df = results_df[results_df["Risk_Category"].isin(risk_filter)]
                if search_id and "customerID" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["customerID"].astype(str).str.contains(search_id, case=False)]

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=360
                )

                # Export Button
                csv_download = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Enriched Batch Predictions (CSV)",
                    data=csv_download,
                    file_name="customer_churn_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# -----------------------------------------------------------------------------
# PAGE 4: ANALYTICS & EDA
# -----------------------------------------------------------------------------
elif page_selection == "📊 Analytics & EDA":
    st.markdown("""
        <div>
            <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 4px;">Exploratory Data Analysis & Churn Patterns</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                Investigate associations between customer attributes, service adoption, pricing tiers, and retention outcomes.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if dataset is None:
        st.error("⚠️ Dataset not available. Please ensure data is loaded.")
    else:
        tab_contracts, tab_tenure, tab_services, tab_charges = st.tabs([
            "📋 Contract & Billing",
            "⏳ Tenure & Lifecycle",
            "🌐 Services & Technology",
            "💰 Monthly & Total Charges"
        ])

        with tab_contracts:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.plotly_chart(
                    plot_churn_by_categorical(dataset, "Contract", "Churn by Contract Structure"),
                    use_container_width=True
                )
            with c_col2:
                st.plotly_chart(
                    plot_churn_by_categorical(dataset, "PaymentMethod", "Churn by Customer Payment Method"),
                    use_container_width=True
                )

        with tab_tenure:
            st.plotly_chart(plot_tenure_cohorts(dataset), use_container_width=True)
            st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.4); border-left: 4px solid #38bdf8; padding: 14px; border-radius: 6px; color: #cbd5e1;">
                    <b>Tenure Insight:</b> The first 12 months is the peak attrition window (~47.4% churn). 
                    Customers who cross 24 months show dramatic loyalty improvement, with long-term (49-72m) churn dropping under 7%.
                </div>
            """, unsafe_allow_html=True)

        with tab_services:
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.plotly_chart(
                    plot_churn_by_categorical(dataset, "TechSupport", "Impact of Tech Support Add-On"),
                    use_container_width=True
                )
            with s_col2:
                st.plotly_chart(
                    plot_churn_by_categorical(dataset, "OnlineSecurity", "Impact of Online Security Add-On"),
                    use_container_width=True
                )

        with tab_charges:
            st.plotly_chart(plot_monthly_vs_total_charges(dataset), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Key Analytical Observations")
        insights = get_key_eda_insights()
        for item in insights:
            with st.expander(f"📌 {item['category']} — Observation & Retention Implication", expanded=False):
                st.markdown(f"**Observed Association:** {item['observation']}")
                st.markdown(f"**Business Retention Strategy:** {item['business_implication']}")
                st.caption("Note: Associations reflect observed empirical patterns and should not be confused with strict causality.")


# -----------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE & EXPLAINABILITY
# -----------------------------------------------------------------------------
elif page_selection == "⚡ Model Performance":
    st.markdown("""
        <div>
            <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 4px;">Model Evaluation & Performance Benchmarking</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                Transparently compare multiple candidate algorithms trained on identical stratified test partitions (20% held-out test set = 1,409 customers).
            </p>
        </div>
    """, unsafe_allow_html=True)

    if metrics_data is None:
        st.warning("Model metrics file `models/model_metrics.json` not found. Please run training pipeline first.")
    else:
        best_model = metrics_data.get("best_model_name", "Random Forest")
        all_models_metrics = metrics_data.get("metrics", {})

        st.success(f"🏆 **Production Algorithm Selected:** `{best_model}` (Selected based on balanced Recall & ROC-AUC to maximize churn capture)")

        # Comparison Table
        table_rows = []
        for m_name, m_vals in all_models_metrics.items():
            table_rows.append({
                "Model Algorithm": m_name,
                "Accuracy": f"{m_vals['accuracy'] * 100:.2f}%",
                "Precision": f"{m_vals['precision'] * 100:.2f}%",
                "Recall (Churn Catch)": f"{m_vals['recall'] * 100:.2f}%",
                "F1-Score": f"{m_vals['f1_score']:.4f}",
                "ROC-AUC": f"{m_vals['roc_auc']:.4f}"
            })

        metrics_df = pd.DataFrame(table_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # ROC Comparison and Confusion Matrix
        eval_col1, eval_col2 = st.columns([1.1, 0.9])
        with eval_col1:
            st.plotly_chart(plot_roc_curves_comparison(all_models_metrics), use_container_width=True)

        with eval_col2:
            best_cm = all_models_metrics.get(best_model, {}).get("confusion_matrix", {})
            if best_cm:
                st.plotly_chart(plot_confusion_matrix_heatmap(best_cm, model_name=best_model), use_container_width=True)

        # Feature Importance / Coefficients Section
        st.markdown("---")
        st.markdown("### 🔬 Model Explainability & Key Feature Drivers")
        feat_data = metrics_data.get("feature_importances", {})

        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            if "Random Forest" in feat_data:
                rf_feats = pd.DataFrame(list(feat_data["Random Forest"].items()), columns=["Feature", "Importance"])
                fig_rf = px.bar(
                    rf_feats.head(10).sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Feature Importances (Random Forest)",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_rf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
                st.plotly_chart(fig_rf, use_container_width=True)

        with feat_col2:
            if "Logistic Regression" in feat_data:
                lr_feats = pd.DataFrame(list(feat_data["Logistic Regression"].items()), columns=["Feature", "Coefficient"])
                fig_lr = px.bar(
                    lr_feats.head(10).sort_values("Coefficient", ascending=True),
                    x="Coefficient",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Feature Coefficients (Logistic Regression Log-Odds)",
                    color="Coefficient",
                    color_continuous_scale="RdBu"
                )
                fig_lr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"))
                st.plotly_chart(fig_lr, use_container_width=True)

        # Beginner Friendly Metrics Guide
        st.markdown("---")
        st.markdown("### 📚 Evaluation Metrics Guide for Beginners")
        explanations = explain_evaluation_metrics()
        for metric, explanation in explanations.items():
            st.markdown(f"**• {metric}:** {explanation}")


# -----------------------------------------------------------------------------
# PAGE 6: STRATEGIC RETENTION INSIGHTS
# -----------------------------------------------------------------------------
elif page_selection == "💡 Strategic Retention Insights":
    st.markdown("""
        <div>
            <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 4px;">Strategic Customer Retention Playbook</h2>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                Actionable business frameworks for translating predictive churn probabilities into profitable retention campaigns.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="kpi-card" style="margin-bottom: 20px;">
                <h4 style="color: #38bdf8; margin-top: 0;">1. Contract Commitment Playbook</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    <b>Challenge:</b> Month-to-month contracts experience over 4x higher churn rates than annual contracts.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Loyalty Lock-in Offer:</b> Present month-to-month customers in month 3 with a 15% discount if they convert to a 12-month commitment.</li>
                    <li><b>Hardware Subsidies:</b> Bundle smart home or upgraded Wi-Fi router hardware exclusively with annual agreements.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="kpi-card">
                <h4 style="color: #10B981; margin-top: 0;">2. Automated Billing Optimization</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    <b>Challenge:</b> Electronic Check customers churn at 45%, largely due to recurring payment friction and manual touchpoints.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Auto-Pay Incentive:</b> Offer a permanent $5/month billing credit for enrollment in Automated Bank Transfer or Credit Card.</li>
                    <li><b>SMS Payment Reminders:</b> Pre-alert customers before payment deadlines to reduce accidental lapses.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="kpi-card" style="margin-bottom: 20px;">
                <h4 style="color: #F59E0B; margin-top: 0;">3. Early-Life Customer Onboarding</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    <b>Challenge:</b> First-year subscribers represent nearly half of all lost customers.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Day 30 Health Check:</b> Automated proactive satisfaction check to verify line stability and equipment speed.</li>
                    <li><b>Complimentary VIP Support:</b> 60 days of prioritized tech assistance to prevent early frustration.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="kpi-card">
                <h4 style="color: #A855F7; margin-top: 0;">4. Ethical AI & Fair Lending Principles</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    <b>Standard:</b> AI systems must adhere to non-discriminatory retention practices.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Exclusion of Protected Attributes:</b> Identifiers (`customerID`) are fully removed from training.</li>
                    <li><b>Service-Based Offers:</b> Retention proposals are rooted in plan features and pricing fairness, never demographic profiling.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
