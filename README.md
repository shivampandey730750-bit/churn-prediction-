# Customer Churn Prediction System 🛡️
### *A Full-Stack Machine Learning Intelligence Suite for Customer Retention*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Objective & Problem Statement

Customer churn occurs when subscribers discontinue doing business with a company or cancel their service contracts. In modern telecommunications, SaaS, and subscription business models, acquiring a new customer costs **5 to 7 times more** than retaining an existing one.

The **Customer Churn Prediction System** is a professional, beginner-friendly Machine Learning platform built from scratch. It solves the proactive retention challenge by:
1. **Predicting Attrition**: Identifying which customers are likely to discontinue service.
2. **Quantifying Probability**: Estimating calibrated churn probabilities ($0.0$ to $1.0$).
3. **Classifying Risk Tiers**: Segmenting accounts into **Low**, **Medium**, and **High Risk** categories.
4. **Prescribing Retention Actions**: Generating non-discriminatory, service-based recommendations (e.g., contract upgrades, auto-pay incentives, VIP onboarding).
5. **Interactive SaaS Dashboard**: Empowering business stakeholders with intuitive single-customer simulation, batch CSV scoring, and transparent model evaluation.

> [!NOTE]  
> **Statistical Disclaimer**: All model predictions represent empirical probability estimates based on historical patterns; they are decision-support insights rather than deterministic certainties.

---

## 🛠️ 2. Technology Stack

- **Core Programming Language**: Python 3.10+
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Machine Learning & Preprocessing**: Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
- **Model Algorithms**: Logistic Regression, Random Forest, Decision Tree
- **Model Persistence**: Joblib
- **Data Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **Web Dashboard**: Streamlit (SaaS Dark Theme)
- **Unit Testing**: Pytest
- **Development & Version Control**: VS Code, Git, GitHub

---

## 📂 3. Project Directory Architecture

```text
churn prediction/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Primary IBM Telco dataset (7,043 rows)
│   └── sample_batch_test.csv                 # 25-account sample file for instant batch testing
│
├── models/
│   ├── best_model_pipeline.joblib            # Production Scikit-learn pipeline (preprocessor + model)
│   ├── random_forest_pipeline.joblib         # Fitted Random Forest candidate pipeline
│   ├── logistic_regression_pipeline.joblib   # Fitted Logistic Regression candidate pipeline
│   └── model_metrics.json                    # Serialized test evaluation metrics & feature importances
│
├── reports/                                  # Exported figures, plots, and analysis assets
│
├── src/
│   ├── __init__.py                           # Module initializer
│   ├── data_preprocessing.py                 # Data loading, cleaning, validation & ColumnTransformer
│   ├── eda.py                                # Summary metrics, Plotly visualizations & business observations
│   ├── train_model.py                        # Reproducible multi-model training & model selection script
│   ├── evaluate_model.py                     # Confusion matrix, ROC curves & metric explanations
│   └── predict.py                            # Inference engine, probability scoring & retention suggestions
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                      # Automated test suite (6 passing unit/integration tests)
│
├── app.py                                    # Modern SaaS Streamlit web dashboard
├── requirements.txt                          # Pinned dependency definitions
├── .gitignore                                # Git ignore file for Python, caches, and venv
└── README.md                                 # Full documentation & execution manual
```

---

## 📊 4. Dataset Overview

The system uses the standard, canonical **IBM Telco Customer Churn** dataset containing **7,043 customers** and **21 attributes**:
- **Identifiers**: `customerID` (automatically excluded from model features to prevent memorization).
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Account Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Contract & Billing**: `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Target Variable**: `Churn` (`Yes` = 1, `No` = 0). Overall baseline churn rate: **26.54%** (1,869 churners vs. 5,174 retained).

### Missing Value & Anomaly Treatment:
In the raw data, customers with `tenure = 0` contain whitespace strings (`" "`) in `TotalCharges`. The cleaning pipeline detects these blanks, coerces them to `NaN`, and accurately imputes `TotalCharges = MonthlyCharges * tenure` (evaluating to `0.0`), preventing data loss or parsing exceptions.

---

## 🚀 5. Quickstart & Execution Instructions (Windows PowerShell)

Follow these exact commands in **Windows PowerShell** or the **VS Code Terminal**:

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\shiva\OneDrive\Pictures\Desktop\churn prediction"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Train and Evaluate the Models
Run the automated end-to-end training pipeline. This script trains Logistic Regression, Random Forest, and Decision Tree on an 80/20 stratified split, evaluates them on the held-out test set, and saves the best model pipeline:
```powershell
python src/train_model.py
```

### Step 6: Run Automated Tests
Execute the test suite to verify data validation, cleaning, probability bounds, and inference:
```powershell
pytest tests/ -v
```

### Step 7: Launch the Streamlit Dashboard
Start the interactive SaaS web application:
```powershell
streamlit run app.py
```
Streamlit will automatically open your default browser at: `http://localhost:8501`.

---

## 📈 6. Model Evaluation & Benchmark Results

All models were evaluated on the **same held-out test partition (1,409 customers)** using a fixed random seed (`random_state=42`) and target stratification:

| Candidate Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Selected) 🏆 | **75.09%** | **52.01%** | **79.68%** | **0.6294** | **0.8421** |
| **Logistic Regression** | 73.81% | 50.43% | 78.34% | 0.6136 | 0.8415 |
| **Decision Tree** | 75.44% | 52.60% | 75.67% | 0.6206 | 0.8318 |

### Why Recall and F1-Score Trump Accuracy in Churn Detection:
- In customer retention, the cost of a **False Negative** (failing to detect a customer who actually churns) is disastrous: permanent loss of recurring revenue and customer lifetime value (LTV).
- Conversely, a **False Positive** (reaching out with a retention offer to someone who would have stayed) only costs a minor incentive or customer service call.
- Because **73.5%** of the dataset does not churn, a dummy model predicting "No" on every record achieves 73.5% accuracy but catches **0% of churners**!
- **Random Forest** was selected as the production model due to its highest **ROC-AUC (0.8421)** and **Recall (79.68%)**, correctly capturing ~8 out of every 10 departing subscribers.

---

## 💡 7. Model Explainability & Retention Playbook

### Top Churn Factors Identified by the Models:
1. **Contract Structure**: Month-to-month contracts exhibit an attrition rate of **~42%**, compared to **~11%** for 1-year and **~2.8%** for 2-year contracts.
2. **Tenure Window**: Customers in months 0–12 represent the highest attrition cohort (~47%). Beyond 24 months, churn drops dramatically.
3. **Payment Method**: Paying via Electronic Check strongly correlates with high churn (~45%) vs. Automated Bank Transfer and Credit Card (~15–18%).
4. **Support & Security Add-Ons**: Customers with **Tech Support** and **Online Security** churn nearly **3x less** than customers without them.

### Prescriptive Strategic Playbook:
- **Annual Loyalty Upgrade**: Offer a $10/month credit for customers who transition from month-to-month to a 12-month agreement.
- **Automated Billing Incentive**: Provide a one-time $10 statement credit to migrate users off Electronic Check to Automatic Bank Transfer or Credit Card.
- **VIP Onboarding Support**: Deliver proactive check-in calls and 60 days of complimentary technical assistance during the initial 90-day onboarding window.

---

## 🖥️ 8. Dashboard Walkthrough (Streamlit App Pages)

1. **🏠 Home Dashboard**: High-level KPI summary cards, churn ratio donut chart, contract churn breakdown, and cohort analysis.
2. **🔮 Single Customer Prediction**: Interactive parameter form with one-click presets (**High Risk**, **Low Risk**, **Moderate Risk**), dynamic probability gauge, risk classification, and tailored retention advice.
3. **📁 Batch Prediction**: Bulk CSV upload interface, automatic schema validation, one-click test sample loading, risk category filtering, and one-click enriched CSV export.
4. **📊 Analytics & EDA**: Tabbed exploration of Contract structures, Tenure cohorts, Internet service types, and Monthly charges.
5. **⚡ Model Performance**: Benchmark comparison table, Confusion Matrix heatmaps, multi-model ROC curve overlays, feature importance charts, and beginner metric guides.
6. **💡 Strategic Retention Insights**: Actionable, ethical frameworks translating predictions into profitable, non-discriminatory retention campaigns.

---

## 🎓 9. College Presentation & Viva Voce Talking Points

When presenting this project for your **B.Tech AI/ML Capstone / Viva**, highlight:
1. **Data Leakage Prevention**: We fitted `ColumnTransformer` (scaling and one-hot encoding) strictly on training data inside a Scikit-Learn `Pipeline`.
2. **Class Imbalance Handling**: Applied `class_weight='balanced'` and stratified splitting to guarantee fair learning of the minority churn class.
3. **Metric Justification**: Explained why ROC-AUC and Recall are the primary optimization criteria rather than misleading overall accuracy.
4. **Full-Stack Execution**: Delivered a real, usable software product featuring automated tests (`pytest`), REST-style prediction functions, and a modern Streamlit user interface.

---

## ⚖️ 10. Ethical AI & Fair Practices
- Personal identifiers (`customerID`) are permanently stripped before modeling.
- The system never uses sensitive demographic traits to deny services or unfairly penalize customers.
- All recommendations focus on product enhancements, transparent billing, and service satisfaction.
