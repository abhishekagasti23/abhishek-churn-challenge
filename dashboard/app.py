import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Local project imports

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import TARGET

from src.feature_engineering import (
    add_engineered_features,
)

# Streamlit page settings

st.set_page_config(
    page_title="UrbanPulse Churn Dashboard",
    layout="wide",
)

sns.set_style("whitegrid")

# Load churn dataset


@st.cache_data
def load_data():

    data_path = ROOT_DIR / "data" / "raw" / "churn_data.csv"

    df = pd.read_csv(data_path)

    # Needed for spend_vs_plan calculation
    plan_medians = df.groupby("plan_type")["monthly_spend"].median().to_dict()

    df = add_engineered_features(
        df,
        plan_medians,
    )

    return df


# Load trained pipeline


@st.cache_resource
def load_pipeline():

    pipeline_path = ROOT_DIR / "models" / "pipeline.pkl"

    return joblib.load(pipeline_path)


# Dashboard layout

st.title("UrbanPulse Churn Prediction Dashboard")

st.markdown("""
This dashboard gives a quick overview of the churn prediction project,
including churn patterns, feature engineering insights,
and a few live sample predictions.
""")

# Load data and model

df = load_data()

pipeline = load_pipeline()

# Top-level project stats

st.subheader("Project Overview")

churn_rate = df[TARGET].mean() * 100

high_risk_pct = (df["risk_flag"] == 2).mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{len(df):,}",
)

col2.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%",
)

col3.metric(
    "Best Model",
    "Random Forest",
)

col4.metric(
    "High Risk Users",
    f"{high_risk_pct:.1f}%",
)

# Churn distribution

st.subheader("Churn Distribution")

fig, ax = plt.subplots(figsize=(5, 3))

counts = df[TARGET].value_counts()

ax.bar(
    ["Retained", "Churned"],
    counts.values,
)

ax.set_ylabel("Customers")

ax.set_title("Class Distribution")

st.pyplot(fig)

# Churn by plan type

st.subheader("Churn by Plan Type")

plan_churn = df.groupby("plan_type")[TARGET].mean().sort_values(ascending=False) * 100

fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(
    plan_churn.index,
    plan_churn.values,
)

ax.set_ylabel("Churn Rate (%)")

ax.set_title("Plan Type vs Churn")

st.pyplot(fig)

# Churn by contract type

st.subheader("Churn by Contract Type")

contract_churn = (
    df.groupby("contract_type")[TARGET].mean().sort_values(ascending=False) * 100
)

fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(
    contract_churn.index,
    contract_churn.values,
)

ax.set_ylabel("Churn Rate (%)")

ax.set_title("Contract Type vs Churn")

st.pyplot(fig)

# Common churn signals

st.subheader("Key Churn Drivers")

st.markdown("""
Top churn signals observed in the project:

- Long inactivity periods
- Low feature adoption
- Free plan users
- Month-to-Month contracts
- High support burden

Random Forest ended up performing best overall once the engineered
behavioural features were added.
""")

# Try a sample prediction

st.subheader("Sample Prediction")

with st.form("prediction_form"):

    tenure = st.slider(
        "Tenure (months)",
        1,
        72,
        12,
    )

    spend = st.number_input(
        "Monthly Spend",
        value=120.0,
    )

    tickets = st.slider(
        "Support Tickets",
        0,
        10,
        2,
    )

    login_days = st.slider(
        "Last Login Days Ago",
        0,
        90,
        20,
    )

    adoption = st.slider(
        "Feature Adoption Score",
        0.0,
        1.0,
        0.5,
    )

    plan_type = st.selectbox(
        "Plan Type",
        ["Free", "Starter", "Pro", "Enterprise"],
    )

    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-Month", "Annual", "Two-Year"],
    )

    region = st.selectbox(
        "Region",
        ["NA", "EMEA", "APAC", "LATAM"],
    )

    referral = st.selectbox(
        "Has Referral",
        ["Yes", "No"],
    )

    referral = 1 if referral == "Yes" else 0

    integrations = st.slider(
        "Integrations",
        0,
        15,
        3,
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:

    sample = pd.DataFrame(
        [
            {
                "customer_id": "demo_user",
                "tenure_months": tenure,
                "monthly_spend": spend,
                "num_support_tickets": tickets,
                "last_login_days_ago": login_days,
                "feature_adoption_score": adoption,
                "plan_type": plan_type,
                "contract_type": contract_type,
                "region": region,
                "has_referral": referral,
                "num_integrations": integrations,
            }
        ]
    )

    plan_medians = df.groupby("plan_type")["monthly_spend"].median().to_dict()

    sample = add_engineered_features(
        sample,
        plan_medians,
    )

    X = sample.drop(columns=["customer_id"])

    probability = pipeline.predict_proba(X)[0][1]

    st.success(f"Predicted Churn Probability: {probability:.2%}")

    if probability >= 0.7:

        st.error("High churn risk")

    elif probability >= 0.4:

        st.warning("Moderate churn risk")

    else:

        st.info("Low churn risk")

st.markdown("---")

st.markdown("Built as part of the UrbanPulse churn prediction project.")
