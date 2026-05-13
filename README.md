# UrbanPulse Churn Prediction Challenge

End-to-end churn prediction system for a SaaS platform using machine learning, explainability, FastAPI deployment, and a Streamlit dashboard.

This project was completed as part of the UrbanPulse Data Scientist / AI Engineer Build Challenge. The goal was to build a complete churn prediction workflow starting from synthetic data generation through model deployment and business-facing insights.

---

> Full end-to-end pipeline reproduction:
>
> ```bash
> python run_pipeline.py
> ```

---

# Quick Start (Reproduce Results)

After installing dependencies, the full workflow can be reproduced end-to-end with a single command:

```bash
python run_pipeline.py
```

This command will:
- generate/load the dataset
- run preprocessing and feature engineering
- train and evaluate models
- generate SHAP explainability outputs
- save the trained pipeline artifact

Optional commands:

Run FastAPI API:

```bash
uvicorn api.app:app --reload
```

Run Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Open FastAPI Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

# Project Overview

The project focuses on predicting customer churn for a mid-sized SaaS platform and identifying the main reasons customers are likely to leave.

The workflow includes:
- Synthetic dataset generation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and comparison
- SHAP explainability
- FastAPI deployment
- Streamlit dashboard
- Basic production-readiness considerations


The final selected model was a Random Forest classifier, which produced the best balance between ROC-AUC, recall, and overall stability across validation and test sets.

Final test performance:

| Metric | Score |
|---|---|
| ROC-AUC | ~0.70 |
| Recall (Churn Class) | ~0.59 |
| F1 Score | ~0.47 |

The strongest churn indicators were:
- low engagement
- long inactivity periods
- low feature adoption
- support burden for newer customers

---

# Repository Structure

```text
abhishek-churn-challenge/
│
├── README.md
├── report.pdf
├── requirements.txt
├── run_pipeline.py
│
├── data/
│   └── churn_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
│
├── api/
│   ├── app.py
│   ├── schemas.py
│   └── middleware.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── pipeline.pkl
│
├── reports/
│   └── shap_summary.png
│
├── tests/
│   └── test_pipeline.py
│
└── .github/
    └── workflows/
        └── python-ci.yml
```

---

# Approach Summary

The project was built in three main stages:

1. Understand the churn patterns through EDA and business-focused analysis
2. Engineer behavioural features that better capture engagement and churn risk
3. Train and compare multiple models while balancing predictive performance with explainability

The focus was not only on model accuracy, but also on producing business-friendly insights and a deployable workflow.

---

# Dataset

The dataset is synthetic and simulates SaaS customer behaviour.

Main features include:
- tenure_months
- monthly_spend
- num_support_tickets
- last_login_days_ago
- feature_adoption_score
- plan_type
- contract_type
- region
- has_referral
- num_integrations

Target:
- churned (1 = churned, 0 = retained)

The dataset was expanded to 50,000 rows and additional churn logic was introduced to create more realistic behavioural patterns.

---

# Feature Engineering

Five engineered features were created:

| Feature | Purpose |
|---|---|
| engagement_score | Captures overall customer activity |
| ticket_rate | Support burden relative to tenure |
| spend_vs_plan | Spending relative to plan peers |
| loyalty_score | Long-term commitment signal |
| risk_flag | Rule-based churn warning flag |

These engineered features improved model performance significantly compared to using only the raw dataset.

---

# Models Used

The following models were trained and compared:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Tree ensemble benchmark |
| XGBoost | Gradient boosting model |
| LightGBM | Final selected model |
| Neural Network | Bonus comparison model |

Random Forest produced the best overall performance while remaining stable and interpretable across validation and test sets.

---

# Explainability

SHAP was used to explain:
- global feature importance
- individual predictions
- business-level churn drivers

Top churn drivers:
1. engagement_score
2. last_login_days_ago
3. feature_adoption_score

The explainability workflow helped connect model predictions to practical retention recommendations.

---

# Dashboard

A lightweight Streamlit dashboard was built to:
- visualise churn insights
- upload customer data
- generate predictions
- display churn risk explanations

Run dashboard:

```bash
streamlit run dashboard/app.py
```

---

# FastAPI Endpoint

The final model was deployed using FastAPI.

API features:
- JSON prediction endpoint
- churn probability
- binary churn prediction
- risk level
- top churn driver
- request validation
- logging middleware

Run API:

```bash
uvicorn api.app:app --reload
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "customer_id": "CUST_1001",
  "tenure_months": 8,
  "monthly_spend": 49.99,
  "num_support_tickets": 3,
  "last_login_days_ago": 45,
  "feature_adoption_score": 0.18,
  "plan_type": "Free",
  "contract_type": "Month-to-Month",
  "region": "NA",
  "has_referral": 0,
  "num_integrations": 1
}
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd abhishek-churn-challenge
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Jupyter Notebooks

```bash
jupyter notebook
```

Then open:
- notebooks/01_eda.ipynb
- notebooks/02_feature_engineering.ipynb
- notebooks/03_modelling.ipynb

---

## Train Models

```bash
python -m src.train
```

This will:
- load/generate data
- engineer features
- train models
- evaluate performance
- save the trained pipeline

---

## Run Full Pipeline

```bash
python run_pipeline.py
```

---

## Run FastAPI

```bash
uvicorn api.app:app --reload
```

---

## Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Reproducing Results

To reproduce the project from a fresh environment:

```bash
python run_pipeline.py
```

This recreates:
- dataset generation
- feature engineering
- model training
- evaluation
- SHAP outputs
- serialized pipeline artifacts

---

# Production Readiness Notes

The project includes basic production-oriented considerations such as:
- reusable sklearn pipelines
- FastAPI deployment
- request validation
- middleware logging
- explainability support
- model serialization
- lightweight CI workflow using GitHub Actions
- monitoring and drift discussion
- integration testing structure

Potential future improvements:
- MLflow experiment tracking
- Grafana monitoring dashboards
- automated retraining
- threshold optimisation
- drift detection automation
- cloud deployment

---

# Limitations

Some important limitations:
- The dataset is synthetic rather than real production data
- Hyperparameter tuning was limited
- No full Optuna optimisation run was completed
- Threshold optimisation was not implemented
- Monitoring setup is still lightweight
- spend_vs_plan has an issue during single-record API inference

These trade-offs and future improvements are discussed further in the written report.

---

# AI Tool Disclosure

Throughout development, I used AI tools in a focused way for debugging support, brainstorming implementation strategies, refactoring parts of the codebase, and improving documentation clarity. I also referred to official documentation and public resources for libraries such as scikit-learn, FastAPI, SHAP, LightGBM, XGBoost, Streamlit, and TensorFlow/Keras, especially for preprocessing workflows, API usage patterns, explainability setup, and library-specific syntax.

The project itself was developed incrementally with repeated testing, validation, and version-controlled updates across the modelling and integration stages. Consulting documentation and reference materials is a routine part of machine learning and software development, similar to reviewing Stack Overflow discussions or official library examples while working through implementation details.

All AI-generated outputs were reviewed, modified, and integrated manually before use. The overall project design, experimentation, feature engineering, modelling decisions, debugging, integration work, and final implementation were completed and validated by me throughout the project.

This challenge focused heavily on reasoning, experimentation, communication, and practical problem-solving rather than just code generation. While AI tools and documentation resources helped speed up parts of the workflow, I still needed to independently design the churn logic, engineer behavioural features, compare modelling approaches, interpret SHAP outputs, debug API integration issues, and connect the technical work back to business-facing recommendations.

---

# Report

The complete written report is included in:

```text
report.pdf
```

The report covers:
- modelling decisions
- trade-offs
- explainability
- deployment considerations
- thought-process questions
- limitations and future improvements

---

# Submission Notes

This project was developed as a take-home challenge focused on:
- problem-solving
- reasoning
- experimentation
- explainability
- communication
- production awareness

The goal was to build a realistic and well-reasoned end-to-end machine learning workflow rather than an overly engineered production system.