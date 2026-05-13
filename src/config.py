"""
Project configuration.
"""

from pathlib import Path


# Paths


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = (
    RAW_DATA_DIR / "churn_data.csv"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "processed_churn_data.csv"
)

MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = (
    MODELS_DIR / "best_model.pkl"
)

PIPELINE_PATH = (
    MODELS_DIR / "pipeline.pkl"
)

REPORTS_DIR = ROOT_DIR / "reports"

EDA_FIGURES_DIR = (
    REPORTS_DIR / "eda_figures"
)

SHAP_FIGURES_DIR = (
    REPORTS_DIR / "shap_figures"
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts"


# General Settings


RANDOM_SEED = 42

N_SAMPLES = 50000

TARGET = "churned"


# Train / Validation / Test Split


TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# Features

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "num_support_tickets",
    "last_login_days_ago",
    "feature_adoption_score",
    "num_integrations",
]

CATEGORICAL_FEATURES = [
    "plan_type",
    "contract_type",
    "region",
]

BOOLEAN_FEATURES = [
    "has_referral",
]

ENGINEERED_FEATURES = [
    "engagement_score",
    "ticket_rate",
    "spend_vs_plan",
    "loyalty_score",
    "risk_flag",
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BOOLEAN_FEATURES
    + ENGINEERED_FEATURES
)

# Model Settings

SUPPORTED_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "neural_network",
]

N_TRIALS = 50

CV_FOLDS = 5

# API Settings

API_HOST = "0.0.0.0"

API_PORT = 8000
