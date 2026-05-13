"""
Tests for preprocessing pipeline.
"""

import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)

from src.feature_engineering import (
    add_engineered_features,
)

from src.pipeline import (
    build_pipeline,
)

# Sample plan median values
PLAN_MEDIANS = {
    "Free": 80,
    "Starter": 120,
    "Pro": 200,
    "Enterprise": 400,
}


def sample_data():
    """
    Create small sample dataset
    for pipeline testing.
    """

    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "tenure_months": [
                12,
                24,
                6,
            ],
            "monthly_spend": [
                100,
                250,
                80,
            ],
            "num_support_tickets": [
                1,
                0,
                3,
            ],
            "last_login_days_ago": [
                5,
                12,
                45,
            ],
            "feature_adoption_score": [
                0.7,
                0.8,
                0.2,
            ],
            "plan_type": [
                "Starter",
                "Pro",
                "Free",
            ],
            "contract_type": [
                "Annual",
                "Two-Year",
                "Month-to-Month",
            ],
            "region": [
                "NA",
                "EMEA",
                "APAC",
            ],
            # Keep numeric instead of bool
            # for sklearn preprocessing
            "has_referral": [
                1,
                0,
                0,
            ],
            "num_integrations": [
                3,
                8,
                1,
            ],
            "churned": [
                0,
                0,
                1,
            ],
        }
    )


def test_pipeline_build():
    """
    Test pipeline creation.
    """

    model = LogisticRegression()

    pipeline = build_pipeline(model)

    assert pipeline is not None


def test_pipeline_fit():
    """
    Test full pipeline training.
    """

    df = sample_data()

    # Add engineered features
    df = add_engineered_features(
        df,
        PLAN_MEDIANS,
    )

    # Separate features and target
    X = df.drop(
        columns=[
            "customer_id",
            "churned",
        ]
    )

    y = df["churned"]

    model = LogisticRegression(
        max_iter=1000,
    )

    pipeline = build_pipeline(model)

    # Train pipeline
    pipeline.fit(X, y)

    # Generate predictions
    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


def test_predict_proba():
    """
    Test probability predictions.
    """

    df = sample_data()

    df = add_engineered_features(
        df,
        PLAN_MEDIANS,
    )

    X = df.drop(
        columns=[
            "customer_id",
            "churned",
        ]
    )

    y = df["churned"]

    model = LogisticRegression(
        max_iter=1000,
    )

    pipeline = build_pipeline(model)

    pipeline.fit(X, y)

    # Predict class probabilities
    probabilities = pipeline.predict_proba(X)

    # Should return probabilities
    # for both classes
    assert probabilities.shape[1] == 2
