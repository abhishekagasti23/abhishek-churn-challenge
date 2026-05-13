"""
Tests for feature engineering functions.
"""

import pandas as pd

from src.feature_engineering import (
    add_engineered_features,
    engagement_score,
    loyalty_score,
    risk_flag,
    spend_vs_plan,
    ticket_rate,
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
    Create sample customer data
    for testing.
    """

    return pd.DataFrame(
        {
            "customer_id": ["C0001"],

            "tenure_months": [12],

            "monthly_spend": [150],

            "num_support_tickets": [2],

            "last_login_days_ago": [35],

            "feature_adoption_score": [0.2],

            "plan_type": ["Starter"],

            "contract_type": ["Annual"],

            "region": ["NA"],

            "has_referral": [True],

            "num_integrations": [3],

            "churned": [1],
        }
    )


def test_engagement_score():
    """
    Test engagement score output.
    """

    df = sample_data()

    result = engagement_score(df)

    assert len(result) == 1

    assert result.iloc[0] >= 0


def test_ticket_rate():
    """
    Test support ticket rate formula.
    """

    df = sample_data()

    result = ticket_rate(df)

    expected = 2 / (12 + 1)

    assert round(result.iloc[0], 4) == round(
        expected,
        4,
    )


def test_spend_vs_plan():
    """
    Test customer spend
    relative to plan median.
    """

    df = sample_data()

    result = spend_vs_plan(
        df,
        PLAN_MEDIANS,
    )

    expected = 150 / 120

    assert round(result.iloc[0], 4) == round(
        expected,
        4,
    )


def test_loyalty_score():
    """
    Test loyalty score output.
    """

    df = sample_data()

    result = loyalty_score(df)

    assert result.iloc[0] > 0


def test_risk_flag():
    """
    Test churn risk flag logic.
    """

    df = sample_data()

    result = risk_flag(df)

    assert result.iloc[0] == 1


def test_add_engineered_features():
    """
    Ensure all engineered
    features are added.
    """

    df = sample_data()

    result = add_engineered_features(
        df,
        PLAN_MEDIANS,
    )

    expected_columns = [
        "engagement_score",
        "ticket_rate",
        "spend_vs_plan",
        "loyalty_score",
        "risk_flag",
    ]

    for col in expected_columns:

        assert col in result.columns