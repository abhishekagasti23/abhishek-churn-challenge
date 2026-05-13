"""
Feature engineering functions for churn prediction.
"""

import numpy as np
import pandas as pd


def engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Customer engagement score based on:
    - product adoption
    - login activity
    - integrations
    """

    # Smooth adoption distribution
    adoption = np.log1p(df["feature_adoption_score"] * 10) / np.log1p(10)

    # More recent activity = higher score
    login_score = 1 - (df["last_login_days_ago"] / df["last_login_days_ago"].max())

    # First few integrations matter most
    integration_score = np.sqrt(df["num_integrations"]) / np.sqrt(
        df["num_integrations"].max()
    )

    score = 0.5 * adoption + 0.3 * login_score + 0.2 * integration_score

    return score.rename("engagement_score")


def ticket_rate(df: pd.DataFrame) -> pd.Series:
    """
    Support tickets relative to tenure.
    """

    rate = df["num_support_tickets"] / (df["tenure_months"] + 1)

    # Compress large outliers
    return np.log1p(rate).rename("ticket_rate")


def spend_vs_plan(
    df: pd.DataFrame,
    plan_medians: dict,
) -> pd.Series:
    """
    Compare customer spend
    against plan median.
    """

    median_spend = df["plan_type"].map(plan_medians).replace(0, 1)

    ratio = df["monthly_spend"] / median_spend

    return ratio.rename("spend_vs_plan")


def loyalty_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Loyalty score based on:
    - contract type
    - tenure
    - referrals
    """

    contract_map = {
        "Month-to-Month": 1,
        "Annual": 2,
        "Two-Year": 3,
    }

    contract_weight = df["contract_type"].map(contract_map)

    # Long-term + referred users
    # tend to retain better
    score = (
        contract_weight * np.log1p(df["tenure_months"]) * (1 + 0.2 * df["has_referral"])
    )

    return score.rename("loyalty_score")


def risk_flag(df: pd.DataFrame) -> pd.Series:
    """
    Flag customers with
    multiple churn signals.
    """

    at_risk = (
        (df["last_login_days_ago"] > 30)
        & (df["feature_adoption_score"] < 0.25)
        & (df["num_support_tickets"] >= 2)
    )

    high_risk = (
        at_risk
        & (df["plan_type"] == "Free")
        & (df["contract_type"] == "Month-to-Month")
    )

    # 0 = low risk
    # 1 = at risk
    # 2 = high risk
    flag = at_risk.astype(int) + high_risk.astype(int)

    return flag.rename("risk_flag")


def add_engineered_features(
    df: pd.DataFrame,
    plan_medians: dict,
) -> pd.DataFrame:
    """
    Add engineered features
    to the dataset.
    """

    df = df.copy()

    df["engagement_score"] = engagement_score(df)

    df["ticket_rate"] = ticket_rate(df)

    df["spend_vs_plan"] = spend_vs_plan(
        df,
        plan_medians,
    )

    df["loyalty_score"] = loyalty_score(df)

    df["risk_flag"] = risk_flag(df)

    return df
