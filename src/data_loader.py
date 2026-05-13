"""
Generate and load synthetic churn dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    N_SAMPLES,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET,
)


def generate_dataset(
    n: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic churn dataset.
    """

    np.random.seed(seed)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": np.random.randint(1, 72, n),
            "monthly_spend": np.round(
                np.random.exponential(120, n),
                2,
            ),
            "num_support_tickets": np.random.poisson(1.5, n),
            "last_login_days_ago": np.random.randint(0, 90, n),
            # Lower adoption values are
            # more common in SaaS products
            "feature_adoption_score": np.round(
                np.random.beta(2, 5, n),
                3,
            ),
            "plan_type": np.random.choice(
                ["Free", "Starter", "Pro", "Enterprise"],
                n,
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "contract_type": np.random.choice(
                ["Month-to-Month", "Annual", "Two-Year"],
                n,
                p=[0.5, 0.35, 0.15],
            ),
            "region": np.random.choice(
                ["NA", "EMEA", "APAC", "LATAM"],
                n,
            ),
            # Keep numeric for simpler preprocessing
            "has_referral": np.random.choice(
                [1, 0],
                n,
                p=[0.3, 0.7],
            ),
            "num_integrations": np.random.randint(
                0,
                15,
                n,
            ),
        }
    )

    # Churn Logic

    churn_prob = np.full(n, 0.12)

    # Inactive customers are more likely to churn
    churn_prob += np.where(
        df["last_login_days_ago"] > 30,
        0.12,
        0,
    )

    # Low feature adoption increases churn risk
    low_adoption = df["feature_adoption_score"] < 0.25

    churn_prob += np.where(
        low_adoption,
        0.10,
        0,
    )

    # Inactivity + low adoption together
    # create stronger churn behavior
    churn_prob += np.where(
        (low_adoption & (df["last_login_days_ago"] > 30)),
        0.10,
        0,
    )

    # Free users churn more often
    churn_prob += np.where(
        df["plan_type"] == "Free",
        0.08,
        0,
    )

    # Enterprise users with long contracts retain better
    churn_prob -= np.where(
        ((df["plan_type"] == "Enterprise") & (df["contract_type"] == "Two-Year")),
        0.15,
        0,
    )

    # New customers with many support tickets
    # are higher risk
    churn_prob += np.where(
        ((df["tenure_months"] < 6) & (df["num_support_tickets"] >= 3)),
        0.10,
        0,
    )

    # Longer tenure reduces churn,
    # but with diminishing returns
    churn_prob -= 0.01 * np.sqrt(df["tenure_months"])

    # Referral users retain slightly better
    churn_prob -= 0.05 * df["has_referral"]

    # Small noise improves realism
    churn_prob += np.random.normal(
        0,
        0.02,
        n,
    )

    churn_prob = churn_prob.clip(
        0.04,
        0.95,
    )

    df[TARGET] = (np.random.rand(n) < churn_prob).astype(int)

    return df


def validate_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic dataset validation.
    """

    required_columns = [
        "customer_id",
        "tenure_months",
        "monthly_spend",
        "num_support_tickets",
        "last_login_days_ago",
        "feature_adoption_score",
        "plan_type",
        "contract_type",
        "region",
        "has_referral",
        "num_integrations",
        TARGET,
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Handle missing values if any exist
    if df.isnull().sum().sum() > 0:

        print("\nMissing values detected.")

        for col in df.columns:

            if df[col].dtype == "object":

                df[col] = df[col].fillna("Unknown")

            else:

                df[col] = df[col].fillna(df[col].median())

        print("Missing values handled.")

    return df


def load_or_generate(
    path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Load dataset if it exists,
    otherwise generate a new one.
    """

    if path.exists():

        df = pd.read_csv(path)

    else:

        df = generate_dataset()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            path,
            index=False,
        )

    df = validate_dataset(df)

    return df


if __name__ == "__main__":

    df = load_or_generate()

    print("\nDataset Summary")
    print("-" * 40)

    print(f"Shape: {df.shape}")

    print(f"Churn Rate: " f"{df[TARGET].mean():.2%}")

    print("\nClass Distribution:")

    print(df[TARGET].value_counts())
