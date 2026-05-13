"""
Batch prediction script.
"""

import argparse
import joblib
import pandas as pd

from src.config import (
    PIPELINE_PATH,
)

from src.feature_engineering import (
    add_engineered_features,
)


def assign_risk(probability):
    """
    Convert churn probability
    into business risk level.
    """

    if probability < 0.30:
        return "Low"

    elif probability < 0.55:
        return "Medium"

    elif probability < 0.75:
        return "High"

    return "Critical"


def load_pipeline():
    """
    Load trained pipeline.
    """

    return joblib.load(
        PIPELINE_PATH
    )


def predict(
    input_path,
    output_path,
):
    """
    Run batch prediction
    on customer dataset.
    """

    # Load input dataset
    df = pd.read_csv(
        input_path
    )

    # Convert column back to boolean
    if "has_referral" in df.columns:

        df["has_referral"] = (
            df["has_referral"].astype(bool)
        )

    # Calculate plan medians
    # for engineered features
    plan_medians = (
        df.groupby("plan_type")[
            "monthly_spend"
        ]
        .median()
        .to_dict()
    )

    # Add engineered features
    df_features = (
        add_engineered_features(
            df,
            plan_medians,
        )
    )

    # Remove columns not used
    # during model training
    drop_cols = [
        "customer_id",
    ]

    if "churned" in df_features.columns:
        drop_cols.append("churned")

    X = df_features.drop(
        columns=drop_cols,
    )

    # Load trained pipeline
    pipeline = load_pipeline()

    # Predict churn probabilities
    probabilities = (
        pipeline.predict_proba(X)[:, 1]
    )

    # Convert probabilities into
    # binary churn predictions
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Build final output
    output = df.copy()

    output["churn_probability"] = (
        probabilities.round(4)
    )

    output["prediction"] = predictions

    output["risk_level"] = [
        assign_risk(p)
        for p in probabilities
    ]

    # Save predictions
    output.to_csv(
        output_path,
        index=False,
    )

    print("\nPredictions saved successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    predict(
        args.input,
        args.output,
    )