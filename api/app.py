"""
FastAPI app for churn prediction.
"""

import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException

from api.middleware import (
    RequestLoggingMiddleware,
)

from api.schemas import (
    ChurnRequest,
    ChurnResponse,
    HealthResponse,
)

from src.explain import (
    explain_prediction,
)

from src.feature_engineering import (
    add_engineered_features,
)

from src.predict import (
    assign_risk,
    load_pipeline,
)

## FastAPI app

app = FastAPI(
    title="UrbanPulse Churn API",
    description=(
        "API for predicting SaaS customer churn " "using machine learning models."
    ),
    version="1.0.0",
)

## Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)


## Load trained model pipeline

pipeline = None

try:

    pipeline = load_pipeline()

    print("\nPipeline loaded successfully.")

except Exception as e:

    print(f"\nFailed to load pipeline: {e}")

## Root endpoint


@app.get("/")
async def root():
    """
    Root API endpoint.
    """

    return {"message": "UrbanPulse Churn API is running."}


## Health check endpoint


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():
    """
    Simple API health check.
    """

    status = "ok" if pipeline is not None else "error"

    return {
        "status": status,
    }


## Prediction endpoint


@app.post(
    "/predict",
    response_model=ChurnResponse,
)
async def predict(
    request: ChurnRequest,
):
    """
    Predict customer churn probability.
    """

    # Ensure model is loaded
    if pipeline is None:

        raise HTTPException(
            status_code=500,
            detail="Model not loaded.",
        )

    # Convert request into DataFrame
    df = pd.DataFrame([request.model_dump()])

    # Calculate plan medians
    # for engineered features
    plan_medians = df.groupby("plan_type")["monthly_spend"].median().to_dict()

    # Add engineered features
    df_features = add_engineered_features(
        df,
        plan_medians,
    )

    # Remove columns not used by model
    X = df_features.drop(
        columns=["customer_id"],
    )

    # Predict churn probability
    probability = float(pipeline.predict_proba(X)[0][1])

    # Convert probability to binary label
    prediction = int(probability >= 0.5)

    # Assign business risk level
    risk_level = assign_risk(probability)

    # Get top feature influencing prediction
    top_driver = explain_prediction(
        pipeline,
        X,
    )

    return {
        "customer_id": request.customer_id,
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "top_driver": top_driver,
    }


if __name__ == "__main__":

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
