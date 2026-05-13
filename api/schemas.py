"""
Pydantic schemas for API requests and responses.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    """
    Input schema for churn prediction.
    """

    # Customer identifier
    customer_id: str

    # Customer tenure in months
    tenure_months: int = Field(
        ge=1,
        le=120,
    )

    # Monthly subscription spend
    monthly_spend: float = Field(
        ge=0,
    )

    # Number of support tickets raised
    num_support_tickets: int = Field(
        ge=0,
    )

    # Days since last login
    last_login_days_ago: int = Field(
        ge=0,
    )

    # Product feature usage score
    feature_adoption_score: float = Field(
        ge=0,
        le=1,
    )

    # Subscription plan
    plan_type: Literal[
        "Free",
        "Starter",
        "Pro",
        "Enterprise",
    ]

    # Contract type
    contract_type: Literal[
        "Month-to-Month",
        "Annual",
        "Two-Year",
    ]

    # Customer region
    region: Literal[
        "NA",
        "EMEA",
        "APAC",
        "LATAM",
    ]

    # Whether customer joined via referral
    has_referral: int = Field(
        ge=0,
        le=1,
    )

    # Connected integrations
    num_integrations: int = Field(
        ge=0,
    )

    # Example request shown in Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C00042",
                "tenure_months": 8,
                "monthly_spend": 95.5,
                "num_support_tickets": 3,
                "last_login_days_ago": 42,
                "feature_adoption_score": 0.18,
                "plan_type": "Starter",
                "contract_type": "Month-to-Month",
                "region": "NA",
                "has_referral": False,
                "num_integrations": 1,
            }
        }
    }


class ChurnResponse(BaseModel):
    """
    Output schema returned by API.
    """

    customer_id: str

    # Predicted churn probability
    churn_probability: float

    # Binary prediction
    prediction: int

    # Business risk label
    risk_level: str

    # Main feature affecting prediction
    top_driver: str


class HealthResponse(BaseModel):
    """
    API health check response.
    """

    status: str
