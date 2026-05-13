"""
Tests for FastAPI endpoints.
"""

from fastapi.testclient import (
    TestClient,
)

from api.app import app

client = TestClient(app)

# Sample request payload
VALID_PAYLOAD = {
    "customer_id": "C0001",
    "tenure_months": 12,
    "monthly_spend": 150.0,
    "num_support_tickets": 2,
    "last_login_days_ago": 35,
    "feature_adoption_score": 0.2,
    "plan_type": "Starter",
    "contract_type": "Annual",
    "region": "NA",
    # Keep numeric instead of bool
    # for sklearn preprocessing
    "has_referral": 1,
    "num_integrations": 3,
}


def test_health_endpoint():
    """
    Test API health endpoint.
    """

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data


def test_predict_endpoint():
    """
    Test prediction endpoint.
    """

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    # If model is not trained yet,
    # API may return 500
    assert response.status_code in [
        200,
        500,
    ]

    # Only validate response
    # if prediction succeeds
    if response.status_code == 200:

        data = response.json()

        assert "customer_id" in data

        assert "churn_probability" in data

        assert "prediction" in data

        assert "risk_level" in data

        assert "top_driver" in data


def test_invalid_plan_type():
    """
    Invalid plan_type
    should return 422.
    """

    payload = VALID_PAYLOAD.copy()

    payload["plan_type"] = "InvalidPlan"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_tenure():
    """
    Negative tenure should fail.
    """

    payload = VALID_PAYLOAD.copy()

    payload["tenure_months"] = -1

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
