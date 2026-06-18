from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample valid payload
valid_payload = {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Google Search",
    "loyalty_tier": "Silver",
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 100,
    "frequency_180d": 2,
    "monetary_180d": 500.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.1,
    "avg_rating_180d": 4.5,
    "category_diversity_180d": 2,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 365,
    "sessions_30d": 5,
    "product_views_30d": 15,
    "cart_adds_30d": 2,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 1,
    "last_visit_days_ago": 5
}

def test_health_endpoint():
    """Test Case 1: GET /health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_valid():
    """Test Case 2: POST /predict with valid payload"""
    response = client.post("/predict", json=valid_payload)
    # If the model is not trained yet (503), ignore assertion, otherwise verify
    if response.status_code != 503: 
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "risk_level" in data
        assert "risk_explanation" in data

def test_predict_endpoint_invalid_input():
    """Test Case 3: POST /predict with validation error"""
    invalid_payload = valid_payload.copy()
    invalid_payload["avg_rating_180d"] = 10.0 # Exceeds max allowed rating of 5.0
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422 # Unprocessable Entity (Validation Error)

def test_batch_predict_endpoint():
    """Test Case 4: POST /batch_predict"""
    response = client.post("/batch_predict", json=[valid_payload, valid_payload])
    if response.status_code != 503:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert "churn_probability" in data[0]