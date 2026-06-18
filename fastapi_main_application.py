import os
import joblib
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Setup App
app = FastAPI(
    title="D2C Customer Churn API",
    description="Predicts the likelihood of a customer churning in the next 60 days.",
    version="1.0.0"
)

# 2. Global Model Loading
MODEL_PATH = "model.pkl"
model = None
OPTIMAL_THRESHOLD = 0.35 # Based on Part 3 calibration

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Run train_model.py first.")

# 3. Pydantic Input Schemas
class CustomerFeatures(BaseModel):
    city_tier: str = Field(..., example="Tier 1")
    age_group: str = Field(..., example="25-34")
    acquisition_channel: str = Field(..., example="Instagram")
    loyalty_tier: str = Field(..., example="Silver")
    preferred_category: str = Field(..., example="Skin Care")
    marketing_consent: str = Field(..., example="Yes")
    
    recency_days: int = Field(..., ge=0, example=120)
    frequency_180d: int = Field(..., ge=0, example=2)
    monetary_180d: float = Field(..., ge=0.0, example=850.50)
    return_rate_180d: float = Field(..., ge=0.0, le=1.0, example=0.0)
    avg_discount_pct_180d: float = Field(..., ge=0.0, le=1.0, example=0.25)
    avg_rating_180d: float = Field(..., ge=1.0, le=5.0, example=3.5)
    category_diversity_180d: int = Field(..., ge=1, example=2)
    
    ticket_count_90d: int = Field(..., ge=0, example=1)
    negative_ticket_rate_90d: float = Field(..., ge=0.0, le=1.0, example=1.0)
    avg_resolution_hours_90d: float = Field(..., ge=0.0, example=24.5)
    
    days_since_signup: int = Field(..., ge=0, example=300)
    sessions_30d: int = Field(..., ge=0, example=2)
    product_views_30d: int = Field(..., ge=0, example=5)
    cart_adds_30d: int = Field(..., ge=0, example=0)
    wishlist_adds_30d: int = Field(..., ge=0, example=0)
    abandoned_carts_30d: int = Field(..., ge=0, example=0)
    email_opens_30d: int = Field(..., ge=0, example=1)
    campaign_clicks_30d: int = Field(..., ge=0, example=0)
    last_visit_days_ago: int = Field(..., ge=0, example=15)

# Output Schema
class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str

# 4. Helper Logic
def generate_risk_explanation(features: CustomerFeatures, prob: float) -> str:
    """Generates a dynamic human-readable explanation based on input features."""
    reasons = []
    if features.recency_days > 90:
        reasons.append("high recency (lack of recent orders)")
    if features.ticket_count_90d >= 1 and features.negative_ticket_rate_90d > 0.5:
        reasons.append("negative support experience")
    if features.return_rate_180d >= 0.3:
        reasons.append("high product return rate")
    if features.sessions_30d < 2:
        reasons.append("low recent web engagement")
    if features.avg_discount_pct_180d > 0.4:
        reasons.append("high discount dependency")

    if not reasons:
        reasons.append("general interaction of behavioral factors")

    explanation = f"Model predicts a {prob:.0%} chance of churn due to: " + ", ".join(reasons) + "."
    return explanation

def predict_single(features: CustomerFeatures) -> ChurnPredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
    
    # Convert to DataFrame matching training format
    df = pd.DataFrame([features.dict()])
    
    # Predict
    try:
        prob = float(model.predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Apply Thresholding
    pred_class = 1 if prob >= OPTIMAL_THRESHOLD else 0
    
    # Determine Risk Level
    if prob >= 0.60:
        risk_level = "high"
    elif prob >= OPTIMAL_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"
        
    explanation = generate_risk_explanation(features, prob)
    
    return ChurnPredictionResponse(
        churn_probability=round(prob, 4),
        predicted_class=pred_class,
        risk_level=risk_level,
        risk_explanation=explanation
    )

# 5. Endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(features: CustomerFeatures):
    return predict_single(features)

@app.post("/batch_predict", response_model=List[ChurnPredictionResponse])
def batch_predict(features_list: List[CustomerFeatures]):
    return [predict_single(feat) for feat in features_list]