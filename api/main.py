from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Predictions API", version="1.0.0")

MODEL_DIR = "./models"
churn_model = None
churn_scaler = None
ltv_model = None
ltv_scaler = None
rec_model = None

@app.on_event("startup")
async def load_models():
    global churn_model, churn_scaler, ltv_model, ltv_scaler, rec_model
    logger.info("Loading models...")
    try:
        if os.path.exists(f"{MODEL_DIR}/churn_model.pkl"):
            with open(f"{MODEL_DIR}/churn_model.pkl", 'rb') as f:
                churn_data = pickle.load(f)
                churn_model = churn_data['model']
                churn_scaler = churn_data['scaler']
            logger.info("Churn model loaded")
        if os.path.exists(f"{MODEL_DIR}/ltv_model.pkl"):
            with open(f"{MODEL_DIR}/ltv_model.pkl", 'rb') as f:
                ltv_data = pickle.load(f)
                ltv_model = ltv_data['model']
                ltv_scaler = ltv_data['scaler']
            logger.info("LTV model loaded")
        if os.path.exists(f"{MODEL_DIR}/recommendation_model.pkl"):
            with open(f"{MODEL_DIR}/recommendation_model.pkl", 'rb') as f:
                rec_model = pickle.load(f)
            logger.info("Recommendation model loaded")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

class CustomerFeatures(BaseModel):
    customer_id: str
    total_spending: float
    total_orders: int
    avg_order_value: float
    days_since_last_purchase: int
    customer_tenure_days: int
    distinct_categories_purchased: int
    completed_orders: int
    cancelled_orders: int
    completion_rate: float

@app.get("/")
async def root():
    return {"message": "ML Predictions API", "docs": "http://localhost:8000/docs"}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if all([churn_model, ltv_model, rec_model]) else "degraded",
        "models_loaded": {
            "churn": churn_model is not None,
            "ltv": ltv_model is not None,
            "recommendations": rec_model is not None
        }
    }

@app.post("/predict/churn")
async def predict_churn(features: CustomerFeatures):
    if churn_model is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    
    feature_list = [
        features.total_spending, features.total_orders, features.avg_order_value,
        features.days_since_last_purchase, features.customer_tenure_days,
        features.distinct_categories_purchased, features.completed_orders,
        features.cancelled_orders, features.completion_rate
    ]
    
    X = np.array(feature_list).reshape(1, -1)
    X_scaled = churn_scaler.transform(X)
    churn_proba = churn_model.predict_proba(X_scaled)[0, 1]
    
    if churn_proba < 0.3:
        risk_level = "Low"
    elif churn_proba < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        "customer_id": features.customer_id,
        "churn_probability": round(float(churn_proba), 4),
        "churn_risk": risk_level
    }

@app.post("/predict/ltv")
async def predict_ltv(features: CustomerFeatures):
    if ltv_model is None:
        raise HTTPException(status_code=503, detail="LTV model not loaded")
    
    feature_list = [
        features.total_spending, features.total_orders, features.avg_order_value,
        features.days_since_last_purchase, features.customer_tenure_days,
        features.distinct_categories_purchased, features.completed_orders,
        features.cancelled_orders, features.completion_rate
    ]
    
    X = np.array(feature_list).reshape(1, -1)
    X_scaled = ltv_scaler.transform(X)
    ltv_pred = ltv_model.predict(X_scaled)[0]
    
    return {
        "customer_id": features.customer_id,
        "predicted_ltv": round(float(ltv_pred), 2)
    }

@app.post("/predict/recommendations")
async def predict_recommendations(features: CustomerFeatures):
    if rec_model is None:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded")
    
    product_popularity = rec_model['product_popularity']
    top_indices = np.argsort(product_popularity)[-3:][::-1]
    recommendations = [f"P{idx:03d}" for idx in top_indices]
    
    return {
        "customer_id": features.customer_id,
        "recommended_products": recommendations
    }

@app.post("/predict/all")
async def predict_all(features: CustomerFeatures):
    churn_result = await predict_churn(features)
    ltv_result = await predict_ltv(features)
    rec_result = await predict_recommendations(features)
    
    return {
        "customer_id": features.customer_id,
        "churn": churn_result,
        "ltv": ltv_result,
        "recommendations": rec_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
