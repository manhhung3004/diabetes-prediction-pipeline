from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for diabetes prediction based on health metrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load("models/saved/diabetes_model.joblib")
    scaler = joblib.load("models/saved/scaler.joblib")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None

class HealthMetrics(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Diabetes Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(metrics: HealthMetrics):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([metrics.dict()])
        
        # Scale the features
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        logger.info(f"Prediction made: {prediction}, Probability: {probability:.3f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 3),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age"
        ],
        "model_loaded": True,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
