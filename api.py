# ============================================================
# api.py — FastAPI application for BTC model inference
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from src.inference.predict import predict_single, predict_latest


app = FastAPI(
    title="Bitcoin Direction Prediction API",
    description="LightGBM Soft-Regime model with HMM regime probabilities",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    data: Dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "LightGBM Soft-Regime",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    return predict_single(request.data)


@app.get("/predict/latest")
def latest_prediction():
    return predict_latest()
