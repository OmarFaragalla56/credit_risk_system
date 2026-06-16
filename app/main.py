"""
main.py
-------
FastAPI entry point for the Credit Risk Intelligence API.

Architecture:
  Streamlit Frontend → POST /predict → InferenceEngine
                                         ├── master_pipeline.pkl  (preprocessing)
                                         └── champion_xgboost.pkl (prediction + SHAP)
"""

import logging
import os

from fastapi import FastAPI, HTTPException

from .custom_transformers import CreditRiskFeatureEngineer  # noqa: F401 (required for pickle)
from .engine import InferenceEngine
from .schemas import BorrowerInput, PredictionResponse

# Configure structured logging for production visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Intelligence API",
    description=(
        "Production-grade MLOps API for Credit Default Risk Evaluation. "
        "Uses a zero-leakage preprocessing pipeline and an Optuna-optimized XGBoost champion model."
    ),
    version="2.0.0",
)

# --- Model Loading ---
# Paths are relative to the project root (where docker-compose.yml lives).
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "master_pipeline.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_xgboost.pkl")

engine = InferenceEngine(pipeline_path=PIPELINE_PATH, model_path=MODEL_PATH)
logger.info("InferenceEngine loaded and ready.")

RISK_THRESHOLD = 0.30


@app.get("/", tags=["Health"])
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "operational",
        "message": "Credit Risk API v2.0 is running. Visit /docs for the Swagger UI.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_risk(data: BorrowerInput):
    """
    Accepts a borrower profile and returns:
      - probability: float — Probability of serious delinquency (0.0–1.0)
      - verdict: str — "Approved" or "Declined"
      - risk_level: str — "Low Risk" or "High Risk"
      - top_reasons: list — Top 5 SHAP feature contributions
      - roadmap_to_yes: str — Counterfactual recommendation
    """
    try:
        raw_payload = data.model_dump()

        # Step 1: Get probability of default
        prob = engine.get_prediction(raw_payload)

        # Step 2: Classify risk
        verdict = "Declined" if prob > RISK_THRESHOLD else "Approved"
        risk_level = "High Risk" if prob > RISK_THRESHOLD else "Low Risk"

        # Step 3: Get SHAP explanations
        top_reasons = engine.get_reasons(raw_payload)

        # Step 4: Generate counterfactual roadmap
        roadmap = engine.get_roadmap_to_yes(raw_payload, current_prob=prob)

        logger.info(
            "Prediction complete | prob=%.4f | verdict=%s | age=%s | income=%s",
            prob, verdict, data.age, data.monthly_income,
        )

        return {
            "probability": round(prob, 4),
            "verdict": verdict,
            "risk_level": risk_level,
            "top_reasons": top_reasons,
            "roadmap_to_yes": roadmap,
        }

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))