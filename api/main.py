"""FastAPI application for M5 Retail Demand Prediction."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from api.predictor import (
    FeatureContractError,
    ModelNotLoadedError,
    PredictionError,
    Predictor,
)
from api.schemas import HealthResponse, PredictionInput, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m5_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application lifecycle: loads model artifacts once at startup."""
    logger.info("Initializing M5 Demand Predictor service...")
    try:
        predictor = Predictor()
        app.state.predictor = predictor
        if predictor.is_ready():
            logger.info("Predictor successfully initialized and ready for requests.")
        else:
            logger.warning("Predictor initialized, but model artifacts were not found.")
    except Exception as e:
        logger.error(f"Startup error loading model: {e}")
        app.state.predictor = None

    yield
    logger.info("Shutting down M5 Demand Prediction service.")


app = FastAPI(
    title="M5 Retail Demand Prediction API",
    description=(
        "Production-ready prediction service serving the pre-trained LightGBM model "
        "for daily item-store unit sales forecasting on the Walmart M5 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/",
    summary="Root metadata endpoint",
    tags=["General"],
)
def root():
    return {
        "service": "M5 Retail Demand Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Model Readiness Check",
    tags=["System"],
)
def health_check():
    """Returns the operational status of the service and confirms model artifact readiness."""
    predictor: Predictor = getattr(app.state, "predictor", None)
    if predictor and predictor.is_ready():
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_type="LightGBM Regressor",
            features_count=len(predictor.feature_names),
        )

    return JSONResponse(
        status_code=status.HTTP(503),
        content={
            "status": "unhealthy",
            "model_loaded": False,
            "detail": "Model artifacts are not loaded in memory.",
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Retail Unit Sales Demand",
    tags=["Inference"],
)
def predict_demand(payload: PredictionInput):
    """Predicts daily unit sales for an item-store observation based on the 34 engineered features.

    Returns non-negative predicted demand.
    """
    predictor: Predictor = getattr(app.state, "predictor", None)
    if not predictor or not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or failed to load.",
        )

    try:
        predicted_value = predictor.predict(payload)
        return PredictionResponse(predicted_demand=predicted_value)

    except FeatureContractError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature contract violation: {str(e)}",
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected prediction failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during demand prediction.",
        )
