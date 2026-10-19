"""Predictor service for M5 LightGBM Demand Forecasting model."""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before model artifacts are loaded."""
    pass


class FeatureContractError(ValueError):
    """Raised when incoming features violate the required schema contract."""
    pass


class PredictionError(RuntimeError):
    """Raised when an internal error occurs during model scoring."""
    pass


class Predictor:
    """Manages model artifact lifecycle, input schema transformation, and inference."""

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None) -> None:
        self.artifacts_dir = self._resolve_artifacts_dir(artifacts_dir)
        self.model: Any = None
        self.feature_names: List[str] = []
        self.categorical_cols: List[str] = []
        self._is_loaded: bool = False

        self.load_artifacts()

    @staticmethod
    def _resolve_artifacts_dir(custom_path: Optional[Union[str, Path]] = None) -> Path:
        """Finds existing artifacts directory among standard locations."""
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                return p

        env_path = os.getenv("ARTIFACTS_DIR")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # Check standard fallback paths relative to cwd or repo structure
        base_dir = Path(__file__).resolve().parent.parent
        candidates = [
            base_dir / "artifacts",
            base_dir / "models" / "tuned",
            Path("artifacts"),
            Path("models/tuned"),
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / "best_lightgbm_model.pkl").exists():
                return candidate.resolve()

        # Default fallback
        return (base_dir / "artifacts").resolve()

    def load_artifacts(self) -> None:
        """Loads model, feature names, and categorical column list from disk."""
        model_file = self.artifacts_dir / "best_lightgbm_model.pkl"
        features_file = self.artifacts_dir / "feature_names.pkl"
        cats_file = self.artifacts_dir / "categorical_cols.pkl"

        if not model_file.exists():
            logger.warning(f"Model artifact not found at {model_file}")
            self._is_loaded = False
            return

        try:
            logger.info(f"Loading model from {model_file}...")
            self.model = joblib.load(model_file)

            if features_file.exists():
                self.feature_names = joblib.load(features_file)
            elif hasattr(self.model, "feature_name_"):
                self.feature_names = list(self.model.feature_name_)
            else:
                raise FeatureContractError("Could not determine feature names from artifacts.")

            if cats_file.exists():
                loaded_cats = joblib.load(cats_file)
                self.categorical_cols = list(dict.fromkeys(loaded_cats))
            else:
                self.categorical_cols = []

            # If booster has pandas_categorical, ensure the known 10 categorical features are all cast
            standard_cats = [
                "item_id", "dept_id", "cat_id", "store_id", "state_id",
                "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            ]
            booster_cats = getattr(self.model.booster_, "pandas_categorical", None)
            if booster_cats and len(booster_cats) == 10:
                self.categorical_cols = [c for c in standard_cats if c in self.feature_names]

            self._is_loaded = True
            logger.info(f"Model successfully loaded with {len(self.feature_names)} features and {len(self.categorical_cols)} categoricals.")

        except Exception as e:
            self._is_loaded = False
            logger.error(f"Failed to load artifacts from {self.artifacts_dir}: {e}")
            raise ModelNotLoadedError(f"Artifact loading failed: {e}") from e

    def is_ready(self) -> bool:
        """Checks if the predictor is initialized and ready for inference."""
        return self._is_loaded and self.model is not None

    def prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Validates feature contract, orders columns according to training schema, and casts categoricals."""
        if not self.is_ready():
            raise ModelNotLoadedError("Predictor model is not loaded.")

        # Check for missing required features
        missing = [f for f in self.feature_names if f not in payload]
        if missing:
            raise FeatureContractError(f"Missing required features: {missing}")

        # Construct 1-row DataFrame enforcing strict feature ordering
        df = pd.DataFrame([payload])[self.feature_names]

        # Apply native categorical typing as required by LightGBM
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def predict(self, input_data: Union[Dict[str, Any], Any]) -> float:
        """Scores a single observation and applies non-negative demand clipping."""
        if not self.is_ready():
            raise ModelNotLoadedError("Model is not loaded. Cannot serve predictions.")

        if hasattr(input_data, "model_dump"):
            data_dict = input_data.model_dump()
        elif hasattr(input_data, "dict"):
            data_dict = input_data.dict()
        elif isinstance(input_data, dict):
            data_dict = input_data
        else:
            raise FeatureContractError(f"Unsupported input type: {type(input_data)}")

        df = self.prepare_features(data_dict)

        try:
            # Use native LightGBM booster directly to bypass scikit-learn cross-version pickle issues ('get_params')
            booster = getattr(self.model, "booster_", None)
            if booster is not None and hasattr(booster, "predict"):
                raw_prediction = booster.predict(df)
            else:
                raw_prediction = self.model.predict(df)

            val = float(raw_prediction[0])

            # Demand cannot be negative: clip to 0
            clipped_prediction = float(np.clip(val, 0.0, None))
            return round(clipped_prediction, 4)

        except Exception as e:
            logger.error(f"Inference computation error: {e}", exc_info=True)
            raise PredictionError(f"Model scoring failed: {e}") from e
