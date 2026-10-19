"""Pydantic schemas for M5 Retail Demand Prediction API."""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """Input payload containing the 34 features required by the LightGBM demand model."""

    # Categorical identifiers
    item_id: str = Field(..., description="Unique Item identifier", json_schema_extra={"example": "HOUSEHOLD_1_457"})
    dept_id: str = Field(..., description="Department identifier", json_schema_extra={"example": "HOUSEHOLD_1"})
    cat_id: str = Field(..., description="Category identifier", json_schema_extra={"example": "HOUSEHOLD"})
    store_id: str = Field(..., description="Store identifier", json_schema_extra={"example": "CA_1"})
    state_id: str = Field(..., description="State identifier (CA, TX, WI)", json_schema_extra={"example": "CA"})

    # Calendar & Date features
    wm_yr_wk: int = Field(..., description="Walmart week index identifier", json_schema_extra={"example": 11105})
    weekday: str = Field(..., description="Day of week name", json_schema_extra={"example": "Saturday"})
    wday: int = Field(..., ge=1, le=7, description="M5 weekday integer (1=Saturday .. 7=Friday)", json_schema_extra={"example": 1})
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)", json_schema_extra={"example": 2})
    year: int = Field(..., ge=2010, le=2030, description="Calendar year", json_schema_extra={"example": 2011})
    event_name_1: str = Field(default="No_Event", description="Primary special event name", json_schema_extra={"example": "No_Event"})
    event_type_1: str = Field(default="No_Event", description="Primary special event type", json_schema_extra={"example": "No_Event"})
    event_name_2: str = Field(default="No_Event", description="Secondary special event name", json_schema_extra={"example": "No_Event"})
    event_type_2: str = Field(default="No_Event", description="Secondary special event type", json_schema_extra={"example": "No_Event"})
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0 or 1)", json_schema_extra={"example": 0})

    # SNAP benefit disbursement flags
    snap_CA: int = Field(..., ge=0, le=1, description="SNAP eligibility flag in California (0 or 1)", json_schema_extra={"example": 0})
    snap_TX: int = Field(..., ge=0, le=1, description="SNAP eligibility flag in Texas (0 or 1)", json_schema_extra={"example": 0})
    snap_WI: int = Field(..., ge=0, le=1, description="SNAP eligibility flag in Wisconsin (0 or 1)", json_schema_extra={"example": 0})

    # Pricing features
    sell_price: float = Field(..., ge=0.0, description="Current selling price of item", json_schema_extra={"example": 9.97})
    price_lag_1: float = Field(..., ge=0.0, description="Previous week selling price", json_schema_extra={"example": 9.97})
    price_diff_1: float = Field(..., description="Price difference vs previous week", json_schema_extra={"example": 0.0})
    price_pct_change: float = Field(..., description="Percentage change in price", json_schema_extra={"example": 0.0})
    price_max: float = Field(..., ge=0.0, description="Historical maximum price of item", json_schema_extra={"example": 10.47})
    price_relative_to_max: float = Field(..., ge=0.0, le=1.5, description="Ratio of sell_price / price_max", json_schema_extra={"example": 0.9522})

    # Lag features
    lag_1: float = Field(..., ge=0.0, description="Sales 1 day prior", json_schema_extra={"example": 0.0})
    lag_7: float = Field(..., ge=0.0, description="Sales 7 days prior (same weekday last week)", json_schema_extra={"example": 0.0})
    lag_14: float = Field(..., ge=0.0, description="Sales 14 days prior", json_schema_extra={"example": 0.0})
    lag_28: float = Field(..., ge=0.0, description="Sales 28 days prior", json_schema_extra={"example": 0.0})

    # Rolling statistics
    rolling_mean_7: float = Field(..., ge=0.0, description="7-day rolling mean of past sales", json_schema_extra={"example": 0.0})
    rolling_std_7: float = Field(..., ge=0.0, description="7-day rolling std of past sales", json_schema_extra={"example": 0.0})
    rolling_mean_14: float = Field(..., ge=0.0, description="14-day rolling mean of past sales", json_schema_extra={"example": 0.0})
    rolling_std_14: float = Field(..., ge=0.0, description="14-day rolling std of past sales", json_schema_extra={"example": 0.0})
    rolling_mean_28: float = Field(..., ge=0.0, description="28-day rolling mean of past sales", json_schema_extra={"example": 0.0})
    rolling_std_28: float = Field(..., ge=0.0, description="28-day rolling std of past sales", json_schema_extra={"example": 0.0})

    # Strict contract: do not allow silent extra/unexpected fields
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "item_id": "HOUSEHOLD_1_457",
                "dept_id": "HOUSEHOLD_1",
                "cat_id": "HOUSEHOLD",
                "store_id": "CA_1",
                "state_id": "CA",
                "wm_yr_wk": 11105,
                "weekday": "Saturday",
                "wday": 1,
                "month": 2,
                "year": 2011,
                "event_name_1": "No_Event",
                "event_type_1": "No_Event",
                "event_name_2": "No_Event",
                "event_type_2": "No_Event",
                "snap_CA": 0,
                "snap_TX": 0,
                "snap_WI": 0,
                "sell_price": 9.97,
                "price_lag_1": 9.97,
                "price_diff_1": 0.0,
                "price_pct_change": 0.0,
                "price_max": 10.47,
                "price_relative_to_max": 0.9522,
                "lag_1": 0.0,
                "lag_7": 0.0,
                "lag_14": 0.0,
                "lag_28": 0.0,
                "rolling_mean_7": 0.0,
                "rolling_std_7": 0.0,
                "rolling_mean_14": 0.0,
                "rolling_std_14": 0.0,
                "rolling_mean_28": 0.0,
                "rolling_std_28": 0.0,
                "is_weekend": 0
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output prediction response."""
    predicted_demand: float = Field(..., description="Predicted unit sales (clipped to non-negative)", json_schema_extra={"example": 6.47})
    model_version: str = Field(default="lightgbm_v1", description="Model identifier")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status", json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., description="Indicates if model artifacts are loaded in memory", json_schema_extra={"example": True})
    model_type: Optional[str] = Field(default="LightGBM Regressor", description="Type of ML model")
    features_count: Optional[int] = Field(default=34, description="Number of expected input features")
