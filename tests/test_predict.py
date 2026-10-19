"""Unit and integration tests for Prediction endpoint and logic."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import Predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """A realistic valid observation from M5 validation sample."""
    return {
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
        "is_weekend": 0,
    }


def test_predict_valid_payload(client: TestClient, valid_payload):
    """POST /predict with valid payload returns HTTP 200 and numeric predicted_demand."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_demand" in data
    assert isinstance(data["predicted_demand"], (int, float))
    assert data["predicted_demand"] >= 0.0


def test_predict_missing_feature(client: TestClient, valid_payload):
    """POST /predict missing a required feature returns HTTP 422 validation error."""
    payload = valid_payload.copy()
    del payload["item_id"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_missing_numerical_feature(client: TestClient, valid_payload):
    """POST /predict missing sell_price returns HTTP 422 validation error."""
    payload = valid_payload.copy()
    del payload["sell_price"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_datatype(client: TestClient, valid_payload):
    """POST /predict with incompatible type (string instead of float) returns HTTP 422."""
    payload = valid_payload.copy()
    payload["sell_price"] = "not_a_number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unexpected_extra_feature(client: TestClient, valid_payload):
    """POST /predict with unexpected field returns HTTP 422 due to strict schema contract."""
    payload = valid_payload.copy()
    payload["unrecognized_field_xyz"] = 123
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_non_negative_demand_guarantee(valid_payload):
    """Directly verifies that the prediction pipeline postprocessing guarantees non-negative demand."""
    predictor = Predictor()

    # Score realistic input
    prediction = predictor.predict(valid_payload)
    assert prediction >= 0.0

    # Mock raw prediction to simulate negative tree output
    original_predict = predictor.model.predict
    booster = getattr(predictor.model, "booster_", None)
    original_booster_predict = booster.predict if booster is not None else None
    try:
        predictor.model.predict = lambda df: [-5.4321]
        if booster is not None:
            booster.predict = lambda df: [-5.4321]
        clipped_result = predictor.predict(valid_payload)
        assert clipped_result == 0.0
    finally:
        predictor.model.predict = original_predict
        if booster is not None and original_booster_predict is not None:
            booster.predict = original_booster_predict
