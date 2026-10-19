"""Unit and integration tests for Health check endpoint."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import Predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_success(client: TestClient):
    """GET /health must return HTTP 200 with model_loaded=True when artifacts are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["features_count"] == 34
    assert "LightGBM" in data["model_type"]


def test_root_endpoint(client: TestClient):
    """GET / returns service metadata and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "/docs" in data["docs_url"]


def test_predictor_model_loading():
    """Predictor directly loads artifacts and reports ready state."""
    predictor = Predictor()
    assert predictor.is_ready() is True
    assert len(predictor.feature_names) == 34
    assert len(predictor.categorical_cols) > 0
    assert predictor.model is not None
