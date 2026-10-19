import pandas as pd
import pytest

from src.models.evaluate import bias, evaluate, mae, naive_lag7_predictions, rmse
from src.utils.config import load_config, resolve
from src.utils.memory import memory_reduction_pct, memory_usage_mb


def test_memory_helpers():
    df = pd.DataFrame({"a": range(1000)})
    assert memory_usage_mb(df) > 0
    assert memory_reduction_pct(100.0, 71.29) == pytest.approx(28.71)


def test_metrics():
    y, p = [1, 2, 3], [1, 2, 5]
    assert mae(y, p) == pytest.approx(2 / 3)
    assert rmse(y, p) == pytest.approx((4 / 3) ** 0.5)
    assert bias(y, p) == pytest.approx(-2 / 3)   # actual - predicted: negative = over-forecast
    assert set(evaluate(y, p)) == {"RMSE", "MAE"}


def test_naive_lag7_is_clipped_at_zero():
    X = pd.DataFrame({"lag_7": [-1.0, 0.0, 4.0]})
    assert naive_lag7_predictions(X).tolist() == [0.0, 0.0, 4.0]


def test_config_loads_and_paths_resolve():
    cfg = load_config()
    assert cfg["ingestion"]["chunk_size"] == 3000
    assert cfg["split"]["valid_start"] == "2016-02-29"
    assert resolve(cfg["paths"]["raw"]).name == "raw"
    assert load_config("features.yaml")["lags"] == [1, 7, 14, 28]
