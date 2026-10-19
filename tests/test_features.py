import numpy as np
import pandas as pd
import pytest

from src.features.build_features import CRITICAL_COLS, build_features_chunk
from src.features.calendar_features import add_is_weekend
from src.features.lag_features import add_lag_features
from src.features.price_features import add_price_features, fill_missing_prices
from src.features.rolling_features import add_rolling_features


def _two_series(n=40):
    dates = pd.date_range("2016-01-01", periods=n)
    rows = []
    for sid, base in (("A", 0), ("B", 1000)):
        for i, d in enumerate(dates):
            rows.append({"id": sid, "date": d, "sales": base + i,
                         "sell_price": 2.0 + (i // 10), "wday": (i % 7) + 1})
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype("category")
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def test_lags_stay_inside_each_series():
    df = add_lag_features(_two_series())
    b = df[df["id"] == "B"].reset_index(drop=True)
    assert b["lag_1"].isna().iloc[0]           # first row of B has no lag (no leakage from A)
    assert b.loc[5, "lag_1"] == b.loc[4, "sales"]
    assert b.loc[10, "lag_7"] == b.loc[3, "sales"]
    assert b["lag_28"].isna().sum() == 28
    assert df["lag_1"].dtype == "float32"


def test_rolling_excludes_current_day():
    df = add_rolling_features(_two_series())
    a = df[df["id"] == "A"].reset_index(drop=True)
    # day 7 (index 7): mean of sales[0..6] = 3.0, NOT including sales[7]
    assert a.loc[7, "rolling_mean_7"] == pytest.approx(3.0)
    assert a["rolling_mean_7"].isna().sum() == 7
    assert a.loc[7, "rolling_std_7"] == pytest.approx(np.std(range(7), ddof=1))
    for w in (7, 14, 28):
        assert {f"rolling_mean_{w}", f"rolling_std_{w}"} <= set(df.columns)


def test_is_weekend_rule_matches_notebook():
    # Documents the ORIGINAL rule (wday >= 6). In M5, wday 1=Saturday ... 7=Friday.
    df = add_is_weekend(pd.DataFrame({"wday": [1, 2, 5, 6, 7]}))
    assert df["is_weekend"].tolist() == [0, 0, 0, 1, 1]
    assert df["is_weekend"].dtype == "int8"


def test_price_features():
    df = _two_series()
    df.loc[0, "sell_price"] = np.nan
    df = fill_missing_prices(df)
    assert not df["sell_price"].isna().any()
    df = add_price_features(df)
    a = df[df["id"] == "A"].reset_index(drop=True)
    assert a.loc[10, "price_diff_1"] == pytest.approx(1.0)   # price steps 2 -> 3 at i=10
    assert a["price_max"].nunique() == 1
    assert (a["price_relative_to_max"] <= 1.0).all()


def test_build_features_chunk_drops_warmup_rows():
    out = build_features_chunk(_two_series(n=60))
    assert not out[CRITICAL_COLS].isna().any().any()
    assert len(out) == 2 * (60 - 28)
