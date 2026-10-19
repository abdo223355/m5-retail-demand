"""Unit tests on tiny hand-made frames (synthetic data is used ONLY for tests)."""
import numpy as np
import pandas as pd

from src.data.ingestion import ID_COLUMNS, get_day_columns, melt_and_merge_chunk
from src.data.preprocessing import preprocess_chunk
from src.data.validation import count_duplicate_keys


def _master_frame():
    return pd.DataFrame({
        "id": ["A_1", "A_1"], "item_id": ["A", "A"], "dept_id": ["D", "D"],
        "cat_id": ["C", "C"], "store_id": ["S", "S"], "state_id": ["CA", "CA"],
        "day": ["d_1", "d_2"], "d": ["d_1", "d_2"], "sales": [3, 0],
        "date": ["2011-01-29", "2011-01-30"], "wm_yr_wk": [11101, 11101],
        "weekday": ["Saturday", "Sunday"], "wday": [1, 2], "month": [1, 1],
        "year": [2011, 2011], "event_name_1": [None, "SuperBowl"],
        "event_type_1": [None, "Sporting"], "event_name_2": [None, None],
        "event_type_2": [None, None], "snap_CA": [0, 1], "snap_TX": [0, 0],
        "snap_WI": [0, 0], "sell_price": [1.5, np.nan],
    })


def test_preprocess_chunk_drops_d_and_downcasts():
    out = preprocess_chunk(_master_frame())
    assert "d" not in out.columns
    assert out["sales"].dtype == "int16"
    assert out["wday"].dtype == "int8"
    assert out["sell_price"].dtype == "float32"
    assert str(out["item_id"].dtype) == "category"


def test_preprocess_chunk_fills_events_but_not_prices():
    out = preprocess_chunk(_master_frame())
    assert out["event_name_1"].tolist() == ["No_Event", "SuperBowl"]
    assert (out["event_name_2"] == "No_Event").all()
    assert out["sell_price"].isna().sum() == 1  # prices are handled in feature engineering


def test_preprocess_chunk_is_idempotent_on_event_category():
    out = preprocess_chunk(_master_frame())
    again = preprocess_chunk(out)
    assert again["event_name_1"].tolist() == out["event_name_1"].tolist()


def test_melt_and_merge_chunk_shape_and_price_join():
    sales = pd.DataFrame({
        "id": ["A_1", "B_1"], "item_id": ["A", "B"], "dept_id": ["D", "D"],
        "cat_id": ["C", "C"], "store_id": ["S", "S"], "state_id": ["CA", "CA"],
        "d_1": [1, 2], "d_2": [3, 4], "d_10": [5, 6],
    })
    calendar = pd.DataFrame({"d": ["d_1", "d_2", "d_10"], "wm_yr_wk": [1, 1, 2], "date": ["x", "y", "z"]})
    prices = pd.DataFrame({"store_id": ["S"], "item_id": ["A"], "wm_yr_wk": [1], "sell_price": [9.99]})
    days = get_day_columns(sales)
    assert days == ["d_1", "d_2", "d_10"]  # numeric, not lexicographic, order

    out = melt_and_merge_chunk(sales, calendar, prices, days)
    assert len(out) == 2 * 3
    assert set(ID_COLUMNS) <= set(out.columns)
    a = out[(out["item_id"] == "A")].set_index("day")["sell_price"]
    assert a["d_1"] == 9.99 and a["d_2"] == 9.99 and np.isnan(a["d_10"])


def test_count_duplicate_keys(tmp_path):
    pd.DataFrame({"id": ["a", "a", "b"], "day": ["d_1", "d_1", "d_1"]}).to_parquet(tmp_path / "c.parquet")
    assert count_duplicate_keys(tmp_path) == 1
