# Feature Engineering

Implemented in `src/features/` (extracted unchanged from `notebooks/05_feature_engineering.ipynb`).
Chunks are sorted by `["id", "date"]`; every group operation is per `id` (one item-store series).

The model uses **34 features** (the 38 columns of a feature chunk minus `sales`, `id`, `day`, `date`).

## Lag features — `lag_features.py`
| Feature | Meaning |
|---|---|
| `lag_1` | sales one day earlier |
| `lag_7` | same weekday, one week earlier |
| `lag_14` | two weeks earlier |
| `lag_28` | four weeks earlier |

Why: trees have no notion of order; lags give them the series' recent history. Weekly lags capture day-of-week seasonality.
Leakage: none inside the dataset (`shift` within each `id`, so values never cross series).
Forecasting caveat: the project has no multi-step forecast. For a horizon > 1 day, `lag_1`…`lag_6` would not be known at prediction time and would need recursive prediction.

## Rolling features — `rolling_features.py`
`rolling_mean_{7,14,28}` and `rolling_std_{7,14,28}`, computed on `sales.shift(1)` so the current day is **excluded**. Smooth short-term demand level and volatility.

## Calendar features — `calendar_features.py`
Raw columns from `calendar.csv` used as features: `weekday`, `wday`, `month`, `year`, `wm_yr_wk`, `event_name_1/2`, `event_type_1/2` (missing → `No_Event`), `snap_CA/TX/WI`.
Engineered: `is_weekend` only. There are **no** engineered day / week-of-year features.

> **Known issue (kept to preserve the original results).** `is_weekend = (wday >= 6)`, but in M5 `wday` 1 = Saturday, 2 = Sunday … 6 = Thursday, 7 = Friday (verified on `calendar.csv`). The feature therefore flags **Thursday and Friday**, not the weekend. Its importance in the baseline is near zero (15 splits). Fix: `wday <= 2`. Changing it requires regenerating the feature chunks and retraining.

## Price features — `price_features.py`
`sell_price` is forward- then back-filled per series, then:

| Feature | Definition |
|---|---|
| `price_lag_1` | previous row's price |
| `price_diff_1` | `sell_price − price_lag_1` |
| `price_pct_change` | relative change vs previous row |
| `price_max` | maximum price of the series |
| `price_relative_to_max` | `sell_price / price_max` (a drop signals a discount) |

Leakage considerations (present in the original code, not changed):
* `price_max` uses the **whole history** of the series, including rows after the prediction date.
* the back-fill uses later prices for the earliest rows of a series.

Sell prices are published in advance in M5, so `price_lag_1` / `price_diff_1` / `price_pct_change` are not a concern; the two points above are.

## Row filtering
Rows with NaN in `lag_28`, `rolling_mean_28`, `rolling_std_28`, `price_lag_1`, `price_pct_change`, `price_relative_to_max` are dropped (first 28 days of each series).

## Importance (baseline LightGBM, split count — `reports/tables/baseline_feature_importance.csv`)
`item_id` 2720 · `lag_1` 2204 · `rolling_mean_7` 1414 · `wm_yr_wk` 1135 · `lag_7` 969 · `lag_28` 953 · `lag_14` 831 · `weekday` 788 · …
In the Day 6 run `state_id`, `event_name_2` and `event_type_2` had ≈ 0 importance and were dropped in the "feature-selected" variant.
