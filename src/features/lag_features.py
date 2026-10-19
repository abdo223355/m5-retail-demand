"""Lag features of daily sales per series (extracted from notebooks/05_feature_engineering.ipynb).

Expects `df` sorted by ["id", "date"]. Lags are computed within each `id`, so no
value ever leaks across series.
"""
import pandas as pd

LAG_DAYS = (1, 7, 14, 28)


def add_lag_features(df: pd.DataFrame, lags=LAG_DAYS) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby("id")["sales"].shift(lag).astype("float32")
        )
    return df
