"""Rolling mean / std of past sales (extracted from notebooks/05_feature_engineering.ipynb).

The window is computed on `sales.shift(1)`, so the current day is excluded.
Expects `df` sorted by ["id", "date"].
"""
import pandas as pd

ROLLING_WINDOWS = (7, 14, 28)


def add_rolling_features(df: pd.DataFrame, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby("id")["sales"]
            .transform(lambda x, w=window: x.shift(1).rolling(w).mean())
            .astype("float32")
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("id")["sales"]
            .transform(lambda x, w=window: x.shift(1).rolling(w).std())
            .astype("float32")
        )
    return df
