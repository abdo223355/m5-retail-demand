"""Price features (extracted from notebooks/05_feature_engineering.ipynb)."""
import pandas as pd


def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward- then back-fill `sell_price` within each series.

    NOTE: the back-fill uses future prices for the earliest rows of a series.
    """
    df["sell_price"] = df.groupby("id")["sell_price"].ffill().bfill()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_lag_1"] = df.groupby("id")["sell_price"].shift(1).astype("float32")
    df["price_diff_1"] = (df["sell_price"] - df["price_lag_1"]).astype("float32")
    df["price_pct_change"] = df.groupby("id")["sell_price"].pct_change().astype("float32")
    # NOTE: max over the WHOLE history of the series (includes future rows).
    df["price_max"] = df.groupby("id")["sell_price"].transform("max").astype("float32")
    df["price_relative_to_max"] = (df["sell_price"] / df["price_max"]).astype("float32")
    return df
