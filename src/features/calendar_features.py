"""Calendar-derived features.

Only `is_weekend` is engineered; weekday / wday / month / year / event / SNAP columns
come straight from calendar.csv.

KNOWN ISSUE (kept as-is to preserve the original results): in M5, wday 1 = Saturday
and 2 = Sunday, so the notebook's rule `wday >= 6` actually flags Thursday and Friday,
not the weekend. See docs/feature_engineering.md.
"""
import pandas as pd


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = (df["wday"] >= 6).astype("int8")
    return df
