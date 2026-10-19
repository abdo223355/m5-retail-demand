"""Feature pipeline over the preprocessed chunks (extracted from notebooks/05)."""
import glob
import os

import pandas as pd

from src.features.calendar_features import add_is_weekend
from src.features.lag_features import add_lag_features
from src.features.price_features import add_price_features, fill_missing_prices
from src.features.rolling_features import add_rolling_features

CRITICAL_COLS = [
    "lag_28", "rolling_mean_28", "rolling_std_28",
    "price_lag_1", "price_pct_change", "price_relative_to_max",
]


def build_features_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_is_weekend(df)
    df = fill_missing_prices(df)
    df = add_price_features(df)

    return df.dropna(subset=CRITICAL_COLS)


def build_features(input_dir, output_dir) -> int:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    for file in files:
        print(f"Processing: {os.path.basename(file)}")
        df = build_features_chunk(pd.read_parquet(file))
        output_path = os.path.join(output_dir, f"features_{os.path.basename(file)}")
        df.to_parquet(output_path, index=False)
        print(f"Saved -> {output_path}")
    print("Feature Engineering completed successfully.")
    return len(files)


def main():
    from src.utils.config import load_config, resolve

    cfg = load_config()
    build_features(
        resolve(cfg["paths"]["preprocessed_chunks"]),
        resolve(cfg["paths"]["feature_chunks"]),
    )


if __name__ == "__main__":
    main()
