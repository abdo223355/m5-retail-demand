"""Chunk preprocessing: category conversion, numeric downcasting, event NaN handling.

`preprocess_chunk` is the final version from notebooks/04_preprocessing.ipynb
(the notebook defines earlier, equivalent drafts of the same function).
"""
import gc
import glob
import os

import pandas as pd

from src.utils.memory import memory_usage_mb

CATEGORY_COLS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def preprocess_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate column ('d' == 'day')
    if "d" in df.columns:
        df.drop(columns=["d"], inplace=True)

    # Category columns
    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    # Numeric downcasting
    df["sales"] = df["sales"].astype("int16")
    df["wm_yr_wk"] = df["wm_yr_wk"].astype("int16")
    df["wday"] = df["wday"].astype("int8")
    df["month"] = df["month"].astype("int8")
    df["year"] = df["year"].astype("int16")
    df["snap_CA"] = df["snap_CA"].astype("int8")
    df["snap_TX"] = df["snap_TX"].astype("int8")
    df["snap_WI"] = df["snap_WI"].astype("int8")
    df["sell_price"] = df["sell_price"].astype("float32")

    # Event missing values
    for col in EVENT_COLS:
        if "No_Event" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(["No_Event"])
        df[col] = df[col].fillna("No_Event")

    return df


def preprocess_chunks(input_dir, output_dir) -> int:
    """Apply `preprocess_chunk` to every Parquet chunk in `input_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = sorted(glob.glob(f"{input_dir}/*.parquet"))
    print(f"Found {len(chunk_files)} chunks")

    for file in chunk_files:
        print(f"\nProcessing {os.path.basename(file)}")
        df = preprocess_chunk(pd.read_parquet(file))
        output_path = os.path.join(output_dir, os.path.basename(file))
        df.to_parquet(output_path, index=False)
        print(f"Saved -> {output_path} | Memory: {memory_usage_mb(df):.2f} MB")
        del df
        gc.collect()
    return len(chunk_files)


def main():
    from src.utils.config import load_config, resolve

    cfg = load_config()
    preprocess_chunks(
        resolve(cfg["paths"]["master_chunks"]),
        resolve(cfg["paths"]["preprocessed_chunks"]),
    )


if __name__ == "__main__":
    main()
