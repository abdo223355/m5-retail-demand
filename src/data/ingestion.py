"""Chunked ingestion: wide sales -> long format, merged with calendar and sell prices.

Extracted verbatim from notebooks/02_data_merging.ipynb. The full merged table
(~58M rows) never exists in memory: item-rows are melted CHUNK_SIZE at a time and
every merged chunk is written to Parquet.
"""
import gc
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CHUNK_SIZE = 3000  # number of item-rows (from sales_train_validation) melted per chunk


def get_day_columns(sales: pd.DataFrame) -> list:
    """Sorted list of the d_1 ... d_N columns."""
    return sorted(
        (col for col in sales.columns if col.startswith("d_")),
        key=lambda col: int(col.split("_", 1)[1]),
    )


def melt_and_merge_chunk(chunk, calendar, sell_prices, day_columns):
    """Melt one chunk of wide sales rows and merge calendar + sell_prices."""
    melted = pd.melt(
        chunk,
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="day",
        value_name="sales",
    )
    merged_calendar = melted.merge(calendar, left_on="day", right_on="d", how="left")
    return merged_calendar.merge(
        sell_prices, on=["item_id", "store_id", "wm_yr_wk"], how="left"
    )


def build_master_chunks(raw_dir, output_dir, chunk_size: int = CHUNK_SIZE) -> int:
    """Read the raw M5 CSVs and write chunk_XXXX.parquet files. Returns total rows written."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sell_prices = pd.read_csv(raw_dir / "sell_prices.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    sales = pd.read_csv(raw_dir / "sales_train_validation.csv")
    day_columns = get_day_columns(sales)

    n_rows = len(sales)
    n_chunks = math.ceil(n_rows / chunk_size)
    total_long_rows = 0

    for i, start in enumerate(range(0, n_rows, chunk_size), start=1):
        end = min(start + chunk_size, n_rows)
        print(f"Processing chunk {i}/{n_chunks} (rows {start} to {end - 1})...")

        master_chunk = melt_and_merge_chunk(
            sales.iloc[start:end], calendar, sell_prices, day_columns
        )
        total_long_rows += len(master_chunk)

        table = pa.Table.from_pandas(master_chunk, preserve_index=False)
        out_path = output_dir / f"chunk_{i:04d}.parquet"
        pq.write_table(table, out_path)
        print(f"  -> saved {len(master_chunk):,} rows to {out_path}")

        del master_chunk, table
        gc.collect()

    print(f"Total rows written across all chunk files: {total_long_rows:,}")
    return total_long_rows


def main():
    from src.utils.config import load_config, resolve

    cfg = load_config()
    build_master_chunks(
        resolve(cfg["paths"]["raw"]),
        resolve(cfg["paths"]["master_chunks"]),
        cfg["ingestion"]["chunk_size"],
    )


if __name__ == "__main__":
    main()
