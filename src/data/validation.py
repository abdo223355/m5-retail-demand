"""Data checks used during ingestion (extracted from notebooks/02_data_merging.ipynb)."""
import pyarrow.dataset as ds


def count_duplicate_keys(dataset_dir, keys=("id", "day")) -> int:
    """Number of duplicated (id, day) pairs in a Parquet dataset.

    Only the key columns are read, so the full table is never loaded.
    """
    dataset = ds.dataset(dataset_dir, format="parquet")
    key_df = dataset.to_table(columns=list(keys)).to_pandas()
    return int(key_df.duplicated().sum())
