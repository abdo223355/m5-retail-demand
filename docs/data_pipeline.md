# Data Pipeline

All numbers below come from the notebook outputs or from the files on disk.

```
Raw M5 CSVs (data/raw)
   │  src/data/ingestion.py        melt 3,000 series at a time, merge calendar + sell_prices
   ▼
Master chunks, Parquet (data/interim/master_dataset_parquet)       11 files
   │  src/data/preprocessing.py    category + downcast + event NaN -> "No_Event"
   ▼
Preprocessed chunks (data/interim/preprocessed_chunks)             11 files
   │  src/features/build_features.py   lags, rolling stats, price features, NaN drop
   ▼
Feature chunks (data/processed/feature_chunks)                     11 files, 38 columns
   │  src/models/train.py          time-based split, LightGBM
   ▼
Model (models/) -> predictions -> evaluation (notebooks 08, 09; src/models/evaluate.py)
```

## Raw data

| File | Content | Shape (from notebook output) |
|---|---|---|
| `sales_train_validation.csv` | daily unit sales, wide (`d_1` … `d_1913`) | 30,490 series × 1,919 columns |
| `calendar.csv` | dates, weekday, month, year, events, SNAP flags | 1,969 × 14 |
| `sell_prices.csv` | weekly price per store/item | 6,841,121 × 4 |

`sales_train_evaluation.csv` and `sample_submission.csv` are loaded in the exploration notebooks but are **not used** by the pipeline.

## Stage 1 — Ingestion and chunk processing (`notebooks/02`)

Wide sales are melted to long format (`id`, `day`, `sales`), joined to the calendar on `day == d`, then to prices on `(item_id, store_id, wm_yr_wk)`.
This is done for `CHUNK_SIZE = 3000` series at a time, and each result is written to Parquet immediately.

* 30,490 series / 3,000 → **11 chunks**; the first 10 hold 5,739,000 rows each (3,000 × 1,913 days), the last one 937,370.
* The full long table therefore has **58,327,370 rows** (30,490 × 1,913). It is never held in memory.
* Check performed: 0 duplicate `(id, day)` pairs across all chunks.

### Why chunking
The first attempt (kept in `notebooks/archive/02_data_merging_draft.ipynb`) melted and merged the whole table in one DataFrame and saved it as a single CSV; that file (`master_df_legacy.csv`) is **7.9 GB**. The author reports a `MemoryError` with that approach; the error itself was not saved in the notebook outputs, so it could not be re-verified. Chunking keeps peak RAM bounded by one chunk instead of the full table.

### Why Parquet
The 11 master chunks take **99.4 MB** on disk (notebook 02 output), versus 7.9 GB for the CSV: columnar storage, compression, per-column reads. Later steps rely on this, e.g. the duplicate check reads only `id` and `day`, and notebook 06 reads only `date` to find the split dates.

## Stage 2 — Cleaning and memory optimization (`notebooks/04`)

* drop column `d` (verified identical to `day`)
* 11 low-cardinality object columns → `category`
* integer columns downcast (`sales`, `wm_yr_wk`, `year` → `int16`; `wday`, `month`, `snap_*` → `int8`), `sell_price` → `float32`
* `event_name_*` / `event_type_*` missing values → `"No_Event"`
* `sell_price` NaNs are **not** filled here (they are rows before an item was priced; handled in feature engineering)

Measured on chunk 1: 1,086.5 MB → 774.6 MB (**−28.7 %**).

## Stage 3 — Feature engineering (`notebooks/05`)
See [feature_engineering.md](feature_engineering.md). Output: 38 columns per row. Each chunk loses its first 28 days per series (warm-up NaNs): chunk 1 goes from 5,739,000 to 5,655,000 rows.

## Stage 4 — Training dataset
Date range after warm-up: 2011-02-26 → 2016-04-24 (1,885 unique dates). See [modeling.md](modeling.md) for the splits.

## Reproducing it
```bash
python -m src.data.ingestion        # data/raw -> data/interim/master_dataset_parquet
python -m src.data.preprocessing    # -> data/interim/preprocessed_chunks
python -m src.features.build_features   # -> data/processed/feature_chunks
```
Paths and the chunk size are in `configs/config.yaml`. Each stage was checked against the existing artifacts on a 20–60-series sample (identical output); the full 58M-row run was not repeated during the refactoring.
