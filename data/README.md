# Data

Nothing in this folder is tracked by Git (see `.gitignore`); only this file is.

## Source
M5 Forecasting – Accuracy (Walmart daily unit sales, 2011-01-29 → 2016-06-19), Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
You need a Kaggle account and must accept the competition rules.

```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy
# unzip the files into data/raw/
```

## Expected layout
```
data/
├── raw/                        original files, never modified
│   ├── sales_train_validation.csv    ~120 MB   (used by the pipeline)
│   ├── calendar.csv                  ~0.1 MB   (used)
│   ├── sell_prices.csv               ~203 MB   (used)
│   ├── sales_train_evaluation.csv    ~122 MB   (exploration notebooks only)
│   └── sample_submission.csv         ~5 MB     (exploration notebooks only)
├── interim/                    regenerable intermediate data
│   ├── master_dataset_parquet/       11 chunks, ~100 MB   <- src/data/ingestion.py
│   ├── preprocessed_chunks/          11 chunks, ~100 MB   <- src/data/preprocessing.py
│   ├── eda_sample.parquet            sample used by notebook 03
│   ├── day2_draft_chunks/            from the archived non-chunked draft (optional)
│   └── master_df_legacy.csv          7.9 GB, from the archived draft (optional, can be deleted)
└── processed/
    ├── feature_chunks/               11 chunks, ~0.5 GB  <- src/features/build_features.py
    └── analysis/                     error tables written by notebooks 08 and 09
```

## What the pipeline does to the data
1. **Ingestion** — wide sales (30,490 series × 1,913 days) → long format, joined with calendar and prices; 58,327,370 rows in 11 Parquet chunks.
2. **Preprocessing** — drop `d`, text → `category`, integer/float downcasting, missing events → `No_Event` (−28.7 % memory on chunk 1).
3. **Features** — lags, rolling statistics, price features; first 28 days of each series dropped (38 columns).

Details: [../docs/data_pipeline.md](../docs/data_pipeline.md).
