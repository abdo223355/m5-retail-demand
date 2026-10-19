"""LightGBM training with the time-based split used in notebooks/06_baseline_model.ipynb.

Only the Day 6 protocol lives here (time-based validation window). The tuning notebook
(07) uses a different row-based split and is intentionally NOT promoted to `src/`;
see docs/modeling.md.
"""
import json
import os

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.evaluate import evaluate

TARGET = "sales"
DROP_COLS = ["sales", "id", "day", "date"]
CATEGORICAL_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]
# Feature selection of notebook 06 (near-zero importance in the baseline)
LOW_IMPORTANCE_COLS = ["state_id", "event_name_2", "event_type_2"]


def time_split(df: pd.DataFrame, valid_start, test_start):
    """Train: date < valid_start; validation: valid_start <= date < test_start."""
    train_df = df[df["date"] < valid_start]
    valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
    return train_df, valid_df


def load_train_valid(feature_dir, valid_start, test_start, n_chunks=4):
    """Load the first `n_chunks` feature chunks and split them by date (as in notebook 06)."""
    files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".parquet"))[:n_chunks]
    valid_start, test_start = pd.Timestamp(valid_start), pd.Timestamp(test_start)

    train_parts, valid_parts = [], []
    for file in files:
        print(f"Processing {file}")
        df = pd.read_parquet(os.path.join(feature_dir, file))
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category")
        train_df, valid_df = time_split(df, valid_start, test_start)
        train_parts.append(train_df)
        valid_parts.append(valid_df)

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    # pd.concat can reset the category dtype
    for col in CATEGORICAL_COLS:
        train_df[col] = train_df[col].astype("category")
        valid_df[col] = valid_df[col].astype("category")
    return train_df, valid_df


def make_xy(df: pd.DataFrame, drop_low_importance: bool = False):
    drop = DROP_COLS + (LOW_IMPORTANCE_COLS if drop_low_importance else [])
    return df.drop(columns=drop), df[TARGET]


def train_model(X_train, y_train, X_valid, y_valid, params: dict, categorical_cols):
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=list(categorical_cols),
    )
    return model


def main():
    from src.utils.config import load_config, resolve

    cfg = load_config()
    split = cfg["split"]
    train_df, valid_df = load_train_valid(
        resolve(cfg["paths"]["feature_chunks"]),
        split["valid_start"], split["test_start"], split["n_chunks"],
    )
    X_train, y_train = make_xy(train_df)
    X_valid, y_valid = make_xy(valid_df)

    model = train_model(
        X_train, y_train, X_valid, y_valid,
        cfg["model"]["baseline_params"], CATEGORICAL_COLS,
    )
    metrics = evaluate(y_valid, model.predict(X_valid))
    print(metrics)

    models_dir = resolve(cfg["paths"]["models"]) / "baseline"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "lightgbm_time_split.pkl")
    with open(resolve(cfg["paths"]["tables"]) / "time_split_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
