# Modeling

**Status:** a LightGBM baseline, a small tuning round and an error analysis exist. There is **no forecast for the 28-day hold-out period** (`test_start` 2016-03-28 onwards is never scored) and no `predict` step. The final forecasting pipeline is currently under development.

## Setup
* Target: `sales` (units sold per item-store-day).
* Model: `lightgbm.LGBMRegressor`, objective `regression`, metric `rmse`, native categorical handling for 10 columns (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, `weekday`, `event_name_1/2`, `event_type_1/2`); no one-hot encoding.
* Metrics: RMSE, MAE, bias; naive lag-7 forecast as a reference (notebook 09).

## Two validation protocols were used — their numbers are not comparable

| | Day 6 (`notebooks/06`, `src/models/train.py`) | Day 7 (`notebooks/07`) |
|---|---|---|
| Split | **time-based**: train < 2016-02-29, validation 2016-02-29 … 2016-03-27 | **row-based 80/20** per chunk: `iloc[:int(0.8*len)]` / rest |
| Data | first 4 feature chunks | chunks 1, 3, 6, 9 |
| Rows | train 21,948,000 · validation 336,000 | train 18,096,000 · validation 4,524,000 |
| What validation holds | the last 4 weeks of all series in those chunks | because chunks are sorted by `id`, the last 20 % of **series** over their whole history |

Consequences:
* Day 7 does not measure forecasting of future dates; it measures prediction for held-out series (using lag/rolling features that are available inside the row).
* Its validation covers only 4 stores (CA_1, CA_3, TX_2, WI_2) and 2 departments (HOUSEHOLD_1, HOUSEHOLD_2).
* The Day 7 table "Baseline 1.9596 → Tuned 1.2600" compares a Day 6 number with a Day 7 number, i.e. two different validation sets. All four Day 7 variants (different learning rates, leaves and sampling) land within 0.001 RMSE of each other, which suggests the drop from ~2.0 to ~1.26 comes from the change of validation set, not from tuning. This is an inference from the logged numbers; it has not been confirmed by re-running Day 6 and Day 7 on the same split.

## Results as logged

| Run | Notebook | Params | RMSE | MAE |
|---|---|---|---|---|
| Baseline, 34 features (time split) | 06 | lr 0.05, 300 trees, 64 leaves | 2.0167 | 1.0197 |
| Feature-selected, 31 features (time split) | 06 | same | 2.0110 | 1.0191 |
| `baseline_full` (saved metrics file) | 07 / `reports/tables/baseline_full_metrics.json` | lr 0.05, 300 trees, 64 leaves | 1.9596 | 0.9613 |
| Day 7 v2 (row split) | 07 | lr 0.03, 500, 128 leaves | 1.2600 | 0.4873 |
| Day 7 v3 | 07 | lr 0.01, 1000, 128 leaves | 1.2603 | 0.4885 |
| **Day 7 v4 (selected)** | 07 | lr 0.03, 500, 64 leaves, 0.8/0.8 sampling | 1.2597 | 0.4873 |
| Day 7 v5 | 07 | lr 0.03, 500, 64 leaves, 0.7/0.7 sampling | 1.2605 | 0.4876 |

The `baseline_full` file was produced outside the saved notebooks, so the protocol behind 1.9596 / 0.9613 cannot be reconstructed; it differs slightly from the 2.0167 printed in notebook 06.

Day 8–9 analysis of v4 on its (Day 7) validation set (4,524,000 rows):
* bias (actual − predicted) = −0.0121 → very slight over-forecasting on average
* naive lag-7: MAE 0.567, RMSE 1.770 vs LightGBM MAE 0.487, RMSE 1.260 (RMSE skill score 0.288) — a fair comparison, since both use the same validation rows
* error is larger for HOUSEHOLD_1 (MAE 0.97, RMSE 2.82) than HOUSEHOLD_2 (MAE 0.41, RMSE 0.73); TX_2 has the highest store RMSE (1.73)

## Model selection
v4 was picked as the lowest RMSE among five Day 7 runs (differences < 0.001). It is saved as `models/tuned/best_lightgbm_model.pkl`. No cross-validation and no systematic search (grid / random / Bayesian) was performed.

## Not done (candidates, not claims)
* score all models on one shared time-based validation set
* forecast the 28 hold-out days; recursive or direct multi-step strategy
* stratified chunking so validation covers all stores and departments
* SHAP: the code exists in notebook 09 but its output was never produced
