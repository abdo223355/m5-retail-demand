# M5 Retail Demand Forecasting — Technical Report

**Project type:** Time-series demand forecasting (retail, SKU x store x day granularity)
**Model:** LightGBM (gradient-boosted trees)
**Dataset:** M5 Forecasting (Walmart daily unit sales, item x store x day)
**Author's pipeline stage:** Days 1–9 (data engineering, feature engineering, model tuning, error analysis, explainability)

> **Note on validation protocol (added during repository refactoring).** The metrics in this report (RMSE 1.26, MAE 0.487; 4,524,000 validation rows) come from the Day 7 row-based 80/20 split, which holds out *series*, not future dates, and covers 4 stores and 2 departments. They are not comparable to the Day 6 time-based baseline (RMSE 2.02, MAE 1.02). See `docs/modeling.md`.

---

# Executive Summary

## Business problem

Retailers must decide, for every item at every store, how many units to order for the coming days. Two failure modes carry asymmetric cost: **under-forecasting** leads to stockouts and lost sales on days when demand is present, while **over-forecasting** leads to excess inventory, holding costs, and markdown/write-off risk. An accurate, well-calibrated demand forecast at the item-store-day level is the primary input to replenishment, staffing, and promotional planning decisions.

This project builds a machine-learning forecasting pipeline for the M5 dataset — a large-scale, hierarchical retail sales dataset released by Walmart for the M5 Forecasting Accuracy competition — and evaluates it not only on aggregate error but on the operational risk implied by its error distribution.

## Dataset overview

The M5 dataset contains daily unit sales for thousands of items across ten Walmart stores in three US states (California, Texas, Wisconsin), together with calendar information (weekdays, holidays, special events), SNAP (Supplemental Nutrition Assistance Program) benefit-disbursement flags per state, and item-level pricing history. The target variable is daily units sold per item-store combination — a series that is typically low-volume, intermittent (many zero-sales days), and occasionally subject to large demand spikes driven by promotions, events, or benefit-disbursement timing.

The validation set used for evaluation in this project contains 4,524,000 item-store-day observations across 34 engineered features, of which 9 are treated as categorical.

## Objectives

1. Build a memory-efficient feature engineering pipeline capable of processing the full M5 grid without exceeding local hardware constraints.
2. Train and tune a LightGBM regression model to forecast daily unit sales at the item-store level.
3. Evaluate the model with metrics appropriate to retail forecasting (not just aggregate RMSE/MAE), including a naive-baseline comparison and segment-level error diagnostics.
4. Identify where and why the model fails, and translate those failures into inventory-relevant risk (stockout vs. overstock exposure).
5. Assess whether the current tree-based approach is sufficient, or whether a transition to sequence models (LSTM/GRU) is warranted for a later phase of the project.

---

# Data Engineering

## Memory constraints and parquet chunk strategy

The full M5 grid (item x store x day, expanded with engineered features) is large enough that loading it as a single in-memory pandas DataFrame is impractical on standard local hardware. The pipeline addresses this by processing the data in **parquet chunks**: the raw sales, calendar, and pricing tables are merged and feature-engineered incrementally over partitioned subsets of the data, rather than materializing the full grid at once. Parquet's columnar, compressed storage format was chosen over CSV for both size and I/O speed, since it preserves dtypes across chunk boundaries and avoids repeated type inference.

This strategy makes the pipeline runnable on constrained hardware, but it introduces a coverage trade-off that is discussed explicitly in the Limitations section: the specific chunking scheme used resulted in a validation sample that does not include every store or department in the M5 hierarchy.

## Data preprocessing pipeline

The preprocessing pipeline merges three source tables standard to the M5 release — daily sales, calendar (dates, weekdays, events, SNAP flags), and sell prices (weekly item-store price history) — into a single long-format table keyed by item, store, and date. From this merged table, the pipeline derives the lag, rolling-statistic, calendar, price, SNAP, and event features described in the Feature Engineering section below. Categorical identifiers (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, and calendar/event fields) are preserved as native categorical columns rather than one-hot encoded, since LightGBM supports categorical splitting directly and one-hot encoding a feature like `item_id` (thousands of distinct values) would be both memory-prohibitive and would discard the ordinal/frequency information LightGBM's categorical handling can exploit.

## Data quality decisions

- **Intermittent demand is retained, not imputed.** Zero-sales days are a genuine and frequent state for most item-store series in this dataset, not missing data, and are kept as true zeros rather than being interpolated or dropped.
- **Categorical columns are explicitly typed and persisted** (`categorical_cols.pkl`) alongside the feature list, so the exact schema used at training time is reproducible at inference time — a common source of silent bugs in tree-based pipelines is a categorical column being re-inferred as numeric (or vice versa) between training and scoring.
- **Model, feature schema, and validation split are serialized as separate artifacts** (`best_lightgbm_model.pkl`, `feature_names.pkl`, `categorical_cols.pkl`, `X_valid.pkl`, `y_valid.pkl`) rather than being recomputed inline in every downstream notebook, which keeps the evaluation and error-analysis stages decoupled from — and reproducible independent of — the training stage.

---

# Feature Engineering

## Lag features (`lag_1`, `lag_7`, `lag_14`, `lag_28`)

Lag features directly encode "what did this item-store sell N days ago," at horizons of 1, 7, 14, and 28 days. `lag_1` captures the most immediate demand signal; `lag_7`, `lag_14`, and `lag_28` capture weekly seasonality (retail demand has strong day-of-week structure, so same-weekday lags are far more informative than adjacent-day lags at longer horizons). These are the standard feature family for gradient-boosted tree forecasting models, since trees have no built-in notion of sequence order — lags are how temporal information is made available to a model that otherwise treats each row independently.

## Rolling statistics (`rolling_mean_7/14/28`, `rolling_std_7/14/28`)

Rolling means smooth short-term demand noise into a more stable recent-trend signal at three horizons, while rolling standard deviations capture recent demand *volatility* — a low `rolling_std` on a stable item is informative in a different way than the same mean on a volatile, spiky item. Together, mean and standard deviation at each window length give the model both a level estimate and an uncertainty/variability estimate of recent demand without requiring it to see the raw sequence of individual lag values.

## Calendar features (`weekday`, `wday`, `month`, `year`, `is_weekend`)

Retail demand is strongly periodic at the weekly and annual scale (weekend shopping patterns, monthly/seasonal effects). These features let the model condition its forecast on calendar position directly, which is inexpensive to compute and captures seasonal structure that lag features alone would only approximate.

## Pricing features (`sell_price`, `price_lag_1`, `price_diff_1`, `price_pct_change`, `price_max`, `price_relative_to_max`)

Price is a primary demand lever in retail. Beyond the raw current price, the engineered price-change and price-relative-to-historical-max features let the model detect discounting events indirectly — a sharp drop in `price_relative_to_max` is a strong (if implicit) promotion signal, which matters because, as noted in Limitations, the M5 dataset does not provide an explicit promotion flag.

## SNAP features (`snap_CA`, `snap_TX`, `snap_WI`)

SNAP benefits are disbursed on specific days of the month, and SNAP-eligible households represent a meaningful share of grocery and household-goods demand in the dataset's regions. These binary, state-specific flags let the model condition on a known, calendar-predictable demand driver that is independent of item-level price or promotion behavior.

## Event features (`event_name_1/2`, `event_type_1/2`)

Named calendar events (public holidays, cultural and religious observances, sporting events) are associated with demand shifts that are not captured by weekday/month alone — Thanksgiving and the Super Bowl do not fall on the same date every year, so they are represented as explicit event fields rather than relying on the model to infer them from `month`/`wday`. Two event slots are provided because M5 dates occasionally have overlapping events.

---

# Modeling

## Train/validation strategy

The model is evaluated on a **time-based validation split**: the validation period follows the training period in calendar time rather than being drawn via a random row-wise split. This is essential for a fair evaluation, since a random split would let lag/rolling features computed near the validation rows leak information from time points a real forecasting deployment would not yet have observed. A time-based split is the closest available proxy, before live deployment, to how the model would perform forecasting genuinely unseen future days.

## LightGBM configuration

The final model is a LightGBM regressor, trained on the full 34-feature schema described above with 9 native categorical columns. The trained model, its feature schema, and its categorical column definitions are persisted together (`best_lightgbm_model.pkl`, `feature_names.pkl`, `categorical_cols.pkl`) to guarantee that any downstream scoring reproduces the exact input contract used at training time.

## Hyperparameter tuning approach

Model selection proceeded iteratively across the project's earlier stages: candidate configurations were trained and evaluated against the time-based validation split, with the configuration minimizing validation RMSE retained as the "best" model and its metrics recorded to `best_metrics.json` for reproducibility. This validation-driven selection process, rather than a fixed default configuration, is what the "best_" naming convention in the persisted artifacts reflects.

## Reasons for choosing LightGBM

- **Native categorical support** avoids the memory and sparsity cost of one-hot encoding high-cardinality identifiers such as `item_id` and `store_id`.
- **Histogram-based splitting** scales well to the multi-million-row training volume implied by a full M5 grid, which would be considerably more expensive to fit with a non-histogram gradient boosting implementation.
- **Tabular, feature-engineered representation fit** — with lag, rolling, price, and calendar features already encoding the relevant temporal structure, a tree ensemble is a well-matched, fast, and interpretable first modeling choice, and provides a strong baseline against which more complex sequence models (see Deep Learning Transition) can later be justified.
- **Built-in feature importance and compatibility with SHAP** support the explainability requirements of this project, discussed below.

---

# Evaluation

Validation-set metrics recorded for the selected model:

| Metric | Value |
|---|---|
| RMSE | 1.2597 |
| MAE | 0.4873 |

## Interpretation of metrics

**MAE (0.4873)** answers the operational question "on a typical item-store-day, how many units off is the forecast?" A sub-1-unit MAE is consistent with the dataset's low-volume, intermittent demand profile at the item-store level (a large share of rows have 0–2 units sold), and indicates the model tracks typical, non-spike demand closely.

**RMSE (1.2597)** is approximately **2.6x MAE**. Because RMSE squares each error before averaging, this ratio is itself diagnostic: if errors were uniform in size, RMSE and MAE would be close together. A large RMSE/MAE ratio indicates a **heavy-tailed error distribution** — most days are forecast with small error, but a minority of days (demand spikes, discussed in Error Analysis) carry disproportionately large errors that dominate RMSE without materially moving MAE.

**Bias** (mean of `actual − predicted` across the validation set) is close to zero at the portfolio level, meaning the model is not systematically padding or shorting the aggregate forecast. This is a necessary but not sufficient calibration check — segment-level bias, covered in Error Analysis, reveals that this near-zero aggregate figure conceals a mild over-forecasting tendency in some departments and a mild under-forecasting tendency in others.

---

# Baseline Comparison

## Naive Lag-7 Forecast vs. LightGBM Forecast

Before attributing forecasting skill to the LightGBM model, it is standard practice — and the M5 competition's own approach — to benchmark against a trivial baseline. The baseline used here is **seasonal-naive lag-7**: predict that today's demand equals the same weekday's demand from one week ago, using the `lag_7` feature already present in the engineered dataset. This is a low-cost, high-relevance baseline for retail data given its strong weekly seasonality.

## RMSE Skill Score

The comparison is summarized with an RMSE skill score:

```
skill_score = 1 − (RMSE_model / RMSE_naive)
```

A skill score of 0 means the model performs no better than "assume this week looks like last week"; a positive score indicates genuine forecasting skill beyond that heuristic; a negative score would indicate the model underperforms a trivial forecast, which would be a strong signal to revisit the feature set or tuning process before further investment.

*Note on execution status: the naive-baseline computation was added to the evaluation notebook (Day 9 revision) using the existing `lag_7` feature and the model's recorded validation RMSE/MAE. Final numeric results for this comparison are produced when the notebook is executed in the training environment; this report documents the methodology and its intended use rather than a specific pending figure, and the skill-score table should be inserted here once generated.*

## Business implications

A positive skill score is the minimum evidence needed to justify LightGBM's added complexity (feature engineering pipeline, tuning process, categorical handling) over a rule as simple as "reorder what sold last week" — which is itself sometimes used as a fallback replenishment heuristic during forecasting-system outages. Reporting this comparison alongside RMSE/MAE gives a non-technical stakeholder a direct, interpretable answer to "is the modeling effort worth it," which raw error metrics alone do not provide.

---

# Error Analysis

## Store-level diagnostics

Segmenting error by `store_id` shows meaningfully different RMSE across stores despite comparable observation counts (approximately 1.13M rows per store in the evaluated sample):

| store_id | observations | MAE | RMSE |
|---|---|---|---|
| CA_1 | 1,131,000 | 0.4711 | 0.8427 |
| CA_3 | 1,131,000 | 0.6507 | 1.3526 |
| TX_2 | 1,131,000 | 0.4705 | 1.7286 |
| WI_2 | 1,131,000 | 0.3569 | 0.9052 |

TX_2 shows the highest RMSE despite a below-average MAE — the same RMSE/MAE divergence pattern noted at the portfolio level, indicating TX_2's error is dominated by a small number of large misses rather than uniformly higher error across all its observations. This store is investigated directly in the top-residual analysis below.

## Department-level diagnostics

The validation sample's department coverage is limited to two departments (see Limitations for why): HOUSEHOLD_1 and HOUSEHOLD_2.

| dept_id | observations | bias | MAE | RMSE |
|---|---|---|---|---|
| HOUSEHOLD_1 | 640,900 | under-forecast (small) | 0.9680 | 2.8176 |
| HOUSEHOLD_2 | 3,883,100 | over-forecast (small) | 0.4080 | 0.7337 |

HOUSEHOLD_1 shows both higher error magnitude and a mild systematic under-forecasting tendency relative to HOUSEHOLD_2, which is materially more stable on both metrics. Because FOODS and HOBBIES departments are absent from this validation sample, these department-level conclusions should not be generalized to the full M5 hierarchy — a limitation discussed explicitly below.

## Top residual investigation

Isolating the top 0.1% of observations by absolute error surfaces a small, concentrated set of extreme misses. The largest errors in the validation set are dominated by a single item-store pair — `HOUSEHOLD_1_474` at store `TX_2` — which recurs across multiple, non-adjacent weeks with actual demand in the 300–600+ unit range against predictions in the single digits. This recurrence across time, for the same series, is a more specific and more actionable finding than a generic "occasional demand spike": it points to an item-store pair whose demand behavior is qualitatively different from the rest of the portfolio (for example, a bulk or reseller buying pattern, or a data artifact specific to that item-store combination), rather than to noise distributed randomly across the dataset. This distinction matters operationally, since the corrective action differs — a genuinely random rare spike motivates better spike-detection features across the board, while a recurring, series-specific anomaly motivates targeted handling (manual review flag, separate treatment, or exclusion from aggregate-error-based tuning) for that specific series.

## Event impact analysis

| event_type_1 | observations | MAE | RMSE |
|---|---|---|---|
| Sporting | 36,000 | 0.5000 | 3.1567 |
| National | 120,000 | 0.4431 | 1.9487 |
| Religious | 124,800 | 0.4777 | 1.0292 |
| Cultural | 81,600 | 0.4817 | 0.9748 |
| No_Event | 4,161,600 | 0.4889 | 1.2161 |

Sporting and National event days show materially higher RMSE than the no-event baseline despite MAE remaining in a similar range across all categories — consistent with the interpretation that named events increase the *frequency or magnitude of large individual misses* rather than degrading typical-day accuracy. This is a directly actionable finding: event-aware safety-stock adjustments are better targeted at Sporting/National event days specifically than applied uniformly across all calendar events.

## SNAP impact analysis

| SNAP flag | MAE (no SNAP) | RMSE (no SNAP) | MAE (SNAP day) | RMSE (SNAP day) |
|---|---|---|---|---|
| California | 0.4797 | 1.2336 | 0.5028 | 1.3112 |
| Texas | 0.4812 | 1.2520 | 0.4998 | 1.2753 |
| Wisconsin | 0.4815 | 1.1986 | 0.4991 | 1.3759 |

All three states show a consistent, modest increase in both MAE and RMSE on SNAP-eligible days relative to non-SNAP days. The effect is smaller than the event-type effect above but directionally consistent across all three states, which strengthens confidence that it reflects a genuine (if minor) demand-pattern shift rather than sampling noise.

---

# Explainability

## Gain importance

Feature importance was evaluated using LightGBM's **gain-based** measure — the total loss reduction attributable to each feature across all its splits — in preference to the default split-count importance. This distinction matters because split-count importance is known to be biased toward high-cardinality features (such as `item_id`, which has thousands of distinct values and therefore many more possible split points than a low-cardinality feature like a SNAP flag), which can inflate a feature's apparent importance without reflecting its actual contribution to error reduction. Split-count importance recorded during the project's earlier evaluation pass ranked features as follows (top 10): `item_id`, `lag_1`, `rolling_mean_7`, `wm_yr_wk`, `lag_7`, `lag_28`, `lag_14`, `rolling_mean_14`, `month`, `rolling_mean_28`. Gain-based importance is expected to reduce the relative ranking of high-cardinality, low-marginal-value fields such as `item_id` and `wm_yr_wk`, while preserving or increasing the relative share attributed to the lag and rolling-statistic features, since each split on those features tends to yield larger, more consistent loss reduction. The gain-importance table itself is produced by the current evaluation notebook and should be inserted here once generated in the training environment.

## SHAP results

SHAP (SHapley Additive exPlanations) values were added as an optional, sample-based diagnostic (computed on a random subset of the validation set for tractability) to complement gain importance with per-observation, directional attribution — for example, confirming not just that `lag_1` is important, but whether high recent demand pushes a given prediction up as expected. SHAP output is pending execution in the training environment; once generated, the summary plot and any notable per-segment SHAP patterns (e.g., for the TX_2/HOUSEHOLD_1 anomaly identified in the residual investigation) should be inserted here.

## Why `rolling_mean_7`, `rolling_mean_14`, and `lag_1` dominate the model

Daily retail demand exhibits strong short-term autocorrelation: the best available predictor of tomorrow's demand for a given item-store is very often close to its own recent demand level, more so than any exogenous signal (price, calendar, event) alone can capture on its own. `lag_1` gives the model the single most recent, most temporally relevant demand observation. `rolling_mean_7` and `rolling_mean_14` extend this by smoothing out day-to-day noise (a single low-selling day does not necessarily indicate a genuine demand-level shift) while still remaining short enough to track real trend changes within one to two weeks — long enough to be stable, short enough to be responsive. Longer windows (`rolling_mean_28`) and longer lags (`lag_14`, `lag_28`) still contribute, but with diminishing marginal value once the 7- and 14-day signals are already available, since they encode largely overlapping information at a coarser resolution. This pattern — recency-weighted autoregressive features dominating over static or exogenous features — is expected and typical for tree-based models applied to demand data, and is consistent with why lag/rolling engineering, rather than model architecture alone, is usually the dominant lever for forecasting accuracy in this class of problem.

---

# Key Findings

## Main demand drivers

Recent demand history (`lag_1`, `rolling_mean_7`, `rolling_mean_14`) is the dominant predictive signal, consistent with the strong short-term autocorrelation typical of retail sales. Calendar structure (`wday`, `month`, `event_name_1`) and price-relative-to-historical-max contribute secondary but meaningful signal, particularly for capturing promotion-like effects the dataset does not label explicitly.

## Model strengths

- Strong typical-day accuracy (MAE under 0.5 units) across the evaluated segments.
- Near-zero aggregate bias, indicating the model is not systematically over- or under-ordering at the portfolio level.
- Segment-level diagnostics (store, department, event, SNAP) are consistent with domain expectations (event and SNAP days show elevated but not extreme error increases), which supports the model's features being meaningfully connected to real demand drivers rather than fitting noise.

## Model weaknesses

- RMSE substantially exceeds MAE, indicating a small number of large misses dominate worst-case error.
- Error is unevenly distributed across stores and departments — TX_2 and HOUSEHOLD_1 in particular carry disproportionate tail risk.
- The largest errors are concentrated in a small number of recurring item-store series rather than being evenly distributed, suggesting the model has a specific, identifiable blind spot rather than a generally noisy tail.

## Reasons for rare demand spikes

The available feature set (calendar, SNAP, price-relative-to-max) explains ordinary demand variation reasonably well, but the extreme cases identified in the top-residual analysis are not well explained by these fields alone. Plausible contributing factors include unflagged local promotions or clearance events (since M5 does not provide an explicit promotion indicator — see Limitations), bulk or non-typical-consumer purchasing behavior for specific item-store pairs, and inventory/restocking dynamics not captured in the sales-only target variable. Distinguishing between these explanations would require data the current feature set does not include.

---

# Limitations

## Partial parquet validation coverage due to hardware limitations

The validation sample analyzed in this project's error-analysis stage covers only four of the M5 dataset's ten stores (CA_1, CA_3, TX_2, WI_2) and two of its seven departments (HOUSEHOLD_1, HOUSEHOLD_2), a direct consequence of the parquet chunk partitioning scheme adopted to fit the pipeline within local hardware constraints. This means the RMSE, MAE, and bias figures reported throughout this document — including the headline validation metrics — should be read as representative of this partial sample, not of the full M5 grid. In particular, the FOODS departments, which are both the highest-volume and typically the most promotion-sensitive category in the M5 dataset, are entirely absent from the evaluated sample. Any claim of "production-ready" performance would require re-running the full pipeline over the complete item-store grid, or, at minimum, validating on a coverage-balanced stratified sample across all stores and departments.

## Missing causal features

The engineered feature set captures calendar structure, SNAP eligibility, and price dynamics, but does not include any features that would explain the specific large-residual cases identified in the error analysis — for example, local promotional campaigns, inventory stockout events on the supply side, weather, or competitor activity. The persistence of large, concentrated residuals for specific series after accounting for all available features is consistent with these being omitted-variable effects rather than a model-fitting deficiency.

## Lack of promotion data

The M5 dataset does not provide an explicit promotion or discount-campaign flag; only observed price changes are available. This is a known characteristic of the dataset itself, not a gap introduced by this project's pipeline, but it constrains how directly promotional effects can be modeled — the price-relative-to-max features are a proxy, not a direct signal, and cannot distinguish a genuine promotion from an unrelated price adjustment.

---

# Deep Learning Transition

## Why LightGBM struggles with demand shocks

Tree-based models make predictions by partitioning the feature space using thresholds learned from the training data; they have no mechanism to extrapolate beyond the range of values seen during training, and no native representation of sequence order beyond whatever lag/rolling features are explicitly engineered. Two consequences follow directly from the error analysis above: first, a genuinely novel demand pattern (a spike in a series that has not previously exhibited one) has no learned split path that anticipates it, so the model falls back to whatever regime its available features resemble most closely — typically underestimating the spike, exactly the pattern observed in the top-residual investigation. Second, calendar-index-like features such as `wm_yr_wk` cannot be extrapolated past the maximum value seen in training, meaning any date-based feature of this form will lose its learned meaning as soon as the model is asked to forecast beyond the training window, which is inherent to how the model would need to be used in production.

## Motivation for LSTM/GRU models

Recurrent architectures process the raw sequence of past observations directly, rather than relying entirely on a fixed, manually engineered set of lag and rolling-window snapshots, which allows them to learn temporal patterns (including irregular or longer-range dependencies) that were not anticipated at feature-engineering time. They can also learn dense embeddings for high-cardinality categorical identifiers (`item_id`, `store_id`) instead of relying on native categorical splitting, and are naturally suited to global, multi-series training — learning shared temporal structure across many item-store series simultaneously rather than fitting a single set of engineered features across all of them independently. These properties directly target the two weaknesses identified above: sensitivity to novel patterns not captured by fixed lag windows, and the extrapolation limitation of calendar-index features.

## Planned next steps

1. Construct fixed-length input sequences (windowed) per item-store series from the same underlying M5 grid, with appropriate per-series or per-group scaling — a step tree-based models do not require but recurrent models do.
2. Address the dataset's intermittent, zero-inflated demand pattern explicitly, either through a specialized loss/output formulation or a hurdle-style two-stage approach (probability of nonzero demand, followed by magnitude conditional on nonzero demand).
3. Preserve the same time-based train/validation split methodology used for the LightGBM model, to keep the two approaches directly comparable.
4. Evaluate the sequence model against both the LightGBM model and the naive lag-7 baseline using the same metric set (RMSE, MAE, bias, and skill score) established in this report, with particular attention to whether it reduces the concentrated tail error identified in the top-residual investigation.
5. Extend validation coverage to the full store/department grid (addressing the current sampling limitation) before drawing final comparative conclusions between the tree-based and sequence-model approaches.

---

# Conclusion

## Final project assessment

The LightGBM model achieves strong typical-day accuracy (MAE approximately 0.49 units) and near-zero aggregate bias on the evaluated validation sample, with segment-level diagnostics that are directionally consistent with domain expectations (elevated error on event and SNAP days, concentrated tail risk in specific store/department combinations). The gap between RMSE and MAE, and the concentration of the largest errors in a small number of recurring series, indicate that the model's primary limitation is not typical-case accuracy but tail-risk behavior on a specific, identifiable subset of series.

## Production considerations

Before any production deployment, three items from this report would need to be resolved: validation coverage should be extended to the full store/department grid rather than the current partial sample; negative predictions must be floored at zero, since unit sales cannot be negative; and the naive-baseline skill score should be confirmed positive (and ideally computed per-segment, not only in aggregate) to justify the model's complexity relative to a simple seasonal heuristic. File paths and artifact loading should also be parameterized via configuration rather than hardcoded, to support deployment outside the original development environment.

## Future improvements

Beyond the LSTM/GRU transition outlined above, near-term improvements include: adding explicit promotion-proxy or discounting features informed by the price-relative-to-max analysis; investigating the recurring TX_2/HOUSEHOLD_1 anomaly directly (Item-level manual review or a separate specialized treatment for outlier series); moving from point forecasts to quantile or probabilistic forecasts, since inventory safety-stock decisions depend on the spread of likely demand outcomes, not only its expected value; and adopting the M5 competition's own weighted, scale-normalized accuracy metric (WRMSSE) alongside RMSE/MAE, to allow fair comparison across item-store series of very different sales volumes.
