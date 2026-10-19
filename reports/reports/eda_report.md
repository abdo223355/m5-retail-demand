# Day 3 — Exploratory Data Analysis (EDA)

## Project

GhostChef AI – Demand Forecasting System

---

# Objective

The objective of this phase is to understand the merged dataset, evaluate data quality, identify potential issues, analyze feature distributions, and extract business insights before preprocessing and feature engineering.

---

# Dataset Overview

| Item | Value |
|------|------:|
| Original Dataset Size | 58,327,370 rows |
| EDA Sample Size | 5,832,737 rows |
| Number of Features | 23 |

> **Note**
>
> Due to the large size of the merged dataset (~58 million rows), a representative random sample (10%) was used for Exploratory Data Analysis (EDA). The complete dataset will be used during preprocessing and model training.

---

# Data Quality Assessment

## Missing Values

### Observation

The following columns contain missing values:

| Column | Missing % |
|---------|----------:|
| event_name_2 | 99.79% |
| event_type_2 | 99.79% |
| event_name_1 | 91.94% |
| event_type_1 | 91.94% |
| sell_price | 21.09% |

### Business Interpretation

- Missing values in event-related columns indicate normal days without special events.
- Missing values in `sell_price` occur because some products were unavailable or had no recorded price during specific weeks.

### Decision

- Keep event columns.
- Replace missing event values with **"No_Event"** during preprocessing.
- Investigate missing `sell_price` values in Day 4.

---

# Duplicate Analysis

## Observation

- Duplicate Rows = **0**
- Duplicate Keys (`item_id`, `store_id`, `day`) = **0**

### Business Interpretation

The merge process preserved data integrity and did not introduce duplicate observations.

### Decision

No duplicate removal is required.

---

# Descriptive Statistics

## Sales

| Metric | Value |
|--------|------:|
| Mean | 1.13 |
| Median | 0 |
| Maximum | 485 |

### Observation

Sales are highly right-skewed.

Most observations correspond to zero or very low sales.

### Business Interpretation

Most products sell only a few units on a typical day, while a small number of observations represent periods of exceptionally high demand.

---

## Sell Price

| Metric | Value |
|--------|------:|
| Mean | 4.41 |
| Median | 3.47 |
| Maximum | 107.32 |

### Observation

Sell prices are concentrated at lower values with a long right tail.

---

# Sales Distribution

## Observation

The sales histogram reveals a highly right-skewed distribution.

Most observations occur at very low sales values.

### Business Interpretation

Daily demand is sparse for many products, while only a limited number of observations represent high-demand periods.

### Decision

No transformation is applied during EDA.

---

# Outlier Analysis

## Observation

The boxplot identified many statistical outliers.

### Business Interpretation

These observations most likely represent genuine demand spikes caused by promotions, holidays, seasonal effects, or naturally high-demand products.

### Decision

Sales outliers will not be removed.

---

# Time Series Analysis

## Observation

Daily aggregated sales show:

- An increasing long-term trend.
- Recurring seasonal fluctuations.
- Several demand peaks.

### Business Interpretation

Customer demand grows over time and follows recurring temporal patterns.

### Decision

Temporal variables will be retained for forecasting.

---

# Monthly Analysis

## Observation

Average sales vary slightly across months.

Highest average demand:

- August
- September
- July

Lowest average demand:

- May
- December

### Business Interpretation

Monthly seasonality exists but its overall impact appears moderate.

---

# Weekly Analysis

## Observation

Highest average sales:

1. Saturday
2. Sunday
3. Friday

Lowest average sales:

- Wednesday

### Business Interpretation

Customer demand is generally higher during weekends.

### Decision

`weekday` will be retained as an important temporal feature.

---

# Event Analysis

## Observation

Average demand differs across event categories.

### Business Interpretation

Special events influence customer purchasing behavior.

### Decision

Event-related features will be retained.

---

# SNAP Analysis

## Observation

Average sales increase during SNAP days.

### Business Interpretation

Government food assistance periods are associated with increased customer demand.

### Decision

SNAP variables should remain in the dataset.

---

# Store Analysis

## Observation

Sales vary considerably across stores.

### Business Interpretation

Customer demand differs by store location.

### Decision

`store_id` is an important categorical feature.

---

# Department Analysis

## Observation

Sales volume differs across departments.

The highest demand was observed in:

- FOODS_3
- HOUSEHOLD_1
- FOODS_2

### Business Interpretation

Different product departments exhibit distinct purchasing patterns.

### Decision

`dept_id` will be retained.

---

# Sell Price Analysis

## Observation

The Pearson correlation between sales and sell price is:

**Correlation = -0.15**

### Business Interpretation

Higher-priced products tend to have slightly lower demand, although the linear relationship is weak.

### Decision

Keep `sell_price` as an important predictive feature.

---

# Price Quantile Analysis

## Observation

Average sales decrease consistently as price increases.

| Price Quantile | Average Sales |
|----------------|--------------:|
| Q1 | 3.31 |
| Q2 | 1.31 |
| Q3 | 1.16 |
| Q4 | 0.76 |
| Q5 | 0.59 |

### Business Interpretation

Lower-priced products generally experience higher average demand.

This relationship is associative and should not be interpreted as causal.

### Decision

Further investigate interactions between price and other variables during Feature Engineering.

---

# Key Business Insights

1. The merged dataset has high overall quality.
2. No duplicate observations were introduced during merging.
3. Missing event values represent normal business days rather than data quality issues.
4. Sales exhibit a highly right-skewed distribution.
5. Statistical outliers correspond to realistic demand spikes.
6. Customer demand shows a long-term increasing trend.
7. Weekly seasonality is clearly present.
8. SNAP periods are associated with higher average demand.
9. Lower-priced products generally achieve higher average sales.
10. Price alone is insufficient to explain customer demand.

---

# Design Decisions

- Use the sampled dataset only for EDA.
- Use the complete dataset during preprocessing and model training.
- Preserve sales outliers.
- Preserve event-related features.
- Preserve SNAP variables.
- Preserve temporal variables.
- Preserve sell price.
- Delay preprocessing decisions until Day 4.

---

# Challenges

- Extremely large merged dataset (~58 million rows).
- High memory consumption after merging.
- Required sampling strategy to perform efficient EDA.

---

# Conclusion

The exploratory analysis successfully evaluated data quality, identified important business patterns, and established a solid understanding of the dataset.

The extracted insights will directly guide the preprocessing, feature engineering, and modeling stages.

---

# Next Step

## Day 4 — Data Preprocessing

Tasks:

- Handle missing values.
- Optimize data types.
- Reduce memory usage.
- Prepare categorical features.
- Prepare the dataset for Feature Engineering.
