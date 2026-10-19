# GhostChef AI -- Day 1 Summary

**Date:** Day 1

## Objective

Understand the business problem, the datasets, and design the project
before writing machine learning code.

------------------------------------------------------------------------

# Part 1 -- Business Understanding

## Business Problem

Restaurants often: - Buy more ingredients than needed → Food Waste. -
Buy fewer ingredients than needed → Stockouts and lost sales.

## Goal

Build an AI system that predicts future demand to: - Reduce waste. -
Reduce stock shortages. - Increase profit. - Improve inventory planning.

------------------------------------------------------------------------

# Part 2 -- Business Requirements

## Users

-   Kitchen Manager
-   Inventory Manager
-   Operations Manager

## Forecast Levels

-   Daily (Version 1)
-   Weekly (Future)
-   Monthly (Future)

Decision: The first release will focus on **Daily Demand Forecasting**.

------------------------------------------------------------------------

# Part 3 -- ML Problem Definition

-   Problem Type: Time Series Forecasting (Regression)
-   Target: Future demand quantity.
-   Prediction Level: Product × Store.
-   Strategy: Global Model.
-   Cold Start will be handled using similar products and metadata.

------------------------------------------------------------------------

# Part 4 -- Data Understanding

Datasets:

1.  sales_train_validation.csv
2.  calendar.csv
3.  sell_prices.csv

Key Findings: - Each row in sales represents one Product inside one
Store. - Columns d_1 ... d_1913 represent daily sales history. -
calendar.csv maps d_x to real calendar dates and events. -
sell_prices.csv stores weekly prices.

------------------------------------------------------------------------

# Part 5 -- Data Schema

Relationships:

sales → calendar (using d)

calendar → sell_prices (using wm_yr_wk)

sell_prices also requires: - item_id - store_id

Important lesson: Prices cannot be merged directly with sales.

------------------------------------------------------------------------

# Part 6 -- Project Architecture

Main folders:

-   data/
-   notebooks/
-   src/
-   models/
-   docs/
-   dashboard/
-   reports/
-   deployment/

Rule: Each notebook has one responsibility.

------------------------------------------------------------------------

# Part 7 -- ML Roadmap

1.  Business Understanding
2.  Data Understanding
3.  Data Merge
4.  EDA
5.  Data Cleaning
6.  Feature Engineering
7.  Baseline Model
8.  Model Improvement
9.  Evaluation
10. Business Insights
11. Dashboard
12. Deployment

------------------------------------------------------------------------

# Part 8 -- Design Decisions

Why Regression? - Predict demand quantity.

Why Daily Forecast? - Better operational planning.

Why Global Model? - Scalable and easier to maintain.

Why Baseline? - Establish a performance reference before advanced
models.

------------------------------------------------------------------------

# Part 9 -- Risk Analysis

Main Risks: - Cold Start products. - Demand variability. - Store
closures. - Price changes. - Special events.

Key Lesson: Not every unusual value is an outlier.

------------------------------------------------------------------------

# Part 10 -- Final Review

Major Takeaways

-   Think Business before AI.
-   Understand data before modeling.
-   Design the architecture before coding.
-   Metrics alone do not define project success.
-   AI supports decisions; managers make decisions.

------------------------------------------------------------------------

# Deliverables Completed

-   Business Understanding
-   Business Requirements
-   ML Problem Definition
-   Data Understanding
-   Data Schema
-   Project Architecture
-   ML Roadmap
-   Design Decisions
-   Risk Analysis

------------------------------------------------------------------------

# Next Day (Day 2)

-   Transform sales from Wide to Long format.
-   Merge datasets.
-   Validate merged dataset.
-   Prepare for EDA.

------------------------------------------------------------------------

## Day 1 Status

**Completed Successfully ✅**
