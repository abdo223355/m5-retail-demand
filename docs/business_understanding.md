# GhostChef AI - Design Decisions

## Project Information

**Project Name:** GhostChef AI - Demand Forecasting & Inventory Optimization

**Objective:**
Develop an AI-powered decision support system that predicts future demand for each product at each store, helping reduce inventory waste, prevent stock shortages, and improve operational efficiency.

---

# Day 1

## Part 1 - Business Understanding

### Business Problem

GhostChef currently relies on manual estimates to determine inventory purchases and food preparation quantities.

This leads to two major operational problems:

* Over-purchasing ingredients, resulting in food waste and unnecessary costs.
* Under-purchasing ingredients, causing stock shortages, canceled orders, and lost revenue.

The company needs a data-driven system that predicts future demand accurately to support inventory planning.

---

### Business Objectives

Primary Objective:

* Improve demand forecasting accuracy to support inventory decisions.

Business Outcomes:

* Reduce food waste.
* Reduce stockouts.
* Increase order fulfillment rate.
* Improve customer satisfaction.
* Increase profitability.
* Optimize inventory management.

---

## Part 2 - Business Requirements

### End Users

The system will support three main users.

#### Kitchen Manager

Needs daily demand predictions to prepare food efficiently.

#### Inventory Manager

Needs recommended purchasing quantities for each product.

#### Operations Manager

Needs dashboards showing overall business performance, demand trends, and operational insights.

---

### Forecast Horizon

The client requires multiple forecasting levels.

* Daily Forecast (highest priority)
* Weekly Forecast
* Monthly Forecast

The first version of the project will focus on **Daily Demand Forecasting**.

Future versions will extend to weekly and monthly forecasting.

---

## Part 3 - Machine Learning Problem Definition

### ML Problem Type

The project is formulated as:

**Time Series Forecasting using Regression.**

Reason:

The target variable is a continuous numerical value (future demand), while historical temporal patterns play a major role in prediction.

---

### Prediction Level (Granularity)

Predictions should be generated for:

**Each Product inside each Store.**

Business example:

Store A → Product X

Store B → Product X

Demand patterns differ across stores even for the same product.

---

### Modeling Strategy

Instead of building one model for every product-store combination, the project will use a **Global Machine Learning Model**.

The model will learn from all products and all stores simultaneously.

Store-specific and product-specific information will be provided through input features such as:

* Store ID
* Product ID
* Category
* Department
* Date
* Price
* Events
* Calendar Features

This approach is more scalable and practical than training thousands of separate models.

---

### Cold Start Consideration

New products without historical sales represent a Cold Start problem.

Possible solutions:

* Use product metadata (category, department, price, etc.).
* Leverage similar products.
* Create similarity groups using clustering as an additional feature if needed.

This decision will be evaluated during feature engineering.

---

## Current Decisions

✔ Business problem clearly defined.

✔ Business objectives identified.

✔ Target users identified.

✔ Forecast horizon selected.

✔ Machine Learning problem defined.

✔ Global Model selected.

✔ Cold Start strategy identified.

---

