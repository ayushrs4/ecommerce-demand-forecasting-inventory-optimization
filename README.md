# E-Commerce Demand Forecasting & Inventory Optimization

An end-to-end pipeline that forecasts weekly product demand and converts those forecasts into concrete inventory decisions (safety stock, reorder points, and risk flags) — benchmarking a statistical time-series model (Prophet) against a feature-engineered gradient-boosted model (XGBoost).

## Overview

Retailers face a constant trade-off: overstock and tie up cash in unsold inventory, or understock and lose sales. This project builds a forecasting pipeline that predicts product-level demand and translates that forecast directly into an actionable reorder recommendation — the same problem quick-commerce and e-commerce operations teams solve daily.

## Pipeline Architecture

```
Synthetic transaction data (3 yrs, 40 products, SQLite)
        │
        ▼
SQL aggregation → weekly demand per product
        │
        ├──► Prophet (per-product time-series model)
        │        └─ provides forecast + confidence interval
        │
        └──► XGBoost (global model, engineered lag/rolling features)
                 └─ provides higher point-forecast accuracy
        │
        ▼
Inventory logic: safety stock + reorder point
  (uses Prophet's confidence interval to size the safety buffer)
        │
        ▼
Power BI dashboard (4 pages): Executive Overview, Demand Forecast
Explorer, Model Comparison, Inventory Recommendations
```

## Key Results

| Metric | Prophet | XGBoost |
|---|---|---|
| Average MAPE | 14.23% | **7.32%** |
| Average MAE (units/week) | 13.40 | **7.65** |
| Products where it won | 0/40 | 40/40 |

- XGBoost improved forecast accuracy by **~48.6%** over Prophet, primarily driven by two engineered features: a 4-week rolling average (`roll_mean_4`, 75% feature importance) and a 1-week lag — recent momentum mattered far more than the yearly seasonal signal (`lag_52`, only ~4% importance) for this dataset.
- Despite Prophet's lower point-forecast accuracy, **its native confidence intervals were used to size safety stock** — a deliberate trade-off, since XGBoost only produces a single point estimate without additional work (e.g. quantile regression).
- The inventory logic flagged **15 of 40 products as Critical** (current stock at or below reorder point) and 2 as Warning, based on a 95% service level (Z = 1.645).

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Forecasting** — Prophet, XGBoost
- **Database** — SQLite, SQL (window functions, aggregation, CASE logic)
- **Visualization** — Power BI, DAX

## Project Structure

```
demand-forecast/
├── data/
│   ├── ecommerce_data.db              # SQLite database (products, transactions, inventory)
│   ├── weekly_demand.csv              # SQL-aggregated weekly demand per product
│   ├── prophet_accuracy_by_product.csv
│   ├── prophet_future_forecast.csv
│   ├── xgboost_accuracy_by_product.csv
│   ├── model_comparison.csv
│   └── inventory_recommendations.csv
├── scripts/
│   ├── generate_data.py               # synthetic data generator (trend + seasonality + noise)
│   ├── forecast_prophet.py            # per-product Prophet forecasting + evaluation
│   ├── forecast_xgboost.py            # global XGBoost model with engineered features
│   └── inventory_optimization.py      # safety stock, reorder point, risk classification
├── dashboard/
│   └── inventory_dashboard.pbix       # 4-page Power BI dashboard
├── requirements.txt
└── README.md
```

## Methodology Notes

**Why generate synthetic data?** To build and test a genuinely complete pipeline (including known ground-truth trend/seasonality), the dataset was generated with realistic patterns: linear growth trend, yearly seasonality (festive-season peak), day-to-day noise, occasional stockout days, and promotional demand spikes.

**Why a time-based train/test split?** The most recent 12 weeks were held out (not a random split) to avoid training on future data — a random split would leak information and produce an unrealistically optimistic accuracy score.

**Why one Prophet model per product but one global XGBoost model?** Prophet is designed to fit per-series and each product has a distinct scale and pattern. XGBoost, given `product_id` as a feature, can instead learn shared cross-product patterns from a single larger model.

**Why `sqrt(lead_time)` in the safety stock formula?** Variance (not standard deviation) accumulates additively across independent time periods; since standard deviation is the square root of variance, uncertainty over a lead time grows with the square root of that lead time, not linearly.

## Dashboard

Four pages: **Executive Overview** (KPI summary + risk distribution), **Demand Forecast Explorer** (actual vs. forecasted demand over time), **Model Comparison** (Prophet vs. XGBoost accuracy, per product), and **Inventory Recommendations** (full action list with conditional formatting on risk status).

*(Screenshots in `/dashboard/screenshots`)*

## Future Enhancements

- Multi-step recursive forecasting for XGBoost (currently only backtested; Prophet drives the production forecast)
- Quantile regression for XGBoost to produce native confidence intervals
- Hyperparameter tuning (grid/Bayesian search) rather than default XGBoost parameters
- Class-imbalance-aware evaluation if extended to a churn-style classification task

## Author

Ayush Raj Singh — [LinkedIn](#) · [GitHub](#)
