"""
forecast_prophet.py
--------------------
Trains a Prophet forecasting model PER PRODUCT on weekly demand,
holds out the last 12 weeks to test accuracy, then forecasts
12 weeks into the future for real use.

WHY ONE MODEL PER PRODUCT (not one big model)?
Different products have different seasonal patterns and scale
(a phone charger and a winter jacket don't sell the same way).
Prophet is designed to be fit per-series. With 40 products this
is fast; at massive scale you'd look at other approaches, but
this is the right call here.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")  # Prophet is chatty with harmless warnings

# ---------------------------------------------------------------
# STEP 1: Load the weekly demand data we built with SQL
# ---------------------------------------------------------------
df = pd.read_csv("data/weekly_demand.csv")
df["week_start"] = pd.to_datetime(df["week_start"])

HOLDOUT_WEEKS = 12   # how many recent weeks we pretend "haven't happened yet"
FORECAST_WEEKS = 12  # how many weeks into the real future we forecast

all_results = []       # will hold accuracy metrics per product
all_forecasts = []     # will hold the actual future forecast values

product_ids = sorted(df["product_id"].unique())

for pid in product_ids:
    # ---------------------------------------------------------------
    # STEP 2: Reshape this one product's data into Prophet's required format
    # ---------------------------------------------------------------
    product_df = df[df["product_id"] == pid].sort_values("week_start")
    prophet_df = product_df[["week_start", "weekly_units"]].rename(
        columns={"week_start": "ds", "weekly_units": "y"}
    )

    if len(prophet_df) < HOLDOUT_WEEKS + 20:
        # Skip products with too little history to reliably train/test
        continue

    # ---------------------------------------------------------------
    # STEP 3: Train/test split -- last 12 weeks held out
    # ---------------------------------------------------------------
    train = prophet_df.iloc[:-HOLDOUT_WEEKS]
    test = prophet_df.iloc[-HOLDOUT_WEEKS:]

    # ---------------------------------------------------------------
    # STEP 4: Fit Prophet on the TRAINING data only
    # ---------------------------------------------------------------
    # yearly_seasonality=True: tells Prophet to look for a repeating
    #   yearly pattern (we know one exists -- we built it in).
    # weekly_seasonality=False: we've already aggregated to weekly level,
    #   so there's no "day of week" pattern left to find.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,  # gives us a 90% confidence interval on forecasts
    )
    model.fit(train)

    # ---------------------------------------------------------------
    # STEP 5: Predict over the TEST period, and check accuracy
    # ---------------------------------------------------------------
    test_future = test[["ds"]]  # Prophet just needs the dates to predict for
    test_pred = model.predict(test_future)

    y_true = test["y"].values
    y_pred = test_pred["yhat"].values
    y_pred = np.clip(y_pred, 0, None)  # forecasts can't be negative demand

    mae = mean_absolute_error(y_true, y_pred)
    # MAPE needs a small epsilon guard in case any true value is 0
    mape = mean_absolute_percentage_error(y_true + 1e-5, y_pred + 1e-5) * 100

    all_results.append({
        "product_id": pid,
        "test_mae": round(mae, 2),
        "test_mape_pct": round(mape, 2),
        "avg_weekly_demand": round(y_true.mean(), 1),
    })

    # ---------------------------------------------------------------
    # STEP 6: Now retrain on ALL data (including those last 12 weeks)
    # and forecast the REAL future -- this is what the business would
    # actually use, unlike the held-out test which was just for scoring.
    # ---------------------------------------------------------------
    full_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,
    )
    full_model.fit(prophet_df)

    future = full_model.make_future_dataframe(periods=FORECAST_WEEKS, freq="W")
    forecast = full_model.predict(future)
    future_only = forecast.tail(FORECAST_WEEKS)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_only["yhat"] = future_only["yhat"].clip(lower=0)
    future_only["product_id"] = pid

    all_forecasts.append(future_only)

# ---------------------------------------------------------------
# STEP 7: Save everything
# ---------------------------------------------------------------
results_df = pd.DataFrame(all_results)
forecasts_df = pd.concat(all_forecasts, ignore_index=True)

results_df.to_csv("data/prophet_accuracy_by_product.csv", index=False)
forecasts_df.to_csv("data/prophet_future_forecast.csv", index=False)

print("=== Prophet Accuracy Summary (across all products) ===")
print(f"Average MAE:  {results_df['test_mae'].mean():.2f} units/week")
print(f"Average MAPE: {results_df['test_mape_pct'].mean():.2f}%")
print()
print("Best 3 products (lowest MAPE):")
print(results_df.nsmallest(3, "test_mape_pct").to_string(index=False))
print()
print("Worst 3 products (highest MAPE):")
print(results_df.nlargest(3, "test_mape_pct").to_string(index=False))
print()
print("Saved: data/prophet_accuracy_by_product.csv")
print("Saved: data/prophet_future_forecast.csv")
