"""
forecast_xgboost.py
--------------------
Builds hand-crafted time-series features (lags, rolling averages,
calendar signals) and trains ONE global XGBoost model across all
products -- then compares its accuracy against Prophet's.

WHY FEATURE ENGINEERING?
XGBoost has no built-in concept of "time" -- it just sees rows of
numbers. To let it use time-series patterns, we manually create
features that describe "what happened recently":
  - lag features:      what did this product sell N weeks ago?
  - rolling averages:  what's the recent trend/level?
  - calendar features: what point in the year is it? (proxy for seasonality)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# STEP 1: Load data and sort properly
# ---------------------------------------------------------------
df = pd.read_csv("data/weekly_demand.csv")
df["week_start"] = pd.to_datetime(df["week_start"])
df = df.sort_values(["product_id", "week_start"]).reset_index(drop=True)

# ---------------------------------------------------------------
# STEP 2: Feature engineering
# ---------------------------------------------------------------
# .groupby("product_id") is CRITICAL here -- it makes sure lag/rolling
# features for Product A never accidentally pull in Product B's numbers.
# Without this, we'd be leaking information across unrelated products.
g = df.groupby("product_id")["weekly_units"]

df["lag_1"] = g.shift(1)    # demand 1 week ago
df["lag_2"] = g.shift(2)    # demand 2 weeks ago
df["lag_4"] = g.shift(4)    # demand 4 weeks ago (roughly 1 month)
df["lag_52"] = g.shift(52)  # demand 52 weeks ago -- SAME WEEK LAST YEAR.
                             # This is how we hand XGBoost a seasonality
                             # signal Prophet gets automatically.

# Rolling averages -- shift(1) first so the current week's own value
# never leaks into its own rolling average (that would be cheating).
df["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
df["roll_mean_8"] = g.transform(lambda s: s.shift(1).rolling(8).mean())

# Calendar features -- proxy for seasonality, since raw week numbers
# alone don't tell XGBoost "this is festive season."
df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
df["month"] = df["week_start"].dt.month

# Rows at the start of each product's history won't have enough lag
# history yet (e.g. lag_52 needs a full year of prior data) -- drop them.
df_model = df.dropna(subset=["lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_mean_8"]).copy()

FEATURES = ["product_id", "lag_1", "lag_2", "lag_4", "lag_52",
            "roll_mean_4", "roll_mean_8", "week_of_year", "month", "promo_days"]
TARGET = "weekly_units"

# ---------------------------------------------------------------
# STEP 3: Time-based train/test split (same principle as Prophet --
# split by DATE, not randomly, so we never train on the future)
# ---------------------------------------------------------------
HOLDOUT_WEEKS = 12
cutoff_date = df_model["week_start"].max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)

train = df_model[df_model["week_start"] <= cutoff_date]
test = df_model[df_model["week_start"] > cutoff_date]

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ---------------------------------------------------------------
# STEP 4: Train XGBoost
# ---------------------------------------------------------------
# n_estimators: how many trees to build (boosting rounds)
# max_depth: how complex each individual tree can be (shallower = less overfitting)
# learning_rate: how much each tree corrects the previous ones (lower = more conservative)
# These are reasonable defaults for a dataset this size -- not heavily tuned,
# which is an honest thing to say if asked ("I used sensible defaults;
# hyperparameter tuning via grid search would be a natural next step").
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# STEP 5: Evaluate
# ---------------------------------------------------------------
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

overall_mae = mean_absolute_error(y_test, y_pred)
overall_mape = mean_absolute_percentage_error(y_test + 1e-5, y_pred + 1e-5) * 100

print("=== XGBoost Accuracy (global model, held-out last 12 weeks) ===")
print(f"Overall MAE:  {overall_mae:.2f} units/week")
print(f"Overall MAPE: {overall_mape:.2f}%")

# Per-product breakdown, so we can compare apples-to-apples against
# Prophet's per-product numbers from forecast_prophet.py
test_results = test.copy()
test_results["prediction"] = y_pred
per_product = test_results.groupby("product_id").apply(
    lambda d: pd.Series({
        "xgb_mae": mean_absolute_error(d["weekly_units"], d["prediction"]),
        "xgb_mape_pct": mean_absolute_percentage_error(d["weekly_units"] + 1e-5, d["prediction"] + 1e-5) * 100,
    })
).reset_index()

per_product.to_csv("data/xgboost_accuracy_by_product.csv", index=False)
print()
print("Saved: data/xgboost_accuracy_by_product.csv")

# ---------------------------------------------------------------
# STEP 6: Feature importance -- WHICH features actually drove predictions?
# This is the interpretability angle for XGBoost, similar to how
# logistic regression coefficients were interpretable in your churn model.
# ---------------------------------------------------------------
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print()
print("=== Feature Importance ===")
print(importance.to_string(index=False))

# ---------------------------------------------------------------
# STEP 7: Compare directly against Prophet, if that file exists
# ---------------------------------------------------------------
try:
    prophet_results = pd.read_csv("data/prophet_accuracy_by_product.csv")
    comparison = prophet_results.merge(per_product, on="product_id", how="inner")
    comparison["winner"] = np.where(
        comparison["xgb_mape_pct"] < comparison["test_mape_pct"], "XGBoost", "Prophet"
    )
    comparison.to_csv("data/model_comparison.csv", index=False)

    print()
    print("=== Prophet vs XGBoost -- head to head ===")
    print(f"Prophet avg MAPE:  {comparison['test_mape_pct'].mean():.2f}%")
    print(f"XGBoost avg MAPE:  {comparison['xgb_mape_pct'].mean():.2f}%")
    print(f"XGBoost wins on {(comparison['winner']=='XGBoost').sum()} / {len(comparison)} products")
    print()
    print("Saved: data/model_comparison.csv")
except FileNotFoundError:
    print()
    print("(Run forecast_prophet.py first if you want the head-to-head comparison file.)")
