"""
inventory_optimization.py
--------------------------
Turns Prophet's forecast (with its uncertainty interval) into concrete
inventory decisions: safety stock, reorder point, and a risk flag per
product ("Healthy" / "Warning" / "Critical").

WHY WE NEED UNCERTAINTY, NOT JUST A POINT FORECAST:
If we only had "expected demand = 100 units/week", we might order
exactly enough for 100 -- but if actual demand comes in at 130 one
week (which happens), we'd stock out. Safety stock exists specifically
to buffer against that variability. This is WHY we used Prophet's
forecast here instead of XGBoost's point estimate.
"""

import pandas as pd
import numpy as np
import sqlite3

# ---------------------------------------------------------------
# STEP 1: Load Prophet's future forecast + current inventory status
# ---------------------------------------------------------------
forecast = pd.read_csv("data/prophet_future_forecast.csv")
conn = sqlite3.connect("data/ecommerce_data.db")
inventory = pd.read_sql("SELECT * FROM inventory_status", conn)
products = pd.read_sql("SELECT product_id, product_name, category FROM products", conn)
conn.close()

# ---------------------------------------------------------------
# STEP 2: Summarize each product's forecast into the numbers we need
# ---------------------------------------------------------------
# Z-score note: Prophet's interval_width=0.90 (set in forecast_prophet.py)
# means yhat_lower/yhat_upper form a 90% confidence interval.
# For a normal distribution, a 90% interval spans +/- 1.645 standard
# deviations from the mean. We use this to back out an implied
# standard deviation of weekly demand -- a standard statistical trick
# for converting a confidence interval into a usable sigma.
Z_90 = 1.645

summary = forecast.groupby("product_id").agg(
    avg_weekly_forecast=("yhat", "mean"),
    avg_upper=("yhat_upper", "mean"),
    avg_lower=("yhat_lower", "mean"),
).reset_index()

summary["implied_std_dev"] = (summary["avg_upper"] - summary["avg_lower"]) / (2 * Z_90)

# ---------------------------------------------------------------
# STEP 3: Merge in current stock and supplier lead time
# ---------------------------------------------------------------
df = summary.merge(inventory, on="product_id").merge(products, on="product_id")
df["lead_time_weeks"] = df["lead_time_days"] / 7

# ---------------------------------------------------------------
# STEP 4: Safety stock formula
# ---------------------------------------------------------------
# Safety Stock = Z * sigma_demand * sqrt(lead time)
# WHY sqrt(lead_time)? Variance adds up over time, not the standard
# deviation directly -- this comes from how variance combines across
# independent periods (a standard result in statistics). A longer lead
# time means more weeks where uncertainty can accumulate, so the
# required buffer grows with the SQUARE ROOT of lead time, not linearly.
# We use a 95% service level here (Z=1.645) -- meaning we're comfortable
# NOT stocking out about 95% of the time. This is a business decision,
# not a fixed rule -- a premium brand might choose 99%, a low-margin
# commodity might accept 90%.
SERVICE_LEVEL_Z = 1.645

df["safety_stock"] = (
    SERVICE_LEVEL_Z * df["implied_std_dev"] * np.sqrt(df["lead_time_weeks"])
).round(0)

# ---------------------------------------------------------------
# STEP 5: Reorder point
# ---------------------------------------------------------------
df["reorder_point"] = (
    df["avg_weekly_forecast"] * df["lead_time_weeks"] + df["safety_stock"]
).round(0)

# ---------------------------------------------------------------
# STEP 6: Risk classification -- turn numbers into a business-readable flag
# ---------------------------------------------------------------
def classify_risk(row):
    if row["current_stock"] <= row["reorder_point"]:
        return "Critical"       # already at or below reorder point -- order NOW
    elif row["current_stock"] <= row["reorder_point"] * 1.25:
        return "Warning"        # getting close, keep an eye on it
    else:
        return "Healthy"        # comfortably stocked

df["risk_status"] = df.apply(classify_risk, axis=1)
df["recommended_order_qty"] = np.where(
    df["risk_status"] != "Healthy",
    (df["reorder_point"] * 1.5 - df["current_stock"]).clip(lower=0).round(0),
    0
)

# ---------------------------------------------------------------
# STEP 7: Save results
# ---------------------------------------------------------------
output_cols = [
    "product_id", "product_name", "category", "current_stock",
    "avg_weekly_forecast", "lead_time_weeks", "safety_stock",
    "reorder_point", "risk_status", "recommended_order_qty"
]
final = df[output_cols].sort_values("risk_status")
final.to_csv("data/inventory_recommendations.csv", index=False)

print("=== Inventory Risk Summary ===")
print(final["risk_status"].value_counts().to_string())
print()
print("=== Products needing action (Critical + Warning) ===")
action_needed = final[final["risk_status"] != "Healthy"]
print(action_needed.to_string(index=False))
print()
print("Saved: data/inventory_recommendations.csv")
