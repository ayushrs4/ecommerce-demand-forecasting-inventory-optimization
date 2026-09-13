"""
generate_data.py
-----------------
Creates a synthetic but realistic e-commerce transaction dataset and
stores it in a SQLite database.

WHY SYNTHETIC DATA?
Real retail transaction data is hard to get publicly (privacy reasons),
so we simulate it -- but we build in the same patterns real data has:
trend (slow growth), seasonality (yearly demand cycle), and noise
(day-to-day randomness). This lets us build and test a real forecasting
pipeline end-to-end.
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility: using a fixed random seed means anyone re-running this
# script gets the exact same "random" data. Good practice for any project
# with a random component -- makes results explainable and repeatable.
np.random.seed(42)

# ---------------------------------------------------------------
# STEP 1: Define the "world" -- products and the time range
# ---------------------------------------------------------------
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Grocery"]
N_PRODUCTS = 40

products = []
for i in range(1, N_PRODUCTS + 1):
    category = np.random.choice(CATEGORIES)
    # Different categories have different typical price ranges --
    # this makes later "profit margin" analysis meaningful.
    base_price = {
        "Electronics": np.random.uniform(1500, 15000),
        "Apparel": np.random.uniform(300, 3000),
        "Home & Kitchen": np.random.uniform(500, 5000),
        "Beauty": np.random.uniform(150, 1500),
        "Grocery": np.random.uniform(50, 800),
    }[category]
    unit_cost = base_price * np.random.uniform(0.55, 0.75)  # cost is 55-75% of price
    products.append({
        "product_id": i,
        "product_name": f"{category[:4].upper()}-{i:03d}",
        "category": category,
        "unit_price": round(base_price, 2),
        "unit_cost": round(unit_cost, 2),
    })
products_df = pd.DataFrame(products)

# 3 years of daily dates -- matches the reference project's timeframe,
# and gives Prophet enough history to learn a yearly seasonal pattern
# (you generally need 2+ full cycles of a season for a model to learn it reliably).
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
date_range = pd.date_range(start_date, end_date, freq="D")

# ---------------------------------------------------------------
# STEP 2: Simulate daily demand per product
# ---------------------------------------------------------------
def simulate_daily_demand(n_days, base_level):
    """
    Builds one product's daily demand series using three components:
    trend + seasonality + noise. This is the same decomposition Prophet
    itself uses internally -- so understanding this function IS understanding
    how Prophet thinks about a time series.
    """
    t = np.arange(n_days)

    # TREND: slow linear growth over the 3 years (e.g. 20% growth total)
    trend = base_level * (1 + 0.20 * (t / n_days))

    # SEASONALITY: a yearly sine wave. 2*pi*t/365 completes one full
    # cycle every 365 days. Amplitude controls how big the swing is.
    # We shift it so the peak lands around late November (festive season).
    day_of_year = t % 365
    seasonal = base_level * 0.35 * np.sin(2 * np.pi * (day_of_year - 300) / 365)

    # NOISE: random day-to-day variation (real demand is never a smooth curve)
    noise = np.random.normal(0, base_level * 0.15, n_days)

    demand = trend + seasonal + noise
    demand = np.clip(demand, 0, None)  # demand can't be negative
    return np.round(demand).astype(int)

all_transactions = []
transaction_id = 1

for _, prod in products_df.iterrows():
    # Each product has its own "typical" daily demand level --
    # some products just sell more than others.
    base_level = np.random.uniform(2, 25)
    daily_units = simulate_daily_demand(len(date_range), base_level)

    for day_idx, date in enumerate(date_range):
        units = daily_units[day_idx]
        if units == 0:
            continue  # no transaction that day for this product

        # Occasionally simulate a "stockout" -- a period where a product
        # sold ZERO despite demand existing. This matters later: it's a
        # realistic data-quality issue an analyst has to notice and handle.
        # We do this by randomly zeroing out ~0.5% of product-days.
        if np.random.random() < 0.005:
            continue

        # Promotions: ~8% of days are "promo days" with a demand boost.
        is_promo = np.random.random() < 0.08
        if is_promo:
            units = int(units * np.random.uniform(1.3, 1.8))

        all_transactions.append({
            "transaction_id": transaction_id,
            "date": date.strftime("%Y-%m-%d"),
            "product_id": prod["product_id"],
            "units_sold": units,
            "is_promo": int(is_promo),
        })
        transaction_id += 1

transactions_df = pd.DataFrame(all_transactions)

# ---------------------------------------------------------------
# STEP 3: Current inventory snapshot (needed later for reorder-point logic)
# ---------------------------------------------------------------
inventory = []
for _, prod in products_df.iterrows():
    inventory.append({
        "product_id": prod["product_id"],
        "current_stock": int(np.random.uniform(20, 400)),
        "lead_time_days": int(np.random.choice([3, 5, 7, 10, 14])),  # supplier delivery time
    })
inventory_df = pd.DataFrame(inventory)

# ---------------------------------------------------------------
# STEP 4: Write everything into a SQLite database
# ---------------------------------------------------------------
db_path = "/home/claude/demand-forecast/data/ecommerce_data.db"
conn = sqlite3.connect(db_path)

products_df.to_sql("products", conn, if_exists="replace", index=False)
transactions_df.to_sql("sales_transactions", conn, if_exists="replace", index=False)
inventory_df.to_sql("inventory_status", conn, if_exists="replace", index=False)

conn.close()

print(f"Products: {len(products_df)}")
print(f"Transactions: {len(transactions_df):,}")
print(f"Date range: {start_date.date()} to {end_date.date()}")
print(f"Saved to: {db_path}")
