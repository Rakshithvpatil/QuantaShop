"""
ingestion/seed_data.py
─────────────────────
Generates realistic mock e-commerce + ERP data and loads it
into DuckDB so the project runs without a real BigCommerce account.

Run: python ingestion/seed_data.py
"""

import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "./data/swiftpulse.duckdb")
os.makedirs("./data", exist_ok=True)

# ── Seed configuration ────────────────────────────────────────────────────────
SKUS = [
    "SHOE-001", "SHOE-002", "BAG-001", "HAT-001",
    "GLASS-001", "GLASS-002", "WATCH-001", "BELT-001",
]
WAREHOUSES = ["Irving-TX", "Chicago-IL", "Seattle-WA"]
N_DAYS = 365       # 1 year of daily order history
N_PRODUCTS = 50    # product catalog size


def generate_orders(skus: list, n_days: int) -> pd.DataFrame:
    """Creates daily order demand with trend + seasonality + noise."""
    rng = np.random.default_rng(42)
    rows = []
    base_date = datetime(2023, 6, 1)

    for sku in skus:
        # Each SKU has its own base demand and seasonal pattern
        base_demand = rng.integers(10, 80)
        trend = rng.uniform(-0.05, 0.15) / n_days  # slight upward/downward trend

        for day in range(n_days):
            date = base_date + timedelta(days=day)
            # Weekly seasonality: weekend dip
            weekly_factor = 0.7 if date.weekday() >= 5 else 1.0
            # Monthly seasonality: peak mid-month
            monthly_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date.day / 30)
            # Trend component
            trend_factor = 1.0 + trend * day
            # Random noise
            noise = rng.normal(1.0, 0.12)

            quantity = max(
                0,
                int(base_demand * weekly_factor * monthly_factor * trend_factor * noise),
            )
            rows.append(
                {
                    "order_id": f"ORD-{sku}-{day:04d}",
                    "sku": sku,
                    "ordered_at": date.isoformat(),
                    "quantity": quantity,
                    "unit_price": round(rng.uniform(19.99, 299.99), 2),
                    "customer_id": f"CUST-{rng.integers(1000, 9999)}",
                    "channel": rng.choice(["web", "mobile", "marketplace"]),
                }
            )

    return pd.DataFrame(rows)


def generate_inventory(skus: list, warehouses: list) -> pd.DataFrame:
    """Mock ERP inventory levels per SKU per warehouse."""
    rng = np.random.default_rng(99)
    rows = []
    for sku in skus:
        for wh in warehouses:
            rows.append(
                {
                    "sku": sku,
                    "warehouse": wh,
                    "quantity_on_hand": int(rng.integers(50, 500)),
                    "reorder_point": int(rng.integers(20, 80)),
                    "lead_time_days": int(rng.integers(3, 14)),
                    "updated_at": datetime.now().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def generate_products(n: int) -> pd.DataFrame:
    """Mock product catalog with descriptions (used by RAG pipeline)."""
    rng = np.random.default_rng(7)
    categories = ["Footwear", "Bags", "Accessories", "Wearables", "Eyewear"]
    rows = []
    for i in range(n):
        cat = rng.choice(categories)
        rows.append(
            {
                "product_id": f"PROD-{i:04d}",
                "sku": f"{cat[:3].upper()}-{i:03d}",
                "name": f"{cat} Product {i}",
                "category": cat,
                "description": (
                    f"High-quality {cat.lower()} product featuring premium materials. "
                    f"Suitable for everyday use. Available in multiple sizes. "
                    f"30-day return policy applies."
                ),
                "price": round(float(rng.uniform(19.99, 499.99)), 2),
                "weight_oz": round(float(rng.uniform(2.0, 32.0)), 1),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("📦 Seeding SwiftPulse DuckDB...")

    con = duckdb.connect(DB_PATH)

    # Orders
    orders_df = generate_orders(SKUS, N_DAYS)
    con.execute("DROP TABLE IF EXISTS raw_orders")
    con.execute("""
        CREATE TABLE raw_orders AS SELECT * FROM orders_df
    """)
    print(f"  ✓ raw_orders: {len(orders_df):,} rows")

    # Inventory
    inv_df = generate_inventory(SKUS, WAREHOUSES)
    con.execute("DROP TABLE IF EXISTS inventory")
    con.execute("CREATE TABLE inventory AS SELECT * FROM inv_df")
    print(f"  ✓ inventory: {len(inv_df)} rows")

    # Products
    prod_df = generate_products(N_PRODUCTS)
    con.execute("DROP TABLE IF EXISTS products")
    con.execute("CREATE TABLE products AS SELECT * FROM prod_df")
    print(f"  ✓ products: {len(prod_df)} rows")

    con.close()
    print(f"\n✅ Database ready at: {DB_PATH}")
    print("   Next: python ingestion/build_features.py")


if __name__ == "__main__":
    main()
