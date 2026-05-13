"""
ingestion/build_features.py
───────────────────────────
Reads raw_orders from DuckDB, computes rolling demand features,
and writes the feature table back to DuckDB.

This replaces the dbt run for a simpler local workflow.
(dbt version lives in ingestion/dbt/models/order_features.sql)

Run: python ingestion/build_features.py
"""

import duckdb
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DUCKDB_PATH", "./data/swiftpulse.duckdb")


FEATURE_SQL = """
-- Drop and recreate feature table
DROP TABLE IF EXISTS order_features;

CREATE TABLE order_features AS
WITH daily AS (
    SELECT
        sku,
        CAST(ordered_at AS DATE)   AS day,
        SUM(quantity)              AS units_sold,
        SUM(quantity * unit_price) AS revenue
    FROM raw_orders
    GROUP BY sku, CAST(ordered_at AS DATE)
),
with_lags AS (
    SELECT
        sku,
        day,
        units_sold,
        revenue,
        -- Rolling averages (demand signal for forecasting)
        AVG(units_sold) OVER (
            PARTITION BY sku
            ORDER BY day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg,
        AVG(units_sold) OVER (
            PARTITION BY sku
            ORDER BY day
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS rolling_14d_avg,
        AVG(units_sold) OVER (
            PARTITION BY sku
            ORDER BY day
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_30d_avg,
        -- Lag features (what sold yesterday, last week)
        LAG(units_sold, 1)  OVER (PARTITION BY sku ORDER BY day) AS lag_1d,
        LAG(units_sold, 7)  OVER (PARTITION BY sku ORDER BY day) AS lag_7d,
        LAG(units_sold, 30) OVER (PARTITION BY sku ORDER BY day) AS lag_30d,
        -- Day-of-week and month (calendar features)
        DAYOFWEEK(day) AS dow,
        MONTH(day)     AS month_num
    FROM daily
)
SELECT * FROM with_lags
WHERE lag_30d IS NOT NULL   -- drop the first 30 rows that have NULL lags
ORDER BY sku, day;
"""


def main():
    print("⚙️  Building feature table...")
    con = duckdb.connect(DB_PATH)
    con.execute(FEATURE_SQL)

    count = con.execute("SELECT COUNT(*) FROM order_features").fetchone()[0]
    skus  = con.execute("SELECT COUNT(DISTINCT sku) FROM order_features").fetchone()[0]
    print(f"  ✓ order_features: {count:,} rows across {skus} SKUs")

    # Quick sanity check — show one SKU's latest features
    sample = con.execute("""
        SELECT sku, day, units_sold, rolling_7d_avg, rolling_30d_avg
        FROM order_features
        WHERE sku = 'SHOE-001'
        ORDER BY day DESC
        LIMIT 5
    """).df()
    print("\n  Sample (SHOE-001 latest 5 days):")
    print(sample.to_string(index=False))

    con.close()
    print("\n✅ Features ready. Next: python models/train_forecast.py")


if __name__ == "__main__":
    main()
