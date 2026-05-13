-- ingestion/dbt_project/models/order_features.sql
-- ──────────────────────────────────────────────────
-- Builds daily demand features per SKU with rolling averages.
-- Output feeds the Chronos-T5 forecasting model.
-- Materialized as a DuckDB table for fast ML feature retrieval.

WITH daily_demand AS (
    SELECT
        sku,
        -- Truncate timestamp to day for grouping
        CAST(ordered_at AS DATE)           AS sale_date,
        SUM(quantity)                       AS units_sold,
        SUM(quantity * unit_price)          AS revenue,
        COUNT(DISTINCT order_id)            AS order_count,
        AVG(unit_price)                     AS avg_price
    FROM {{ ref('raw_orders') }}
    GROUP BY 1, 2
),

with_rolling AS (
    SELECT
        *,
        -- 7-day rolling average demand (forecast input feature)
        AVG(units_sold) OVER (
            PARTITION BY sku
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg,

        -- 14-day rolling average
        AVG(units_sold) OVER (
            PARTITION BY sku
            ORDER BY sale_date
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS rolling_14d_avg,

        -- Day-over-day change (momentum feature)
        units_sold - LAG(units_sold, 1, 0) OVER (
            PARTITION BY sku ORDER BY sale_date
        ) AS day_over_day_delta,

        -- Day of week (0=Monday) — captures weekly seasonality
        EXTRACT(DOW FROM sale_date)         AS day_of_week,

        -- Week number — captures annual seasonality
        EXTRACT(WEEK FROM sale_date)        AS week_of_year
    FROM daily_demand
)

SELECT * FROM with_rolling
ORDER BY sku, sale_date
