"""
ingestion/producers/order_producer.py
──────────────────────────────────────
Simulates a BigCommerce webhook producer.
In production: replace generate_order() with a real BC API call.

Run: python -m ingestion.producers.order_producer
Requires Kafka running on localhost:9092
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Config ────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_ORDERS    = "bc-orders"
TOPIC_INVENTORY = "bc-inventory-alerts"

SKUS = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"]
WAREHOUSES = ["TX-01", "CA-02"]


def make_producer() -> KafkaProducer:
    """Create a Kafka producer with JSON serialization."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",               # wait for all replicas to confirm
        retries=3,
        retry_backoff_ms=500,
    )


def generate_order() -> dict:
    """
    Generate a synthetic BigCommerce-style order event.
    Schema mirrors what BigCommerce webhooks actually send.
    """
    sku = random.choice(SKUS)
    qty = random.randint(1, 10)
    price = round(random.uniform(19.99, 299.99), 2)

    return {
        "order_id":   str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status":     random.choice(["pending", "processing", "shipped"]),
        "channel":    random.choice(["web", "mobile", "marketplace"]),
        "customer": {
            "id":    random.randint(1000, 9999),
            "email": f"customer{random.randint(1,999)}@example.com",
            "state": random.choice(["TX", "CA", "NY", "WA", "FL"]),
        },
        "line_items": [
            {
                "sku":       sku,
                "quantity":  qty,
                "unit_price": price,
                "total":     round(qty * price, 2),
            }
        ],
        "order_total": round(qty * price, 2),
        "warehouse":   random.choice(WAREHOUSES),
    }


def generate_inventory_alert(order: dict) -> dict | None:
    """
    Check if an order might deplete stock — emit an alert if so.
    In production this would query the ERP service.
    """
    sku = order["line_items"][0]["sku"]
    qty = order["line_items"][0]["quantity"]
    simulated_stock = random.randint(0, 200)   # mock current stock

    if simulated_stock - qty < 25:             # below reorder point
        return {
            "alert_type": "low_stock",
            "sku":        sku,
            "remaining":  max(0, simulated_stock - qty),
            "reorder_pt": 25,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
    return None


def main():
    print("🚀 Starting SwiftPulse order producer...")
    print(f"   Kafka: {KAFKA_BOOTSTRAP}")
    print(f"   Topic: {TOPIC_ORDERS}")
    print("   Press Ctrl+C to stop\n")

    try:
        producer = make_producer()
    except NoBrokersAvailable:
        print("❌  Kafka not reachable. Start it with:")
        print("    docker compose -f infra/docker-compose.yml up -d kafka")
        return

    order_count = 0
    try:
        while True:
            order = generate_order()
            order_id = order["order_id"]

            # Send order event
            producer.send(TOPIC_ORDERS, key=order_id, value=order)
            order_count += 1
            print(f"[{order_count:04d}] 📦 Order {order_id[:8]}… "
                  f"| SKU: {order['line_items'][0]['sku']} "
                  f"| Qty: {order['line_items'][0]['quantity']} "
                  f"| ${order['order_total']:.2f}")

            # Send inventory alert if needed
            alert = generate_inventory_alert(order)
            if alert:
                producer.send(TOPIC_INVENTORY, key=alert["sku"], value=alert)
                print(f"         ⚠️  Low stock alert: {alert['sku']} → {alert['remaining']} remaining")

            producer.flush()
            time.sleep(random.uniform(0.5, 2.0))   # variable rate — realistic

    except KeyboardInterrupt:
        print(f"\n✅ Stopped. Sent {order_count} orders.")
        producer.close()


if __name__ == "__main__":
    main()
