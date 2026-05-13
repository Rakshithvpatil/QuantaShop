"""
ingestion/kafka_producer.py
───────────────────────────
Streams mock order events to Kafka topic 'bc-orders'.
Simulates what a real BigCommerce webhook integration would do.

Run: python ingestion/kafka_producer.py
(Requires Kafka running: docker compose up -d kafka)
"""

import json
import time
import random
import os
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC     = os.getenv("KAFKA_ORDERS_TOPIC", "bc-orders")

SKUS = ["SHOE-001", "SHOE-002", "BAG-001", "HAT-001",
        "GLASS-001", "GLASS-002", "WATCH-001", "BELT-001"]
CHANNELS = ["web", "mobile", "marketplace"]


def make_order() -> dict:
    """Generates a single realistic order event."""
    sku = random.choice(SKUS)
    return {
        "order_id":    f"ORD-{random.randint(100000, 999999)}",
        "sku":         sku,
        "quantity":    random.randint(1, 5),
        "unit_price":  round(random.uniform(19.99, 299.99), 2),
        "customer_id": f"CUST-{random.randint(1000, 9999)}",
        "channel":     random.choice(CHANNELS),
        "ordered_at":  datetime.now().isoformat(),
        "source":      "bigcommerce_webhook",
    }


def wait_for_kafka(bootstrap: str, retries: int = 10) -> KafkaProducer:
    """Retries connection so the script can start before Kafka is fully up."""
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
            print(f"✓ Connected to Kafka at {bootstrap}")
            return producer
        except NoBrokersAvailable:
            print(f"  Kafka not ready (attempt {attempt+1}/{retries}), retrying in 3s...")
            time.sleep(3)
    raise RuntimeError("Could not connect to Kafka. Is docker compose running?")


def main():
    print(f"🚀 Starting order producer → topic '{TOPIC}'")
    producer = wait_for_kafka(BOOTSTRAP)

    sent = 0
    try:
        while True:
            order = make_order()
            # Use SKU as partition key so same-SKU orders land on same partition
            producer.send(TOPIC, key=order["sku"], value=order)
            sent += 1
            print(f"  [{sent}] Sent order {order['order_id']} | {order['sku']} x{order['quantity']}")
            # Simulate realistic order rate (~1 per second for demo)
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print(f"\n⏹  Producer stopped. Total sent: {sent}")
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
