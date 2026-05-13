"""
wearable/sensor_simulator.py
─────────────────────────────
Simulates an AI glasses wearable device publishing sensor telemetry
over MQTT — the same protocol embedded firmware uses.

Run: python wearable/sensor_simulator.py
(Requires MQTT broker: docker compose up -d mosquitto)
"""

import json
import math
import os
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_HOST   = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
PUB_TOPIC   = os.getenv("MQTT_TOPIC_TELEMETRY", "swiftpulse/wearable/telemetry")
ALERT_TOPIC = os.getenv("MQTT_TOPIC_ALERTS", "swiftpulse/wearable/alerts")
DEVICE_ID   = "glasses-dev-001"
PUBLISH_HZ  = 2


class SensorState:
    """Maintains realistic physical state for all sensors."""
    def __init__(self):
        self.t           = 0.0
        self.battery_pct = 100.0
        self.hr_phase    = random.uniform(0, math.pi)
        self.anomaly_in  = 0

    def step(self) -> dict:
        self.t += 1.0 / PUBLISH_HZ
        accel_x = 0.35 * math.sin(self.t * 2.1) + random.gauss(0, 0.04)
        accel_y = 0.22 * math.cos(self.t * 2.1) + random.gauss(0, 0.04)
        accel_z = 9.81  + 0.15 * math.sin(self.t * 1.0) + random.gauss(0, 0.06)
        hr = 72 + 10 * math.sin(self.t * 0.03 + self.hr_phase) + random.gauss(0, 1.5)
        lux = max(0, 280 + 180 * math.sin(self.t * 0.008) + random.gauss(0, 10))
        self.battery_pct -= (40 / (8 * 3600 * PUBLISH_HZ))
        battery = max(0, self.battery_pct + random.gauss(0, 0.05))

        if self.anomaly_in > 0:
            accel_x += random.gauss(0, 3.0)
            accel_y += random.gauss(0, 3.0)
            accel_z += random.gauss(-5, 3.0)
            hr += random.gauss(25, 5)
            self.anomaly_in -= 1
        elif random.random() < (1 / (60 * PUBLISH_HZ)):
            self.anomaly_in = int(2 * PUBLISH_HZ)
            print("  WARNING: ANOMALY INJECTED (simulated fall/impact)")

        return {
            "device_id":   DEVICE_ID,
            "timestamp":   datetime.now().isoformat(),
            "accel_x":     round(accel_x, 4),
            "accel_y":     round(accel_y, 4),
            "accel_z":     round(accel_z, 4),
            "heart_rate":  round(hr, 1),
            "ambient_lux": round(lux, 1),
            "battery_pct": round(battery, 2),
        }


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(ALERT_TOPIC)
    else:
        print(f"Connection failed rc={rc}. Is mosquitto running?")


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    print(f"\n  ALERT: score={payload.get('score', 0):.3f} | {payload.get('message', '')}")


def main():
    print(f"SwiftPulse Wearable Simulator | Device: {DEVICE_ID}")
    print(f"Publishing to: {PUB_TOPIC}  @ {PUBLISH_HZ}Hz | Ctrl+C to stop\n")

    client = mqtt.Client(client_id=DEVICE_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    except ConnectionRefusedError:
        print(f"Cannot connect to MQTT at {MQTT_HOST}:{MQTT_PORT}")
        print("Start it with: docker compose up -d mosquitto")
        return

    client.loop_start()
    state = SensorState()
    sent  = 0

    try:
        while True:
            reading = state.step()
            client.publish(PUB_TOPIC, json.dumps(reading), qos=0)
            sent += 1
            if sent % 10 == 0:
                print(
                    f"  [{sent:05d}] HR={reading['heart_rate']:.1f}bpm "
                    f"Az={reading['accel_z']:.2f} "
                    f"Bat={reading['battery_pct']:.1f}% "
                    f"Lux={reading['ambient_lux']:.0f}"
                )
            time.sleep(1.0 / PUBLISH_HZ)
    except KeyboardInterrupt:
        print(f"\nStopped. Total messages: {sent}")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
