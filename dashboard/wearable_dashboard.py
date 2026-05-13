"""
dashboard/wearable_dashboard.py
────────────────────────────────
Real-time wearable sensor dashboard using Streamlit.
Subscribes to MQTT telemetry and displays live sensor charts.

Run:
  1. Start MQTT broker (Docker)
  2. Start sensor simulator: python -m wearable.sensor_simulator
  3. Run dashboard: streamlit run dashboard/wearable_dashboard.py
"""

import json
import queue
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SwiftPulse — Wearable Monitor",
    page_icon="🕶️",
    layout="wide",
)

MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
TOPIC       = "swiftpulse/wearable/telemetry"
BUFFER_SIZE = 60   # 30 seconds of data at 2Hz

# ── Shared state (thread-safe queue) ─────────────────────────
data_queue: queue.Queue = queue.Queue(maxsize=200)

# ── MQTT subscriber thread ────────────────────────────────────
def mqtt_thread():
    """Background thread: subscribe to MQTT and push to queue."""
    try:
        import paho.mqtt.client as mqtt

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                data_queue.put_nowait(payload)
            except Exception:
                pass

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
        client.subscribe(TOPIC)
        client.loop_forever()
    except Exception as e:
        # Push error so UI can display it
        data_queue.put_nowait({"_error": str(e)})


# Start MQTT thread once per Streamlit session
if "mqtt_started" not in st.session_state:
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    st.session_state.mqtt_started = True
    st.session_state.buffers = {
        "ts": deque(maxlen=BUFFER_SIZE),
        "accel_x": deque(maxlen=BUFFER_SIZE),
        "accel_y": deque(maxlen=BUFFER_SIZE),
        "accel_z": deque(maxlen=BUFFER_SIZE),
        "heart_rate": deque(maxlen=BUFFER_SIZE),
        "ambient_lux": deque(maxlen=BUFFER_SIZE),
        "battery_pct": deque(maxlen=BUFFER_SIZE),
        "step_count": deque(maxlen=BUFFER_SIZE),
        "anomaly_flags": deque(maxlen=BUFFER_SIZE),
    }
    st.session_state.anomaly_count = 0
    st.session_state.total_readings = 0

# ── Drain queue into session buffers ─────────────────────────
buf = st.session_state.buffers
mqtt_error = None

while not data_queue.empty():
    item = data_queue.get_nowait()
    if "_error" in item:
        mqtt_error = item["_error"]
        continue

    st.session_state.total_readings += 1
    buf["ts"].append(item.get("timestamp", "")[-8:])  # HH:MM:SS
    buf["accel_x"].append(item.get("accelerometer", {}).get("x", 0))
    buf["accel_y"].append(item.get("accelerometer", {}).get("y", 0))
    buf["accel_z"].append(item.get("accelerometer", {}).get("z", 9.81))
    buf["heart_rate"].append(item.get("heart_rate_bpm", 72))
    buf["ambient_lux"].append(item.get("ambient_lux", 300))
    buf["battery_pct"].append(item.get("battery_pct", 100))
    buf["step_count"].append(item.get("step_count", 0))
    is_anomaly = item.get("anomaly", False)
    buf["anomaly_flags"].append(is_anomaly)
    if is_anomaly:
        st.session_state.anomaly_count += 1

# ── UI ────────────────────────────────────────────────────────
st.title("🕶️ SwiftPulse — Wearable Live Monitor")

if mqtt_error:
    st.warning(f"⚠️ MQTT connection issue: {mqtt_error}")
    st.info("Start the broker: `docker compose -f infra/docker-compose.yml up -d mosquitto`")

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)

ts = list(buf["ts"])
latest_hr   = buf["heart_rate"][-1] if buf["heart_rate"] else 0
latest_batt = buf["battery_pct"][-1] if buf["battery_pct"] else 0
latest_step = buf["step_count"][-1] if buf["step_count"] else 0
latest_anom = buf["anomaly_flags"][-1] if buf["anomaly_flags"] else False

k1.metric("Heart Rate", f"{latest_hr} bpm",
          delta="ALERT" if latest_hr > 180 or latest_hr < 40 else "Normal")
k2.metric("Battery", f"{latest_batt:.0f}%")
k3.metric("Steps", f"{latest_step:,}")
k4.metric("Anomalies", st.session_state.anomaly_count,
          delta="⚠️ Active" if latest_anom else None)
k5.metric("Total Readings", st.session_state.total_readings)

st.divider()

# Two chart columns
col_left, col_right = st.columns(2)

with col_left:
    # Accelerometer chart
    accel_fig = go.Figure()
    if ts:
        accel_fig.add_trace(go.Scatter(x=ts, y=list(buf["accel_x"]),
                                        name="X", line={"color": "#E24B4A"}))
        accel_fig.add_trace(go.Scatter(x=ts, y=list(buf["accel_y"]),
                                        name="Y", line={"color": "#3B8BD4"}))
        accel_fig.add_trace(go.Scatter(x=ts, y=list(buf["accel_z"]),
                                        name="Z", line={"color": "#1D9E75"}))

    accel_fig.update_layout(
        title="Accelerometer (m/s²)",
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h"},
        margin={"t": 40, "b": 20},
        xaxis={"showticklabels": False},
    )
    st.plotly_chart(accel_fig, use_container_width=True)

    # Battery gauge
    batt_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_batt,
        title={"text": "Battery %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1D9E75" if latest_batt > 20 else "#E24B4A"},
            "steps": [
                {"range": [0, 20], "color": "rgba(226,75,74,0.2)"},
                {"range": [20, 50], "color": "rgba(239,159,39,0.2)"},
                {"range": [50, 100], "color": "rgba(29,158,117,0.1)"},
            ],
        }
    ))
    batt_fig.update_layout(height=220, margin={"t": 40, "b": 10},
                           paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(batt_fig, use_container_width=True)

with col_right:
    # Heart rate chart
    hr_fig = go.Figure()
    if ts:
        # Color anomaly regions red
        hr_colors = ["#E24B4A" if a else "#D85A30"
                     for a in buf["anomaly_flags"]]
        hr_fig.add_trace(go.Scatter(
            x=ts, y=list(buf["heart_rate"]),
            mode="lines+markers",
            name="Heart Rate",
            line={"color": "#D85A30", "width": 2},
            marker={"color": hr_colors, "size": 5},
        ))
        # Normal zone
        hr_fig.add_hrect(y0=50, y1=180, fillcolor="rgba(29,158,117,0.05)",
                          line_width=0, annotation_text="Normal range")

    hr_fig.update_layout(
        title="Heart Rate (bpm)",
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 40, "b": 20},
        xaxis={"showticklabels": False},
    )
    st.plotly_chart(hr_fig, use_container_width=True)

    # Ambient light
    lux_fig = go.Figure()
    if ts:
        lux_fig.add_trace(go.Scatter(
            x=ts, y=list(buf["ambient_lux"]),
            fill="tozeroy",
            fillcolor="rgba(239,159,39,0.2)",
            line={"color": "#EF9F27"},
            name="Lux",
        ))
    lux_fig.update_layout(
        title="Ambient Light (lux)",
        height=220,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 40, "b": 10},
        xaxis={"showticklabels": False},
    )
    st.plotly_chart(lux_fig, use_container_width=True)

# ── Anomaly banner ────────────────────────────────────────────
if latest_anom:
    st.error("🚨 ANOMALY DETECTED — Abnormal sensor readings flagged by LSTM autoencoder")

# Auto-refresh every 1 second
time.sleep(1)
st.rerun()
