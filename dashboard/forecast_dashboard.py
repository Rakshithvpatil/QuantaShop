"""
dashboard/forecast_dashboard.py
Run: streamlit run dashboard/forecast_dashboard.py
"""

import os
import time
import requests
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SKUS = ["SHOE-001","SHOE-002","BAG-001","HAT-001","GLASS-001","GLASS-002","WATCH-001","BELT-001"]

st.set_page_config(page_title="SwiftPulse Analytics", page_icon="🚀", layout="wide")
st.title("SwiftPulse — E-Commerce AI Dashboard")
st.caption("Demand Forecasting · Inventory · Wearable Anomaly Detection")

st.sidebar.header("Controls")
selected_sku = st.sidebar.selectbox("Select SKU", SKUS)
horizon      = st.sidebar.slider("Forecast horizon (days)", 1, 7, 7)
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)

def fetch(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

# ── Forecast ──────────────────────────────────────────────────────────────────
st.subheader(f"7-Day Demand Forecast — {selected_sku}")
fc = fetch(f"/forecast/{selected_sku}")

if fc and "forecast" in fc:
    items = fc["forecast"][:horizon]
    days  = [item["date"] for item in items]
    mean  = [item["median"] for item in items]
    low   = [item["lo_80"] for item in items]
    high  = [item["hi_80"] for item in items]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days+days[::-1], y=high+low[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% CI", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=days, y=mean, mode="lines+markers", name="Forecast",
        line=dict(color="#1f77b4", width=2.5), marker=dict(size=7)
    ))
    fig.update_layout(height=360, xaxis_title="Date", yaxis_title="Units",
                      margin=dict(l=40,r=20,t=20,b=40))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Day",  days[mean.index(max(mean))], f"{max(mean)} units")
    c2.metric("Total 7d",  f"{sum(mean)} units")
    c3.metric("Avg/Day",   f"{sum(mean)/len(mean):.1f} units")
else:
    st.warning("Forecast API not reachable. Make sure Tab 2 API is running.")

# ── Inventory ─────────────────────────────────────────────────────────────────
st.subheader(f"Inventory — {selected_sku}")
inv = fetch(f"/inventory/{selected_sku}")
if inv and "warehouses" in inv:
    rows = inv["warehouses"]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="On Hand",
        x=[r["warehouse"] for r in rows],
        y=[r["quantity_on_hand"] for r in rows], marker_color="#2ca02c"))
    fig2.add_trace(go.Bar(name="Reorder Point",
        x=[r["warehouse"] for r in rows],
        y=[r["reorder_point"] for r in rows], marker_color="#d62728"))
    fig2.update_layout(barmode="group", height=280, margin=dict(l=40,r=20,t=10,b=40))
    st.plotly_chart(fig2, use_container_width=True)
    for r in rows:
        if r["quantity_on_hand"] <= r["reorder_point"]:
            st.error(f"REORDER ALERT — {r['warehouse']}: qty={r['quantity_on_hand']} <= reorder={r['reorder_point']}")
else:
    st.info("Run seed_data.py to load inventory data.")

# ── Wearable ──────────────────────────────────────────────────────────────────
st.subheader("Live Wearable Anomaly Feed")
wearable = fetch("/wearable/recent?n=30")
if wearable and wearable.get("events"):
    events    = wearable["events"]
    scores    = [e["score"] for e in events]
    alerts    = [e["is_alert"] for e in events]
    threshold = events[-1]["threshold"]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        y=scores, mode="lines+markers", name="Anomaly score",
        line=dict(color="#ff7f0e"),
        marker=dict(color=["red" if a else "#ff7f0e" for a in alerts], size=8)
    ))
    fig3.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig3.update_layout(height=260, yaxis_title="Reconstruction MSE",
                       margin=dict(l=40,r=20,t=10,b=40))
    st.plotly_chart(fig3, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Recent Alerts", sum(alerts), f"in last {len(events)} readings")
    c2.metric("Latest Score",  f"{scores[-1]:.6f}")
else:
    st.info("No wearable data yet. Run: python wearable/sensor_simulator.py")

if auto_refresh:
    time.sleep(10)
    st.rerun()
