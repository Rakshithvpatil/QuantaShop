"""
api/main.py — SwiftPulse FastAPI Gateway
Run: uvicorn api.main:app --reload --port 8000
"""
import asyncio, json, os
from datetime import datetime, timezone
from collections import deque
from typing import Optional

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "./data/swiftpulse.duckdb")

REQUEST_COUNT   = Counter("swiftpulse_requests_total", "Total API requests", ["endpoint","method","status"])
REQUEST_LATENCY = Histogram("swiftpulse_request_duration_seconds", "Request latency", ["endpoint"])
ANOMALIES_DETECTED = Counter("swiftpulse_anomalies_total", "Anomalies detected")

app = FastAPI(title="SwiftPulse API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory telemetry buffer
telemetry_buffer: deque = deque(maxlen=200)

class RAGQuery(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SensorSequence(BaseModel):
    readings: list[list[float]]

class ForecastResponse(BaseModel):
    sku: str
    horizon_days: int
    method: str
    forecast: list[dict]

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "service": "SwiftPulse API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }

@app.get("/forecast/{sku}", response_model=ForecastResponse)
async def get_forecast(sku: str, horizon: int = 7):
    """Generate demand forecast for a SKU using Chronos-T5."""
    try:
        from models.forecast_model import predict
        result = predict(sku=sku, horizon=horizon)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory/{sku}")
async def get_inventory(sku: str):
    """Read inventory directly from DuckDB — no separate ERP service needed."""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        rows = con.execute(
            "SELECT warehouse, quantity_on_hand, reorder_point, lead_time_days "
            "FROM inventory WHERE sku = ?", [sku]
        ).df()
        con.close()
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"SKU {sku} not found. Run seed_data.py")
        return {
            "sku": sku,
            "warehouses": rows.to_dict(orient="records"),
            "total_qty": int(rows["quantity_on_hand"].sum()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query(body: RAGQuery):
    """Answer product/policy questions using local RAG pipeline."""
    try:
        from rag.build_rag import query
        result = query(question=body.question, top_k=body.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomaly/score")
async def score_anomaly(body: SensorSequence):
    """Score sensor sequence for anomalies."""
    return {"demo": True, "anomaly": False, "score": 0.01, "threshold": 0.05}

@app.get("/anomaly/demo")
async def anomaly_demo():
    return {
        "demo": True,
        "features": ["accel_x", "accel_y", "accel_z", "heart_rate", "ambient_lux", "battery_pct"],
    }

@app.get("/wearable/recent")
async def wearable_recent(n: int = 30):
    """Return last N telemetry events from the in-memory buffer."""
    events = list(telemetry_buffer)[-n:]
    return {"count": len(events), "events": events}

@app.websocket("/wearable/live")
async def wearable_websocket(ws: WebSocket):
    """WebSocket bridge: MQTT telemetry to browser clients."""
    await ws.accept()
    try:
        import paho.mqtt.client as mqtt
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode())
                # Add anomaly score (demo values)
                data["score"]     = float(np.random.uniform(0.001, 0.08))
                data["is_alert"]  = data["score"] > 0.05
                data["threshold"] = 0.05
                telemetry_buffer.append(data)
                queue.put_nowait(data)
            except Exception:
                pass

        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.on_message = on_message
        try:
            mqtt_client.connect("localhost", 1883, keepalive=30)
            mqtt_client.subscribe("swiftpulse/wearable/telemetry")
            mqtt_client.loop_start()
        except Exception:
            await ws.send_json({"error": "MQTT not available", "mode": "demo"})

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                await ws.send_json(payload)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping", "ts": datetime.now().isoformat()})
    except WebSocketDisconnect:
        pass

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain; version=0.0.4")

@app.on_event("startup")
async def start_fake_telemetry():
    """Generates fake wearable data every 2 seconds — no MQTT needed."""
    import random, math
    async def loop():
        t = 0
        while True:
            score     = abs(random.gauss(0.02, 0.015))
            is_alert  = score > 0.05
            telemetry_buffer.append({
                "device_id":   "glasses-dev-001",
                "timestamp":   datetime.now().isoformat(),
                "accel_x":     round(0.35 * math.sin(t * 2.1) + random.gauss(0, 0.04), 4),
                "accel_y":     round(0.22 * math.cos(t * 2.1) + random.gauss(0, 0.04), 4),
                "accel_z":     round(9.81 + random.gauss(0, 0.06), 4),
                "heart_rate":  round(72 + 10 * math.sin(t * 0.03) + random.gauss(0, 1.5), 1),
                "ambient_lux": round(max(0, 280 + random.gauss(0, 10)), 1),
                "battery_pct": round(max(0, 95 - t * 0.001), 2),
                "score":       round(score, 6),
                "is_alert":    is_alert,
                "threshold":   0.05,
            })
            t += 1
            await asyncio.sleep(2)
    asyncio.create_task(loop())
