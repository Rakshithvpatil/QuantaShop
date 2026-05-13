# SwiftPulse

End-to-end e-commerce AI automation and wearable analytics. Covers time series forecasting, multi-GPU ML, RAG, BigCommerce/ERP integration, and IoT/MQTT firmware simulation. Every tool is free.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/swiftpulse.git && cd swiftpulse
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
python ingestion/seed_data.py
python ingestion/build_features.py
python models/train_forecast.py
python models/train_anomaly.py
# In a new terminal:
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/forecast_dashboard.py
```

## URLs

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| RAG chat | http://localhost:7860 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |

## Stack

Chronos-T5, PyTorch DDP, LSTM Autoencoder, LangChain, ChromaDB, Ollama, Kafka, dbt, DuckDB, MLflow, FastAPI, Streamlit, Gradio, Prometheus, Grafana, GitHub Actions — all free.
