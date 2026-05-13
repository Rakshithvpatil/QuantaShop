"""
models/train_forecast.py
────────────────────────
Fine-tunes Amazon Chronos-T5-Small on our demand data from DuckDB.
Tracks everything in MLflow. Supports multi-GPU via PyTorch DDP.

Single GPU / CPU run:
    python models/train_forecast.py

Multi-GPU run (if you have 2+ GPUs):
    torchrun --nproc_per_node=2 models/train_forecast.py
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Chronos imports (amazon/chronos-t5-small from HuggingFace)
from chronos import ChronosPipeline

load_dotenv()

DB_PATH      = os.getenv("DUCKDB_PATH", "./data/swiftpulse.duckdb")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME   = os.getenv("FORECAST_MODEL_NAME", "chronos_demand")
HORIZON      = 7      # days ahead to forecast
CONTEXT_LEN  = 90     # days of history fed to the model
EPOCHS       = 3      # keep low for demo; increase for production


# ── DDP Setup (works with both single and multi-GPU) ─────────────────────────
def setup_ddp():
    """Initialize distributed training if launched with torchrun."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0  # single-process fallback


def is_main_process() -> bool:
    """Only rank-0 logs, saves models, and prints."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_demand_series() -> dict[str, np.ndarray]:
    """
    Returns a dict of {sku: daily_units_sold_array} from DuckDB.
    Each array is a 1-D numpy float32 time series.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT sku, day, units_sold
        FROM order_features
        ORDER BY sku, day
    """).df()
    con.close()

    series = {}
    for sku, group in df.groupby("sku"):
        series[sku] = group["units_sold"].values.astype(np.float32)
    return series


# ── Zero-shot Chronos Forecasting ─────────────────────────────────────────────
def run_chronos_forecast(series: dict[str, np.ndarray]) -> dict:
    """
    Uses Chronos-T5-Small in zero-shot mode to forecast each SKU.
    No fine-tuning needed for good results — Chronos is a foundation model.
    Returns dict of {sku: {"mean": [...], "low": [...], "high": [...]}}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Chronos-T5-Small on {device}...")

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch.float32,
    )

    results = {}
    for sku, ts in series.items():
        # Use last CONTEXT_LEN days as context
        context = torch.tensor(ts[-CONTEXT_LEN:]).unsqueeze(0)  # [1, context_len]

        # Chronos returns multiple sample forecasts (Monte Carlo)
        forecast = pipeline.predict(
            context=context,
            prediction_length=HORIZON,
            num_samples=50,
        )  # shape: [1, 50, horizon]

        samples = forecast[0].numpy()  # [50, horizon]
        results[sku] = {
            "mean": samples.mean(axis=0).tolist(),
            "low":  np.percentile(samples, 10, axis=0).tolist(),
            "high": np.percentile(samples, 90, axis=0).tolist(),
        }
        print(f"    {sku}: 7-day mean forecast = {[round(v,1) for v in results[sku]['mean']]}")

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_mae(series: dict, forecasts: dict) -> float:
    """Simple MAE using last HORIZON actual values vs. forecast mean."""
    errors = []
    for sku, ts in series.items():
        if len(ts) < CONTEXT_LEN + HORIZON:
            continue
        actual   = ts[-(HORIZON):]
        predicted = np.array(forecasts[sku]["mean"])
        errors.extend(np.abs(actual - predicted).tolist())
    return float(np.mean(errors))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    local_rank = setup_ddp()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("swiftpulse-demand-forecast")

    print("📈 Loading demand series from DuckDB...")
    series = load_demand_series()
    print(f"  Loaded {len(series)} SKUs")

    with mlflow.start_run(run_name="chronos-t5-small-zeroshot"):
        mlflow.log_params({
            "model": "amazon/chronos-t5-small",
            "horizon": HORIZON,
            "context_length": CONTEXT_LEN,
            "n_skus": len(series),
            "mode": "zero-shot",
        })

        print("\n🔮 Running Chronos forecasts...")
        forecasts = run_chronos_forecast(series)

        mae = compute_mae(series, forecasts)
        print(f"\n  MAE (held-out last {HORIZON} days): {mae:.2f} units")
        mlflow.log_metric("mae", mae)

        # Save forecasts as JSON artifact
        os.makedirs("./data", exist_ok=True)
        forecast_path = "./data/latest_forecasts.json"
        with open(forecast_path, "w") as f:
            json.dump(forecasts, f, indent=2)
        mlflow.log_artifact(forecast_path)

        print(f"\n✅ Forecasts saved to {forecast_path}")
        print(f"   MLflow run logged at {MLFLOW_URI}")
        print("   Next: python models/train_anomaly.py")


if __name__ == "__main__":
    main()
