"""
models/anomaly_detector.py
───────────────────────────
LSTM Autoencoder for real-time anomaly detection
on wearable sensor streams (accelerometer, heart rate, etc.).

How it works:
  - Encoder compresses a time window into a latent vector
  - Decoder reconstructs the original sequence
  - High reconstruction error (MSE) = anomaly
  - Threshold is set at mean + 3*std of training reconstruction errors

Train: python -m models.anomaly_detector --train
Score: imported by the FastAPI endpoint and MQTT subscriber
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# ── Config ────────────────────────────────────────────────────
INPUT_DIM   = 6      # accel_x, accel_y, accel_z, heart_rate, lux, battery
HIDDEN_DIM  = 64
NUM_LAYERS  = 2
SEQ_LEN     = 30     # 30 readings = 15 seconds at 2Hz
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
MODEL_PATH  = Path("data/models/anomaly_detector.pt")
THRESH_PATH = Path("data/models/anomaly_threshold.json")


# ── Model architecture ────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate time series anomaly detection.
    Encoder compresses sequence → latent; Decoder reconstructs it.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: sequence → (hidden, cell) representing compressed state
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Decoder: broadcast latent → full sequence reconstruction
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Project hidden dim back to original feature space
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, input_dim) — reconstructed sequence
        """
        batch_size, seq_len, _ = x.shape

        # Encode: collapse sequence into latent (hidden, cell)
        _, (hidden, cell) = self.encoder(x)

        # Broadcast last hidden state across all time steps for decoding
        # hidden: (num_layers, batch, hidden_dim)
        latent = hidden[-1]                              # (batch, hidden_dim)
        latent_seq = latent.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq, hidden)

        # Decode: reconstruct sequence from latent
        decoded, _ = self.decoder(latent_seq)            # (batch, seq, hidden_dim)

        # Project to original feature space
        reconstructed = self.output_proj(decoded)        # (batch, seq, input_dim)
        return reconstructed

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample MSE reconstruction error. Shape: (batch,)"""
        with torch.no_grad():
            recon = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=(1, 2))   # mean over seq+features
        return mse


# ── Data utilities ────────────────────────────────────────────
def generate_synthetic_data(n_sequences: int = 500) -> np.ndarray:
    """
    Generate synthetic NORMAL sensor data for training.
    In production: replace with real MQTT recordings.
    """
    np.random.seed(42)
    data = []
    for _ in range(n_sequences):
        t = np.linspace(0, 10, SEQ_LEN)
        seq = np.stack([
            np.sin(t * 0.5) + np.random.randn(SEQ_LEN) * 0.1,   # accel_x
            np.cos(t * 0.4) + np.random.randn(SEQ_LEN) * 0.1,   # accel_y
            9.81 + np.random.randn(SEQ_LEN) * 0.05,              # accel_z
            72 + 10 * np.sin(t * 0.1) + np.random.randn(SEQ_LEN) * 2,  # hr
            300 + 50 * np.sin(t * 0.05) + np.random.randn(SEQ_LEN) * 5,  # lux
            np.linspace(80, 70, SEQ_LEN) + np.random.randn(SEQ_LEN) * 0.5,  # batt
        ], axis=-1)  # (SEQ_LEN, INPUT_DIM)
        data.append(seq)
    return np.array(data, dtype=np.float32)  # (N, SEQ_LEN, INPUT_DIM)


def normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization. Returns (normalized, mean, std)."""
    mean = data.mean(axis=(0, 1), keepdims=True)
    std  = data.std(axis=(0, 1), keepdims=True) + 1e-8
    return (data - mean) / std, mean.squeeze(), std.squeeze()


# ── Training ──────────────────────────────────────────────────
def train():
    print("🧠 Training LSTM Autoencoder for anomaly detection...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Generate training data (normal sensor readings only)
    raw = generate_synthetic_data(n_sequences=800)
    norm, mean, std = normalize(raw)

    # Save normalization stats for inference
    np.save("data/models/norm_mean.npy", mean)
    np.save("data/models/norm_std.npy", std)

    # DataLoader
    tensor = torch.tensor(norm)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True)

    # Model, optimizer, loss
    model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    mlflow.set_experiment("swiftpulse-anomaly")
    with mlflow.start_run(run_name="lstm_autoencoder"):
        mlflow.log_params({
            "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN,
            "epochs": EPOCHS, "lr": LR
        })

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"  Epoch {epoch:02d}/{EPOCHS} | loss: {avg_loss:.6f}")

        # Compute anomaly threshold on training data
        model.eval()
        errors = model.reconstruction_error(tensor).numpy()
        threshold = float(errors.mean() + 3 * errors.std())
        print(f"\n📏 Anomaly threshold (mean + 3σ): {threshold:.6f}")

        with open(THRESH_PATH, "w") as f:
            json.dump({"threshold": threshold}, f)

        mlflow.log_metric("anomaly_threshold", threshold)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.pytorch.log_model(model, "lstm_autoencoder")
        print(f"✅ Model saved to {MODEL_PATH}")


# ── Inference helper ──────────────────────────────────────────
def load_model() -> tuple["LSTMAutoencoder", float, np.ndarray, np.ndarray]:
    """Load model + threshold + normalization stats for inference."""
    model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(THRESH_PATH) as f:
        threshold = json.load(f)["threshold"]

    mean = np.load("data/models/norm_mean.npy")
    std  = np.load("data/models/norm_std.npy")
    return model, threshold, mean, std


def score_sequence(raw_seq: np.ndarray) -> dict:
    """
    Score a single sensor sequence.
    raw_seq: numpy array of shape (SEQ_LEN, INPUT_DIM)
    Returns: {"error": float, "threshold": float, "anomaly": bool}
    """
    model, threshold, mean, std = load_model()
    normalized = (raw_seq - mean) / (std + 1e-8)
    tensor = torch.tensor(normalized[np.newaxis], dtype=torch.float32)
    error = float(model.reconstruction_error(tensor).item())
    return {
        "reconstruction_error": round(error, 6),
        "threshold": round(threshold, 6),
        "anomaly": error > threshold,
    }


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        print("Usage: python -m models.anomaly_detector --train")
