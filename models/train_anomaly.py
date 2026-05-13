"""
models/train_anomaly.py
───────────────────────
Trains an LSTM Autoencoder on simulated wearable sensor data.
High reconstruction error → anomaly (fall, abnormal heart rate, etc.)

Run: python models/train_anomaly.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME  = os.getenv("ANOMALY_MODEL_NAME", "lstm_anomaly")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN     = 50      # 50 timesteps = 25 seconds at 2Hz
INPUT_DIM   = 6       # accel_x, accel_y, accel_z, heart_rate, ambient_lux, battery
HIDDEN_DIM  = 64
NUM_LAYERS  = 2
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
ANOMALY_THRESHOLD_PERCENTILE = 95   # flag top 5% reconstruction errors


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    Encoder compresses the window; decoder reconstructs it.
    Anomaly score = MSE between input and reconstruction.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, input_dim] → reconstructed x (same shape)"""
        _, (hidden, cell) = self.encoder(x)
        # Repeat the final hidden state across all timesteps for decoder input
        dec_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(dec_input, (hidden, cell))
        return self.output_layer(decoded)


# ── Synthetic Training Data ───────────────────────────────────────────────────
def generate_normal_sensor_data(n_sequences: int, seq_len: int, input_dim: int) -> np.ndarray:
    """
    Generates realistic 'normal' wearable sensor windows.
    Shape: [n_sequences, seq_len, input_dim]

    Channels: accel_x, accel_y, accel_z, heart_rate, ambient_lux, battery
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, seq_len)
    data = np.zeros((n_sequences, seq_len, input_dim), dtype=np.float32)

    for i in range(n_sequences):
        phase = rng.uniform(0, 2 * np.pi)
        # Accelerometer: walking gait pattern + noise
        data[i, :, 0] = 0.3 * np.sin(t * 2 + phase)  + rng.normal(0, 0.05, seq_len)  # accel_x
        data[i, :, 1] = 0.2 * np.cos(t * 2 + phase)  + rng.normal(0, 0.05, seq_len)  # accel_y
        data[i, :, 2] = 9.81 + 0.1 * np.sin(t + phase) + rng.normal(0, 0.05, seq_len) # accel_z
        # Heart rate: 65-85 bpm, slow variation
        data[i, :, 3] = (72 + 8 * np.sin(t * 0.1)) / 100.0  + rng.normal(0, 0.01, seq_len)
        # Ambient lux: slow change
        data[i, :, 4] = (300 + 100 * np.sin(t * 0.05)) / 1000.0
        # Battery: slowly decreasing
        data[i, :, 5] = np.linspace(0.9, 0.88, seq_len) + rng.normal(0, 0.002, seq_len)

    return data


class SensorDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Training ─────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def compute_threshold(model, loader) -> float:
    """Computes the anomaly threshold from reconstruction errors on training data."""
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            err = criterion(recon, batch).mean(dim=(1, 2))  # per-sequence MSE
            errors.extend(err.cpu().numpy().tolist())
    threshold = float(np.percentile(errors, ANOMALY_THRESHOLD_PERCENTILE))
    return threshold


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"🔧 Training LSTM Autoencoder on {DEVICE}...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("swiftpulse-anomaly-detection")

    # Generate data
    n_train = 2000
    data = generate_normal_sensor_data(n_train, SEQ_LEN, INPUT_DIM)
    dataset = SensorDataset(data)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    model     = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="lstm-autoencoder"):
        mlflow.log_params({
            "input_dim":  INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "seq_len":    SEQ_LEN,
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
        })

        for epoch in range(1, EPOCHS + 1):
            loss = train(model, loader, optimizer, criterion)
            mlflow.log_metric("train_loss", loss, step=epoch)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.6f}")

        # Compute threshold on training data
        threshold = compute_threshold(model, loader)
        mlflow.log_metric("anomaly_threshold", threshold)
        print(f"\n  Anomaly threshold (p{ANOMALY_THRESHOLD_PERCENTILE}): {threshold:.6f}")

        # Save model + threshold together
        os.makedirs("./data", exist_ok=True)
        torch.save(model.state_dict(), "./data/anomaly_model.pt")
        with open("./data/anomaly_threshold.json", "w") as f:
            json.dump({"threshold": threshold, "seq_len": SEQ_LEN, "input_dim": INPUT_DIM}, f)

        mlflow.pytorch.log_model(model, artifact_path="lstm_anomaly_model")
        mlflow.log_artifact("./data/anomaly_threshold.json")

    print("\n✅ Anomaly model saved.")
    print("   Next: python rag/build_index.py")


if __name__ == "__main__":
    main()
