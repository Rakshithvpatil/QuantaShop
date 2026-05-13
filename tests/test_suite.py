"""
tests/test_suite.py
────────────────────
Unit tests for SwiftPulse components.
Run: pytest tests/ -v
"""

import os
import sys
import numpy as np
import torch
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Anomaly Model ─────────────────────────────────────────────────────────────
class TestLSTMAutoencoder:

    def test_output_shape(self):
        from api.main import LSTMAutoencoder
        model = LSTMAutoencoder(input_dim=6, hidden_dim=32, num_layers=1)
        model.eval()
        x = torch.randn(2, 50, 6)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_anomaly_higher_error(self):
        from api.main import LSTMAutoencoder
        model = LSTMAutoencoder(input_dim=6, hidden_dim=32, num_layers=1)
        model.eval()
        normal  = torch.zeros(1, 50, 6)
        anomaly = torch.zeros(1, 50, 6)
        anomaly[0, 25:, :] = 10.0
        with torch.no_grad():
            e_normal  = torch.nn.functional.mse_loss(model(normal), normal).item()
            e_anomaly = torch.nn.functional.mse_loss(model(anomaly), anomaly).item()
        assert e_anomaly > e_normal


# ── Sensor Simulator ──────────────────────────────────────────────────────────
class TestSensorSimulator:

    def test_battery_drains(self):
        from wearable.sensor_simulator import SensorState
        state = SensorState()
        readings = [state.step() for _ in range(100)]
        assert readings[-1]["battery_pct"] < readings[0]["battery_pct"]

    def test_battery_bounded(self):
        from wearable.sensor_simulator import SensorState
        state = SensorState()
        for _ in range(500):
            r = state.step()
            assert 0.0 <= r["battery_pct"] <= 100.0
            assert r["ambient_lux"] >= 0.0

    def test_anomaly_spikes_hr(self):
        from wearable.sensor_simulator import SensorState
        state = SensorState()
        state.anomaly_in = 4
        readings = [state.step() for _ in range(4)]
        assert max(r["heart_rate"] for r in readings) > 80


# ── Forecast Structure ────────────────────────────────────────────────────────
class TestForecastStructure:

    def test_forecast_shape(self):
        """Mocks ChronosPipeline so test runs without internet or GPU."""
        import models.train_forecast as tf

        dummy_series = {"SKU-A": np.random.rand(120).astype(np.float32)}
        mock_samples = np.random.rand(1, 50, tf.HORIZON).astype(np.float32)

        mock_pipeline = mock.MagicMock()
        mock_pipeline.predict.return_value = mock_samples

        with mock.patch.object(tf, "ChronosPipeline") as MockClass:
            MockClass.from_pretrained.return_value = mock_pipeline
            forecasts = tf.run_chronos_forecast(dummy_series)

        fc = forecasts["SKU-A"]
        assert len(fc["mean"]) == tf.HORIZON
        assert all(lo <= hi for lo, hi in zip(fc["low"], fc["high"]))
