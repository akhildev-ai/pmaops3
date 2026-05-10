from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


sys.path.append(str(ROOT / "sensor-simulator"))
sys.path.append(str(ROOT / "inference-service"))


generator_module = importlib.import_module("simulator.generator")
app_module = importlib.import_module("app.main")

build_machine_profiles = generator_module.build_machine_profiles
generate_records = generator_module.generate_records
app = app_module.app


def test_generator_schema() -> None:
    profiles = build_machine_profiles(machine_count=4, seed=12)
    records = list(generate_records(profiles=profiles, start_time=datetime(2026, 1, 1), periods=1, seed=12))
    assert len(records) == 4
    payload = records[0].to_dict()
    assert payload["machine_id"].startswith("MOTOR")
    assert payload["timestamp"].startswith("2026-01-01T00:00:00")
    assert "anomaly_score" in payload


def test_inference_fallback_prediction() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "temperature": 95,
            "vibration": 0.92,
            "rpm": 1470,
            "pressure": 34,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= payload["failure_probability"] <= 1
