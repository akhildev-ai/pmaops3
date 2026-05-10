from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    mlflow_pyfunc = importlib.import_module("mlflow.pyfunc")
except ImportError:  # pragma: no cover - optional dependency at local authoring time
    mlflow_pyfunc = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "model_package.joblib"


class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=0, le=250)
    vibration: float = Field(..., ge=0, le=10)
    rpm: int = Field(..., ge=0, le=10000)
    pressure: float = Field(..., ge=0, le=400)
    voltage: float = Field(default=415.0, ge=0, le=1000)
    humidity: float = Field(default=40.0, ge=0, le=100)
    anomaly_score: float = Field(default=0.0, ge=0)
    degradation_index: float = Field(default=0.0, ge=0, le=1)


class PredictionResponse(BaseModel):
    failure_probability: float
    risk_level: str
    recommended_action: str
    model_source: str
    model_version: str
    scored_at: str


class Predictor:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_columns = [
            "temperature",
            "vibration",
            "rpm",
            "pressure",
            "voltage",
            "humidity",
            "anomaly_score",
            "degradation_index",
            "temp_vibration_ratio",
            "pressure_rpm_ratio",
            "stress_index",
        ]
        self.model_source = "heuristic"
        self.model_version = "fallback-v1"
        self._load()

    def _load(self) -> None:
        mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
        local_model_path = Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_MODEL_PATH)))

        if mlflow_model_uri:
            if mlflow_pyfunc is None:
                raise RuntimeError("MLflow model URI configured but mlflow is not installed.")
            self.model = mlflow_pyfunc.load_model(mlflow_model_uri)
            self.model_source = "mlflow"
            self.model_version = mlflow_model_uri
            return

        if local_model_path.exists():
            artifact = joblib.load(local_model_path)
            self.model = artifact["model"]
            self.feature_columns = artifact["feature_columns"]
            self.model_source = "artifact"
            self.model_version = artifact.get("source_run_id", artifact.get("created_at", "local"))

    def _features(self, payload: PredictionRequest) -> pd.DataFrame:
        row = payload.model_dump()
        row["temp_vibration_ratio"] = row["temperature"] / max(row["vibration"], 0.05)
        row["pressure_rpm_ratio"] = row["pressure"] / max(row["rpm"], 100)
        row["stress_index"] = (
            row["temperature"] * 0.35
            + row["vibration"] * 48.0
            + row["pressure"] * 0.25
            + row["degradation_index"] * 16.0
        )
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, payload: PredictionRequest) -> float:
        features = self._features(payload)
        if self.model is None:
            stress = features.iloc[0]["stress_index"]
            anomaly = float(features.iloc[0]["anomaly_score"])
            probability = min(max((stress / 100.0) + (anomaly / 120.0), 0.02), 0.99)
            return probability

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(features)[0][1])

        result = self.model.predict(features)
        if isinstance(result, pd.DataFrame):
            if "failure_probability" in result.columns:
                return float(result.iloc[0]["failure_probability"])
            return float(result.iloc[0][0])

        if isinstance(result, np.ndarray):
            return float(result[0])

        return float(result)


def classify_risk(probability: float) -> tuple[str, str]:
    if probability >= 0.8:
        return "HIGH", "Inspect bearing within 12 hours"
    if probability >= 0.5:
        return "MEDIUM", "Schedule maintenance in next shift"
    return "LOW", "Continue monitoring"


app = FastAPI(title="Industrial Predictive Maintenance Inference API", version="0.1.0")
predictor = Predictor()
request_counter = 0
error_counter = 0
latencies: list[float] = []
risk_counter: Counter[str] = Counter()


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    global request_counter, error_counter

    started = time.perf_counter()
    request_counter += 1
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            error_counter += 1
        return response
    finally:
        latencies.append(time.perf_counter() - started)
        if len(latencies) > 500:
            del latencies[:-500]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_source": predictor.model_source}


@app.get("/version")
def version() -> dict[str, str]:
    return {"model_source": predictor.model_source, "model_version": predictor.model_version}


@app.get("/metrics", response_class=None)
def metrics() -> str:
    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    high_risk = risk_counter.get("HIGH", 0)
    medium_risk = risk_counter.get("MEDIUM", 0)
    low_risk = risk_counter.get("LOW", 0)
    return "\n".join(
        [
            "# HELP inference_requests_total Total API requests.",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {request_counter}",
            "# HELP inference_errors_total Total API errors.",
            "# TYPE inference_errors_total counter",
            f"inference_errors_total {error_counter}",
            "# HELP inference_latency_seconds_avg Average request latency in seconds.",
            "# TYPE inference_latency_seconds_avg gauge",
            f"inference_latency_seconds_avg {average_latency:.6f}",
            "# HELP inference_predictions_total Prediction counts by risk level.",
            "# TYPE inference_predictions_total counter",
            f'inference_predictions_total{{risk_level="HIGH"}} {high_risk}',
            f'inference_predictions_total{{risk_level="MEDIUM"}} {medium_risk}',
            f'inference_predictions_total{{risk_level="LOW"}} {low_risk}',
        ]
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        probability = predictor.predict(payload)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    risk_level, action = classify_risk(probability)
    risk_counter[risk_level] += 1
    return PredictionResponse(
        failure_probability=round(probability, 4),
        risk_level=risk_level,
        recommended_action=action,
        model_source=predictor.model_source,
        model_version=predictor.model_version,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
