from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


mlflow = None
try:
    mlflow = importlib.import_module("mlflow")
except ImportError:
    mlflow = None


FEATURE_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a predictive-maintenance baseline model.")
    parser.add_argument("--input", type=Path, required=True, help="Path to telemetry parquet file.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_package.joblib"))
    parser.add_argument("--experiment", default="industrial-predictive-maintenance")
    parser.add_argument("--register-model-name", default=os.getenv("MLFLOW_REGISTER_MODEL_NAME", ""))
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["temp_vibration_ratio"] = enriched["temperature"] / enriched["vibration"].clip(lower=0.05)
    enriched["pressure_rpm_ratio"] = enriched["pressure"] / enriched["rpm"].clip(lower=100)
    enriched["stress_index"] = (
        enriched["temperature"] * 0.35
        + enriched["vibration"] * 48.0
        + enriched["pressure"] * 0.25
        + enriched["degradation_index"] * 16.0
    )
    return enriched


def candidate_models() -> dict[str, Any]:
    return {
        "xgboost": XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(max_iter=400),
        "random_forest": RandomForestClassifier(n_estimators=220, random_state=42),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = 0.5
    if y_test.nunique() > 1:
        roc_auc = float(roc_auc_score(y_test, probabilities))
    return {
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def build_pipeline(base_model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base_model),
        ]
    )


def mlflow_enabled() -> bool:
    return mlflow is not None and os.getenv("ENABLE_MLFLOW", "false").lower() in {"1", "true", "yes"}


def log_model_with_mlflow(model: Pipeline, artifact_path: str) -> None:
    if mlflow is None:
        return
    sklearn_module = importlib.import_module("mlflow.sklearn")
    sklearn_module.log_model(model, artifact_path=artifact_path)


def main() -> None:
    args = parse_args()
    dataframe = pd.read_parquet(args.input)
    dataframe = engineer_features(dataframe)

    use_mlflow = mlflow_enabled()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if use_mlflow and tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if use_mlflow:
        mlflow.set_experiment(args.experiment)

    x = dataframe[FEATURE_COLUMNS]
    y = dataframe["failed"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training data must include both failed and healthy records. Increase generation horizon or failure probability.")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    best_model: Pipeline | None = None
    best_name = ""
    best_metrics: dict[str, float] = {"roc_auc": float("-inf")}
    best_run_id = ""

    for model_name, base_model in candidate_models().items():
        pipeline = build_pipeline(base_model)
        run_id = ""

        if use_mlflow:
            with mlflow.start_run(run_name=model_name) as run:
                pipeline.fit(x_train, y_train)
                metrics = evaluate_model(pipeline, x_test, y_test)
                mlflow.log_params({"model_name": model_name, "feature_count": len(FEATURE_COLUMNS)})
                mlflow.log_metrics(metrics)
                log_model_with_mlflow(pipeline, artifact_path="model")
                run_id = run.info.run_id
        else:
            pipeline.fit(x_train, y_train)
            metrics = evaluate_model(pipeline, x_test, y_test)

        if metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_model = pipeline
            best_name = model_name
            best_metrics = metrics
            best_run_id = run_id

    if best_model is None:
        raise RuntimeError("No model was trained.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": best_run_id,
    }
    joblib.dump(artifact, args.output)

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(artifact["metrics"], indent=2), encoding="utf-8")

    if args.register_model_name and best_run_id and use_mlflow:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=args.register_model_name)

    if not use_mlflow:
        print("MLflow tracking disabled. Set ENABLE_MLFLOW=true in Databricks or CI when MLflow is available.")

    print(f"Best model: {best_name}")
    print(json.dumps(best_metrics, indent=2))
    print(f"Saved artifact to {args.output}")


if __name__ == "__main__":
    main()
