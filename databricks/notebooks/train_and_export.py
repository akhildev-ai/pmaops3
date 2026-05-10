# Databricks notebook source
from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
import os

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "rolling_avg_temp",
    "rolling_avg_vibration",
    "pressure_variance",
    "rpm_stability_score",
    "max_anomaly_score",
    "avg_degradation_index",
    "temp_vibration_ratio",
    "vibration_pressure_ratio",
    "stability_penalty",
]


def get_runtime_value(name: str, default: str = "") -> str:
    if "dbutils" in globals():
        try:
            dbutils.widgets.text(name, default)
        except Exception:
            pass
        try:
            value = dbutils.widgets.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.getenv(name.upper(), default)


def build_bronze(raw_df):
    from pyspark.sql import functions as F

    return raw_df.withColumn("ingested_at", F.current_timestamp())


def build_silver(bronze_df):
    from pyspark.sql import functions as F

    return (
        bronze_df.dropDuplicates(["machine_id", "timestamp"])
        .withColumn("timestamp", F.to_timestamp("timestamp"))
        .fillna(
            {
                "temperature": 0.0,
                "vibration": 0.0,
                "rpm": 0,
                "pressure": 0.0,
                "voltage": 0.0,
                "humidity": 0.0,
                "anomaly_score": 0.0,
                "degradation_index": 0.0,
                "failed": 0,
            }
        )
        .filter(F.col("temperature").between(0, 200))
        .filter(F.col("vibration").between(0, 5))
        .filter(F.col("pressure").between(0, 250))
    )


def build_gold(silver_df):
    from pyspark.sql import functions as F

    by_machine = silver_df.groupBy("machine_id", "machine_type", F.window("timestamp", "6 hours"))
    return by_machine.agg(
        F.avg("temperature").alias("rolling_avg_temp"),
        F.avg("vibration").alias("rolling_avg_vibration"),
        F.variance("pressure").alias("pressure_variance"),
        F.stddev("rpm").alias("rpm_stability_score"),
        F.max("anomaly_score").alias("max_anomaly_score"),
        F.avg("degradation_index").alias("avg_degradation_index"),
        F.max("failed").alias("failed"),
    ).select(
        "machine_id",
        "machine_type",
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "rolling_avg_temp",
        "rolling_avg_vibration",
        "pressure_variance",
        "rpm_stability_score",
        "max_anomaly_score",
        "avg_degradation_index",
        "failed",
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["pressure_variance"] = enriched["pressure_variance"].fillna(0.0)
    enriched["rpm_stability_score"] = enriched["rpm_stability_score"].fillna(0.0)
    enriched["temp_vibration_ratio"] = enriched["rolling_avg_temp"] / enriched["rolling_avg_vibration"].clip(lower=0.05)
    enriched["vibration_pressure_ratio"] = enriched["rolling_avg_vibration"] / (enriched["pressure_variance"].abs() + 0.05)
    enriched["stability_penalty"] = (
        enriched["avg_degradation_index"] * 8.0
        + enriched["max_anomaly_score"] * 0.04
        + enriched["rpm_stability_score"].abs() * 0.3
    )
    return enriched


def candidate_models() -> dict[str, object]:
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


def build_pipeline(base_model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base_model),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = 0.5
    if y_test.nunique() > 1:
        roc_auc = float(roc_auc_score(y_test, probabilities))
    return {
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def maybe_write_delta(df, path: str) -> None:
    if not path:
        return
    df.write.format("delta").mode("overwrite").save(path)


def upload_file(local_path: Path, destination_uri: str) -> None:
    if destination_uri.startswith("s3://"):
        import boto3

        parsed = urlparse(destination_uri)
        boto3.client("s3").upload_file(str(local_path), parsed.netloc, parsed.path.lstrip("/"))
        return

    if destination_uri.startswith("dbfs:/"):
        target_path = Path(destination_uri.replace("dbfs:/", "/dbfs/"))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target_path)
        return

    target_path = Path(destination_uri)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, target_path)


def main() -> None:
    if "spark" not in globals():
        raise RuntimeError("This notebook must run in Databricks or another Spark runtime.")

    raw_data_uri = get_runtime_value("raw_data_uri")
    register_model_name = get_runtime_value("register_model_name")
    model_export_uri = get_runtime_value("model_export_uri")
    experiment_path = get_runtime_value("experiment_path", "/Shared/industrial-ai-platform/experiments/dev")
    bronze_output_uri = get_runtime_value("bronze_output_uri")
    silver_output_uri = get_runtime_value("silver_output_uri")
    gold_output_uri = get_runtime_value("gold_output_uri")

    if not raw_data_uri:
        raise RuntimeError("raw_data_uri is required.")
    if not model_export_uri:
        raise RuntimeError("model_export_uri is required so deployment can fetch the trained artifact.")

    mlflow.set_experiment(experiment_path)

    raw_df = spark.read.parquet(raw_data_uri)
    bronze_df = build_bronze(raw_df)
    silver_df = build_silver(bronze_df)
    gold_df = build_gold(silver_df)

    maybe_write_delta(bronze_df, bronze_output_uri)
    maybe_write_delta(silver_df, silver_output_uri)
    maybe_write_delta(gold_df, gold_output_uri)

    gold_pdf = engineer_features(gold_df.toPandas())
    if gold_pdf.empty:
        raise RuntimeError("No data available after Gold feature generation.")

    x = gold_pdf[FEATURE_COLUMNS]
    y = gold_pdf["failed"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training data must include both failed and healthy records.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_model: Pipeline | None = None
    best_name = ""
    best_metrics: dict[str, float] = {"roc_auc": float("-inf")}
    best_run_id = ""

    for model_name, base_model in candidate_models().items():
        pipeline = build_pipeline(base_model)
        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(x_train, y_train)
            metrics = evaluate_model(pipeline, x_test, y_test)
            mlflow.log_params({"model_name": model_name, "feature_count": len(FEATURE_COLUMNS)})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            if metrics["roc_auc"] > best_metrics["roc_auc"]:
                best_model = pipeline
                best_name = model_name
                best_metrics = metrics
                best_run_id = run.info.run_id

    if best_model is None:
        raise RuntimeError("No model was trained.")

    if register_model_name and best_run_id:
        mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=register_model_name)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": best_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": best_run_id,
    }

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        model_path = temp_dir_path / "model_package.joblib"
        metrics_path = temp_dir_path / "model_metrics.json"
        joblib.dump(artifact, model_path)
        metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        upload_file(model_path, model_export_uri)

        metrics_export_uri = model_export_uri.rsplit(".", 1)[0] + ".json" if "." in model_export_uri else model_export_uri + ".json"
        upload_file(metrics_path, metrics_export_uri)

    summary = {
        "raw_data_uri": raw_data_uri,
        "register_model_name": register_model_name,
        "model_export_uri": model_export_uri,
        "best_model": best_name,
        "metrics": best_metrics,
        "rows_used": len(gold_pdf),
    }
    print(json.dumps(summary, indent=2))


main()
