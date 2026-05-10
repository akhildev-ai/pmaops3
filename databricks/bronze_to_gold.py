from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def build_bronze(raw_df: DataFrame) -> DataFrame:
    return raw_df.withColumn("ingested_at", F.current_timestamp())


def build_silver(bronze_df: DataFrame) -> DataFrame:
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


def build_gold(silver_df: DataFrame) -> DataFrame:
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
