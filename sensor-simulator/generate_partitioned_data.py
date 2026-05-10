from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from simulator.generator import build_machine_profiles, generate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate partitioned synthetic telemetry for S3 ingestion.")
    parser.add_argument("--machines", type=int, default=16, help="Number of machine profiles to simulate.")
    parser.add_argument("--days", type=int, default=7, help="Number of days of telemetry to generate.")
    parser.add_argument("--frequency-minutes", type=int, default=15, help="Sampling interval in minutes.")
    parser.add_argument("--anomaly-probability", type=float, default=0.03, help="Per-event anomaly probability.")
    parser.add_argument("--failure-probability", type=float, default=0.005, help="Per-event base failure probability.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/raw"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    periods = max(int((args.days * 24 * 60) / args.frequency_minutes), 1)
    start_time = datetime.now(timezone.utc) - timedelta(days=args.days)
    profiles = build_machine_profiles(machine_count=args.machines, seed=args.seed)
    records = [
        record.to_dict()
        for record in generate_records(
            profiles=profiles,
            start_time=start_time,
            periods=periods,
            frequency_minutes=args.frequency_minutes,
            anomaly_probability=args.anomaly_probability,
            failure_probability=args.failure_probability,
            seed=args.seed,
        )
    ]

    dataframe = pd.DataFrame.from_records(records)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    dataframe["year"] = dataframe["timestamp"].dt.strftime("%Y")
    dataframe["month"] = dataframe["timestamp"].dt.strftime("%m")
    dataframe["day"] = dataframe["timestamp"].dt.strftime("%d")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(args.output_dir, index=False, partition_cols=["year", "month", "day"])
    print(f"Wrote {len(dataframe)} telemetry rows into partitioned dataset at {args.output_dir}")


if __name__ == "__main__":
    main()