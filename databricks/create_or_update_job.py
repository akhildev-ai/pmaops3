from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "job-template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a Databricks job from a JSON template.")
    parser.add_argument("--host", required=True, help="Databricks workspace host, for example https://dbc-xxxx.cloud.databricks.com")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Path to Databricks job JSON template.")
    parser.add_argument("--job-id", type=int, help="Existing Databricks job id to update. If omitted, a new job is created.")
    parser.add_argument("--replace", action="append", default=[], help="Template replacement in OLD=NEW form. Can be repeated.")
    return parser.parse_args()


def load_template(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_replacements(payload: dict[str, Any], replacements: list[str]) -> dict[str, Any]:
    serialized = json.dumps(payload)
    for replacement in replacements:
        if "=" not in replacement:
            raise ValueError(f"Invalid --replace value: {replacement}. Expected OLD=NEW.")
        old_value, new_value = replacement.split("=", 1)
        serialized = serialized.replace(old_value, new_value)
    return json.loads(serialized)


def api_request(host: str, path: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{host.rstrip('/')}{path}"
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Databricks API call failed for {url}: {exc.code} {message}") from exc


def main() -> None:
    args = parse_args()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required.")

    payload = load_template(args.template)
    payload = apply_replacements(payload, args.replace)

    if args.job_id is None:
        response = api_request(args.host, "/api/2.1/jobs/create", token, payload)
        print(json.dumps({"action": "created", "job_id": response.get("job_id")}, indent=2))
        return

    reset_payload = {"job_id": args.job_id, "new_settings": payload}
    api_request(args.host, "/api/2.1/jobs/reset", token, reset_payload)
    print(json.dumps({"action": "updated", "job_id": args.job_id}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
