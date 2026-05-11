from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from urllib import error, parse, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger and wait for a Databricks job run.")
    parser.add_argument("--host", required=True, help="Databricks workspace host, for example https://dbc-xxxx.cloud.databricks.com")
    parser.add_argument("--job-id", required=True, type=int, help="Databricks job id to run.")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--param", action="append", default=[], help="Notebook parameter in key=value form. Can be repeated.")
    return parser.parse_args()


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def api_request(host: str, path: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    url = f"{host.rstrip('/')}{path}"
    req = request.Request(url=url, data=body, headers=headers(token), method="POST" if payload is not None else "GET")
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Databricks API call failed for {url}: {exc.code} {message}") from exc


def api_get(host: str, path: str, token: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
    suffix = ""
    if query:
        suffix = "?" + parse.urlencode(query)
    return api_request(host, f"{path}{suffix}", token, payload=None)


def notebook_params(raw_params: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw_params:
        if "=" not in item:
            raise ValueError(f"Invalid --param value: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def get_job_settings(host: str, job_id: int, token: str) -> dict[str, Any]:
    job_response = api_get(host, "/api/2.2/jobs/get", token, {"job_id": job_id})
    return job_response.get("settings", {})


def build_run_request(job_id: int, settings: dict[str, Any], params: dict[str, str]) -> tuple[str, dict[str, Any]]:
    tasks = settings.get("tasks", [])
    if tasks:
        payload: dict[str, Any] = {"job_id": job_id}
        if params:
            payload["job_parameters"] = params
        task_keys = [task["task_key"] for task in tasks if "task_key" in task]
        if task_keys:
            payload["only"] = task_keys
        return "/api/2.2/jobs/run-now", payload

    if "notebook_task" in settings:
        payload = {"job_id": job_id}
        if params:
            payload["notebook_params"] = params
        return "/api/2.1/jobs/run-now", payload

    raise RuntimeError(
        "Unsupported Databricks job format: expected settings.tasks or settings.notebook_task in jobs/get response."
    )


def main() -> None:
    args = parse_args()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required.")

    params = notebook_params(args.param)
    settings = get_job_settings(args.host, args.job_id, token)
    run_path, run_payload = build_run_request(args.job_id, settings, params)

    response = api_request(args.host, run_path, token, run_payload)
    run_id = response["run_id"]
    print(f"Triggered Databricks job {args.job_id} with run_id {run_id}")

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        time.sleep(args.poll_seconds)
        state_response = api_request(
            args.host,
            "/api/2.1/jobs/runs/get?" + parse.urlencode({"run_id": run_id}),
            token,
        )
        state = state_response.get("state", {})
        life_cycle_state = state.get("life_cycle_state", "UNKNOWN")
        result_state = state.get("result_state", "PENDING")
        print(f"Run {run_id} status: {life_cycle_state} / {result_state}")

        if life_cycle_state in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            if result_state != "SUCCESS":
                raise RuntimeError(f"Databricks run {run_id} failed with state {life_cycle_state} / {result_state}")
            print(f"Databricks run {run_id} completed successfully")
            return

    raise TimeoutError(f"Timed out waiting for Databricks run {run_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
