from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from urllib import error, parse, request


SINGLE_TASK_FIELDS = {
    "notebook_task",
    "spark_jar_task",
    "spark_python_task",
    "spark_submit_task",
    "python_wheel_task",
    "pipeline_task",
    "run_job_task",
    "dbt_task",
    "sql_task",
    "condition_task",
    "for_each_task",
    "clean_rooms_notebook_task",
}


class DatabricksApiError(RuntimeError):
    def __init__(self, url: str, code: int, message: str) -> None:
        super().__init__(f"Databricks API call failed for {url}: {code} {message}")
        self.url = url
        self.code = code
        self.message = message


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
        raise DatabricksApiError(url, exc.code, message) from exc


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


def get_job_details(host: str, job_id: int, token: str) -> dict[str, Any]:
    return api_get(host, "/api/2.2/jobs/get", token, {"job_id": job_id})


def build_run_requests(job_id: int, job_details: dict[str, Any], params: dict[str, str]) -> list[tuple[str, dict[str, Any], str]]:
    settings = job_details.get("settings", {})
    tasks = settings.get("tasks") or job_details.get("tasks") or []
    requests_to_try: list[tuple[str, dict[str, Any], str]] = []

    if tasks:
        payload: dict[str, Any] = {"job_id": job_id}
        if params:
            payload["job_parameters"] = params
        task_keys = [task["task_key"] for task in tasks if "task_key" in task]
        if task_keys:
            payload["only"] = task_keys
        requests_to_try.append(("/api/2.2/jobs/run-now", payload, "Jobs API 2.2 with discovered task keys"))

    single_task_field = next(
        (
            field
            for field in SINGLE_TASK_FIELDS
            if field in settings or field in job_details
        ),
        None,
    )
    if single_task_field == "notebook_task":
        payload = {"job_id": job_id}
        if params:
            payload["notebook_params"] = params
        requests_to_try.append(("/api/2.1/jobs/run-now", payload, "Jobs API 2.1 with notebook_params"))
    elif single_task_field:
        payload = {"job_id": job_id}
        if params:
            payload["job_parameters"] = params
        requests_to_try.append(("/api/2.2/jobs/run-now", payload, f"Jobs API 2.2 for detected {single_task_field}"))

    generic_payload: dict[str, Any] = {"job_id": job_id}
    if params:
        generic_payload["job_parameters"] = params
    requests_to_try.append(("/api/2.2/jobs/run-now", generic_payload, "Jobs API 2.2 generic run-now"))

    if params:
        requests_to_try.append(
            (
                "/api/2.1/jobs/run-now",
                {"job_id": job_id, "notebook_params": params},
                "Jobs API 2.1 notebook fallback",
            )
        )

    requests_to_try.append(("/api/2.1/jobs/run-now", {"job_id": job_id}, "Jobs API 2.1 bare run-now"))

    deduplicated: list[tuple[str, dict[str, Any], str]] = []
    seen: set[tuple[str, str]] = set()
    for path, payload, description in requests_to_try:
        key = (path, json.dumps(payload, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((path, payload, description))

    return deduplicated


def trigger_run(
    host: str,
    token: str,
    job_id: int,
    job_details: dict[str, Any],
    params: dict[str, str],
) -> dict[str, Any]:
    settings = job_details.get("settings", {})
    attempts = build_run_requests(job_id, job_details, params)
    failures: list[str] = []

    for path, payload, description in attempts:
        try:
            print(f"Trigger attempt: {description}")
            return api_request(host, path, token, payload)
        except DatabricksApiError as exc:
            failures.append(f"{description}: {exc.code} {exc.message}")
            if exc.code not in {400, 404}:
                raise

    settings_keys = ", ".join(sorted(settings.keys())) or "<none>"
    root_keys = ", ".join(sorted(job_details.keys())) or "<none>"
    raise RuntimeError(
        "Unable to trigger Databricks job after trying supported run-now variants. "
        f"settings keys: {settings_keys}. response keys: {root_keys}. "
        f"Attempts: {' | '.join(failures)}"
    )


def main() -> None:
    args = parse_args()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN is required.")

    params = notebook_params(args.param)
    job_details = get_job_details(args.host, args.job_id, token)
    print(f"DEBUG jobs/get response: {json.dumps(job_details, indent=2, default=str)}")
    response = trigger_run(args.host, token, args.job_id, job_details, params)
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
