# Industrial AI Platform

Industrial Predictive Maintenance AI Platform built for a cloud-first workflow using AWS, Databricks, MLflow, GitHub Actions, and synthetic machine telemetry. The first implementation path is optimized for speed: get the end-to-end pipeline working in the cloud first, then codify infrastructure with Terraform and add Kubernetes as a later deployment target.

## Current MVP scope

- Synthetic telemetry generation for motors, pumps, compressors, and conveyors.
- S3-first batch landing path that can later expand into streaming.
- Databricks-ready Bronze, Silver, and Gold feature pipeline reference.
- Databricks-integrated MLflow workflow for experiment tracking and model registry.
- FastAPI inference service with health, version, predict, and metrics endpoints.
- GitHub Actions CI that generates data, trains a baseline model, and runs smoke tests.

## Repository structure

```text
industrial-ai-platform/
├── sensor-simulator/
├── streaming/
├── databricks/
├── ml-training/
├── inference-service/
├── frontend/
├── monitoring/
├── terraform/
├── k8s/
├── edge-node/
├── .github/workflows/
├── architecture-diagram/
├── requirements.txt
└── README.md
```

## End-to-end flow

1. `sensor-simulator` generates historical telemetry and demo traffic.
2. Raw telemetry is intended to land in S3 as partitioned parquet files.
3. `databricks/bronze_to_gold.py` defines the first ETL contract for Delta-oriented processing.
4. `ml-training/train_baseline.py` engineers features, trains candidate models, and can track runs in MLflow when executed in Databricks or another MLflow-enabled environment.
5. The best model artifact is saved for deployment and can also be registered in the MLflow Model Registry from Databricks.
6. `inference-service/app/main.py` serves predictions from MLflow, a local artifact, or a heuristic fallback.
7. `.github/workflows/ci.yml` validates the generator, training path, and API behavior on every push and pull request.

## Quickstart

1. Create or activate a Python 3.12 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to run MLflow tracking outside Databricks, install the optional Databricks requirements:

```bash
pip install -r requirements-databricks.txt
```

3. Generate synthetic telemetry:

```bash
python sensor-simulator/generate_historical_data.py --machines 16 --days 30 --frequency-minutes 30 --output artifacts/historical_telemetry.parquet
```

4. Train the baseline model locally without requiring MLflow:

```bash
python ml-training/train_baseline.py --input artifacts/historical_telemetry.parquet --output artifacts/model_package.joblib
```

To enable MLflow tracking in Databricks or another MLflow-enabled runtime, set `ENABLE_MLFLOW=true` and the appropriate tracking configuration before running the same command.

5. Start the inference service:

```bash
uvicorn inference-service.app.main:app --reload
```

## MLflow usage in this project

MLflow is part of the main MVP flow, but the intended primary runtime is Databricks-integrated MLflow rather than a mandatory local installation.

- Training runs log parameters, metrics, and model artifacts when `ENABLE_MLFLOW=true` and MLflow is available.
- The best run can be registered into the MLflow Model Registry via `--register-model-name` or `MLFLOW_REGISTER_MODEL_NAME` from Databricks or another MLflow-enabled runtime.
- The inference API can load a deployed model from `MLFLOW_MODEL_URI`.
- `/version` exposes which model source is active.

## AWS-first deployment direction

The intended fast-path deployment target is AWS ECS Fargate with GitHub Actions building and pushing images remotely. Terraform and EKS are intentionally deferred until the application contracts and AWS resource shape stabilize.

## Remote deployment setup

The repository includes a first-pass deployment workflow in [.github/workflows/deploy-inference.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/deploy-inference.yml). It builds the API container on GitHub-hosted runners, pushes the image to ECR, updates the ECS task definition, and deploys the service to ECS Fargate.

Create these AWS resources manually before using it:

- ECR repository for the inference image.
- ECS cluster and ECS service.
- ECS task execution role.
- CloudWatch log group for ECS.
- GitHub OIDC IAM role that can push to ECR and update ECS.

Configure these GitHub repository variables:

- `AWS_REGION`
- `ECR_REPOSITORY`
- `ECS_CLUSTER`
- `ECS_SERVICE`
- `INFERENCE_BASE_URL` as an optional smoke-test URL after the load balancer exists

Configure these GitHub repository secrets:

- `AWS_ROLE_TO_ASSUME`

Update [inference-service/deploy/ecs-task-definition.json](c:/Users/akhil.mangalarapu/Pictures/pmaops/inference-service/deploy/ecs-task-definition.json) before the first deploy:

- replace `REPLACE_WITH_EXECUTION_ROLE_ARN`
- replace `REPLACE_WITH_AWS_REGION`
- confirm the task family name
- confirm CPU and memory sizing
- add any runtime environment variables you want ECS to inject

The current container deploys the API in heuristic or local-artifact mode. Connecting the running service to a promoted Databricks or MLflow model artifact is the next step after the ECS deployment path is working.

## Full GitHub Actions path

The repository now has the minimum workflow set to run CI and CD completely from GitHub-hosted runners:

- [.github/workflows/ci.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/ci.yml) validates the codebase.
- [.github/workflows/upload-telemetry.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/upload-telemetry.yml) generates partitioned synthetic telemetry and uploads it to S3.
- [.github/workflows/databricks-train.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/databricks-train.yml) triggers a Databricks job for ingestion, training, and MLflow registration.
- [.github/workflows/deploy-inference.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/deploy-inference.yml) builds the inference image, optionally pulls a promoted model artifact from S3, and deploys it to ECS.

The fastest setup guide is in [docs/cloud-setup-checklist.md](c:/Users/akhil.mangalarapu/Pictures/pmaops/docs/cloud-setup-checklist.md).

For Databricks-specific setup guidance, see [databricks/README.md](c:/Users/akhil.mangalarapu/Pictures/pmaops/databricks/README.md).
