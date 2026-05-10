# Cloud Setup Checklist

This file is the fastest setup reference for running the repository entirely through GitHub Actions.

## GitHub environments

- Environment name: `dev`

## GitHub repository variables

- `AWS_REGION` = AWS region for all resources, for example `us-east-1`
- `RAW_DATA_BUCKET` = S3 bucket name for raw telemetry, for example `industrial-ai-dev-raw-<account-id>`
- `RAW_DATA_PREFIX` = S3 prefix for raw telemetry, for example `predictive-maintenance/raw`
- `ECR_REPOSITORY` = ECR repository name, for example `industrial-ai/inference-service`
- `ECS_CLUSTER` = ECS cluster name, for example `industrial-ai-dev-cluster`
- `ECS_SERVICE` = ECS service name, for example `industrial-ai-inference-dev`
- `INFERENCE_BASE_URL` = public load balancer URL, for example `https://api-dev.example.com`
- `MODEL_ARTIFACT_S3_URI` = optional model artifact object for deploy builds, for example `s3://industrial-ai-dev-models/exported/model_package.joblib`
- `DATABRICKS_HOST` = workspace base URL, for example `https://dbc-xxxxxxxx.cloud.databricks.com`
- `DATABRICKS_JOB_ID` = Databricks job id that runs ingestion and training
- `MLFLOW_REGISTER_MODEL_NAME` = MLflow registered model name, for example `industrial_predictive_maintenance_model`

## GitHub repository secrets

- `AWS_ROLE_TO_ASSUME` = IAM role ARN trusted by GitHub OIDC, for example `arn:aws:iam::<account-id>:role/github-actions-industrial-ai-dev`
- `DATABRICKS_TOKEN` = Databricks PAT for the training workflow

## AWS resources to create

- S3 raw bucket name: `industrial-ai-dev-raw-<account-id>`
- S3 model bucket name: `industrial-ai-dev-models-<account-id>`
- ECR repository name: `industrial-ai/inference-service`
- ECS cluster name: `industrial-ai-dev-cluster`
- ECS service name: `industrial-ai-inference-dev`
- ECS task definition family: `industrial-ai-inference-dev`
- CloudWatch log group: `/ecs/industrial-ai-inference-dev`
- IAM OIDC role: `github-actions-industrial-ai-dev`
- ECS task execution role: `ecsTaskExecutionRole-industrial-ai-dev`
- Application Load Balancer name: `industrial-ai-dev-alb`
- Target group name: `industrial-ai-inference-dev-tg`

## AWS console links

- ECR: https://console.aws.amazon.com/ecr/repositories
- ECS clusters: https://console.aws.amazon.com/ecs/v2/clusters
- ECS task definitions: https://console.aws.amazon.com/ecs/v2/task-definitions
- S3: https://console.aws.amazon.com/s3/buckets
- CloudWatch log groups: https://console.aws.amazon.com/cloudwatch/home#logsV2:log-groups
- IAM identity providers: https://console.aws.amazon.com/iamv2/home#/identity_providers
- IAM roles: https://console.aws.amazon.com/iam/home#/roles
- Load balancers: https://console.aws.amazon.com/ec2/home#LoadBalancers:
- Target groups: https://console.aws.amazon.com/ec2/home#TargetGroups:

## Databricks resources to create

- Workspace folder: `/Shared/industrial-ai-platform`
- Notebook path: `/Shared/industrial-ai-platform/notebooks/train_and_export`
- Experiment: `/Shared/industrial-ai-platform/experiments/dev`
- Job: `industrial-ai-train-dev`
- Registered model: `industrial_predictive_maintenance_model`
- Optional Unity Catalog catalog: `industrial_ai`
- Optional schema: `predictive_maintenance_dev`

## Databricks UI links

- Jobs: `https://<your-workspace>/jobs`
- Experiments: `https://<your-workspace>/ml/experiments`
- Models: `https://<your-workspace>/ml/models`
- Compute: `https://<your-workspace>/compute`
- Workspace: `https://<your-workspace>/workspace`

## Workflow order

1. `.github/workflows/ci.yml` validates the codebase.
2. `.github/workflows/upload-telemetry.yml` generates partitioned data and uploads it to S3.
3. `.github/workflows/databricks-train.yml` triggers the Databricks ingestion and training job.
4. `.github/workflows/deploy-inference.yml` builds the inference image, optionally pulls a model artifact from S3, and deploys it to ECS.

## Databricks job contract

The Databricks job should accept these notebook parameters from GitHub Actions:

- `raw_data_uri`
- `register_model_name`
- `model_export_uri`

The job should export the selected model artifact to the S3 object referenced by `MODEL_ARTIFACT_S3_URI` so the ECS deploy workflow can bundle it into the image.
