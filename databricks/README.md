# Databricks Integration

The GitHub Actions path assumes Databricks is the primary MLflow runtime. The workflow in [.github/workflows/databricks-train.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/databricks-train.yml) triggers a pre-created Databricks job through the Jobs API and waits for completion.

The notebook source for that job is in [databricks/notebooks/train_and_export.py](c:/Users/akhil.mangalarapu/Pictures/pmaops/databricks/notebooks/train_and_export.py). Import that file into your Databricks workspace as a notebook at `/Shared/industrial-ai-platform/notebooks/train_and_export`, then create a Databricks job that points to it.

Use [databricks/job-template.json](c:/Users/akhil.mangalarapu/Pictures/pmaops/databricks/job-template.json) as the starting payload if you want to create the job through the Databricks Jobs UI or API.

If you want to create or update the job through the Databricks API directly, use [databricks/create_or_update_job.py](c:/Users/akhil.mangalarapu/Pictures/pmaops/databricks/create_or_update_job.py). Example:

```bash
python databricks/create_or_update_job.py \
	--host https://dbc-xxxxxxxx.cloud.databricks.com \
	--replace 123456789012=<your-account-id> \
	--replace industrial-ai-dev-raw=your-raw-bucket \
	--replace industrial-ai-dev-models=your-model-bucket
```

Recommended Databricks resources:

- Workspace folder: `/Shared/industrial-ai-platform`
- Notebook path: `/Shared/industrial-ai-platform/notebooks/train_and_export`
- Job name: `industrial-ai-train-dev`
- MLflow registered model: `industrial_predictive_maintenance_model`
- Experiment path: `/Shared/industrial-ai-platform/experiments/dev`
- Catalog: `industrial_ai`
- Schema: `predictive_maintenance_dev`

Recommended Databricks job parameters:

- `raw_data_uri`
- `register_model_name`
- `model_export_uri`
- `experiment_path`
- `bronze_output_uri`
- `silver_output_uri`
- `gold_output_uri`

Recommended Databricks job tasks:

1. Ingest raw parquet from S3 into a Bronze Delta table.
2. Clean and transform into Silver and Gold tables.
3. Train the baseline model and register the best model in MLflow.
4. Export the approved model artifact to S3 for ECS deployment.

The GitHub Actions training trigger now passes `model_export_uri`, so the Databricks job can publish the exact `model_package.joblib` artifact that [deploy-inference.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/deploy-inference.yml) downloads before building the ECS image.
