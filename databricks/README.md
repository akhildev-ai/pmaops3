# Databricks Integration

The GitHub Actions path assumes Databricks is the primary MLflow runtime. The workflow in [.github/workflows/databricks-train.yml](c:/Users/akhil.mangalarapu/Pictures/pmaops/.github/workflows/databricks-train.yml) triggers a pre-created Databricks job through the Jobs API and waits for completion.

Recommended Databricks resources:

- Workspace folder: `/Shared/industrial-ai-platform`
- Job name: `industrial-ai-train-dev`
- MLflow registered model: `industrial_predictive_maintenance_model`
- Experiment path: `/Shared/industrial-ai-platform/experiments/dev`
- Catalog: `industrial_ai`
- Schema: `predictive_maintenance_dev`

Recommended Databricks job parameters:

- `raw_data_uri`
- `register_model_name`

Recommended Databricks job tasks:

1. Ingest raw parquet from S3 into a Bronze Delta table.
2. Clean and transform into Silver and Gold tables.
3. Train the baseline model and register the best model in MLflow.
4. Export the approved model artifact to S3 for ECS deployment.
