# ZenML GitFlow Template

A template repository for building production ML pipelines with ZenML and GitHub Actions. Fork this repo and adapt it to your own ML problem.

## What This Does

When you open a pull request:
- **To staging branch**: Creates a pipeline snapshot and runs it on your staging stack
- **To main branch**: Creates a production snapshot (manual approval required to run)

All configuration (model names, tags, snapshot naming) is centralized in `project_config.yaml`.

## Quick Start

1. Fork this repository
2. Set up your ZenML server and stacks
3. Add `ZENML_API_KEY` to your GitHub repository secrets
4. Edit `project_config.yaml` with your project details
5. Update `.github/workflows/pipeline_run.yaml` with your ZenML server URL and stack names

## Project Structure

```
├── project_config.yaml      # Central configuration (edit this first)
├── pipeline/
│   └── training_pipeline.py
├── steps/                   # Pipeline steps
├── configs/
│   ├── local.yml           # Local development settings
│   ├── staging.yml         # Staging environment settings
│   └── production.yml      # Production environment settings
├── build.py                # Creates snapshots (used by CI/CD)
├── run.py                  # Runs pipeline locally
└── promote.py              # Promotes model versions to production
```

## Configuration

Edit `project_config.yaml` to customize your project:

```yaml
project:
  name: "my-project"

model:
  name: "MyModel"
  description: "My ML model"
  tags: ["classification", "v1"]

pipeline:
  name: "my_training_pipeline"
  run_name_prefix: "training"
  tags: ["training"]

snapshot:
  prefix: "my_model"  # Generates: STG_my_model_abc1234
```

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline locally:

```bash
python run.py --environment local
```

## CI/CD Workflow

The GitHub Actions workflow in `.github/workflows/pipeline_run.yaml` handles automation.

Update these environment variables for your setup:

```yaml
env:
  ZENML_STORE_URL: https://your-workspace.zenml.io
  ZENML_PROJECT: your-project
  ZENML_STAGING_STACK: your-staging-stack
  ZENML_PRODUCTION_STACK: your-production-stack
```

## Adapting for Your ML Problem

1. Replace the steps in `steps/` with your own data loading, preprocessing, and training logic
2. Update `pipeline/training_pipeline.py` to wire up your steps
3. Adjust parameters in `configs/*.yml` for your use case
4. Update `project_config.yaml` with your model and pipeline names

## Commands

Create a snapshot manually:

```bash
python build.py --environment staging --stack my-stack
```

Create and run:

```bash
python build.py --environment staging --stack my-stack --run
```

Promote a model to production:

```bash
python promote.py --version 1
```

## Requirements

- Python 3.10+
- ZenML server (cloud or self-hosted)
- Remote stack with orchestrator, artifact store, and container registry

## Links

- [ZenML Documentation](https://docs.zenml.io/)
- [Pipeline Snapshots](https://docs.zenml.io/concepts/snapshots)
- [Model Control Plane](https://docs.zenml.io/concepts/models)
