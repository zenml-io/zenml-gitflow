#!/usr/bin/env bash

set -Eeo pipefail

zenml data-validator register deepchecks_data_validator --flavor=deepchecks
zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow
zenml stack register local_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -dv deepchecks_data_validator
zenml stack set local_stack