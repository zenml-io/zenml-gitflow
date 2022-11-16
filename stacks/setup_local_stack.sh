#!/usr/bin/env bash

set -Eeo pipefail

zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow || \
  msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}mlflow_tracker"
zenml stack register local_gitflow_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}local_gitflow_stack"

zenml stack set local_gitflow_stack