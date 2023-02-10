#!/usr/bin/env bash

msg() {
  echo >&2 -e "${1-}"
}

set -Eeo pipefail

zenml data-validator register evidently --flavor=evidently || \
  msg "${WARNING}Reusing preexisting data validator ${NOFORMAT}evidently"
zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow || \
  msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}local_mlflow_tracker"
zenml model-deployer register local_mlflow_deployer --flavor=mlflow || \
  msg "${WARNING}Reusing preexisting model deployer ${NOFORMAT}local_mlflow_deployer"
zenml stack register local_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv evidently || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}local_stack"
zenml stack set local_stack