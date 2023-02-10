#!/usr/bin/env bash

msg() {
  echo >&2 -e "${1-}"
}

set -Eeo pipefail

zenml artifact-store register local --flavor local || \
  msg "${WARNING}Reusing preexisting artifact store ${NOFORMAT}local"
zenml orchestrator register local --flavor local || \
  msg "${WARNING}Reusing preexisting orchestrator ${NOFORMAT}local"
zenml data-validator register evidently --flavor=evidently || \
  msg "${WARNING}Reusing preexisting data validator ${NOFORMAT}evidently"
zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow || \
  msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}local_mlflow_tracker"
zenml model-deployer register local_mlflow_deployer --flavor=mlflow || \
  msg "${WARNING}Reusing preexisting model deployer ${NOFORMAT}local_mlflow_deployer"
zenml stack register devweek_local_stack \
    -a local \
    -o local \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv evidently || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}devweek_local_stack"
zenml stack set devweek_local_stack
zenml stack share -r devweek_local_stack