#!/usr/bin/env bash

set -Eeo pipefail

setup_stack () {
  zenml experiment-tracker register mlflow_tracker  --flavor=mlflow || \
    msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}mlflow_tracker"
  zenml data-validator register evidently_validator --flavor=evidently
  zenml stack register local_gitflow_stack \
      -a default \
      -o default \
      -dv evidently_validator \
      -e mlflow_tracker || \
    msg "${WARNING}Reusing preexisting stack ${NOFORMAT}local_gitflow_stack"

  zenml stack set local_gitflow_stack

  zenml experiment-tracker register aws_mlflow_tracker  --flavor=mlflow --tracking_uri="https://ac8e6c63af207436194ab675ee71d85a-1399000870.us-east-1.elb.amazonaws.com/mlflow" --tracking_username="admin" --tracking_password="zenml" || \
    msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}mlflow_tracker"
  zenml orchestrator register multi_tenant_kubeflow \
    --flavor=kubeflow \
    --kubernetes_context=fullvanilladeployment \
    --kubeflow_hostname=https://www.kubeflow.zenml.io/pipeline

  zenml artifact-store register s3_store -f s3 --path=s3://zenfiles|| \
    msg "${WARNING}Reusing preexisting artifact_store ${NOFORMAT}s3_store"

  zenml container-registry register ecr_registry --flavor=aws --uri=715803424590.dkr.ecr.us-east-1.amazonaws.com 

  zenml stack register kubeflow_gitflow_stack \
      -a s3_store \
      -c ecr_registry \
      -o multi_tenant_kubeflow \
      -dv evidently_validator \
      -e aws_mlflow_tracker || \
    msg "${WARNING}Reusing preexisting stack ${NOFORMAT}kubeflow_gitflow_stack"

  zenml stack set kubeflow_gitflow_stack
}

pre_run () {
  zenml integration install s3 aws kubeflow sklearn mlflow evidently facets
}

pre_run_forced () {
  zenml integration install s3 aws kubeflow sklearn mlflow evidently facets -y
}

post_run () {
  # cleanup the last local ZenML daemon started by the example
  pkill -n -f zenml.services.local.local_daemon_entrypoint
}
