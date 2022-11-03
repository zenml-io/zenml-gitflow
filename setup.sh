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

  zenml model-deployer register seldon_model_deployer --flavor=seldon --kubernetes_context=fullvanilladeployment  --kubenernetes_namespace=seldon-workloads --base_url=a0ffe798a9241437f969a005b2540275-728628904.eu-central-1.elb.amazonaws.com --secret=aws_seldon_secret

  zenml secrets-manager register aws_secrets_manager --flavor=aws --region_name=eu-central-1

  zenml stack register kubeflow_gitflow_stack \
      -a s3_store \
      -c ecr_registry \
      -o multi_tenant_kubeflow \
      -x aws_secrets_manager \
      -d seldon_model_deployer \
      -e aws_mlflow_tracker || \
    msg "${WARNING}Reusing preexisting stack ${NOFORMAT}kubeflow_gitflow_stack"

  zenml stack set kubeflow_gitflow_stack

  zenml secrets-manager secret register -s aws_seldon_secret s3-store --rclone_config_s3_env_auth=True

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
