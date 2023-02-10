#!/usr/bin/env bash

msg() {
  echo >&2 -e "${1-}"
}

set -Eeo pipefail

# These settings are hard-coded at the moment
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.eu-central-1.amazonaws.com
aws eks --region eu-central-1 update-kubeconfig --name kubeflowmultitenant --alias kubeflowmultitenant

zenml secrets-manager register aws_secrets_manager \
  --flavor=aws --region_name=eu-central-1 || \
  msg "${WARNING}Reusing preexisting secrets manager ${NOFORMAT}aws_secrets_manager"

zenml experiment-tracker register aws_mlflow_tracker --flavor=mlflow \
  --tracking_insecure_tls=true \
  --tracking_uri="https://mlflow.develaws.zenml.io/" \
  --tracking_username="{{mlflow_secret.tracking_username}}" \
  --tracking_password="{{mlflow_secret.tracking_password}}" || \
  msg "${WARNING}Reusing preexisting experiment tracker ${NOFORMAT}aws_mlflow_tracker"

zenml orchestrator register aws_multi_tenant_kubeflow \
  --flavor=kubeflow \
  --kubernetes_context=kubeflowmultitenant \
  --kubeflow_hostname=https://www.kubeflowshowcase.zenml.io/pipeline \
  --synchronous=true \
  --timeout=1200 || \
  msg "${WARNING}Reusing preexisting orchestrator ${NOFORMAT}multi_tenant_kubeflow"

zenml artifact-store register s3_store -f s3 --path=s3://zenfiles || \
  msg "${WARNING}Reusing preexisting artifact store ${NOFORMAT}s3_store"

zenml image-builder register local_image_builder -f local || \
  msg "${WARNING}Reusing preexisting image builder ${NOFORMAT}local_image_builder"

zenml container-registry register ecr_registry --flavor=aws \
  --uri=715803424590.dkr.ecr.us-east-1.amazonaws.com  || \
  msg "${WARNING}Reusing preexisting container registry ${NOFORMAT}ecr_registry"

# For GKE clusters, the host is the GKE cluster IP address.
# export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
# For EKS clusters, the host is the EKS cluster IP hostname.
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export INGRESS_URL="http://${INGRESS_HOST}:${INGRESS_PORT}"

zenml model-deployer register kserve_s3 --flavor=kserve \
  --kubernetes_context=kubeflowmultitenant \
  --kubernetes_namespace=zenml-workloads \
  --base_url=$INGRESS_URL --secret=kservesecret || \
  msg "${WARNING}Reusing preexisting model deployer ${NOFORMAT}kserve_s3"

zenml data-validator register evidently --flavor=evidently || \
  msg "${WARNING}Reusing preexisting data validator ${NOFORMAT}evidently"

zenml stack register devweek_aws_stack \
    -a s3_store \
    -c ecr_registry \
    -o aws_multi_tenant_kubeflow \
    -x aws_secrets_manager \
    -d kserve_s3 \
    -dv evidently \
    -e aws_mlflow_tracker \
    -i local_image_builder || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}devweek_aws_stack"

zenml stack set devweek_aws_stack
zenml stack share -r devweek_aws_stack

zenml secrets-manager secret register -s kserve_s3 kservesecret --credentials="@~/.aws/credentials" 

echo "In the following prompt, please set the `tracking_username` key with value of your MLflow username and `tracking_password` key with value of your MLflow password. "
zenml secrets-manager secret register mlflow_secret \
  --tracking_username=<VALUE_1> \
  --tracking_password=<VALUE_2>
