#!/usr/bin/env bash

set -Eeo pipefail

# These settings are hard-coded at the moment
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.us-east-1.amazonaws.com
aws eks --region us-east-1 update-kubeconfig --name zenhacks-cluster --alias zenml-eks

zenml orchestrator register multi_tenant_kubeflow \
  --flavor=kubeflow \
  --kubernetes_context=kubeflowmultitenant \
  --kubeflow_hostname=https://www.kubeflowshowcase.zenml.io/pipeline

zenml artifact-store register s3_store -f s3 --path=s3://zenfiles

zenml image-builder register local_image_builder -f local
zenml container-registry register ecr_registry --flavor=aws --uri=715803424590.dkr.ecr.us-east-1.amazonaws.com 

# For GKE clusters, the host is the GKE cluster IP address.
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
# For EKS clusters, the host is the EKS cluster IP hostname.
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export INGRESS_URL="http://${INGRESS_HOST}:${INGRESS_PORT}"


zenml stack register aws_gitflow_stack \
    -a s3_store \
    -c ecr_registry \
    -o multi_tenant_kubeflow \
    -i local_image_builder || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}kubeflow_gitflow_stack"

zenml stack set aws_gitflow_stack
zenml stack share aws_gitflow_stack