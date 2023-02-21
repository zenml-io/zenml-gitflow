#!/bin/sh -e
set -x

export ZENML_DEBUG=1
export ZENML_LOGGING_VERBOSITY=INFO
export ZENML_ANALYTICS_OPT_IN=false

pip3 install --user -r requirements.txt --default-timeout=3600
eval $(aws sts assume-role --role-arn $AWS_ROLE_ARN --role-session-name SandboxCodeSpaces --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text | awk '{print "export AWS_ACCESS_KEY_ID="$1"\nexport AWS_SECRET_ACCESS_KEY="$2"\nexport AWS_SESSION_TOKEN="$3}')
zenml connect --url https://demoserver.zenml.io --username $ZENML_USERNAME --password $ZENML_PASSWORD
zenml stack set devweek_local_stack
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.eu-central-1.amazonaws.com
aws eks --region eu-central-1 update-kubeconfig --name kubeflowmultitenant --alias kubeflowmultitenant