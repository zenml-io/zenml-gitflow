#!/usr/bin/env bash

set -Eeo pipefail

zenml stack register local_gitflow_stack \
    -a default \
    -o default \

zenml stack set local_gitflow_stack