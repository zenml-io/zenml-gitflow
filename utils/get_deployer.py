#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from zenml.client import Client
from functools import partial


def get_stack_deployer(model_name: str = "model"):
    stack_model_deployer = Client().active_stack.model_deployer
    if stack_model_deployer is None:
        raise ValueError(
            "Cannot run the deployment or end-to-end pipeline: no model "
            "deployer found in stack. "
        )
    elif stack_model_deployer.flavor == "mlflow":

        from zenml.integrations.mlflow.steps import (
            mlflow_model_deployer_step,
        )

        if model_name != "model":
            raise ValueError(
                "Cannot run the deployment or end-to-end pipeline: "
                "model name must be `model` when using the MLFlow "
                "deployer."
            )

        model_deployer = partial(
            mlflow_model_deployer_step,
            model_name=model_name,
            timeout=120,
        )
    elif stack_model_deployer.flavor == "kserve":

        from zenml.integrations.kserve.services import (
            KServeDeploymentConfig,
        )
        from zenml.integrations.kserve.steps import (
            kserve_model_deployer_step,
        )

        model_deployer = partial(
            kserve_model_deployer_step,
            service_config=KServeDeploymentConfig(
                model_name=model_name,
                replicas=1,
                predictor="sklearn",
                resources={"requests": {"cpu": "200m", "memory": "500m"}},
            ),
            timeout=120,
        )
    else:
        raise ValueError(
            f"Cannot run the deployment or end-to-end pipeline: "
            f"model deployer flavor `{stack_model_deployer.flavor}` not "
            f"supported by the pipeline."
        )
    return model_deployer
