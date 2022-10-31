#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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

from zenml.pipelines import pipeline
import requests

from zenml.client import Client
from zenml.integrations.kubeflow.flavors.kubeflow_orchestrator_flavor import (
    KubeflowOrchestratorSettings,
)


orchestrator = Client().active_stack.orchestrator

if orchestrator.flavor == "kubeflow":
    import os
    NAMESPACE = os.getenv("KUBEFLOW_NAMESPACE")  # This is the user namespace for the profile you want to use
    USERNAME = os.getenv("KUBEFLOW_USERNAME")  # This is the username for the profile you want to use
    PASSWORD = os.getenv("KUBEFLOW_PASSWORD")  # This is the password for the profile you want to use

    def get_kfp_token(username: str, password: str) -> str:
        """Get token for kubeflow authentication."""
        # Resolve host from active stack
        orchestrator = Client().active_stack.orchestrator

        if orchestrator.flavor != "kubeflow":
            raise AssertionError(
                "You can only use this function with an "
                "orchestrator of flavor `kubeflow` in the "
                "active stack!"
            )

        try:
            kubeflow_host = orchestrator.config.kubeflow_hostname
        except AttributeError:
            raise AssertionError(
                "You must configure the Kubeflow orchestrator "
                "with the `kubeflow_hostname` parameter which ends "
                "with `/pipeline` (e.g. `https://mykubeflow.com/pipeline`). "
                "Please update the current kubeflow orchestrator with: "
                f"`zenml orchestrator update {orchestrator.name} "
                "--kubeflow_hostname=<MY_KUBEFLOW_HOST>`"
            )

        session = requests.Session()
        response = session.get(kubeflow_host)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"login": username, "password": password}
        session.post(response.url, headers=headers, data=data)
        session_cookie = session.cookies.get_dict()["authservice_session"]
        return session_cookie


    token = get_kfp_token(USERNAME, PASSWORD)
    session_cookie = "authservice_session=" + token
    kubeflow_settings = KubeflowOrchestratorSettings(
        client_args={"cookies": session_cookie}, user_namespace=NAMESPACE
    )
else:
    kubeflow_settings = {}

@pipeline(
    settings={
        "orchestrator.kubeflow": kubeflow_settings
    }
)
def production_train_and_deploy_pipeline(
    training_data_loader,
    trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    """Train, evaluate, and deploy a model."""
    X_train, X_test, y_train, y_test = training_data_loader()
    model = trainer(X_train=X_train, y_train=y_train)
    test_acc = evaluator(X_test=X_test, y_test=y_test, model=model)
    deployment_decision = deployment_trigger(test_acc)
    model_deployer(deployment_decision, model)
