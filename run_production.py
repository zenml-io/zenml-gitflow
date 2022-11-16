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


from zenml.client import Client
from zenml.config import DockerSettings

from pipelines import prod_train_and_deploy_pipeline
from steps import (
    TrainerParams,
    deployment_trigger,
    evaluator,
    production_data_loader,
    sklearn_model_deployer,
    svc_trainer_mlflow,
)
from utils.kubeflow_helper import get_kubeflow_settings


def main():

    docker_settings = DockerSettings(required_integrations=["sklearn"])

    experiment_tracker = Client().active_stack.experiment_tracker

    if experiment_tracker is None:
        raise AssertionError("Experiment Tracker needs to exist in the  stack!")
    
    # initialize and run the training pipeline
    training_pipeline_instance = prod_train_and_deploy_pipeline(
        importer=production_data_loader(),
        trainer=svc_trainer_mlflow(
            params=TrainerParams(
                degree=1,
            )
        ).configure(experiment_tracker=experiment_tracker.name),
        evaluator=evaluator(),
        deployment_trigger=deployment_trigger(),
        model_deployer=sklearn_model_deployer,
    )

    # Validate whether stack infra is ready
    Client().active_stack.validate(fail_if_secrets_missing=True)

    # Run pipeline
    training_pipeline_instance.run(
        settings={
            "orchestrator.kubeflow": get_kubeflow_settings(),
            "docker": docker_settings,
        }
    )


if __name__ == "__main__":
    main()
