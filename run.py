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

import argparse
import zenml
from zenml.client import Client
from zenml.config import DockerSettings

from pipelines import (
    development_pipeline,
    staging_train_and_deploy_pipeline,
    prod_train_and_deploy_pipeline
)

from steps.data_loaders import (
    development_data_loader,
    production_data_loader,
    staging_data_loader,
)
from steps.evaluators import evaluator
from steps.trainers import TrainerParams, svc_trainer_mlflow

from utils.kubeflow_helper import get_kubeflow_settings


def main(stage: str = "local"):
    """Main runner for all three pipelines.

    Args:
        stage: One of "local", "staging", and "production".
        Defaults to "local".

    Raises:
        AssertionError: "If experiment tracker not in stack."
    """

    experiment_tracker = Client().active_stack.experiment_tracker

    settings = {}
    
    if experiment_tracker is None:
        raise AssertionError("Experiment Tracker needs to exist in the stack!")
    
    if stage == "local":
        # initialize and run the training pipeline
        training_pipeline_instance = development_pipeline(
            importer=development_data_loader(),
            trainer=svc_trainer_mlflow(
                params=TrainerParams(
                    degree=1,
                )
            ).configure(experiment_tracker=experiment_tracker.name),
            evaluator=evaluator(),
        )

    elif stage == "staging":
        # initialize the staging pipeline with a new data loader        
        docker_settings = DockerSettings(
            required_integrations=["sklearn", "mlflow"],
            requirements=["pandas==1.4.0"],
            build_options={
                "buildargs": {
                    "ZENML_VERSION": f"{zenml.__version__}"
                },
            },
        )
        
        training_pipeline_instance = staging_train_and_deploy_pipeline(
            importer=staging_data_loader(),
            trainer=svc_trainer_mlflow(
                params=TrainerParams(
                    degree=1,
                )
            ).configure(experiment_tracker=experiment_tracker.name),
            evaluator=evaluator(),
        )
        
        settings = {
            "orchestrator.kubeflow": get_kubeflow_settings(),
            "docker": docker_settings,
        }

    elif stage == "production":
        from steps.deployment_triggers import deployment_trigger
        from steps.model_deployers import sklearn_model_deployer
        
        # docker settings for production
        docker_settings = DockerSettings(
            required_integrations=["sklearn", "kserve", "mlflow"],
            requirements=["pandas==1.4.0"],
        )
        
        # initialize and run the training pipeline in production
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
        settings = {
            "orchestrator.kubeflow": get_kubeflow_settings(),
            "docker": docker_settings,
        }

    # Run pipeline
    training_pipeline_instance.run(
        settings=settings
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stage",
        default="local",
        help="Toggles which pipeline to run. One of `local`, "
        "`staging`, and `production`. Defaults to `local`",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    
    assert args.stage in ["local", "staging", "production"]
    main(args.stage)
