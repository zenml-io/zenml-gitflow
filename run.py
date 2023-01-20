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
from zenml.config.docker_settings import PythonEnvironmentExportMethod

from pipelines import (
    gitflow_training_pipeline, gitflow_train_and_deploy_pipeline,
)

from steps.data_loaders import (
    development_data_loader,
    production_data_loader,
    staging_data_loader,
    data_loader,
    data_splitter,
)
from steps.model_evaluators import model_evaluator
from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from steps.result_checkers import model_train_results_checker
from steps.model_trainers import TrainerParams, svc_trainer
from steps.data_validators import (
    data_drift_detector,
    data_integrity_checker,
)
from steps.model_validators import model_drift_detector
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from utils.kubeflow_helper import get_kubeflow_settings
from utils.report_generators import deepcheck_suite_to_pdf


def main(stage: str = "local", disable_caching: bool = False):
    """Main runner for all three pipelines.

    Args:
        stage: One of "local", "staging", and "production".
        Defaults to "local".

    Raises:
        AssertionError: "If experiment tracker not in stack."
    """

    settings = {}

    if stage == "local":
        # initialize and run the training pipeline

        training_pipeline_instance = gitflow_training_pipeline(
            importer=data_loader(),
            data_splitter=data_splitter(),
            data_integrity_checker=data_integrity_checker,
            train_test_data_drift_detector=data_drift_detector,
            model_trainer=svc_trainer(
                params=TrainerParams(
                    degree=1,
                )
            ),
            model_evaluator=model_evaluator(),
            train_test_model_drift_detector=model_drift_detector,
            result_checker=model_train_results_checker(),
        )

    elif stage == "staging":
        # initialize the staging pipeline with a new data loader
        docker_settings = DockerSettings(
            install_stack_requirements=False,
            requirements = "requirements-staging.txt",
            apt_packages=["python3-opencv"], # required to make OpenCV work in a container (which is used by Deepchecks)
        )

        training_pipeline_instance = gitflow_training_pipeline(
            importer=data_loader(),
            data_splitter=data_splitter(),
            data_integrity_checker=data_integrity_checker,
            train_test_data_drift_detector=data_drift_detector,
            model_trainer=svc_trainer(
                params=TrainerParams(
                    degree=1,
                )
            ),
            model_evaluator=model_evaluator(),
            train_test_model_drift_detector=model_drift_detector,
            result_checker=model_train_results_checker(),
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
            install_stack_requirements=False,
            requirements="requirements-prod.txt",
        )

        # initialize and run the training pipeline in production
        training_pipeline_instance = gitflow_train_and_deploy_pipeline(
            importer=production_data_loader(),
            trainer=svc_trainer(
                params=TrainerParams(
                    degree=1,
                )
            ),
            evaluator=model_evaluator(),
            deployment_trigger=deployment_trigger(),
            model_deployer=sklearn_model_deployer,
        )
        settings = {
            "orchestrator.kubeflow": get_kubeflow_settings(),
            "docker": docker_settings,
        }

    # Run pipeline
    training_pipeline_instance.run(
        settings=settings, enable_cache=not disable_caching
    )

    if stage in ["local", "staging"]:
        pipeline_run = training_pipeline_instance.get_runs()[-1]
        data_integrity_step = pipeline_run.get_step(
            step="data_integrity_checker"
        )
        data_drift_step = pipeline_run.get_step(
            step="train_test_data_drift_detector"
        )
        model_drift_step = pipeline_run.get_step(
            step="train_test_model_drift_detector"
        )

        if stage == "local":
            DeepchecksVisualizer().visualize(data_integrity_step)
            DeepchecksVisualizer().visualize(data_drift_step)
            DeepchecksVisualizer().visualize(model_drift_step)

            print(
                "Now run \n "
                f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
                "To inspect your experiment runs within the mlflow UI.\n"
            )
        elif stage == "staging":
            deepcheck_suite_to_pdf(
                data_integrity_step, "data_integrity_report.md"
            )
            deepcheck_suite_to_pdf(data_drift_step, "data_drift_report.md")
            deepcheck_suite_to_pdf(model_drift_step, "model_drift_report.md")


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
    parser.add_argument(
        "-d",
        "--disable-caching",
        default=False,
        help="Disables caching for the pipeline. Defaults to False",
        type=bool,
        required=False,
    )
    args = parser.parse_args()

    assert args.stage in ["local", "staging", "production"]
    assert isinstance(args.disable_caching, bool)
    main(args.stage, args.disable_caching)
