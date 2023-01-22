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
from enum import Enum
import zenml
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.config.docker_settings import PythonEnvironmentExportMethod

from pipelines import (
    gitflow_training_pipeline,
    gitflow_extended_training_pipeline,
    gitflow_end_to_end_pipeline,
)

from steps.data_loaders import (
    DataSplitterStepParameters,
    development_data_loader,
    production_data_loader,
    staging_data_loader,
    data_loader,
    data_splitter,
)
from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer
from steps.model_appraisers import (
    ModelAppraisalStepParams,
    model_train_appraiser,
    model_train_reference_appraiser,
)
from steps.model_loaders import (
    ServedModelLoaderStepParameters,
    TrainedModelLoaderStepParameters,
    served_model_loader,
    trained_model_loader,
)

from steps.model_trainers import TrainerParams, svc_trainer

from steps.data_validators import (
    data_drift_detector,
    data_integrity_checker,
)
from steps.model_evaluators import (
    ModelScorerStepParams,
    model_scorer,
    model_evaluator,
    optional_model_scorer,
    train_test_model_evaluator,
)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from utils.kubeflow_helper import get_kubeflow_settings
from utils.report_generators import (
    deepcheck_suite_to_pdf,
    get_result_and_write_report,
    get_result_and_write_report,
)
from utils.tracker_helper import LOCAL_MLFLOW_UI_PORT, get_tracker_name


RANDOM_STATE = 38
TRAIN_TEST_SPLIT = 0.2
MIN_TRAIN_ACCURACY = 0.9
MIN_TEST_ACCURACY = 0.8
MAX_SERVE_TRAIN_ACCURACY_DIFF = 0.1
MAX_SERVE_TEST_ACCURACY_DIFF = 0.05


class Pipeline(str, Enum):

    TRAIN = "train"
    PRE_DEPLOY = "pre-deploy"
    END_TO_END = "end-to-end"


def main(
    pipeline_name: Pipeline = Pipeline.TRAIN,
    disable_caching: bool = False,
    ignore_checks: bool = False,
    requirements_file: str = "requirements.txt",
    model_name: str = "model",
):
    """Main runner for all pipelines.

    Args:
        pipeline: One of "train", "pre-deploy", and "end-to-end".
        disable_caching: Whether to disable caching. Defaults to False.
        ignore_checks: Whether to ignore model appraisal checks. Defaults to False.
        requirements_file: The requirements file to use to ensure reproducibility.
            Defaults to "requirements.txt".
        model_name: The name of the model to use. Defaults to "model".
    """

    settings = {}
    pipeline_args = {}
    if disable_caching:
        pipeline_args["enable_cache"] = False

    docker_settings = DockerSettings(
        install_stack_requirements=False,
        requirements=requirements_file,
    )
    settings["docker"] = docker_settings

    client = Client()
    if pipeline_name in [Pipeline.PRE_DEPLOY, Pipeline.END_TO_END]:
        model_deployer = client.active_stack.model_deployer
        if model_deployer is None:
            raise ValueError(
                "Cannot run the deployment or end-to-end pipeline: no model "
                "deployer found in stack. "
            )
        elif model_deployer.flavor == "mlflow":

            from zenml.integrations.mlflow.steps import (
                MLFlowDeployerParameters,
                mlflow_model_deployer_step,
            )

            if model_name != "model":
                raise ValueError(
                    "Cannot run the deployment or end-to-end pipeline: "
                    "model name must be `model` when using the MLFlow "
                    "deployer."
                )

            mlflow_model_deployer = mlflow_model_deployer_step(
                params=MLFlowDeployerParameters(
                    model_name=model_name,
                    timeout=120,
                ),
            )

            model_deployer_step = mlflow_model_deployer

        elif model_deployer.flavor == "kserve":

            from zenml.integrations.kserve.services import (
                KServeDeploymentConfig,
            )
            from zenml.integrations.kserve.steps import (
                KServeDeployerStepParameters,
                kserve_model_deployer_step,
            )

            model_deployer_step = kserve_model_deployer_step(
                params=KServeDeployerStepParameters(
                    service_config=KServeDeploymentConfig(
                        model_name=model_name,
                        replicas=1,
                        predictor="sklearn",
                        resources={
                            "requests": {"cpu": "200m", "memory": "500m"}
                        },
                    ),
                    timeout=120,
                )
            )
        else:
            raise ValueError(
                f"Cannot run the deployment or end-to-end pipeline: "
                f"model deployer flavor `{model_deployer.flavor}` not "
                f"supported by the pipeline."
            )

    orchestrator = client.active_stack.orchestrator
    assert orchestrator is not None, "Orchestrator not in stack."
    if orchestrator.flavor == "kubeflow":
        settings["orchestrator.kubeflow"] = get_kubeflow_settings()

    if pipeline_name == Pipeline.TRAIN:

        pipeline_instance = gitflow_training_pipeline(
            importer=data_loader(),
            data_splitter=data_splitter(
                params=DataSplitterStepParameters(
                    test_size=TRAIN_TEST_SPLIT,
                    random_state=RANDOM_STATE,
                )
            ),
            data_integrity_checker=data_integrity_checker,
            train_test_data_drift_detector=data_drift_detector,
            model_trainer=svc_trainer(),
            model_scorer=model_scorer(
                params=ModelScorerStepParams(
                    accuracy_metric_name="test_accuracy",
                )
            ),
            model_evaluator=model_evaluator,
            train_test_model_evaluator=train_test_model_evaluator,
            model_appraiser=model_train_appraiser(
                params=ModelAppraisalStepParams(
                    train_accuracy_threshold=MIN_TRAIN_ACCURACY,
                    test_accuracy_threshold=MIN_TEST_ACCURACY,
                    ignore_data_integrity_failures=ignore_checks,
                    ignore_train_test_data_drift_failures=ignore_checks,
                    ignore_model_evaluation_failures=ignore_checks,
                )
            ),
        )

    elif pipeline_name == Pipeline.PRE_DEPLOY:

        pipeline_instance = gitflow_extended_training_pipeline(
            importer=data_loader(),
            data_splitter=data_splitter(
                params=DataSplitterStepParameters(
                    test_size=TRAIN_TEST_SPLIT,
                    random_state=RANDOM_STATE,
                )
            ),
            data_integrity_checker=data_integrity_checker,
            train_test_data_drift_detector=data_drift_detector,
            model_trainer=svc_trainer(),
            model_scorer=model_scorer(
                params=ModelScorerStepParams(
                    accuracy_metric_name="test_accuracy",
                )
            ),
            model_evaluator=model_evaluator,
            train_test_model_evaluator=train_test_model_evaluator,
            served_model_loader=served_model_loader(
                params=ServedModelLoaderStepParameters(
                    model_name=model_name,
                    step_name="model_deployer",
                )
            ),
            served_model_train_scorer=optional_model_scorer(
                name="served_model_train_scorer",
                params=ModelScorerStepParams(
                    accuracy_metric_name="reference_train_accuracy",
                ),
            ),
            served_model_test_scorer=optional_model_scorer(
                name="served_model_test_scorer",
                params=ModelScorerStepParams(
                    accuracy_metric_name="reference_test_accuracy",
                ),
            ),
            model_appraiser=model_train_reference_appraiser(
                params=ModelAppraisalStepParams(
                    train_accuracy_threshold=MIN_TRAIN_ACCURACY,
                    test_accuracy_threshold=MIN_TEST_ACCURACY,
                    max_train_accuracy_diff=MAX_SERVE_TRAIN_ACCURACY_DIFF,
                    max_test_accuracy_diff=MAX_SERVE_TEST_ACCURACY_DIFF,
                    ignore_data_integrity_failures=ignore_checks,
                    ignore_train_test_data_drift_failures=ignore_checks,
                    ignore_model_evaluation_failures=ignore_checks,
                    ignore_reference_model=ignore_checks,
                )
            ),
        )

    elif pipeline_name == Pipeline.END_TO_END:

        pipeline_instance = gitflow_end_to_end_pipeline(
            importer=data_loader(),
            data_splitter=data_splitter(
                params=DataSplitterStepParameters(
                    test_size=TRAIN_TEST_SPLIT,
                    random_state=RANDOM_STATE,
                )
            ),
            data_integrity_checker=data_integrity_checker,
            train_test_data_drift_detector=data_drift_detector,
            model_trainer=svc_trainer(),
            model_scorer=model_scorer(
                params=ModelScorerStepParams(
                    accuracy_metric_name="test_accuracy",
                )
            ),
            model_evaluator=model_evaluator,
            train_test_model_evaluator=train_test_model_evaluator,
            served_model_loader=served_model_loader(
                params=ServedModelLoaderStepParameters(
                    model_name=model_name,
                    step_name="model_deployer",
                )
            ),
            served_model_train_scorer=optional_model_scorer(
                name="served_model_train_scorer",
                params=ModelScorerStepParams(
                    accuracy_metric_name="reference_train_accuracy",
                ),
            ),
            served_model_test_scorer=optional_model_scorer(
                name="served_model_test_scorer",
                params=ModelScorerStepParams(
                    accuracy_metric_name="reference_test_accuracy",
                ),
            ),
            model_appraiser=model_train_reference_appraiser(
                params=ModelAppraisalStepParams(
                    train_accuracy_threshold=MIN_TRAIN_ACCURACY,
                    test_accuracy_threshold=MIN_TEST_ACCURACY,
                    max_train_accuracy_diff=MAX_SERVE_TRAIN_ACCURACY_DIFF,
                    max_test_accuracy_diff=MAX_SERVE_TEST_ACCURACY_DIFF,
                    ignore_data_integrity_failures=ignore_checks,
                    ignore_train_test_data_drift_failures=ignore_checks,
                    ignore_model_evaluation_failures=ignore_checks,
                    ignore_reference_model=ignore_checks,
                )
            ),
            model_deployer=model_deployer_step,
        )

    else:
        raise ValueError(f"Pipeline name `{pipeline_name}` not supported. ")

    # Run pipeline
    pipeline_instance.run(settings=settings, **pipeline_args)

    pipeline_run = pipeline_instance.get_runs()[-1]
    data_integrity_step = pipeline_run.get_step(step="data_integrity_checker")
    data_drift_step = pipeline_run.get_step(
        step="train_test_data_drift_detector"
    )
    model_evaluator_step = pipeline_run.get_step(step="model_evaluator")
    train_test_model_evaluator_step = pipeline_run.get_step(
        step="train_test_model_evaluator"
    )
    model_appraiser_step = pipeline_run.get_step(step="model_appraiser")
    report, result = get_result_and_write_report(
        model_appraiser_step, "model_train_results.md"
    )
    print(report)
    if get_tracker_name() and get_tracking_uri().startswith("file"):
        # If mlflow is used as a tracker, print the command to run the UI
        # The reports are accessible as artifacts in the mlflow tracker
        print(
            "NOTE: you have to manually start the MLflow UI by running e.g.:\n "
            f"    mlflow ui --backend-store-uri {get_tracking_uri()} -p {LOCAL_MLFLOW_UI_PORT}\n"
            "to be able inspect your experiment runs within the mlflow UI.\n"
        )
    else:
        # If no tracker is used, open the reports in the browser
        DeepchecksVisualizer().visualize(data_integrity_step)
        DeepchecksVisualizer().visualize(data_drift_step)
        DeepchecksVisualizer().visualize(model_evaluator_step)
        DeepchecksVisualizer().visualize(train_test_model_evaluator_step)

        deepcheck_suite_to_pdf(data_integrity_step, "data_integrity_report.pdf")
        deepcheck_suite_to_pdf(data_drift_step, "data_drift_report.pdf")
        deepcheck_suite_to_pdf(
            model_evaluator_step, "model_evaluator_report.pdf"
        )
        deepcheck_suite_to_pdf(
            train_test_model_evaluator_step,
            "train_test_model_evaluator_report.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        default="train",
        help="Toggles which pipeline to run. One of `train`, "
        "`pre-deploy`, and `end-to-end`. Defaults to `train`",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="model",
        help="Name of the model to train/deploy. Defaults to `model`",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-r",
        "--requirements",
        default="requirements.txt",
        help="Path to file with frozen python requirements needed to run the "
        "pipelines on the active stack. Defaults to `requirements.txt`",
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
    parser.add_argument(
        "-i",
        "--ignore-checks",
        default=False,
        help="Ignore model training checks. Defaults to False",
        type=bool,
        required=False,
    )
    args = parser.parse_args()

    assert args.pipeline in [
        Pipeline.TRAIN,
        Pipeline.PRE_DEPLOY,
        Pipeline.END_TO_END,
    ]
    assert isinstance(args.disable_caching, bool)
    assert isinstance(args.ignore_checks, bool)
    main(
        pipeline_name=Pipeline(args.pipeline),
        disable_caching=args.disable_caching,
        ignore_checks=args.ignore_checks,
        requirements_file=args.requirements,
        model_name=args.model,
    )
