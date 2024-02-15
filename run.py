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

import argparse
from typing import TYPE_CHECKING, Optional

from zenml import Model
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.enums import ExecutionStatus
from zenml.integrations.deepchecks import DeepchecksIntegration
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.utils.enum_utils import StrEnum

from configs.global_conf import (
    MAX_SERVE_TEST_ACCURACY_DIFF,
    MAX_SERVE_TRAIN_ACCURACY_DIFF,
    MIN_TEST_ACCURACY,
    MIN_TRAIN_ACCURACY,
    MODEL_NAME,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT,
    WARNINGS_AS_ERRORS,
)
from pipelines import gitflow_end_to_end_pipeline, gitflow_training_pipeline
from utils.kubeflow_helper import get_kubeflow_settings
from utils.report_generators import get_result_and_write_report
from utils.tracker_helper import LOCAL_MLFLOW_UI_PORT, get_tracker_name

if TYPE_CHECKING:
    from zenml.models import PipelineRunResponse


class Pipeline(StrEnum):
    TRAIN = "train"
    END_TO_END = "end-to-end"


def main(
    pipeline_name: Pipeline = Pipeline.TRAIN,
    disable_caching: bool = False,
    ignore_checks: bool = False,
    model_name: str = "model",
    dataset_version: Optional[str] = None,
    version: Optional[str] = None,
    github_pr_url: Optional[str] = None,
    org_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Main runner for all pipelines.

    Args:
        pipeline: One of "train", "pre-deploy", and "end-to-end".
        disable_caching: Whether to disable caching. Defaults to False.
        ignore_checks: Whether to ignore model appraisal checks. Defaults to False.
        model_name: The name to use for the trained/deployed model. Defaults to
            "model".
        dataset_version: The dataset version to use to train the model. If not
            set, the original dataset shipped with sklearn will be used.
        version: The version of the model to be created.
        github_pr_url: The URL of the GitHub pull request.
        org_id: The ID of the organization in ZenML Cloud.
        tenant_id: The ID of the tenant in ZenML Cloud.

    """

    settings = {}
    pipeline_args = {}
    if disable_caching:
        pipeline_args["enable_cache"] = False
    pipeline_args["model"] = Model(name=MODEL_NAME, version=version)

    docker_settings = DockerSettings(
        install_stack_requirements=False,
        requirements="requirements.txt",
        required_integrations=[
            "sklearn",
            "mlflow",
            "deepchecks",
            "s3",
            "kubernetes",
        ],
        apt_packages=DeepchecksIntegration.APT_PACKAGES,  # for Deepchecks
    )
    settings["docker"] = docker_settings

    client = Client()

    orchestrator = client.active_stack.orchestrator
    assert orchestrator is not None, "Orchestrator not in stack."
    if orchestrator.flavor == "kubeflow":
        settings["orchestrator.kubeflow"] = get_kubeflow_settings()

    common_params = dict(
        dataset_version=dataset_version,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        accuracy_metric_name="test_accuracy",
        train_accuracy_threshold=MIN_TRAIN_ACCURACY,
        test_accuracy_threshold=MIN_TEST_ACCURACY,
        warnings_as_errors=WARNINGS_AS_ERRORS,
        ignore_data_integrity_failures=ignore_checks,
        ignore_train_test_data_drift_failures=ignore_checks,
        ignore_model_evaluation_failures=ignore_checks,
        ignore_reference_model=ignore_checks,
        max_depth=5,
        github_pr_url=github_pr_url,
        org_id=org_id,
        tenant_id=tenant_id,
    )

    if pipeline_name == Pipeline.TRAIN:
        run_info: PipelineRunResponse = gitflow_training_pipeline.with_options(
            settings=settings, **pipeline_args
        )(**common_params)

    elif pipeline_name == Pipeline.END_TO_END:
        run_info: (
            PipelineRunResponse
        ) = gitflow_end_to_end_pipeline.with_options(
            settings=settings, **pipeline_args
        )(
            max_train_accuracy_diff=MAX_SERVE_TRAIN_ACCURACY_DIFF,
            max_test_accuracy_diff=MAX_SERVE_TEST_ACCURACY_DIFF,
            model_name=model_name,
            **common_params,
        )

    else:
        raise ValueError(f"Pipeline name `{pipeline_name}` not supported. ")

    # refresh run_info
    run_info = client.get_pipeline_run(run_info.id)

    if run_info.status == ExecutionStatus.FAILED:
        print("Pipeline failed. Check the logs for more details.")
        exit(1)
    elif run_info.status == ExecutionStatus.RUNNING:
        print(
            "Pipeline is still running. The post-execution phase cannot "
            "proceed. Please make sure you use an orchestrator with a "
            "synchronous mode of execution."
        )
        exit(1)

    model_appraiser_step = run_info.steps["model_appraiser"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        default="train",
        help="Toggles which pipeline to run. One of `train` and `end-to-end`. "
        "Defaults to `train`",
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
        "-d",
        "--dataset",
        default=None,
        help="Dataset to use for training. One of `staging`, and `production`. "
        "Leave unset, to use the original dataset shipped with sklearn.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-dc",
        "--disable-caching",
        default=False,
        help="Disables caching for the pipeline. Defaults to False",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--ignore-checks",
        default=False,
        help="Ignore model training checks. Defaults to False",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-gp",
        "--github-pr-url",
        default=None,
        help="GitHub PR URL",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        help="Model Version to create.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--org-id",
        default=None,
        help="ZenML Cloud Organization ID.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--tenant-id",
        default=None,
        help="ZenML Cloud Tenant ID.",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    assert args.pipeline in [
        Pipeline.TRAIN,
        Pipeline.END_TO_END,
    ]
    assert isinstance(args.disable_caching, bool)
    assert isinstance(args.ignore_checks, bool)
    main(
        pipeline_name=Pipeline(args.pipeline),
        disable_caching=args.disable_caching,
        ignore_checks=args.ignore_checks,
        model_name=args.model,
        dataset_version=args.dataset,
        version=args.version,
        github_pr_url=args.github_pr_url,
        org_id=args.org_id,
        tenant_id=args.tenant_id,
    )
