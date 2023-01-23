#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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
from typing import Optional

from pipelines import gitflow_data_generation_pipeline
from steps.data_generators import DataGeneratorStepParameters, data_generator

from steps.data_loaders import (
    DataLoaderStepParameters,
    data_loader,
)
from zenml.integrations.deepchecks.visualizers import DeepchecksVisualizer

from steps.data_validators import (
    data_drift_detector,
    data_integrity_checker,
)
from zenml.enums import ExecutionStatus
from zenml.io.fileio import copy


def main(
    disable_caching: bool = False,
    dataset_version: Optional[str] = None,
    size: int = -1,
    filename: str = "./df.csv"
):
    """Generate new data for model training.

    Args:
        dataset_version: The dataset version to use as input. If not set,
            the original dataset shipped with sklearn will be used.
        size: The size of the dataset to generate. Defaults to -1, which
            means the size of the dataset will be the same as the input
            dataset.
        filename: The filename to save the generated data to. Defaults to
            "./df.csv".
    """

    settings = {}
    pipeline_args = {}
    if disable_caching:
        pipeline_args["enable_cache"] = False

    pipeline_instance = gitflow_data_generation_pipeline(
        importer=data_loader(
            params=DataLoaderStepParameters(
                version=dataset_version,
            ),
        ),
        data_generator = data_generator(
            params=DataGeneratorStepParameters(
                size=size,
            ),
        ),
        data_integrity_checker=data_integrity_checker,
        data_drift_detector=data_drift_detector,
    )

    # Run pipeline
    pipeline_instance.run(settings=settings, **pipeline_args)

    pipeline_run = pipeline_instance.get_runs()[-1]

    if pipeline_run.status == ExecutionStatus.FAILED:
        print("Pipeline failed. Check the logs for more details.")
        exit(1)
    elif pipeline_run.status == ExecutionStatus.RUNNING:
        print(
            "Pipeline is still running. The post-execution phase cannot "
            "proceed. Please make sure you use an orchestrator with a "
            "synchronous mode of execution."
        )
        exit(1)

    data_generation_step = pipeline_run.get_step(step="data_generator")
    data_integrity_step = pipeline_run.get_step(step="data_integrity_checker")
    data_drift_step = pipeline_run.get_step(step="data_drift_detector")

    # If no tracker is used, open the reports in the browser
    DeepchecksVisualizer().visualize(data_integrity_step)
    DeepchecksVisualizer().visualize(data_drift_step)

    copy(f"{data_generation_step.output.uri}/df.csv", filename, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help="Dataset version to use as source. Defaults to None, which "
        "means the original dataset shipped with sklearn will be used.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-dc",
        "--disable-caching",
        default=False,
        help="Disables caching for the pipeline. Defaults to False",
        type=bool,
        required=False,
    )
    parser.add_argument(
        "-s",
        "--size",
        default=-1,
        help="Size of the dataset to generate. Defaults to -1, which "
        "means the size of the dataset will be the same as the input "
        "dataset.",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--filename",
        default="./df.csv",
        help="Filename to save the generated data to. Defaults to "
        "`./df.csv`.",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    assert isinstance(args.disable_caching, bool)
    assert isinstance(args.size, int)
    main(
        disable_caching=args.disable_caching,
        dataset_version=args.dataset,
        size=args.size,
        filename=args.filename,
    )
