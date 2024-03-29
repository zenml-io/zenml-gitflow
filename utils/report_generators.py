#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

"""Utility functions for generating model training and evaluation reports."""

from typing import Tuple, cast

from zenml.models import StepRunResponse


def get_result_and_write_report(
    step: StepRunResponse,
    output_path: str,
) -> Tuple[str, bool]:
    """Get the MarkDown report and result generated by the model appraisal step
    and save the report to a file.

    Args:
        step: The step that created and returned the report and result.
        output_path: The path to the file to be generated.

    Returns:
        The report and result generated by the step.
    """
    artifact_view = step.outputs["report"]
    report = cast(str, artifact_view.load())
    with open(output_path, "w") as f:
        f.write(report)

    artifact_view = step.outputs["result"]
    result = cast(bool, artifact_view.load())

    return report, result
