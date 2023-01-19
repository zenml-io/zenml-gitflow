#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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

from typing import cast
from deepchecks import SuiteResult
import pdfkit

from zenml.post_execution import StepView


def deepcheck_suite_to_pdf(
    step: StepView,
    output_path: str,
) -> None:
    """Generates a PDF file with the results of a DeepChecks suite saved as a step artifact."""

    for artifact_view in step.outputs.values():
        # filter out anything but data analysis artifacts
        if artifact_view.data_type.endswith(".SuiteResult"):
            suite = cast(SuiteResult, artifact_view.read())

            suite_html = suite.html_serializer.serialize(
                full_html=True,
                include_requirejs=True,
                include_plotlyjs=True,
            )
            pdfkit.from_string(suite_html, output_path)
            return
