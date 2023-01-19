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

from typing import List
from zenml.steps import Output, step
from utils.tracker_helper import get_tracker_name, log_text

from deepchecks import SuiteResult


@step(
    experiment_tracker=get_tracker_name(),
)
def model_train_results_checker(
    data_integrity_report: SuiteResult,
    train_test_data_drift_report: SuiteResult,
    train_test_model_drift_report: SuiteResult,
) -> Output(
    data_integrity_result=bool, data_drift_result=bool, model_drift_result=bool
):
    """Check and log Deepchecks suite results to the experiment tracker."""

    # Log Deepchecks suite results to the experiment tracker.
    results: List[bool] = []
    for suite_result, name in [
        (data_integrity_report, "data_integrity_report"),
        (train_test_data_drift_report, "train_test_data_drift_report"),
        (train_test_model_drift_report, "train_test_model_drift_report"),
    ]:
        suite_html = suite_result.html_serializer.serialize(
            full_html=True,
        )
        log_text(suite_html, f"{name}.html")
        print(f"Deepchecks {name} results: {suite_result.passed()}")
        results.append(suite_result.passed())

    # Return Deepchecks suite results.
    return results
