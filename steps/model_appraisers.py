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

import tempfile
from typing import List, Optional, Tuple, Union
from zenml.steps import BaseParameters, Output, step
from utils.tracker_helper import (
    get_current_tracker_run_url,
    get_tracker_name,
    log_text,
)

from deepchecks import SuiteResult


class ModelAppraisalStepParams(BaseParameters):
    """Parameters for the training appraisal step."""

    train_accuracy_threshold: float = 0.9
    test_accuracy_threshold: float = 0.8
    ignore_data_integrity_failures: bool = False
    ignore_train_test_data_drift_failures: bool = False
    ignore_model_evaluation_failures: bool = False
    ignore_reference_model: bool = False
    max_train_accuracy_diff: float = 0.1
    max_test_accuracy_diff: float = 0.05


def model_analysis(
    params: ModelAppraisalStepParams,
    train_accuracy: float,
    test_accuracy: float,
    data_integrity_report: SuiteResult,
    train_test_data_drift_report: SuiteResult,
    model_evaluation_report: SuiteResult,
    train_test_model_evaluation_report: SuiteResult,
    reference_train_accuracy: Optional[float] = None,
    reference_test_accuracy: Optional[float] = None,
) -> Tuple[bool, str]:
    """Analyze the model training and evaluation results and make a decision
    about serving the model.

    The gathered results are analyzed and a decision is made about whether the
    model should be served or not. The decision is based on the accuracy of the
    model on the training and test data, the data integrity report, the train-test
    data drift report, the model evaluation report, and the train-test model
    evaluation report. If accuracy values are provided for a reference model,
    the difference between the reference model and the trained model is also
    taken into account.

    Args:
        params: Training appraisal configuration parameters.
        train_accuracy: The accuracy of the trained model on the training data.
        test_accuracy: The accuracy of the trained model on the test data.
        data_integrity_report: The data integrity report.
        train_test_data_drift_report: The train-test data drift report.
        model_evaluation_report: The model evaluation report.
        train_test_model_evaluation_report: The train-test model evaluation
            report.
        reference_train_accuracy: Accuracy of the reference model measured on
            the training data (omitted if there is no reference model).
        reference_test_accuracy: Accuracy of the reference model measured on
            the test data (omitted if there is no reference model).

    Returns:
        A tuple of the appraisal decision and a report message.
    """

    results: List[Tuple[bool, int, int, int, List[str]]] = []
    passed = True
    for suite_result, name, ignored in [
        (
            data_integrity_report,
            "data_integrity_report",
            params.ignore_data_integrity_failures,
        ),
        (
            train_test_data_drift_report,
            "train_test_data_drift_report",
            params.ignore_train_test_data_drift_failures,
        ),
        (
            model_evaluation_report,
            "model_evaluation_report",
            params.ignore_model_evaluation_failures,
        ),
        (
            train_test_model_evaluation_report,
            "train_test_model_evaluation_report",
            params.ignore_model_evaluation_failures,
        ),
    ]:
        # Log Deepchecks suite results to the experiment tracker.

        # serialize and save the suite result as an HTML file in the experiment
        # tracker
        with tempfile.NamedTemporaryFile(
            mode="w", delete=True, suffix=".html", encoding="utf-8"
        ) as f:
            suite_result.save_as_html(f)
            with open(f.name, "r") as f:
                suite_html = f.read()
        log_text(suite_html, f"{name}.html")
        results.append(
            (
                suite_result.passed(),
                len(suite_result.get_passed_checks()),
                len(suite_result.get_not_passed_checks()),
                len(suite_result.get_not_ran_checks()),
                [
                    check_result.check.name()
                    for check_result in suite_result.get_not_passed_checks()
                ],
            )
        )
        if not suite_result.passed() and not ignored:
            passed = False

    if train_accuracy < params.train_accuracy_threshold:
        passed = False

    if test_accuracy < params.test_accuracy_threshold:
        passed = False

    report = f"""
# Model training results
    
Overall decision: {'**PASSED**' if passed else '**FAILED**'}

## Summary of checks

### Model accuracy on training dataset

Description: Checks how well the model performs on the training dataset.
The model accuracy on the training set should be at least {params.train_accuracy_threshold}.

Result: {'**PASSED**' if train_accuracy >= params.train_accuracy_threshold else '**FAILED**'}

- min. accuracy: {params.train_accuracy_threshold}
- actual accuracy: {train_accuracy}

### Model accuracy on evaluation dataset

Description: Checks how well the model performs on the evaluation dataset.
The model accuracy on the evaluation set should be at least {params.test_accuracy_threshold}.

Result: {'**PASSED**' if test_accuracy >= params.test_accuracy_threshold else '**FAILED**'}

- min. accuracy: {params.test_accuracy_threshold}
- actual accuracy: {test_accuracy}

### Data integrity checks {'(ignored)' if params.ignore_data_integrity_failures else ''}

Description: A set of data quality checks that verify whether the input data is
valid and can be used for training (no duplicate samples, no missing values or
problems with string or categorical features, no significant outliers, no
inconsistent labels, etc.).

Results: {'**PASSED**' if results[0][0] else '**FAILED**'}

- passed checks: {results[0][1]}
- failed checks: {results[0][2]} ({', '.join(results[0][4])})
- skipped checks: {results[0][3]}

### Train-test data drift checks {'(ignored)' if params.ignore_train_test_data_drift_failures else ''}

Description: Compares the training and evaluation datasets to verify that their
distributions are similar and there is no potential data leakage that may
contaminate your model or perceived results.

Results: {'**PASSED**' if results[1][0] else '**FAILED**'}

- passed checks: {results[1][1]}
- failed checks: {results[1][2]} ({', '.join(results[1][4])})
- skipped checks: {results[1][3]}

### Model evaluation checks {'(ignored)' if params.ignore_model_evaluation_failures else ''}

Description: Runs a set of checks to evaluate the model performance, detect
overfitting, and verify that the model is not biased.

Results: {'**PASSED**' if results[2][0] else '**FAILED**'}

- passed checks: {results[2][1]}
- failed checks: {results[2][2]} ({', '.join(results[2][4])})
- skipped checks: {results[2][3]}

### Train-test model comparison checks {'(ignored)' if params.ignore_model_evaluation_failures else ''}

Description: Runs a set of checks to compare the model performance between the
test dataset and the evaluation dataset.

Results: {'**PASSED**' if results[2][0] else '**FAILED**'}

- passed checks: {results[3][1]}
- failed checks: {results[3][2]} ({', '.join(results[3][4])})
- skipped checks: {results[3][3]}
"""

    if (
        reference_train_accuracy is not None
        and reference_test_accuracy is not None
    ):

        if not params.ignore_reference_model:
            if (
                reference_train_accuracy - train_accuracy
            ) > params.max_train_accuracy_diff:
                passed = False

            if (
                reference_test_accuracy - test_accuracy
            ) > params.max_test_accuracy_diff:
                passed = False

        report += f"""
### Model comparison with reference model on training dataset {'(ignored)' if params.ignore_reference_model else ''}

Description: Compares the performance of the trained model on the training
dataset against a reference model that is already deployed in production.
The difference in accuracy on the training dataset should not exceed
{params.max_train_accuracy_diff}.

Result: {'**PASSED**' if train_accuracy >= params.train_accuracy_threshold else '**FAILED**'}

- trained model accuracy: {train_accuracy}
- reference model accuracy: {reference_train_accuracy}
- (absolute) difference in accuracy: {abs(reference_train_accuracy - train_accuracy)}

### Model comparison with reference model on evaluation dataset {'(ignored)' if params.ignore_reference_model else ''}

Description: Compares the performance of the trained model on the evaluation
dataset against a reference model that is already deployed in production.
The difference in accuracy on the evaluation dataset should not exceed
{params.max_test_accuracy_diff}.

Result: {'**PASSED**' if test_accuracy >= params.test_accuracy_threshold else '**FAILED**'}

- trained model accuracy: {test_accuracy}
- reference model accuracy: {reference_test_accuracy}
- (absolute) difference in accuracy: {abs(reference_test_accuracy - test_accuracy)}
"""
    experiment_tracker_run_url = get_current_tracker_run_url()

    report += f"""
## Other results

{'- [experiment tracker run](' + experiment_tracker_run_url + ')' if experiment_tracker_run_url else ''}

"""
    return passed, report


@step(
    experiment_tracker=get_tracker_name(),
)
def model_train_appraiser(
    params: ModelAppraisalStepParams,
    train_accuracy: float,
    test_accuracy: float,
    data_integrity_report: SuiteResult,
    train_test_data_drift_report: SuiteResult,
    model_evaluation_report: SuiteResult,
    train_test_model_evaluation_report: SuiteResult,
) -> Output(result=bool, report=str):
    """Analyze the training results and make a decision about the model."""

    return model_analysis(
        params=params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        model_evaluation_report=model_evaluation_report,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
        reference_train_accuracy=None,
        reference_test_accuracy=None,
    )


@step(
    experiment_tracker=get_tracker_name(),
)
def model_train_reference_appraiser(
    params: ModelAppraisalStepParams,
    train_accuracy: float,
    test_accuracy: float,
    reference_train_accuracy: float,
    reference_test_accuracy: float,
    data_integrity_report: SuiteResult,
    train_test_data_drift_report: SuiteResult,
    model_evaluation_report: SuiteResult,
    train_test_model_evaluation_report: SuiteResult,
) -> Output(result=bool, report=str):
    """Analyze the training results and make a decision about the model."""

    return model_analysis(
        params=params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        model_evaluation_report=model_evaluation_report,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
        reference_train_accuracy=reference_train_accuracy
        if reference_train_accuracy
        else None,
        reference_test_accuracy=reference_test_accuracy
        if reference_test_accuracy
        else None,
    )
