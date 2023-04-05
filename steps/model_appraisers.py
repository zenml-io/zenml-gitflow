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

"""Model appraisal steps used to analyze the model training and evaluation
results, to generate human-readable reports, and to make a decision about
serving the model."""

import json
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from zenml.steps import BaseParameters, Output, step
from utils.tracker_helper import (
    get_current_tracker_run_url,
    get_tracker_name,
    log_text,
)

class ModelAppraisalStepParams(BaseParameters):
    """Parameters for the model training appraisal step.

    Attributes:
        train_accuracy_threshold: The minimum accuracy of the model on the
            training data.
        test_accuracy_threshold: The minimum accuracy of the model on the test
            data.
        warnings_as_errors: Whether to treat Evidently warnings as errors.
        ignore_data_integrity_failures: Whether to ignore data integrity
            failures reported by Evidently on the input data.
        ignore_train_test_data_drift_failures: Whether to ignore train-test data
            drift check failures reported by Evidently on the train/test
            datasets.
        ignore_model_evaluation_failures: Whether to ignore model evaluation
            failures reported by Evidently on the model.
        ignore_reference_model: Whether to ignore the reference model in the
            model appraisal.
        max_train_accuracy_diff: The maximum difference between the accuracy of
            the trained model and the reference model on the training data.
        max_test_accuracy_diff: The maximum difference between the accuracy of
            the trained model and the reference model on the test data.
    """

    train_accuracy_threshold: float = 0.7
    test_accuracy_threshold: float = 0.7
    warnings_as_errors: bool = False
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
    data_quality_report: Dict[str, Any],
    data_quality_html: str,
    train_test_data_drift_report: Dict[str, Any],
    train_test_data_drift_html: str,
    model_evaluation_report: Dict[str, Any],
    model_evaluation_html: str,
    train_test_model_evaluation_report: Dict[str, Any],
    train_test_model_evaluation_html: str,
    reference_test_accuracy: Optional[float] = None,
    train_serve_model_comparison_report: Optional[Dict[str, Any]] = None,
    train_serve_model_comparison_html: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Analyze the model training and evaluation results, generate a report and
    make a decision about serving the model.

    The gathered results are analyzed and a decision is made about whether the
    model should be served or not. The decision is based on the accuracy of the
    model on the training and test data, the data quality report, the train-test
    data drift report, the model evaluation report, and the train-test model
    evaluation report. If accuracy values are provided for a reference model,
    the performance difference between the reference model and the trained model
    is also taken into account.

    The function also generates a human-readable report that is returned
    alongside the decision.

    Args:
        params: Training appraisal configuration parameters.
        train_accuracy: The accuracy of the trained model on the training data.
        test_accuracy: The accuracy of the trained model on the test data.
        data_quality_report: The data quality report.
        data_quality_html: The data quality html report.
        train_test_data_drift_report: The train-test data drift report.
        train_test_data_drift_html: The train-test data drift html report.
        model_evaluation_report: The model evaluation report.
        model_evaluation_html: The model evaluation html report.
        train_test_model_evaluation_report: The train-test model evaluation
            report.
        train_test_model_evaluation_html: The train-test model evaluation
            html report.
        reference_test_accuracy: Accuracy of the reference model measured on
            the test data (omitted if there is no reference model).
        train_serve_model_comparison_report: The train-serve model comparison
            report (omitted if there is no reference model).
        train_serve_model_comparison_html: The train-serve model comparison
            html report (omitted if there is no reference model).

    Returns:
        A tuple of the appraisal decision and a report message.
    """
    results: List[bool] = []
    passed = True
    for report, html_report, name, ignored in [
        (
            data_quality_report,
            data_quality_html,
            "data_quality_report",
            params.ignore_data_integrity_failures,
        ),
        (
            train_test_data_drift_report,
            train_test_data_drift_html,
            "train_test_data_drift_report",
            params.ignore_train_test_data_drift_failures,
        ),
        (
            model_evaluation_report,
            model_evaluation_html,
            "model_evaluation_report",
            params.ignore_model_evaluation_failures,
        ),
        (
            train_test_model_evaluation_report,
            train_test_model_evaluation_html,
            "train_test_model_evaluation_report",
            params.ignore_model_evaluation_failures,
        ),
        (
            train_serve_model_comparison_report,
            train_serve_model_comparison_html,
            "train_serve_model_comparison_report",
            params.ignore_reference_model,
        ),
    ]:
        # Log Evidently suite results to the experiment tracker.

        # save the HTML report as an HTML file in the experiment
        # tracker
        if report is None:
            results.append(True)
            continue

        log_text(html_report, f"{name}.html")
        check_passed = True
        # TODO: Check results here
        results.append(check_passed)
        if not check_passed and not ignored:
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

### Data quality checks {'(ignored)' if params.ignore_data_integrity_failures else ''}

Description: A set of data quality checks that verify whether the input data is
valid and can be used for training (no duplicate samples, no missing values or
problems with string or categorical features, no significant outliers, no
inconsistent labels, etc.).

Result: {'**PASSED**' if results[0] else '**FAILED**'}

### Train-test data drift checks {'(ignored)' if params.ignore_train_test_data_drift_failures else ''}

Description: Compares the training and evaluation datasets to verify that their
distributions are similar and there is no potential data leakage that may
contaminate your model or perceived results.

Result: {'**PASSED**' if results[1] else '**FAILED**'}

### Model evaluation checks {'(ignored)' if params.ignore_model_evaluation_failures else ''}

Description: Runs a set of checks to evaluate the model performance, detect
overfitting, and verify that the model is not biased.

Result: {'**PASSED**' if results[2] else '**FAILED**'}

### Train-test model comparison checks {'(ignored)' if params.ignore_model_evaluation_failures else ''}

Description: Runs a set of checks to compare the model performance between the
test dataset and the evaluation dataset.

Result: {'**PASSED**' if results[3] else '**FAILED**'}
"""

    if reference_test_accuracy is not None:
        if not params.ignore_reference_model:
            if (
                reference_test_accuracy - test_accuracy
            ) > params.max_test_accuracy_diff:
                passed = False

        report += f"""
### Model comparison with reference model on evaluation dataset {'(ignored)' if params.ignore_reference_model else ''}

Description: Compares the performance of the trained model on the evaluation
dataset against a reference model that is already deployed in production.
The difference in accuracy on the evaluation dataset should not exceed
{params.max_test_accuracy_diff}.

Result: {'**PASSED**' if test_accuracy >= params.test_accuracy_threshold else '**FAILED**'}

- trained model accuracy: {test_accuracy}
- reference model accuracy: {reference_test_accuracy}
- (absolute) difference in accuracy: {abs(reference_test_accuracy - test_accuracy)}

### Train-serve model comparison checks {'(ignored)' if params.ignore_reference_model else ''}

Description: Runs a set of checks to compare the performance of the trained
and the model currently deployed in production against the evaluation dataset.

Result: {'**PASSED**' if results[4] else '**FAILED**'}

"""
    experiment_tracker_run_url = get_current_tracker_run_url()

    report += f"""
## Other results

{'- [experiment tracker run](' + experiment_tracker_run_url + ')' if experiment_tracker_run_url else ''}

"""
    log_text(report, f"model_report.md")

    return passed, report


@step(
    experiment_tracker=get_tracker_name(),
)
def model_train_appraiser(
    params: ModelAppraisalStepParams,
    train_accuracy: float,
    test_accuracy: float,
    data_quality_report: str,
    data_quality_html: str,
    train_test_data_drift_report: str,
    train_test_data_drift_html: str,
    model_evaluation_report: str,
    model_evaluation_html: str,
    train_test_model_evaluation_report: str,
    train_test_model_evaluation_html: str,
) -> Output(result=bool, report=str):
    """Analyze the training results, generate a report and make a decision about
    serving the model.

    This step is the last step in the model training pipeline. It analyzes the
    results collected from various steps in the pipeline (e.g model and data
    Evdently reports, model scoring accuracy values) and makes a decision
    regarding the model quality and whether it should be served or not.
    It also generates a report that summarizes the results of the analysis.

    Args:
        params: Model training appraisal step parameters (e.g. thresholds,
            silenced checks, etc.).
        train_accuracy: Accuracy of the trained model on the training dataset.
        test_accuracy: Accuracy of the trained model on the evaluation dataset.
        data_quality_report: The data quality report.
        data_quality_html: The data quality html report.
        train_test_data_drift_report: The train-test data drift report.
        train_test_data_drift_html: The train-test data drift html report.
        model_evaluation_report: The model evaluation report.
        model_evaluation_html: The model evaluation html report.
        train_test_model_evaluation_report: The train-test model evaluation
            report.
        train_test_model_evaluation_html: The train-test model evaluation
            html report.

    Returns:
        A tuple of the model appraisal result and the model appraisal report.
    """
    return model_analysis(
        params=params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_quality_report=json.loads(data_quality_report),
        data_quality_html=data_quality_html,
        train_test_data_drift_report=json.loads(train_test_data_drift_report),
        train_test_data_drift_html=train_test_data_drift_html,
        model_evaluation_report=json.loads(model_evaluation_report),
        model_evaluation_html=model_evaluation_html,
        train_test_model_evaluation_report=json.loads(train_test_model_evaluation_report),
        train_test_model_evaluation_html=train_test_model_evaluation_html,
    )


@step(
    experiment_tracker=get_tracker_name(),
)
def model_train_reference_appraiser(
    params: ModelAppraisalStepParams,
    train_accuracy: float,
    test_accuracy: float,
    reference_test_accuracy: float,
    data_quality_report: str,
    data_quality_html: str,
    train_test_data_drift_report: str,
    train_test_data_drift_html: str,
    model_evaluation_report: str,
    model_evaluation_html: str,
    train_test_model_evaluation_report: str,
    train_test_model_evaluation_html: str,
    train_serve_model_comparison_report: str,
    train_serve_model_comparison_html: str,
) -> Output(result=bool, report=str):
    """Analyze the training results, generate a report and make a decision about
    serving the model.

    This is a variant of the model_train_appraiser step that also compares the
    trained model against a reference model (e.g. that is already deployed in
    production). It analyzes the
    results collected from various steps in the pipeline (e.g model and data
    Evidently reports, model scoring accuracy values) and makes a decision
    regarding the model quality and whether it should be served or not.
    It also generates a report that summarizes the results of the analysis.

    Args:
        params: Model training appraisal step parameters (e.g. thresholds,
            silenced checks, etc.).
        train_accuracy: Accuracy of the trained model on the training dataset.
        test_accuracy: Accuracy of the trained model on the evaluation dataset.
        reference_test_accuracy: Accuracy of the reference model on the
            evaluation dataset.
        data_quality_report: The data quality report.
        data_quality_html: The data quality html report.
        train_test_data_drift_report: The train-test data drift report.
        train_test_data_drift_html: The train-test data drift html report.
        model_evaluation_report: The model evaluation report.
        model_evaluation_html: The model evaluation html report.
        train_test_model_evaluation_report: The train-test model evaluation
            report.
        train_test_model_evaluation_html: The train-test model evaluation
            html report.
        train_serve_model_comparison_report: The train-serve model comparison
            report.
        train_serve_model_comparison_html: The train-serve model comparison
            html report.

    Returns:
        A tuple of the model appraisal result and the model appraisal report.
    """

    return model_analysis(
        params=params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_quality_report=json.loads(data_quality_report),
        data_quality_html=data_quality_html,
        train_test_data_drift_report=json.loads(train_test_data_drift_report),
        train_test_data_drift_html=train_test_data_drift_html,
        model_evaluation_report=json.loads(model_evaluation_report),
        model_evaluation_html=model_evaluation_html,
        train_test_model_evaluation_report=json.loads(train_test_model_evaluation_report),
        train_test_model_evaluation_html=train_test_model_evaluation_html,
        reference_test_accuracy=reference_test_accuracy
        if reference_test_accuracy
        else None,
        train_serve_model_comparison_report=train_serve_model_comparison_report
        if reference_test_accuracy
        else None,
        train_serve_model_comparison_html=train_serve_model_comparison_html
        if reference_test_accuracy
        else None,
    )
