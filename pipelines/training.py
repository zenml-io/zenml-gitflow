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

from typing import Optional

from zenml import pipeline

from steps import (
    data_drift_detector,
    data_integrity_checker,
    data_loader,
    data_splitter,
    decision_tree_trainer,
    metadata_logger,
    model_evaluator,
    model_scorer,
    model_train_appraiser,
    train_test_model_evaluator,
)


@pipeline
def gitflow_training_pipeline(
    dataset_version: Optional[str] = None,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    accuracy_metric_name: str = "accuracy",
    max_depth: int = 5,
    extra_hyperparams: dict = {},
    train_accuracy_threshold: float = 0.7,
    test_accuracy_threshold: float = 0.7,
    warnings_as_errors: bool = False,
    ignore_data_integrity_failures: bool = False,
    ignore_train_test_data_drift_failures: bool = False,
    ignore_model_evaluation_failures: bool = False,
    ignore_reference_model: bool = False,
    max_train_accuracy_diff: float = 0.1,
    max_test_accuracy_diff: float = 0.05,
    github_pr_url: Optional[str] = None,
    org_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Pipeline that trains and evaluates a new model."""
    metadata_logger(github_pr_url=github_pr_url)
    data = data_loader(version=dataset_version)
    data_integrity_report = data_integrity_checker(dataset=data)
    train_dataset, test_dataset = data_splitter(
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        dataset=data,
    )
    train_test_data_drift_report = data_drift_detector(
        reference_dataset=train_dataset, target_dataset=test_dataset
    )
    model, train_accuracy = decision_tree_trainer(
        random_state=random_state,
        max_depth=max_depth,
        train_dataset=train_dataset,
        extra_hyperparams=extra_hyperparams,
    )
    test_accuracy = model_scorer(
        accuracy_metric_name=accuracy_metric_name,
        dataset=test_dataset,
        model=model,
    )
    train_test_model_evaluation_report = train_test_model_evaluator(
        model=model,
        reference_dataset=train_dataset,
        target_dataset=test_dataset,
    )
    model_evaluation_report = model_evaluator(
        model=model,
        dataset=test_dataset,
    )
    model_train_appraiser(
        id="model_appraiser",
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        model_evaluation_report=model_evaluation_report,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
        train_accuracy_threshold=train_accuracy_threshold,
        test_accuracy_threshold=test_accuracy_threshold,
        warnings_as_errors=warnings_as_errors,
        ignore_data_integrity_failures=ignore_data_integrity_failures,
        ignore_train_test_data_drift_failures=ignore_train_test_data_drift_failures,
        ignore_model_evaluation_failures=ignore_model_evaluation_failures,
        ignore_reference_model=ignore_reference_model,
        max_train_accuracy_diff=max_train_accuracy_diff,
        max_test_accuracy_diff=max_test_accuracy_diff,
        org_id=org_id,
        tenant_id=tenant_id,
    )
