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

from zenml.pipelines import pipeline


@pipeline
def staging_pipeline(
    importer,
    data_splitter,
    data_integrity_checker,
    train_test_data_drift_detector,
    model_trainer,
    model_evaluator,
    train_test_model_drift_detector,
    result_checker,
):
    """Load, check and split data, then train and evaluate a model."""
    data = importer()
    data_integrity_report = data_integrity_checker(dataset=data)
    train_dataset, test_dataset = data_splitter(data)
    train_test_data_drift_report = train_test_data_drift_detector(
        reference_dataset=train_dataset, target_dataset=test_dataset
    )
    model, _ = model_trainer(train_dataset=train_dataset)
    model_evaluator(test_dataset=test_dataset, model=model)
    train_test_model_drift_report = train_test_model_drift_detector(
        model=model,
        reference_dataset=train_dataset,
        target_dataset=test_dataset,
    )
    result_checker(
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        train_test_model_drift_report=train_test_model_drift_report,
    )
