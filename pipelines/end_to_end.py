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

from zenml.pipelines import pipeline


@pipeline
def devweek_end_to_end_pipeline(
    importer,
    data_splitter,
    data_quality_profiler,
    train_test_data_drift_detector,
    model_trainer,
    train_model_scorer,
    test_model_scorer,
    model_evaluator,
    train_test_model_evaluator,
    served_model_loader,
    served_model_test_scorer,
    train_serve_model_comparison,
    model_appraiser,
    model_deployer,
):
    """Train and serve a new model if it performs better than the model
    currently served."""
    data = importer()
    served_model = served_model_loader()
    data_quality_report, data_quality_html = data_quality_profiler(
        dataset=data,
    )
    train_dataset, test_dataset = data_splitter(data)

    (
        train_test_data_drift_report,
        train_test_data_drift_html,
    ) = train_test_data_drift_detector(
        reference_dataset=train_dataset, comparison_dataset=test_dataset
    )

    model = model_trainer(train_dataset=train_dataset)
    train_dataset_with_predictions, train_accuracy = train_model_scorer(
        dataset=train_dataset, model=model
    )
    test_dataset_with_predictions, test_accuracy = test_model_scorer(
        dataset=test_dataset, model=model
    )

    (
        train_test_model_evaluation_report,
        train_test_model_evaluation_html,
    ) = train_test_model_evaluator(
        reference_dataset=train_dataset_with_predictions,
        comparison_dataset=test_dataset_with_predictions,
    )

    model_evaluation_report, model_evaluation_html = model_evaluator(
        dataset=test_dataset_with_predictions,
    )

    (
        served_test_dataset_with_predictions,
        served_test_accuracy,
    ) = served_model_test_scorer(dataset=test_dataset, model=served_model)

    (
        train_serve_model_comparison_report,
        train_serve_model_comparison_html,
    ) = train_serve_model_comparison(
        reference_dataset=served_test_dataset_with_predictions,
        comparison_dataset=test_dataset_with_predictions,
    )

    deploy_decision, report = model_appraiser(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_quality_report=data_quality_report,
        data_quality_html=data_quality_html,
        train_test_data_drift_report=train_test_data_drift_report,
        train_test_data_drift_html=train_test_data_drift_html,
        model_evaluation_report=model_evaluation_report,
        model_evaluation_html=model_evaluation_html,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
        train_test_model_evaluation_html=train_test_model_evaluation_html,
        reference_test_accuracy=served_test_accuracy,
        train_serve_model_comparison_report=train_serve_model_comparison_report,
        train_serve_model_comparison_html=train_serve_model_comparison_html,
    )
    model_deployer(deploy_decision=deploy_decision, model=model)
