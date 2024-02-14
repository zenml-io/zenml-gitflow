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

from zenml import pipeline
from zenml.client import Client
from steps.model_trainers import (
    decision_tree_trainer,
    DecisionTreeTrainerParams,
)
from steps.data_loaders import (
    data_loader,
    DataLoaderStepParameters,
    data_splitter,
    DataSplitterStepParameters,
)
from steps.data_validators import (
    data_drift_detector,
    data_integrity_checker,
)
from steps.model_evaluators import (
    ModelScorerStepParams,
    train_test_model_evaluator,
    model_scorer,
    model_evaluator,
    optional_model_scorer,
)
from steps.model_appraisers import (
    ModelAppraisalStepParams,
    model_train_reference_appraiser,
)
from utils import get_stack_deployer
from steps.model_loaders import (
    ServedModelLoaderStepParameters,
    served_model_loader,
)


@pipeline
def gitflow_end_to_end_pipeline(
    trainer_params: DecisionTreeTrainerParams = DecisionTreeTrainerParams(),
    model_name: str = "model",
    loader_params: DataLoaderStepParameters = DataLoaderStepParameters(),
    splitter_params: DataSplitterStepParameters = DataSplitterStepParameters(),
    model_scorer_params: ModelScorerStepParams = ModelScorerStepParams(),
    model_appraiser_params: ModelAppraisalStepParams = ModelAppraisalStepParams(),
):
    """Train and serve a new model if it performs better than the model
    currently served."""

    data = data_loader(params=loader_params)
    served_model = served_model_loader(
        params=ServedModelLoaderStepParameters(
            model_name=model_name,
            step_name="model_deployer",
        )
    )
    data_integrity_report = data_integrity_checker(dataset=data)
    train_dataset, test_dataset = data_splitter(
        params=splitter_params, dataset=data
    )
    train_test_data_drift_report = data_drift_detector(
        reference_dataset=train_dataset, target_dataset=test_dataset
    )
    model, train_accuracy = decision_tree_trainer(
        params=trainer_params, train_dataset=train_dataset
    )
    test_accuracy = model_scorer(
        params=model_scorer_params, dataset=test_dataset, model=model
    )

    served_train_accuracy = optional_model_scorer(
        id="served_model_train_scorer",
        params=ModelScorerStepParams(
            accuracy_metric_name="reference_train_accuracy",
        ),
        dataset=train_dataset,
        model=served_model,
    )
    served_test_accuracy = optional_model_scorer(
        id="served_model_test_scorer",
        params=ModelScorerStepParams(
            accuracy_metric_name="reference_test_accuracy",
        ),
        dataset=train_dataset,
        model=served_model,
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
    deploy_decision, report = model_train_reference_appraiser(
        params=model_appraiser_params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        reference_train_accuracy=served_train_accuracy,
        reference_test_accuracy=served_test_accuracy,
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        model_evaluation_report=model_evaluation_report,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
    )
    get_stack_deployer(model_name=model_name)(
        deploy_decision=deploy_decision, model=model
    )
