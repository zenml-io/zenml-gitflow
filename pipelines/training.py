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

from zenml.pipelines import pipeline
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
)
from steps.model_appraisers import (
    ModelAppraisalStepParams,
    model_train_appraiser,
)


@pipeline
def gitflow_training_pipeline(
    trainer_params: DecisionTreeTrainerParams = DecisionTreeTrainerParams(),
    loader_params: DataLoaderStepParameters = DataLoaderStepParameters(),
    splitter_params: DataSplitterStepParameters = DataSplitterStepParameters(),
    model_scorer_params: ModelScorerStepParams = ModelScorerStepParams(),
    model_appraiser_params:ModelAppraisalStepParams=ModelAppraisalStepParams(),
):
    """Pipeline that trains and evaluates a new model."""
    data = data_loader(params=loader_params)
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
        params=model_appraiser_params,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        data_integrity_report=data_integrity_report,
        train_test_data_drift_report=train_test_data_drift_report,
        model_evaluation_report=model_evaluation_report,
        train_test_model_evaluation_report=train_test_model_evaluation_report,
    )
