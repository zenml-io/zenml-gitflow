#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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

"""Model scoring and evaluation steps used to check the model performance"""

from typing import List, Tuple
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    EvidentlyProfileParameters,
    evidently_profile_step,
)

from zenml.steps import BaseParameters, Output, step

from steps.data_loaders import (
    DATASET_PREDICTION_COLUMN_NAME,
    DATASET_TARGET_COLUMN_NAME,
)
from utils.tracker_helper import get_tracker_name, log_metric
from steps.evidently import CustomEvidentlyProfileStep


class ModelScorerStepParams(BaseParameters):
    """Parameters for the model scorer step.

    Attributes:
        accuracy_metric_name: The name of the metric used to log the accuracy
            in the experiment tracker.
    """

    accuracy_metric_name: str = "accuracy"


def score_model(
    dataset: pd.DataFrame,
    model: ClassifierMixin,
) -> Tuple[pd.DataFrame, float]:
    """Calculate the model accuracy on a given dataset.

    Args:
        dataset: The dataset to score the model on.
        model: The model to score.

    Returns:
        The predictions dataset and the accuracy of the model on the dataset.
    """
    X = dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = dataset[DATASET_TARGET_COLUMN_NAME]

    p = model.predict(X)
    acc = accuracy_score(y, p)

    predictions = dataset.copy()
    predictions[DATASET_PREDICTION_COLUMN_NAME] = p

    return predictions, acc


@step(
    experiment_tracker=get_tracker_name(),
)
def model_scorer(
    params: ModelScorerStepParams,
    dataset: pd.DataFrame,
    model: ClassifierMixin,
) -> Output(predictions=pd.DataFrame, accuracy=float,):
    """Calculate and log the model accuracy on a given dataset.

    If the experiment tracker is enabled, the scoring accuracy
    will be logged to the experiment tracker.

    Args:
        params: The parameters for the model scorer step.
        dataset: The dataset to score the model on.
        model: The model to score.

    Returns:
        The accuracy of the model on the dataset and a copy of the input
        dataframe with the predictions added as a new column.
    """
    predictions, acc = score_model(dataset, model)
    log_metric(params.accuracy_metric_name, acc)
    print(f"{params.accuracy_metric_name}: {acc}")
    return predictions, acc


@step(
    experiment_tracker=get_tracker_name(),
)
def optional_model_scorer(
    params: ModelScorerStepParams,
    dataset: pd.DataFrame,
    model: List[ClassifierMixin],
) -> Output(predictions=pd.DataFrame, accuracy=float,):
    """Calculate and log the model accuracy on a given dataset with an optional
    model.

    This is a variation of the model_scorer step that can handle an optional
    model. This is useful for cases where the model loaded by a previous step
    is not available, e.g. when the model is not found in the model registry.

    If the experiment tracker is enabled, the scoring accuracy will be logged to
    the experiment tracker.

    Args:
        params: The parameters for the model scorer step.
        dataset: The dataset to score the model on.
        model: Optional model to score.

    Returns:
        The accuracy of the model on the dataset and a copy of the input
        dataframe with the predictions added as a new column. If no model is
        provided, a 0.0 accuracy and an the original dataset with no prediction
        column are returned instead.
    """
    if not len(model):
        return dataset, 0.0
    predictions, acc = score_model(dataset, model[0])
    log_metric(params.accuracy_metric_name, acc)
    print(f"{params.accuracy_metric_name}: {acc}")
    return predictions, acc


# Evidently train-test model evaluation step.
train_test_model_evaluator = evidently_profile_step(
    step_name="train_test_model_evaluator",
    params=EvidentlyProfileParameters(
        column_mapping=EvidentlyColumnMapping(
            target=DATASET_TARGET_COLUMN_NAME,
            prediction=DATASET_PREDICTION_COLUMN_NAME,
        ),
        profile_sections=[
            "classificationmodelperformance",
        ],
        verbose_level=1,
    ),
)

# Evidently single dataset model evaluation step.
model_evaluator = CustomEvidentlyProfileStep(
    name="model_evaluator",
    params=EvidentlyProfileParameters(
        column_mapping=EvidentlyColumnMapping(
            target=DATASET_TARGET_COLUMN_NAME,
            prediction=DATASET_PREDICTION_COLUMN_NAME,
        ),
        profile_sections=[
            "classificationmodelperformance",
        ],
        verbose_level=1,
    ),
)
