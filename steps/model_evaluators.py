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

from typing import Optional
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.client import Client
from zenml.steps import step, Output

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME
from utils.tracker_helper import get_tracker_name, log_metric


@step(
    experiment_tracker=get_tracker_name(),
)
def model_evaluator(
    test_dataset: pd.DataFrame,
    model: ClassifierMixin,
) -> float:
    """Calculate the accuracy on the test set"""
    X = test_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME]).to_numpy()
    y = test_dataset[DATASET_TARGET_COLUMN_NAME].to_numpy()
    test_acc = model.score(X, y)
    log_metric("test_accuracy", test_acc)
    print(f"Test accuracy: {test_acc}")
    return test_acc
