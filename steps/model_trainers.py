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

# import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from zenml.client import Client
from zenml.steps import BaseParameters, Output, step
from sklearn.tree import DecisionTreeClassifier

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME
from utils.tracker_helper import enable_autolog, get_tracker_name


class TrainerParams(BaseParameters):
    random_state: int = 42
    C: int = 1.320498
    kernel: str = "rbf"
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False

@step(
    experiment_tracker=get_tracker_name(),
)
def svc_trainer(
    params: TrainerParams,
    train_dataset: pd.DataFrame,
) -> Output(model=ClassifierMixin, accuracy=float):
    """Train a sklearn SVC classifier."""
    enable_autolog()

    X = train_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = train_dataset[DATASET_TARGET_COLUMN_NAME]
    # model = SVC(
    #     C=params.C,
    #     kernel=params.kernel,
    #     degree=params.degree,
    #     coef0=params.coef0,
    #     shrinking=params.shrinking,
    #     probability=params.probability,
    # )
    model = DecisionTreeClassifier(
        max_depth=5, random_state=params.random_state
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc}")
    return model, train_acc
