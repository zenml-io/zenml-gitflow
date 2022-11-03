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

import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from zenml.client import Client
from zenml.steps import step, BaseParameters


class TrainerParams(BaseParameters):
    C: int = 1.0
    kernel: str ="rbf"
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False

@step
def svc_trainer(
    params: TrainerParams,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier."""
    model = SVC(
        C=params.C,
        kernel=params.kernel,
        degree=params.degree,
        coef0=params.coef0,
        shrinking=params.shrinking,
        probability=params.probability,
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())
    print(f"Train accuracy: {train_acc}")
    return model


@step
def svc_trainer_mlflow(
    params: TrainerParams,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ClassifierMixin:
    """Train a sklearn SVC classifier and log to MLflow."""
    mlflow.sklearn.autolog()  # log all model hparams and metrics to MLflow
    model = SVC(
        C=params.C,
        kernel=params.kernel,
        degree=params.degree,
        coef0=params.coef0,
        shrinking=params.shrinking,
        probability=params.probability,
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())
    print(f"Train accuracy: {train_acc}")
    return model
