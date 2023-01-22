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

from typing import List, Optional, Tuple
from sklearn.base import ClassifierMixin
from zenml.environment import Environment
from zenml.steps import BaseParameters, Output, step
from zenml.services import BaseService
from utils.deployer_helper import load_deployed_model, load_trained_model
from utils.tracker_helper import get_tracker_name, log_model


class TrainedModelLoaderStepParameters(BaseParameters):

    train_pipeline_name: Optional[str] = None
    train_pipeline_step_name: str
    train_pipeline_step_output_name: Optional[str] = None


@step(
    enable_cache=False,
    experiment_tracker=get_tracker_name(),
)
def trained_model_loader(
    params: TrainedModelLoaderStepParameters,
) -> ClassifierMixin:
    pipeline_name = (
        params.train_pipeline_name
        or Environment().step_environment.pipeline_name
    )
    model = load_trained_model(
        pipeline_name=pipeline_name,
        step_name=params.train_pipeline_step_name,
        output_name=params.train_pipeline_step_output_name,
    )
    if model:
        # Log the model to the experiment tracker. In the case of the local
        # MLflow tracker, this is needed to serve the model.
        log_model(model, "model")
        return [model]

    raise ValueError("No model found")


class ServedModelLoaderStepParameters(BaseParameters):
    model_name: str
    step_name: str


@step(enable_cache=False)
def served_model_loader(
    params: ServedModelLoaderStepParameters,
) -> List[ClassifierMixin]:
    model_server, model = load_deployed_model(
        model_name=params.model_name,
        step_name=params.step_name,
    )
    if model:
        return [model]

    return []
