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

from sklearn.base import ClassifierMixin
from typing import Optional, Tuple
from zenml.client import Client
from zenml.post_execution import StepView


def get_deployed_model(
    model_name: str,
) -> Tuple[Optional[ClassifierMixin], str, str, str]:
    """Load and return the model with the given name being currently served."""

    model_deployer = Client().active_stack.model_deployer
    if model_deployer is None:
        raise ValueError("No model deployer found in the active stack.")
    models = model_deployer.find_model_server(model_name=model_name)

    if len(models) == 0:
        print(f"No model with name {model_name} is currently deployed.")
        return None, "", "", ""

    pipeline_name = models[0].config.pipeline_name
    pipeline_run_id = models[0].config.pipeline_run_id
    step_name = models[0].config.pipeline_step_name

    pipeline_run = Client().get_pipeline_run(name_id_or_prefix=pipeline_run_id)
    steps = Client().list_run_steps(pipeline_run_id=pipeline_run.id)
    step = next((step for step in steps if step.name == step_name), None)
    if step is None:
        print(
            f"Could not find the pipeline step run with name {step_name} in "
            f"pipeline run {pipeline_run_id} of pipeline {pipeline_name} that "
            f"was used to deploy the model {model_name}."
        )
        return None, "", "", ""

    step_view = StepView(step)
    step_model_input = step_view.inputs.values()[0]
    model = step_model_input.read(output_data_type=ClassifierMixin)
    return model, pipeline_name, pipeline_run_id, step_name
