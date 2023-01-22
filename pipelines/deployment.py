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
def gitflow_deployment_pipeline(
    importer,
    trained_model_loader,
    served_model_loader,
    train_serve_model_evaluator,
    model_deployer,
):
    """Load a newly trained and the currently served model, compare them, then deploy the new model if better."""
    data = importer()

    trained_model = trained_model_loader()
    served_model = served_model_loader() 
    deploy_decision = train_serve_model_evaluator(
        model=trained_model,
        reference_model=served_model,
        dataset=data,
    )
    model_deployer(deploy_decision = deploy_decision, model=trained_model)
