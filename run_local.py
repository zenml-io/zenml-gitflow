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


from pipelines import development_pipeline
from steps import (
    TrainerParams,
    development_data_loader,
    evaluator,
    svc_trainer_mlflow,
)
from zenml.client import Client


def main():

    experiment_tracker = Client().active_stack.experiment_tracker

    if experiment_tracker is None:
        raise AssertionError("Experiment Tracker needs to exist in the  stack!")
    
    # initialize and run the training pipeline
    training_pipeline_instance = development_pipeline(
        importer=development_data_loader(),
        trainer=svc_trainer_mlflow(
            params=TrainerParams(
                degree=7,
            )
        ).configure(experiment_tracker=experiment_tracker.name),
        evaluator=evaluator(),
    )
    training_pipeline_instance.run()


if __name__ == "__main__":
    main()
