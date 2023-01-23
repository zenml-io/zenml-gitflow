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

from zenml.pipelines import pipeline


@pipeline
def gitflow_data_generation_pipeline(
    importer,
    data_generator,
    data_integrity_checker,
    data_drift_detector,
    # exporter,
):
    """Pipeline simulates new data ETL by synthetically generating new data."""
    data = importer()
    new_data = data_generator(dataset=data)
    data_integrity_report = data_integrity_checker(dataset=data)
    data_drift_report = data_drift_detector(
        reference_dataset=data, target_dataset=new_data
    )
    # exporter(dataset=new_data)
