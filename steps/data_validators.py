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

"""Data validation steps used to check the input data quality and to ensure that
the training and validation data have the same distribution."""

from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    EvidentlyProfileParameters,
    evidently_profile_step,
)
from steps.evidently import CustomEvidentlyProfileStep

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME

# Evidently data quality profiler step
data_quality_profiler = CustomEvidentlyProfileStep(
    name="data_quality_profiler",
    params=EvidentlyProfileParameters(
        column_mapping=EvidentlyColumnMapping(
            target=DATASET_TARGET_COLUMN_NAME,
            prediction=DATASET_TARGET_COLUMN_NAME,
        ),
        profile_sections=[
            "dataquality",
        ],
        verbose_level=1,
    ),
)

# Evidently train-test data similarity check step
data_drift_detector = evidently_profile_step(
    step_name="data_drift_detector",
    params=EvidentlyProfileParameters(
        column_mapping=EvidentlyColumnMapping(
            target=DATASET_TARGET_COLUMN_NAME,
            prediction=DATASET_TARGET_COLUMN_NAME,
        ),
        profile_sections=[
            "categoricaltargetdrift",
            "datadrift",
        ],
        verbose_level=1,
    ),
)
