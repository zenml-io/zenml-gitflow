#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
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

from functools import partial

from zenml.integrations.deepchecks.steps import (
    deepchecks_data_drift_check_step,
    deepchecks_data_integrity_check_step,
)
from zenml.integrations.deepchecks.validation_checks import (
    DeepchecksDataDriftCheck,
)

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME

# Deepchecks data integrity check step

data_integrity_checker = partial(
    deepchecks_data_integrity_check_step,
    id="data_integrity_checker",
    dataset_kwargs=dict(
        label=DATASET_TARGET_COLUMN_NAME,
        cat_features=[],
    ),
)


# Deepchecks train-test data similarity check step
data_drift_detector = partial(
    deepchecks_data_drift_check_step,
    id="data_drift_detector",
    dataset_kwargs=dict(label=DATASET_TARGET_COLUMN_NAME, cat_features=[]),
    check_kwargs={
        DeepchecksDataDriftCheck.TABULAR_FEATURE_LABEL_CORRELATION_CHANGE: dict(
            condition_feature_pps_in_train_less_than=dict(
                threshold=1.0,  # essentially turns off the label correlation check
            ),
        )
    },
)
