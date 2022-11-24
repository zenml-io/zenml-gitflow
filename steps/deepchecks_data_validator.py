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
import pandas as pd
from deepchecks.core.suite import SuiteResult
from zenml.integrations.deepchecks.data_validators import DeepchecksDataValidator
from zenml.integrations.deepchecks.validation_checks import DeepchecksDataIntegrityCheck
from zenml.steps import step


@step
def deepchecks_data_validator(
    dataset: pd.DataFrame,
) -> SuiteResult:
    """Custom data integrity check step with Deepchecks

    Args:
        dataset: input Pandas DataFrame

    Returns:
        Deepchecks test suite execution result
    """

    # validation pre-processing (e.g. dataset preparation) can take place here

    data_validator = DeepchecksDataValidator.get_active_data_validator()
    suite = data_validator.data_validation(
        dataset=dataset,
        check_list=[
            DeepchecksDataIntegrityCheck.TABULAR_OUTLIER_SAMPLE_DETECTION,
            DeepchecksDataIntegrityCheck.TABULAR_STRING_LENGTH_OUT_OF_BOUNDS,
        ],
    )

    # validation post-processing (e.g. interpret results, take actions) can happen here

    return suite
