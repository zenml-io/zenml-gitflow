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

"""Steps that synthetically generate data to be used to simulate new data
ingestion."""

import pandas as pd
from zenml.steps import BaseParameters, step


class DataGeneratorStepParameters(BaseParameters):
    """Parameters for the data generator step.

    Attributes:
        size: Size of the dataset to generate. If -1, a dataset with the same
            size as the original dataset is generated.

    """

    size: int = -1


@step(enable_cache=False)
def data_generator(
    dataset: pd.DataFrame,
    params: DataGeneratorStepParameters,
) -> pd.DataFrame:
    """Synthetically generate a dataset with the same characteristics and
    distribution as the input dataset.

    Args:
        params: Parameters for the data_loader step (data version to load).

    Returns:
        The dataset with the indicated version.
    """
    from sdv.tabular import GaussianCopula

    model = GaussianCopula()
    model.fit(dataset)
    sample = model.sample(params.size if params.size != -1 else len(dataset))
    return sample
