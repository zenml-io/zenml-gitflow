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
from evidently.model_profile import Profile
from evidently.dashboard import Dashboard
from evidently.model_profile.sections import DataQualityProfileSection
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.tabs import DataQualityTab
from zenml.steps import step, Output


@step
def evidently_data_validator(
    dataset: pd.DataFrame,
) -> Output(profile=Profile, dashboard=str):
    """Custom data quality profiler step with Evidently

    Args:
        dataset: a Pandas DataFrame

    Returns:
        Evidently Profile generated for the dataset
    """

    # validation pre-processing (e.g. dataset preparation) can take place here
    profile = Profile(sections=[DataQualityProfileSection()])
    profile.calculate(
        reference_data=dataset,
    )
    dashboard = Dashboard(tabs=[DataQualityTab()])
    dashboard.calculate(dataset)
    
    # validation post-processing (e.g. interpret results, take actions) can happen here

    return [profile, dashboard.html()]
