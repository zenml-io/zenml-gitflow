import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection
from evidently.pipeline.column_mapping import ColumnMapping
from zenml.steps import step


@step
def data_quality_profiler(
    dataset: pd.DataFrame,
) -> Profile:
    """Custom data quality profiler step with Evidently

    Args:
        dataset: a Pandas DataFrame

    Returns:
        Evidently Profile generated for the dataset
    """

    # validation pre-processing (e.g. dataset preparation) can take place here

    profile = Profile(sections=[DataQualityProfileSection])
    profile.calculate(
        reference_data=dataset,
    )

    # validation post-processing (e.g. interpret results, take actions) can happen here

    return profile
