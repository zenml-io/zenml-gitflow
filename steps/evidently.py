from typing import cast
import pandas as pd
from evidently.model_profile import Profile  # type: ignore[import]

from zenml.integrations.evidently.steps import (
    EvidentlyProfileParameters,
)
from zenml.integrations.evidently.data_validators import EvidentlyDataValidator

from zenml.steps import BaseStep, Output


class CustomEvidentlyProfileStep(BaseStep):
    """Custom Evidently Profile Step that only takes in one dataset."""

    def entrypoint(
        self,
        dataset: pd.DataFrame,
        params: EvidentlyProfileParameters,
    ) -> Output(  # type:ignore[valid-type]
        profile=Profile, dashboard=str
    ):
        """Main entrypoint for the Evidently single-dataset profiler step.

        Args:
            dataset: a Pandas DataFrame
            params: the parameters for the step

        Raises:
            ValueError: If ignored_cols is an empty list
            ValueError: If column is not found in reference or comparison
                dataset

        Returns:
            profile: Evidently Profile generated for the data drift
            dashboard: HTML report extracted from an Evidently Dashboard
              generated for the data drift
        """
        data_validator = cast(
            EvidentlyDataValidator,
            EvidentlyDataValidator.get_active_data_validator(),
        )
        column_mapping = None

        if params.ignored_cols is None:
            pass

        elif not params.ignored_cols:
            raise ValueError(
                f"Expects None or list of columns in strings, but got {params.ignored_cols}"
            )

        elif not (set(params.ignored_cols).issubset(set(dataset.columns))):
            raise ValueError("Column is not found in input dataset")

        else:
            dataset = dataset.drop(labels=list(params.ignored_cols), axis=1)

        if params.column_mapping:
            column_mapping = params.column_mapping.to_evidently_column_mapping()
        profile, dashboard = data_validator.data_profiling(
            dataset=dataset,
            profile_list=params.profile_sections,
            column_mapping=column_mapping,
            verbose_level=params.verbose_level,
            profile_options=params.profile_options,
            dashboard_options=params.dashboard_options,
        )
        return [profile, dashboard.html()]
