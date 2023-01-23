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

"""Data loader steps for the Iris classification pipeline."""

from enum import Enum
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml.steps import BaseParameters, Output, step
import requests


DATASET_TARGET_COLUMN_NAME = "target"


def download_dataframe(
    bucket: str = "zenmlpublicdata",
    env: str = "staging",
    df_name: str = "X_train",
    df_type: str = "dataframe",
) -> pd.DataFrame:
    filename = "df.csv"
    url = f"https://{bucket}.s3.eu-central-1.amazonaws.com/{env}/{df_name}/{filename}"
    r = requests.get(url, allow_redirects=True)

    with open(filename, "wb") as f:
        f.write(r.content)

    with open(filename, "rb") as f:
        df = pd.read_csv(f)

    if df_type == "series":
        # Taking the first column if its a series as the assumption
        # is that there will only be one
        df = df[df.columns[0]]

    return df


class DataVersion(str, Enum):
    """Enum for data versions.
    
    This encodes the possible data versions that can be loaded by the
    various pipelines:

    - EXPERIMENTATION: Data version used for local experimentation.
    - STAGING: Data version used for staging changes before they are
        deployed to production.
    - PRODUCTION: Data version used in production.
    """

    EXPERIMENTATION = "experimentation"
    STAGING = "staging"
    PRODUCTION = "production"


class DataLoaderStepParameters(BaseParameters):
    """Parameters for the data_loader step.

    Attributes:
        version: data version to load. Use an actual data version number or
            one of the DataVersion enum values.
    """

    version: DataVersion = DataVersion.EXPERIMENTATION


@step
def data_loader(
    params: DataLoaderStepParameters,
) -> pd.DataFrame:
    """Load the dataset with the indicated version.
    
    Args:
        params: Parameters for the data_loader step (data version to load).

    Returns:
        The dataset with the indicated version.
    """
    if params.version == DataVersion.EXPERIMENTATION:
        # We use the original data shipped with scikit-learn for experimentation
        iris = load_iris(as_frame=True).frame
        return iris

    else:
        # We use data stored in the public S3 bucket for staging and
        # production
        if params.version == DataVersion.STAGING:
            env = "staging"
        else:
            env = "production"

        X_train = download_dataframe(env=env, df_name="X_train")
        X_test = download_dataframe(env=env, df_name="X_test")
        y_train = download_dataframe(
            env=env, df_name="y_train", df_type="series"
        )
        y_test = download_dataframe(env=env, df_name="y_test", df_type="series")
        return X_train + X_test + y_train + y_test


class DataSplitterStepParameters(BaseParameters):
    """Parameters for the data_splitter step.

    Attributes:
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether or not to shuffle the data before splitting.
        random_state: Controls the shuffling applied to the data before
            applying the split. Pass an int for reproducible and cached output
            across multiple step runs.
    """

    test_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42


@step
def data_splitter(
    params: DataSplitterStepParameters,
    dataset: pd.DataFrame, 
) -> Output(train=pd.DataFrame, test=pd.DataFrame,):
    """Split the dataset into train and test (validation) subsets.

    Args:
        params: Parameters for the data_splitter step (split proportions,
            shuffling, random state).
        dataset: The dataset to split.
    
    Returns:
        The train and test (validation) subsets of the dataset.
    """
    train, test = train_test_split(
        dataset,
        test_size=params.test_size,
        shuffle=params.shuffle,
        random_state=params.random_state,
    )
    return train, test
