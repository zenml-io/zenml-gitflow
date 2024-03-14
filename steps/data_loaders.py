#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
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

from typing import Optional, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step

DATASET_TARGET_COLUMN_NAME = "target"


def download_dataframe(
    version: str,
    bucket: str = "zenmlpublicdata",
) -> pd.DataFrame:
    url = f"https://{bucket}.s3.eu-central-1.amazonaws.com/gitflow_data/{version}.csv"
    df = pd.read_csv(url)
    return df


@step
def data_loader(version: Optional[str] = None) -> pd.DataFrame:
    """Load the dataset with the indicated version.

    Args:
        version: The version of the dataset to load.

    Returns:
        The dataset with the indicated version.
    """
    if version is None:
        # We use the original data shipped with scikit-learn for experimentation
        dataset = load_breast_cancer(as_frame=True).frame
        return dataset

    else:
        # We use data stored in the public S3 bucket for specified versions
        dataset = download_dataframe(version=version)
        return dataset


@step
def data_splitter(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"]]:
    """Split the dataset into train and test (validation) subsets.

    Args:
        dataset: The dataset to split.
        test_size: The size of the test subset.
        shuffle: Whether to shuffle the dataset.
        random_state: The random state to use for shuffling.

    Returns:
        The train and test (validation) subsets of the dataset.
    """
    train, test = train_test_split(
        dataset,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    return train, test
