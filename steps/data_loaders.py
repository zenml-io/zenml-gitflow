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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step
import requests


def download_dataframe(
    bucket: str = "zenmlpublicdata",
    env: str = "staging",
    df_name: str = "X_train",
    df_type: str = "dataframe",
    
) -> pd.DataFrame:
    url = f'https://{bucket}.s3.eu-central-1.amazonaws.com/{env}/{df_name}/df.parquet.gzip'
    r = requests.get(url, allow_redirects=True)

    with open('df.parquet.gzip', 'wb') as f:
        f.write(r.content)

    with open('df.parquet.gzip', 'rb') as f:
        df = pd.read_parquet(f)

    if df_type == "series":
        # Taking the first column if its a series as the assumption
        # is that there will only be one
        df = df[df.columns[0]]

    return df

@step
def development_data_loader() -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Load the local dataset."""
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
    )
    return X_train, X_test, y_train, y_test


@step(enable_cache=False)
def staging_data_loader() -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Load the static staging dataset."""
    X_train = download_dataframe(env="staging", df_name="X_train")
    X_test = download_dataframe(env="staging", df_name="X_test")
    y_train = download_dataframe(env="staging", df_name="y_train", df_type="series")
    y_test = download_dataframe(env="staging", df_name="y_test", df_type="series")

    return X_train, X_test, y_train, y_test


@step(enable_cache=False)
def production_data_loader() -> Output(
    X_train=pd.DataFrame,
    X_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Load the production production dataset."""
    X_train = download_dataframe(env="production", df_name="X_train")
    X_test = download_dataframe(env="production", df_name="X_test")
    y_train = download_dataframe(env="production", df_name="y_train", df_type="series")
    y_test = download_dataframe(env="production", df_name="y_test", df_type="series")

    return X_train, X_test, y_train, y_test
