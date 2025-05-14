import pandas as pd
import numpy as np
import datetime
from typing import Annotated
from zenml import step
from zenml import log_metadata, step

@step
def clean_data(data: pd.DataFrame) -> Annotated[pd.DataFrame, "cleaned_data"]:
    """Clean the dataset by handling missing values and outliers."""
    # Store pre-cleaning stats
    pre_cleaning_shape = data.shape
    pre_cleaning_nulls = data.isnull().sum().sum()
    
    # Handle missing values
    cleaned_data = data.copy()
    
    # Fill missing brand_rating with median
    cleaned_data["brand_rating"].fillna(cleaned_data["brand_rating"].median(), inplace=True)
    
    # Fill missing num_reviews with 0
    cleaned_data["num_reviews"].fillna(0, inplace=True)
    
    # Fill missing shipping_weight with mean
    cleaned_data["shipping_weight"].fillna(cleaned_data["shipping_weight"].mean(), inplace=True)
    
    # Handle outliers in price (capping at 3 std devs)
    mean_price = cleaned_data["price"].mean()
    std_price = cleaned_data["price"].std()
    cleaned_data["price"] = cleaned_data["price"].clip(
        lower=mean_price - 3*std_price,
        upper=mean_price + 3*std_price
    )
    
    # Log cleaning impact
    post_cleaning_shape = cleaned_data.shape
    post_cleaning_nulls = cleaned_data.isnull().sum().sum()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    post_cleaning_stats = cleaned_data[numeric_cols].describe().to_dict()
    
    log_metadata(
        artifact_name="cleaned_data",
        infer_artifact=True,
        metadata={
            "pre_cleaning_rows": pre_cleaning_shape[0],
            "pre_cleaning_missing_values": int(pre_cleaning_nulls),
            "post_cleaning_rows": post_cleaning_shape[0],
            "post_cleaning_missing_values": int(post_cleaning_nulls),
            "outliers_handled": "price",
            "descriptive_statistics": post_cleaning_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return cleaned_data