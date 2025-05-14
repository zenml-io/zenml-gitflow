from utils.utils import mock_data
import pandas as pd
import numpy as np
import datetime
from typing import Annotated
from zenml import step
from zenml import log_metadata, step



@step
def load_data(n_samples: int = 1000) -> Annotated[pd.DataFrame, "raw_data"]:
    """Load synthetic product price data with various features."""
    # Create synthetic e-commerce dataset
    np.random.seed(42)
    
    df = mock_data(n_samples)
    
    # Calculate and log detailed metrics
    missing_stats = df.isnull().sum().to_dict()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    descriptive_stats = df[numeric_cols].describe().to_dict()
    
    # Add category-specific metrics for enhanced reporting
    category_stats = {}
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        category_stats[category] = {
            "count": len(cat_df),
            "avg_price": float(cat_df["price"].mean()),
            "avg_cost": float(cat_df["manufacturing_cost"].mean()),
            "avg_weight": float(cat_df["shipping_weight"].dropna().mean()),
            "avg_profit_margin": float(((cat_df["price"] - cat_df["manufacturing_cost"]) / cat_df["price"]).mean())
        }
    
    categorical_stats = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_stats[col] = df[col].value_counts().to_dict()
    
    log_metadata(
        artifact_name="raw_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": missing_stats,
            "descriptive_statistics": descriptive_stats,
            "categorical_distributions": categorical_stats,
            "category_specific_stats": category_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df
