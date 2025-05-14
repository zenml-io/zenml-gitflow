import pandas as pd
import numpy as np
import datetime
from typing import Annotated, Dict
from zenml import step
from zenml import log_metadata, step

@step
def analyze_data(data: pd.DataFrame) -> Annotated[Dict, "data_analysis"]:
    """Analyze the dataset and compute various statistics."""
    analysis = {}
    
    # Correlation analysis for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    analysis["correlation"] = numeric_data.corr().to_dict()
    
    # Category distribution
    analysis["category_distribution"] = data["category"].value_counts().to_dict()
    
    # Price statistics by category
    price_by_category = data.groupby("category")["price"].agg(["mean", "min", "max", "std"]).to_dict()
    analysis["price_by_category"] = price_by_category
    
    # Discount impact analysis
    discount_impact = data.groupby("discount_offered")["price"].mean().to_dict()
    analysis["discount_impact"] = discount_impact
    
    log_metadata(
        artifact_name="data_analysis",
        infer_artifact=True,
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "price_range": (float(data["price"].min()), float(data["price"].max())),
            "most_common_category": data["category"].value_counts().index[0],
            "price_correlation_factors": sorted(
                [(col, analysis["correlation"]["price"].get(col, 0)) 
                 for col in numeric_data.columns if col != "price"],
                key=lambda x: abs(x[1]),
                reverse=True
            )
        }
    )
    
    return analysis
