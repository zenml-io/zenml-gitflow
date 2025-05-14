from utils.utils import generate_data_report
import pandas as pd
import numpy as np
import datetime
from typing import Annotated, Dict
from zenml import step
from zenml import log_metadata, step
from zenml.config import DockerSettings
from zenml.types import HTMLString

@step()
def generate_data_analysis_report(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    analysis: Dict
) -> Annotated[HTMLString, "data_analysis_report"]:
    """Generate an HTML report with Plotly visualizations of the data analysis."""
    
    # Log basic metadata about the report
    log_metadata(
        artifact_name="data_analysis_report",
        infer_artifact=True,
        metadata={
            "generated_at": datetime.datetime.now().isoformat(),
            "report_type": "Data Analysis Report",
            "visualizations": [
                "Price by Category", 
                "Manufacturing Cost by Category",
                "Shipping Weight by Category",
                "Profit Margin by Category",
                "Price Distribution"
            ]
        }
    )
    
    return HTMLString(generate_data_report(cleaned_data=cleaned_data, raw_data=raw_data, analysis=analysis))