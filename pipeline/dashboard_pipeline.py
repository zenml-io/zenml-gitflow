import os
import numpy as np
from typing import Annotated, Dict, List
import json
import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import joblib

from zenml import log_metadata, step, pipeline, Model
from zenml.config import DockerSettings
from zenml.types import HTMLString

from utils.utils import generate_data_report
from steps.load_data import load_data
from steps.analyze_data import analyze_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.generate_data_analysis_report import generate_data_analysis_report


@pipeline(
    enable_cache=True,
    model=Model(
        name="PricePredictionModel",
        description="End-to-end pipeline for price prediction.",
    ),
    settings={
        "docker": DockerSettings(
            python_package_installer="uv",
            requirements=["pandas", "numpy", "scikit-learn", "plotly"],
        ),
    },
)
def price_prediction_pipeline(epochs: int = 15, data_analysis: bool = True):
    """Pipeline that demonstrates ZenML's visualization and reporting capabilities."""
    
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    model, model_report = train_model(cleaned_data, epochs=epochs)

    if data_analysis:
        data_analysis = analyze_data(raw_data)
        
        # Generate two separate reports
        data_report = generate_data_analysis_report(raw_data, cleaned_data, data_analysis)
    

if __name__ == "__main__":
    price_prediction_pipeline(epochs=10) 