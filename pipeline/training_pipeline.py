from zenml import Model, pipeline
from zenml.config import DockerSettings

from steps.analyze_data import analyze_data
from steps.clean_data import clean_data
from steps.generate_data_analysis_report import generate_data_analysis_report
from steps.load_data import load_data
from steps.train_model import train_model
from utils.project_config import get_config


# Load project configuration
config = get_config()


@pipeline(
    name=config.pipeline.name,
    enable_cache=True,
    model=Model(
        name=config.model.name,
        description=config.model.description,
        tags=config.model.tags,
    ),
    settings={
        "docker": DockerSettings(
            python_package_installer="uv",
            requirements=["pandas", "numpy", "scikit-learn", "plotly"],
        ),
    },
    # Use substitutions for dynamic run names
    # The {run_name_prefix} will be replaced by the value from config
    substitutions={
        "run_name_prefix": config.pipeline.run_name_prefix,
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
        generate_data_analysis_report(raw_data, cleaned_data, data_analysis)


if __name__ == "__main__":
    price_prediction_pipeline(epochs=10)

