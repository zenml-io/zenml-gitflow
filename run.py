import os

import click
from zenml.client import Client

from pipeline.dashboard_pipeline import price_prediction_pipeline


@click.command()
@click.option(
    "--environment",
    type=click.Choice(["local", "staging", "production"]),
    default="local",
    show_default=True,
    help="Environment to run the pipeline in.",
)
def main(environment: str):
    """
    CLI to run a pipeline with specified parameters.
    """
    pipeline = price_prediction_pipeline.with_options(
        config_path=f"configs/{environment}.yml",
    )
    
    pipeline()


if __name__ == "__main__":
    main()