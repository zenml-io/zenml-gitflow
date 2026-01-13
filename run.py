import click

from pipeline.training_pipeline import price_prediction_pipeline
from utils.project_config import get_config


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
    CLI to run the pipeline locally with specified environment configuration.
    
    This script is for local development and testing. For CI/CD deployments,
    use build.py to create snapshots instead.
    
    Configuration is loaded from:
    - project_config.yaml (central configuration)
    - configs/{environment}.yml (environment-specific overrides)
    """
    config = get_config()
    
    click.echo(f"Running pipeline: {config.pipeline.name}")
    click.echo(f"Environment: {environment}")
    click.echo(f"Model: {config.model.name}")
    
    pipeline = price_prediction_pipeline.with_options(
        config_path=f"configs/{environment}.yml",
    )
    
    pipeline()


if __name__ == "__main__":
    main()
