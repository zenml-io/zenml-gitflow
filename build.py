import os
from typing import Optional

import click
from zenml.client import Client

from pipeline.dashboard_pipeline import price_prediction_pipeline
from utils.project_config import get_snapshot_name


@click.command()
@click.option(
    "--environment",
    type=click.Choice(["staging", "production"]),
    default="staging",
    show_default=True,
    help="Environment to run the pipeline in.",
)
@click.option(
    "--stack",
    type=str,
    required=True,
    help="Stack to run the pipeline in.",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Name of the pipeline snapshot. If not provided, auto-generated from project_config.yaml.",
)
@click.option(
    "--git-sha",
    type=str,
    default=None,
    envvar="ZENML_GITHUB_SHA",
    help="Git SHA for snapshot naming (auto-detected from ZENML_GITHUB_SHA env var).",
)
@click.option(
    "--run",
    is_flag=True,
    help="Whether to also run the pipeline after creating the snapshot.",
)
def main(
    environment: str,
    stack: str,
    name: Optional[str] = None,
    git_sha: Optional[str] = None,
    run: bool = False,
):
    """
    CLI to build a pipeline snapshot with specified parameters.

    Snapshot names are auto-generated from project_config.yaml settings:
    - Staging: STG_{prefix}_{git_sha}
    - Production: PROD_{prefix}_{git_sha}

    Optionally runs the pipeline if the `--run` flag is set.
    """
    client = Client()

    # Set the active stack
    remote_stack = client.get_stack(stack)
    os.environ["ZENML_ACTIVE_STACK_ID"] = str(remote_stack.id)

    # Generate snapshot name from config if not provided
    if name is None:
        name = get_snapshot_name(environment=environment, git_sha=git_sha)
        click.echo(f"Generated snapshot name: {name}")

    # Create a pipeline snapshot
    click.echo(f"Creating snapshot '{name}' for {environment} environment...")
    snapshot = price_prediction_pipeline.with_options(
        config_path=f"configs/{environment}.yml",
    ).create_snapshot(
        name=name,
    )
    click.echo(f"Snapshot created successfully: {snapshot.id}")

    if run:
        click.echo("Triggering pipeline run...")
        config = snapshot.config_template
        run_response = client.trigger_pipeline(
            snapshot_name_or_id=snapshot.id,
            run_configuration=config,
        )
        click.echo(f"Pipeline run triggered: {run_response.id}")


if __name__ == "__main__":
    main()
