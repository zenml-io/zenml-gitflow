import os

import click
from zenml.client import Client

from pipeline.dashboard_pipeline import price_prediction_pipeline


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
    help="Stack to run the pipeline in.",
)
@click.option(
    "--name",
    type=str,
    help="Name of the pipeline snapshot.",
)
@click.option(
    "--run",
    is_flag=True,
    help="Whether to also run."
)
def main(environment: str, stack: str, name: str = None, run: bool = False):
    """
    CLI to build a pipeline snapshot with specified parameters.

    Optionally runs the pipeline if the `--run` flag is set.
    """
    client = Client()

    remote_stack = client.get_stack(stack)
    os.environ["ZENML_ACTIVE_STACK_ID"] = str(remote_stack.id)

    # Create a pipeline snapshot (replaces deprecated run templates)
    snapshot = price_prediction_pipeline.with_options(
        config_path=f"configs/{environment}.yml",
    ).create_snapshot(
        name=name,
    )

    if run:
        config = snapshot.config_template
        client.trigger_pipeline(
            snapshot_name_or_id=snapshot.id,
            run_configuration=config,
        )


if __name__ == "__main__":
    main()
