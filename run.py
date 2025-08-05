import os
import click

from pipeline.dashboard_pipeline import price_prediction_pipeline

from zenml.client import Client

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
    help="Name of the pipeline template.",
)
@click.option(
    "--run",
    type=bool,
    help="Whether to also run.",
)
def main(environment: str, stack: str, name: str = None,  run: bool = False):
    """
    CLI to run a pipeline with specified parameters.
    """
    client = Client()

    remote_stack = client.get_stack(stack)
    os.environ["ZENML_ACTIVE_STACK_ID"] = str(remote_stack.id)

    template = price_prediction_pipeline.with_options(
            config_path=f"configs/{environment}.yml",
        ).create_run_template(
            name=name,
        )

    if run:
        config = template.config_template
        client.trigger_pipeline(
            template_id=template.id,
            run_configuration=config,
        )


if __name__ == "__main__":
    main()
