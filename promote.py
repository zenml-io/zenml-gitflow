"""
Promote a model version to production stage.

Usage:
    python promote.py --version 1
    python promote.py -v 1
"""

import argparse

from zenml.client import Client
from zenml.enums import ModelStages

from utils.project_config import get_model_name


def main():
    parser = argparse.ArgumentParser(
        description="Promote a model version to production."
    )
    parser.add_argument(
        "-v",
        "--version",
        help="The version to promote to production.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stage",
        help="The stage to promote to.",
        type=str,
        choices=["staging", "production", "archived"],
        default="production",
    )
    args = parser.parse_args()

    # Get model name from central config
    model_name = get_model_name()
    
    # Map string stage to ModelStages enum
    stage_map = {
        "staging": ModelStages.STAGING,
        "production": ModelStages.PRODUCTION,
        "archived": ModelStages.ARCHIVED,
    }
    stage = stage_map[args.stage]

    # Promote the model
    Client().get_model_version(model_name, args.version).set_stage(
        stage, force=True
    )
    
    print(f"✅ Model `{model_name}` version `{args.version}` promoted to {args.stage}!")


if __name__ == "__main__":
    main()
