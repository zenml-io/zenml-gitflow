import argparse

from zenml.client import Client
from zenml.enums import ModelStages

from configs.global_conf import MODEL_NAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        help="The version to promote to production.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    Client().get_model_version(MODEL_NAME, args.version).set_stage(
        ModelStages.PRODUCTION, force=True
    )
    print(
        f"Model `{MODEL_NAME}` version `{args.version}` promoted to production!"
    )
