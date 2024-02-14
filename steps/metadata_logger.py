from typing import Optional

from zenml import get_step_context, log_model_metadata, step


@step(enable_cache=False)
def metadata_logger(github_pr_url: Optional[str] = None):
    model = get_step_context().model
    if not model.version.isnumeric():
        log_model_metadata(
            {
                "GitHub commit": (
                    f"https://github.com/zenml-io/zenml-gitflow/commit/{model.version}"
                ),
                "GitHub PullRequest": github_pr_url,
            }
        )
