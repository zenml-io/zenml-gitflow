# ZenML - GitFlow

This repository showcases how ZenML can be used with GitFlow for machine learning CI/CD. This allows data scientists to automatically test their models on staging and deploy to production.

The workflow works as follows:

1) A data scientist wants to make improvements to the ML pipeline. He clones the repository, creates a new branch, and experiments with new models or data processing steps on his local machine.

2) Once the data scientist thinks he has improved the pipeline, he creates a pull request for his branch on GitHub. This automatically triggers a GitHub Action that will run the same pipeline in the staging environment (e.g. KubeFlow on an EKS cluster), potentially with different test data. As long as the pipeline does not run successfully in the staging environment, the PR cannot be merged.

3) Once the PR has been reviewed and passes all checks, the branch is merged into develop. This automatically triggers another GitHub Action that now runs a pipeline in the production environment, which trains the same model on production data, and then automatically deploys it.
