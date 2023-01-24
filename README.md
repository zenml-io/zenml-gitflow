# ZenML - GitHub Action Workflow

<div align="center">
  <a href="https://zenml.io">
    <img src="_assets/zenml_logo.png" alt="Logo" width="400">
  </a>

  <h3 align="center">Build portable, production-ready MLOps pipelines.</h3>

  <p align="center">
    A simple yet powerful open-source framework that integrates all your ML tools.
    <br />
    <a href="https://docs.zenml.io/"><strong>Explore the docs »</strong></a>
    <br />
    <div align="center">
      Join our <a href="https://zenml.io/slack-invite" target="_blank">
      <img width="25" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> </a> and be part of the ZenML family.
    </div>
    <br />
    <a href="https://zenml.io/features">Features</a>
    ·
    <a href="https://zenml.io/roadmap">Roadmap</a>
    ·
    <a href="https://github.com/zenml-io/zenml/issues">Report Bug</a>
    ·
    <a href="https://zenml.io/discussion">Vote New Features</a>
    ·
    <a href="https://blog.zenml.io/">Read Blog</a>
    ·
    <a href="#-meet-the-team">Meet the Team</a>
    <br />
    <br />
    <a href="https://www.linkedin.com/company/zenml/">
    <img src="https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555" alt="Logo">
    </a>
    <a href="https://twitter.com/zenml_io">
    <img src="https://img.shields.io/badge/-Twitter-black.svg?style=for-the-badge&logo=twitter&colorB=555" alt="Logo">
    </a>
    <a href="https://www.youtube.com/c/ZenML">
    <img src="https://img.shields.io/badge/-YouTube-black.svg?style=for-the-badge&logo=youtube&colorB=555" alt="Logo">
    </a>
  </p>
</div>


## 🖼️ Overview

This repository showcases how ZenML can be used for machine learning with a Git 
Ops workflow. This allows data scientists to automatically test their models on 
staging and deploy to production. Here's an architecture diagram of what this 
can look like:

<img src="_assets/pipeline_architecture.png" alt="Pipeline being run on staging/production stack through ci/cd" width="800"/>


The workflow works as follows:

A data scientist wants to make improvements to the ML pipeline. They clone the 
repository, create a new branch, and experiment with new models or data 
processing steps on their local machine.

<img src="_assets/pipeline_local.png" alt="Pipeline with local stack" width="500"/>


Once the data scientist thinks they have improved the pipeline, they create a 
pull request for his branch on GitHub. This automatically triggers a GitHub Action 
that will run the same pipeline in the staging environment (e.g. a pipeline 
running on a cloud stack in GCP), potentially with different test data. As long
as the  pipeline does not run successfully in the staging environment, the PR
cannot be  merged.

<img src="_assets/pipeline_staging.png" alt="Pipeline with staging stack" width="500"/>

Once the PR has been reviewed and passes all checks, the branch is merged into 
main. This automatically triggers another GitHub Action that now runs a 
pipeline in the production environment, which trains the same model on 
production data, and then automatically deploys it.

<img src="_assets/pipeline_prod.png" alt="Pipeline with production stack" width="500"/>



## 👀 Set up

In order to run this repository, the following secrets need to be set in the repository:

```shell
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
GCP_CREDENTIALS
KUBEFLOW_NAMESPACE
KUBEFLOW_PASSWORD
KUBEFLOW_PRODUCTION_NAMESPACE
KUBEFLOW_PRODUCTION_PASSWORD
KUBEFLOW_PRODUCTION_USERNAME
KUBEFLOW_USERNAME
ZENML_HOST
ZENML_PASSWORD
ZENML_USERNAME

# Currently not used but WIP
KUBEFLOW_STAGING_NAMESPACE
KUBEFLOW_STAGING_PASSWORD
KUBEFLOW_STAGING_USERNAME
ZENML_PRODUCTION_STACK
ZENML_STAGING_STACK
```

## 🏇 How to run

1. Clone the repository

```
git clone git@github.com:zenml-io/zenml-gitflow.git
```

2. Install local requirements

```
pip install -r requirements.txt
```

3. Run pipeline locally

```
python run.py
```

4. Make changes in a branch and create a PR to `main`: Will trigger a run in staging GCP Vertex!

5. Merge to `main` after staging github action is completed: Will trigger a run in production AWS Kubeflow!

