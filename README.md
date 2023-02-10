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

This repository showcases how ZenML can be used to incrementally build an
end-to-end production ML pipeline, starting locally and eventually scaling it
onto a production-ready tool stack. The pipeline contains built-in best
practices implemented using data validators such as Evidently, experiment
trackers such as MLflow, and model serving tools such as KServe.

The end-to-end pipeline can be shown as being constructed in several stages:

**Stage 1 - _local experimentation_**

This stage features the first version of the pipeline, which is
_a model training pipeline_ that can be run locally as part of an iterative
experimentation workflow. It only goes as far as training a model without
actually serving it, but it already covers experiment tracking through a local
MLflow server and model evaluation and data validation performed using
Evidently.

**Stage 2 - _staging for production_**

In this second stage, a number of steps that are important in production are
added to the model training pipeline, resulting in _an end-to-end pipeline_ that
also covers model serving. The end-to-end pipeline includes a local
MLflow model serving step and other steps that together implement the same
continuous deployment workflow that will be used in production.

This stage can be thought of as a prelude to running continuous model training
and deployment in production. The end-to-end pipeline is still run on a local
machine and using local-only components, but it is now ready to be ported to a
production-ready stack.

Aside from the model deployment step, the end-to-end pipeline also features a
steps that are used to compare the performance of a newly trained model against
the performance of the model already being served, if one exists.

**Stage 3 - _production_**

The third stage is about porting and run the end-to-end pipeline to a
production-ready stack with no changes to the machine learning code. This is
where the ability to build portable pipelines with ZenML really comes through.

You'll need to configure a production-ready ZenML stack, or use the one already
pre-configured for you). The pre-configured stack consists of a remote MLflow
server, an S3 artifact store, a multi-tenant Kubeflow orchestrator and a KServe
model server, all running in AWS.

This stage also requires connecting your local ZenML client to a remote ZenML
server which is reachable from the production-ready AWS environment. Please note
that a remote ZenML server could also have been used in the previous stages,
but here it becomes mandatory.

These are some of the MLOps best practices that are showcased in this example: 

* **Experiment Tracking**: All experiments are logged with an experiment tracker
(MLflow), which allows for easy comparison of different runs and models and
provides quick access to visualization and validation reports.
* **Data and Model validation**: The pipelines include a set of Evdently-powered
steps that verify the quality of the data and evaluate the model after
training. The results are gathered, analyzed and then a report is generated with
a summary of the findings and a suggested course of action. This provides useful
insights into the quality of the data and the performance of the model and helps to
catch potential issues early on, before the model is deployed to production.
* **Pipeline Tracking**: All pipeline runs and their artifacts are of course
versioned and logged with ZenML. This enables features such as lineage tracking,
provenance, caching and reproducibility.
* **Continuous Deployment**: When a new model is trained, it is automatically
deployed to production. There are also additional checks in place to ensure that
the model is not deployed if it is not fit for production or performs worse than
the model currently deployed.
* **Software Dependency Management**: All software dependencies are managed
in a way that guarantees full reproducibility and are automatically installed
by ZenML in the pipeline runtime environments. Python package versions are frozen
and pinned to ensure that the pipeline runs are fully reproducible.
* **Reproducible Randomness**: All randomness is controlled and seeded to ensure
reproducibility and caching of the pipeline runs.

## 🏇 How to run

1. Clone the repository and checkout the `devweek-demo` branch

```
git clone git@github.com:zenml-io/zenml-gitflow.git
git checkout -b devweek-demo
```

2. Install local requirements in a virtual environment

```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. (optional) Connect to a remote ZenML server

You should use the https://demo.zenml.io/ remote ZenML server early on, even
if this is only required in the final stage, for the following reasons:

* all stacks showcased in this example are already configured in the demo
server, so you don't have to configure them yourself:
  * [the local stack](https://demo.zenml.io/workspaces/default/stacks/187e8d06-b3af-4f20-8435-fa1a628a09dd/component?page=1)
  * [the AWS stack](https://demo.zenml.io/workspaces/default/stacks/c421ce43-9a10-4738-a91b-f71f0e25290f/component?page=1)
* the demo server is an open server that is available to everyone, so you can
share the link to the ZenML UI with anyone who is interested in seeing live
results of your pipeline runs:
  * [training pipeline runs](https://demo.zenml.io/workspaces/default/pipelines/355bc31f-8fe4-4e84-904a-b2f62b33fc1e/runs?page=1)
  * [end-to-end pipeline runs](https://demo.zenml.io/workspaces/default/pipelines/a1de5f98-2320-4934-befb-5c2e46e45c18/runs?page=1)

To connect to the demo server, you can run the following command with the
provided username and password credentials:

```
zenml connect --url https://demoserver.zenml.io --username ZENML_USERNAME --password ZENML_PASSWORD
```

### 📦 Stage 1: local experimentation

1. Set up your local stack

If you are already connected to the demo server, you can skip configuring
the local stack and use the one that is already configured in the demo server:

```bash
zenml stack use devweek_local_stack
```

If you are not connected to the demo server, you can set up your local stack
by running:

```
stacks/setup_local_stack.sh
```

NOTE: this script registers a stack that consists of the following components:

* a local orchestrator and artifact store
* a local MLflow tracking server
* a local MLflow model deployer (only used in the next stage)
* Evidently as a data/model validator

2. Run the training pipeline locally

```
python run.py
```

3. View the results

A report will be printed to the console and the model and the model metrics
and check results generated by Evidently will be saved in the MLflow experiment
tracker. To view the results in the local MLflow UI you'll have to manually
start the MLFlow UI server on your local machine, as instructed in the run
output.

If you're connected to the demo server, the pipeline run can also be viewed
[in the ZenML dashboard](https://demo.zenml.io/workspaces/default/pipelines/355bc31f-8fe4-4e84-904a-b2f62b33fc1e/runs?page=1).

The following Evidently reports will be generated:

* a data quality report for the entire data used as input
* a data drift report that compares the training and test datasets and
points out any differences that might indicate causes for variance in the
trained model
* a model evaluation report that measures the performance of the trained model
on the test dataset
* a train-test model evaluation report that is a side-by-side comparison of
how the trained model performs on the training and test datasets

4. Look at the code

You can take a look at the training pipeline definition in `pipelines/training.py`
and the steps that are part of it in `steps/`.

5. Make changes and run the pipeline again

You can make changes to the various step parameters configured in `run.py` and
then run the pipeline again to see how the results change. Some steps will be
cached, but you can force a re-run by passing the `--disable-caching` flag to
the `run.py` script.

### 📦 Stage 2: staging for production

1. Run the end-to-end pipeline locally

Now you can exercise the end-to-end workflow locally by running it on the same
stack. The end-to-end pipeline adds continuous model deployment to model
training. It also includes steps that compare the performance of the recently
trained model with the performance of the model currently deployed:

```
python run.py --pipeline=end-to-end
```

2. View the results

A report will be printed to the console and the model and the model metrics
and check results generated by Evidently will be saved in the MLflow experiment
tracker.

If you're connected to the demo server, the pipeline run can also be viewed
[in the ZenML dashboard](https://demo.zenml.io/workspaces/default/pipelines/a1de5f98-2320-4934-befb-5c2e46e45c18/runs?page=1).

The pipeline results in an MLflow model being deployed locally. You can view
all the deployed models by running:

```bash
zenml model-deployer models list
```

The first time you run the end-to-end pipeline locally, the same Evidently
reports will be generated as in the previous stage. However, if you run the
pipeline again, you will also get a new Evidently report that compares the
performance of the newly trained model with the performance of the model that
is currently deployed.

4. Look at the code

You can take a look at the training pipeline definition in `pipelines/end_to_end.py`
and the steps that are part of it in `steps/`.

5. Make changes and run the pipeline again

You can make changes to the various step parameters configured in `run.py` and
then run the pipeline again to see how the results change. Some steps will be
cached, but you can force a re-run by passing the `--disable-caching` flag to
the `run.py` script.

If a new model is trained and if it passes the model evaluation checks, it will
be deployed in place of the currently deployed model.

NOTE: there is a known issue with updating a deployed MLflow model. If you
run into an error when the end-to-end pipeline is trying to deploy a new model,
you can simply re-run the pipeline again and it should work.

### 📦 Stage 3: production

This is where you scale your pipeline to run on a remote ZenML server (if you
aren't already using one) and on a cloud ZenML stack with zero code changes.

In this stage you will use an AWS cloud stack. The stack is already configured
in the demo ZenML server, but you still need to install some prerequisites and
set up local credentials in order to use it on your local machine. You'll need:

* an AWS account
* the AWS CLI installed and configured to use your AWS account
* Docker installed
* the Kubernetes client (kubectl) installed 

1. Connect to the https://demo.zenml.io/ ZenML server

```
zenml connect --url https://demoserver.zenml.io --username ZENML_USERNAME --password ZENML_PASSWORD
```

2. Authenticate to use the AWS cloud stack components (ECR, EKS, the
multi-tenant Kubeflow orchestrator) locally

You'll need your AWS account credentials to be properly configured on your
local machine for this to work:

```bash
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.eu-central-1.amazonaws.com
aws eks --region eu-central-1 update-kubeconfig --name kubeflowmultitenant --alias kubeflowmultitenant

# These are needed to authenticate to the Kubeflow orchestrator
export KUBEFLOW_USERNAME=...
export KUBEFLOW_PASSWORD=...
export KUBEFLOW_NAMESPACE=...
```

3. Set up your cloud stack

Since you are connected to the demo server, you can skip configuring
the local stack and use the one that is already configured in the demo server:

```bash
zenml stack use devweek_aws_stack
```

You can take a look at the `stacks/setup_aws_stack.py` script to see how the
stack was configured.

4. Run the end-to-end pipeline on the cloud stack

Now you can finally run the same end-to-end pipeline that you used in the
previous stage, only this time you'll be using a full-fledge production-ready
stack:

```
python run.py --pipeline=end-to-end
```

5. View the results

A report will be printed to the console and the model and the model metrics
and check results generated by Evidently will be saved in the remote MLflow
experiment tracker and can be accessed using the URL printed on screen.

NOTE: You'll need an MLFlow tracker username and password to access the remote
MLFlow UI.

The pipeline run can also be viewed
[in the ZenML dashboard](https://demo.zenml.io/workspaces/default/pipelines/a1de5f98-2320-4934-befb-5c2e46e45c18/runs?page=1)
and [in the Kubeflow dashboard](https://www.kubeflowshowcase.zenml.io/_/pipeline/?ns=production-demo#/runs).

NOTE: You'll need to use the same `KUBEFLOW_USERNAME` and `KUBEFLOW_PASSWORD`
credentials to access the Kubeflow dashboard. 

The pipeline results in a KServe model being deployed locally. You can view
all the deployed models by running:

```bash
zenml model-deployer models list
```

The same Evidently reports will be generated as in the previous stage.

6. Make changes and run the pipeline again

You can make changes to the various step parameters configured in `run.py` and
then run the pipeline again to see how the results change. Some steps will be
cached, but you can force a re-run by passing the `--disable-caching` flag to
the `run.py` script.

If a new model is trained and if it passes the model evaluation checks, it will
be deployed in place of the currently deployed KServe model.

## 📦 Software requirements management

In building ML pipelines and their dependencies for production, you want to make
sure that your builds are predictable and deterministic. This requires
Python packages to be pinned to very specific versions, including ZenML itself.

A series of `requirements.in` files are maintained to contain loosely defined
package dependencies that are either required by the project or derived from the
ZenML integrations. A tool like `pip-compile` is then used to generate frozen
versions of those requirements that are used when running pipelines to guarantee
deterministic builds and pipeline runs.

* `requirements-base.in` contains the base requirements for the project,
including ZenML itself. Note that these are not requirements that are otherwise
managed by ZenML as integrations (e.g. evidently, scikit-learn, etc.).
* `requirements.in` contains the requirements needed for both the local and AWS
stacks.
* `requirements.txt` are frozen package requirements that you use to run
pipelines locally.

### 🏇 How to update requirements

The frozen requirements files need to be updated whenever you make a change to
the project dependencies. This includes the following:

* when you add, update or remove a new package to the project that is not
covered by ZenML or a ZenML integration
* when you update ZenML itself to a new version (which may also include updates
to the ZenML integrations and their requirements)
* when you add or remove a ZenML integration to/from the project (also reflected
in the code changes or stacks that you use for the various workflows)

Whenever one of these happens, you need to update the `requirements.in` files to
reflect the changes.

Example: you update ZenML to version 0.50.0:

1. install the new ZenML version and the `pip-tools` package:

```
pip install zenml[server]==0.50.0
pip install pip-tools
```

2. Update the ZenML version in `requirements.in`:

```
zenml[server]==0.50.0
```

3. Run `zenml integration export-requirements` commands to extract the
dependencies required by all the integrations used for each workflow and update
the `requirements.in` file:

```
zenml integration export-requirements sklearn mlflow evidently kubeflow s3 aws kserve
```

3. Run `pip-compile` to update the frozen requirements file:

```
pip-compile -v -o requirements.txt requirements-base.in requirements.in  --resolver=backtracking
```

4. Update the local python virtual environment:

```
pip install -r requirements.txt
```
