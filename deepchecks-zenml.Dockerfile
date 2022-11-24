ARG ZENML_VERSION=0.22.0
FROM zenmldocker/zenml:${ZENML_VERSION} AS base

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y