ARG ZENML_HOST
ARG ZENML_USERNAME
ARG ZENML_PASSWORD

FROM zenmldocker/zenml:0.21.1-py3.8
WORKDIR /app
ENV ZENML_CONFIG_PATH=/app/.zenconfig
ENV ZENML_STORE_URL=$ZENML_HOST
ENV ZENML_STORE_PASSWORD=$ZENML_PASSWORD
ENV ZENML_STORE_USERNAME=$ZENML_USERNAME
ENV ZENML_STORE_TYPE=rest

COPY .zenml_integration_requirements .
RUN pip install --no-cache-dir -r .zenml_integration_requirements
COPY . .
RUN chmod -R a+rw .