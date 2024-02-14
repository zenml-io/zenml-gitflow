install-base:
	pip3 install -r requirements.txt
	zenml integration install sklearn mlflow deepchecks -y

install-gcp: install-base
	zenml integration install gcp -y

install-aws: install-base
	zenml integration install s3 aws kubernetes -y
