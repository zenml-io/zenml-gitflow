install-base:
	pip install uv
	uv pip install -r requirements.txt
	zenml integration install sklearn mlflow deepchecks -y --uv

install-gcp: install-base
	pip install uv
	zenml integration install gcp -y --uv	
