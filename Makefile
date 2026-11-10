PY ?= python3
export PYTHONPATH := src

install:      ## dépendances de dev
	$(PY) -m pip install -r requirements-dev.txt
index:        ## construit l'index
	$(PY) -c "from copilot.config import get_settings; from copilot.service import build_index; print(len(build_index(get_settings()).chunks), 'chunks')"
run:          ## lance l'API en local (moteur extractif)
	uvicorn copilot.api:app --reload --port 8080
test:
	$(PY) -m pytest -q
eval:         ## évaluation + quality gate + MLflow
	$(PY) -m copilot.evaluate --gate
load:         ## test de charge (API lancée sur :8080)
	$(PY) scripts/load_test.py --url http://localhost:8080 --requests 600 --concurrency 20
up:           ## stack complète : API + Ollama (Qwen2.5) + Prometheus + Grafana
	docker compose up -d --build
k8s:          ## déploiement Kubernetes (vLLM GPU + API)
	kubectl apply -k deploy/k8s
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
.PHONY: install index run test eval load up k8s mlflow-ui
