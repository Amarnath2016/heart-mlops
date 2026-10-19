# Heart Disease Risk - End-to-End MLOps (Assignment 1)

Predicts heart-disease risk from the UCI Heart Disease (Cleveland) dataset and serves it as a
containerised, monitored Flask API.

## Quick start (Python 3.11)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py      # or place processed.cleveland.data in data/raw/ and add --skip-download
python -m src.eda                    # figures -> reports/figures
python -m src.train                  # MLflow runs -> ./mlruns, model -> models/
mlflow ui --backend-store-uri ./mlruns    # http://127.0.0.1:5000
pytest -v
```

## Serve
```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
scripts/sample_request.sh            # POST /predict
```
Endpoints: `POST /predict`, `GET /health`, `GET /model-info`, `GET /metrics`.

## Kubernetes (Docker Desktop or Minikube)
```bash
# Minikube only: eval $(minikube docker-env) before `docker build`, and run `minikube tunnel` for the LoadBalancer
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
curl http://localhost/health        # Docker Desktop; Minikube: use the EXTERNAL-IP shown
```

## Monitoring
```bash
docker compose -f monitoring/docker-compose.yml up --build
python scripts/load_test.py http://localhost:8000 200
# Prometheus http://localhost:9090   Grafana http://localhost:3000 (admin/admin) -> "Heart Disease API"
```

## Report
Put screenshots in `screenshots/` (names are listed in `scripts/generate_report.py`), then
`python scripts/generate_report.py --repo-url <url> --name "<name>" --id <id>` -> `reports/MLOps_Assignment_Report.docx`
(export to PDF from Word if required).

## Layout
`src/` pipeline + API - `tests/` unit tests - `scripts/` download, load test, report - `k8s/` manifests -
`monitoring/` Prometheus/Grafana - `.github/workflows/ci.yml` CI - `data/processed/` cleaned dataset.
