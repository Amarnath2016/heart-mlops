#!/usr/bin/env bash
# End-to-end runner for the MLOps assignment (Linux VM / lab terminal).
#   ./run_pipeline.sh              interactive: pauses at every screenshot point
#   ./run_pipeline.sh --no-pause   run straight through (screenshots are only listed)
# Optional steps (Docker, Kubernetes, monitoring) are skipped automatically when the tool is missing.
# Env overrides: REPO_URL, STUDENT_NAME, STUDENT_ID, MLFLOW_PORT, API_PORT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_URL="${REPO_URL:-https://github.com/Amarnath2016/heart-mlops}"
STUDENT_NAME="${STUDENT_NAME:-Amarnath K}"
STUDENT_ID="${STUDENT_ID:-2025ae05664}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
API_PORT="${API_PORT:-8000}"
PAUSE=1; [[ "${1:-}" == "--no-pause" ]] && PAUSE=0
BG_PIDS=()
SUMMARY=()

mkdir -p screenshots reports/figures
step()  { echo; echo "================ $* ================"; }
note()  { echo "  -> $*"; }
have()  { command -v "$1" >/dev/null 2>&1; }
skip()  { echo "  [SKIPPED] $*"; SUMMARY+=("SKIPPED: $*"); }
ok()    { SUMMARY+=("OK:      $*"); }
shot()  { # shot <file> <what to capture>
  echo; echo "  📸 SCREENSHOT -> screenshots/$1"; echo "     Capture: $2"
  SUMMARY+=("SCREENSHOT screenshots/$1 : $2")
  if (( PAUSE )); then read -r -p "     Press Enter once it is saved... " _ || true; fi
}
cleanup() { for p in "${BG_PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
must()  { "$@" || { echo "FAILED: $*"; exit 1; }; }
wait_http() { # wait_http <url> [tries]
  for _ in $(seq 1 "${2:-30}"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
compose() { if docker compose version >/dev/null 2>&1; then docker compose "$@"; else docker-compose "$@"; fi; }

# ---------------------------------------------------------------- 0. environment
step "0. Python environment"
python3 --version
[[ -d .venv ]] || must python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
must pip install --quiet --upgrade pip
must pip install --quiet -r requirements.txt
ok "virtualenv + requirements"

# CI workflow must live in .github/workflows
if [[ ! -f .github/workflows/ci.yml && -f github-actions-ci.yml ]]; then
  mkdir -p .github/workflows && cp github-actions-ci.yml .github/workflows/ci.yml
  note "copied github-actions-ci.yml -> .github/workflows/ci.yml"
fi

# ---------------------------------------------------------------- 1. data, EDA, training
step "1. Data download, EDA, training (MLflow)"
must python scripts/download_data.py
must python -m src.eda
must python -m src.train
ok "data, EDA figures (reports/figures), model (models/model.joblib), MLflow runs (mlruns/)"
shot training_output.png "terminal showing data shape, missing values, both models' ROC-AUC and the 'Selected:' line (scroll up if needed)"

# ---------------------------------------------------------------- 2. tests + lint
step "2. Unit tests and lint"
must pytest -v
ok "pytest"
shot pytest_results.png "the passing pytest list"
if flake8 src tests scripts; then ok "flake8"; else echo "  flake8 reported issues (fix before pushing; CI will fail on them)"; SUMMARY+=("WARN: flake8 issues"); fi

# ---------------------------------------------------------------- 3. MLflow UI
step "3. MLflow UI"
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port "$MLFLOW_PORT" >/tmp/mlflow_ui.log 2>&1 &
BG_PIDS+=($!)
if wait_http "http://localhost:${MLFLOW_PORT}" 30; then
  note "MLflow UI running at http://localhost:${MLFLOW_PORT} - open it in the lab browser"
  shot mlflow_ui.png "runs list of experiment 'heart-disease-risk' (both runs)"
  shot mlflow_compare.png "tick both runs -> Compare -> metrics view"
  shot mlflow_run_detail.png "(optional) one run: params, metrics and artifacts (ROC / confusion matrix)"
  ok "MLflow UI"
else
  skip "MLflow UI did not start (see /tmp/mlflow_ui.log)"
fi
cleanup; BG_PIDS=()

# ---------------------------------------------------------------- 4. Git commit / push
step "4. Commit and push to GitHub"
if have git && [[ -d .git ]]; then
  git add -A
  git commit -m "Add pipeline results, cleaned data, trained model" || note "nothing new to commit"
  if (( PAUSE )); then
    read -r -p "  Push to GitHub now? (needs username + personal access token) [y/N] " ans || ans=n
    if [[ "$ans" =~ ^[Yy]$ ]]; then git push && ok "git push" || SUMMARY+=("WARN: git push failed"); fi
  else
    note "run 'git push' yourself (credentials required)"
  fi
  note "Now open ${REPO_URL}/actions and wait for the run to finish"
  shot github_actions.png "browser: Actions tab -> green run with lint-test, train, docker jobs"
  shot github_actions_artifacts.png "browser: same run page, Artifacts section at the bottom"
else
  skip "not a git checkout"
fi

# ---------------------------------------------------------------- 5. Docker
DOCKER_OK=0
step "5. Docker image and container"
if have docker && docker info >/dev/null 2>&1; then
  DOCKER_OK=1
  must docker build -t heart-api:latest .
  shot docker_build.png "end of 'docker build' output"
  docker rm -f heart-api >/dev/null 2>&1 || true
  must docker run -d --name heart-api -p "${API_PORT}:8000" heart-api:latest
  if wait_http "http://localhost:${API_PORT}/health" 30; then
    echo; echo "--- /predict ---"; scripts/sample_request.sh "http://localhost:${API_PORT}"
    echo; echo "--- docker logs ---"; docker logs heart-api | tail -n 10
    shot docker_predict.png "docker run + /predict response (prediction + confidence)"
    shot api_logs.png "JSON request logs from 'docker logs heart-api'"
    ok "docker build + run + predict"
  else
    docker logs heart-api | tail -n 30; skip "container did not become healthy"
  fi
  docker rm -f heart-api >/dev/null 2>&1 || true
else
  skip "Docker not available (do steps 5-7 on a machine with Docker Desktop)"
fi

# ---------------------------------------------------------------- 6. Kubernetes
step "6. Kubernetes deployment"
if (( DOCKER_OK )) && have kubectl && kubectl cluster-info >/dev/null 2>&1; then
  have minikube && minikube image load heart-api:latest >/dev/null 2>&1 || true
  must kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
  kubectl rollout status deployment/heart-api --timeout=180s || true
  echo; kubectl get pods,svc
  shot kubectl_get_all.png "'kubectl get pods,svc' - pods Running, service heart-api (LoadBalancer)"
  kubectl port-forward svc/heart-api 8080:80 >/tmp/pf.log 2>&1 &
  BG_PIDS+=($!)
  if wait_http "http://localhost:8080/health" 20; then
    scripts/sample_request.sh http://localhost:8080
    shot k8s_predict.png "/predict response served from the Kubernetes service (via port-forward; use the EXTERNAL-IP if your cluster gives one)"
    ok "kubernetes deploy + predict"
  else
    skip "service not reachable via port-forward (see /tmp/pf.log, 'kubectl describe pods')"
  fi
  cleanup; BG_PIDS=()
else
  skip "kubectl/cluster not available (start Docker Desktop Kubernetes or 'minikube start')"
fi

# ---------------------------------------------------------------- 7. Monitoring
step "7. Prometheus + Grafana monitoring"
if (( DOCKER_OK )) && { docker compose version >/dev/null 2>&1 || have docker-compose; }; then
  must compose -f monitoring/docker-compose.yml up --build -d
  if wait_http "http://localhost:${API_PORT}/health" 40; then
    python scripts/load_test.py "http://localhost:${API_PORT}" 300
    sleep 10
    note "Prometheus: http://localhost:9090/targets    Grafana: http://localhost:3000 (admin/admin) -> 'Heart Disease API'"
    shot prometheus_targets.png "Prometheus Status -> Targets: heart-api is UP"
    shot grafana_dashboard.png "Grafana dashboard 'Heart Disease API' with request rate, latency, predictions"
    ok "monitoring stack"
  else
    skip "monitoring API did not start"
  fi
  compose -f monitoring/docker-compose.yml down >/dev/null 2>&1 || true
else
  skip "Docker Compose not available"
fi

# ---------------------------------------------------------------- 8. Report
step "8. Report"
note "Make sure all screenshots are saved in ./screenshots/ with the names listed above"
must python scripts/generate_report.py --repo-url "$REPO_URL" --name "$STUDENT_NAME" --id "$STUDENT_ID"
ok "reports/MLOps_Assignment_Report.docx (open in Word, fix [MISSING] entries, add pytest/training screenshots, Save as PDF)"

step "SUMMARY"
printf '%s\n' "${SUMMARY[@]}"
echo; echo "Screenshots present:"; ls -1 screenshots | grep -v '^.gitkeep$' || echo "  (none yet)"
