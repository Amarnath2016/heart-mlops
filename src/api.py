"""Flask serving API: /predict, /health, /metrics, /model-info.

Request logging is structured JSON on stdout (and optionally a file); Prometheus-style
metrics are exposed in text format at /metrics with no extra dependency.
"""
import json
import logging
import os
import threading
import time
import uuid

from flask import Flask, Response, jsonify, request

from src import config
from src.predict import ValidationError, load_metadata, load_model, predict_one

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    """Tiny thread-safe Prometheus text-format registry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}            # (endpoint, method, status) -> count
        self.predictions = {0: 0, 1: 0}
        self.buckets = {b: 0 for b in LATENCY_BUCKETS}
        self.lat_sum = 0.0
        self.lat_count = 0

    def observe_request(self, endpoint, method, status, seconds):
        with self.lock:
            key = (endpoint, method, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            if endpoint == "/predict":
                self.lat_sum += seconds
                self.lat_count += 1
                for b in LATENCY_BUCKETS:
                    if seconds <= b:
                        self.buckets[b] += 1

    def observe_prediction(self, cls):
        with self.lock:
            self.predictions[cls] += 1

    def render(self):
        with self.lock:
            lines = ["# HELP api_requests_total HTTP requests", "# TYPE api_requests_total counter"]
            for (ep, m, s), c in sorted(self.requests.items()):
                lines.append(f'api_requests_total{{endpoint="{ep}",method="{m}",status="{s}"}} {c}')
            lines += ["# HELP predictions_total Predictions by class", "# TYPE predictions_total counter"]
            for cls, c in self.predictions.items():
                lines.append(f'predictions_total{{class="{cls}"}} {c}')
            lines += ["# HELP predict_latency_seconds /predict latency", "# TYPE predict_latency_seconds histogram"]
            for b in LATENCY_BUCKETS:  # buckets are already cumulative (<=)
                lines.append(f'predict_latency_seconds_bucket{{le="{b}"}} {self.buckets[b]}')
            lines.append(f'predict_latency_seconds_bucket{{le="+Inf"}} {self.lat_count}')
            lines.append(f"predict_latency_seconds_sum {self.lat_sum:.6f}")
            lines.append(f"predict_latency_seconds_count {self.lat_count}")
        return "\n".join(lines) + "\n"


def setup_logger():
    logger = logging.getLogger("heart-api")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handlers = [logging.StreamHandler()]
        log_file = os.environ.get("API_LOG_FILE")
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        for h in handlers:
            h.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(h)
        logger.propagate = False
    return logger


def create_app(model=None, metadata=None):
    app = Flask(__name__)
    logger = setup_logger()
    metrics = Metrics()
    app.config["MODEL"] = model if model is not None else load_model(
        os.environ.get("MODEL_PATH", config.MODEL_PATH))
    app.config["META"] = metadata if metadata is not None else load_metadata()

    @app.before_request
    def _start():
        request.environ["t0"] = time.perf_counter()
        request.environ["rid"] = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

    @app.after_request
    def _log(resp):
        dt = time.perf_counter() - request.environ.get("t0", time.perf_counter())
        ep = request.path if request.path in {"/predict", "/health", "/metrics", "/model-info"} else "other"
        metrics.observe_request(ep, request.method, resp.status_code, dt)
        if ep != "/metrics":
            logger.info(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request_id": request.environ.get("rid"),
                "method": request.method, "path": request.path, "status": resp.status_code,
                "latency_ms": round(dt * 1000, 2), "remote": request.remote_addr,
                **({"prediction": request.environ["pred"]} if "pred" in request.environ else {}),
            }))
        return resp

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/model-info")
    def model_info():
        m = app.config["META"]
        return jsonify({k: m.get(k) for k in ("model_name", "cv_metrics", "test_metrics", "features", "trained_at")})

    @app.get("/metrics")
    def prom():
        return Response(metrics.render(), mimetype="text/plain; version=0.0.4")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        try:
            result = predict_one(app.config["MODEL"], payload)
        except ValidationError as exc:
            return jsonify(error=str(exc)), 400
        metrics.observe_prediction(result["prediction"])
        request.environ["pred"] = result["prediction"]
        return jsonify(result)

    return app


def get_app():  # gunicorn factory: gunicorn "src.api:get_app()"
    return create_app()
