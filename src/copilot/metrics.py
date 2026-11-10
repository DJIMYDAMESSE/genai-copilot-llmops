"""Métriques Prometheus exposées sur /metrics (scrapées par Prometheus, affichées dans Grafana)."""

import os

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess

REQUESTS = Counter("copilot_requests_total", "Requêtes traitées", ["endpoint", "status"])
LATENCY = Histogram(
    "copilot_request_latency_seconds",
    "Latence de bout en bout",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
LLM_LATENCY = Histogram("copilot_llm_latency_seconds", "Latence d'appel au LLM", ["model"], buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60))
TOKENS = Counter("copilot_llm_tokens_total", "Tokens consommés", ["model", "kind"])
NO_ANSWER = Counter("copilot_no_answer_total", "Réponses « information introuvable » (refus)")
BLOCKED = Counter("copilot_blocked_total", "Requêtes bloquées par les garde-fous", ["reason"])
FALLBACK = Counter("copilot_llm_fallback_total", "Bascules vers le moteur local suite à une panne LLM")
TOOL_CALLS = Counter("copilot_tool_calls_total", "Appels d'outils métier", ["tool"])
INDEX_CHUNKS = Gauge("copilot_index_chunks", "Nombre de chunks dans l'index", multiprocess_mode="max")


def render() -> tuple[bytes, str]:
    """Agrège les métriques de tous les workers uvicorn si PROMETHEUS_MULTIPROC_DIR est défini."""
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
