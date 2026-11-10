# ---- build : dépendances dans un virtualenv isolé ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- runtime : image minimale, utilisateur non root ----
FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    COPILOT_INDEX_PATH=/app/artifacts/index.pkl \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY --from=build /opt/venv /opt/venv
COPY src/ src/
COPY data/corpus/ data/corpus/
# L'index est construit au build : démarrage rapide et artefact reproductible
RUN python -c "from copilot.config import get_settings; from copilot.service import build_index; build_index(get_settings())" \
    && chown -R app:app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
# métriques partagées entre workers : répertoire vidé à chaque démarrage
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn copilot.api:app --host 0.0.0.0 --port 8080 --workers 2"]
