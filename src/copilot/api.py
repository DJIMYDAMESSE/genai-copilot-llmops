"""API FastAPI du copilote."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from . import metrics as m
from .config import get_settings
from .prompts import NO_ANSWER
from .service import build_agent, build_index

logging.basicConfig(level=logging.INFO, format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}')
log = logging.getLogger("copilot")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    STATE["settings"] = s
    STATE["agent"] = build_agent(s)
    m.INDEX_CHUNKS.set(len(STATE["agent"].retriever.chunks))
    log.info('"copilote prêt : %d chunks, provider=%s, modèle=%s"', len(STATE["agent"].retriever.chunks), s.llm_provider, s.llm_model)
    yield
    STATE.clear()


app = FastAPI(title="NovaGrid GenAI Copilot", version=__version__, lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Quels sont les délais de prise en charge d'un incident P1 ?"])


class Source(BaseModel):
    id: int
    doc_id: str
    source: str
    relevance: float
    cited: bool


class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    grounded: bool
    tool: str | None
    pii_masked: list[str]
    blocked: bool
    model: str | None
    fallback_used: bool
    prompt_version: str
    usage: dict
    steps: list[str]
    latency_ms: float


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = STATE["settings"].api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Clé API invalide")


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        m.REQUESTS.labels(endpoint, "500").inc()
        raise
    if endpoint != "/metrics":
        m.REQUESTS.labels(endpoint, str(response.status_code)).inc()
        m.LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready():
    if "agent" not in STATE:
        raise HTTPException(503, "index non chargé")
    s = STATE["settings"]
    return {
        "status": "ready",
        "chunks": len(STATE["agent"].retriever.chunks),
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model if s.llm_provider != "local" else "local-extractive-v1",
    }


@app.get("/metrics")
def metrics():
    body, ctype = m.render()
    return PlainTextResponse(body, media_type=ctype)


@app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask(req: AskRequest):
    s = STATE["settings"]
    if len(req.question) > s.max_question_chars:
        raise HTTPException(413, f"Question trop longue (max {s.max_question_chars} caractères)")
    rid = uuid.uuid4().hex[:12]
    # l'agent est synchrone (appels LLM bloquants) : exécution dans un pool de threads
    out = await run_in_threadpool(STATE["agent"].ask, req.question)
    if out["blocked"]:
        m.BLOCKED.labels("prompt_injection").inc()
    if out["answer"] == NO_ANSWER:
        m.NO_ANSWER.inc()
    if out["tool"]:
        m.TOOL_CALLS.labels(out["tool"]).inc()
    if out["fallback_used"]:
        m.FALLBACK.inc()
    if out["model"] and out["model"] != "tool":
        m.TOKENS.labels(out["model"], "prompt").inc(out["usage"]["prompt_tokens"])
        m.TOKENS.labels(out["model"], "completion").inc(out["usage"]["completion_tokens"])
        m.LLM_LATENCY.labels(out["model"]).observe(out["usage"]["llm_latency_ms"] / 1000)
    log.info(
        '{"request_id":"%s","latency_ms":%s,"tool":"%s","grounded":%s,"pii":"%s"}',
        rid,
        out["latency_ms"],
        out["tool"],
        str(out["grounded"]).lower(),
        ",".join(out["pii_masked"]),
    )
    return {"request_id": rid, **out}


@app.post("/v1/reindex", dependencies=[Depends(require_api_key)])
async def reindex():
    s = STATE["settings"]
    retriever = await run_in_threadpool(build_index, s)
    STATE["agent"] = build_agent(s, retriever)
    m.INDEX_CHUNKS.set(len(retriever.chunks))
    return {"status": "reindexed", "chunks": len(retriever.chunks)}
