"""Faux serveur LLM compatible OpenAI (même contrat que vLLM / Ollama).

Sert à tester le chemin réseau complet API → serveur LLM sans GPU : latence simulée,
erreurs injectables, comptage de tokens. Il NE génère PAS de texte intelligent : il
reformule l'extrait [1] reçu dans le prompt.

    python scripts/mock_llm_server.py --port 8001 --latency-ms 400 --error-rate 0.0
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re

import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()
CFG = {"latency_ms": 300, "error_rate": 0.0}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"data": [{"id": "mock-qwen2.5-7b-instruct", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    await asyncio.sleep(CFG["latency_ms"] / 1000 * random.uniform(0.7, 1.3))
    if random.random() < CFG["error_rate"]:
        raise HTTPException(503, "overloaded")
    user = body["messages"][-1]["content"]
    question = user.rsplit("QUESTION :", 1)[-1].lower()
    qwords = {w for w in re.findall(r"\w{4,}", question)}
    m = re.search(r"\[1\] (.+?)(?:\n\n\[2\]|\n\nQUESTION)", user, re.DOTALL)
    units = re.split(r"(?<=[.!?])\s+| • ", m.group(1).strip()) if m else []
    # choisit l'unité de l'extrait [1] qui recouvre le plus la question
    best = max(units, key=lambda u: len(qwords & set(re.findall(r"\w{4,}", u.lower()))), default="")
    answer = (
        f"D'après la documentation : {best.strip(' •').rstrip('.')}. [1]"
        if best
        else "Je ne trouve pas cette information dans la documentation."
    )
    return {
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(user.split()), "completion_tokens": len(answer.split())},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--latency-ms", type=int, default=300)
    ap.add_argument("--error-rate", type=float, default=0.0)
    a = ap.parse_args()
    CFG.update(latency_ms=a.latency_ms, error_rate=a.error_rate)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
