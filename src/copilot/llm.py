"""Couche LLM interchangeable.

- `OpenAICompatibleLLM` : tout serveur exposant l'API /v1/chat/completions
  (vLLM, Ollama, llama.cpp server, TGI, Azure OpenAI...). Timeout, retries avec
  backoff exponentiel, comptage des tokens.
- `LocalExtractiveLLM` : moteur extractif déterministe sans GPU ni téléchargement,
  utilisé en développement, en CI et comme solution de repli (dégradation gracieuse).
- `FallbackLLM` : bascule automatique vers le moteur local si le LLM distant échoue.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from .config import Settings
from .prompts import NO_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .retriever import tokenize


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    fallback_used: bool = False
    prompt_version: str = PROMPT_VERSION


class LLMError(RuntimeError):
    pass


class LocalExtractiveLLM:
    """Sélectionne les phrases des extraits les plus pertinentes pour la question et
    les restitue avec leurs citations. Ne peut pas halluciner : tout est extrait."""

    name = "local-extractive-v1"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[str]) -> LLMResult:
        t0 = time.perf_counter()
        q = set(tokenize(question))
        scored = []
        for ci, ctx in enumerate(contexts, start=1):
            # une puce = une unité (elle porte son libellé, ex. « P1 – Critique ») ; sinon découpage en phrases
            units = [u for b in ctx.split(" • ") for u in ([b] if len(b) < 260 else re.split(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂ0-9])", b))]
            for pos, sent in enumerate(units):
                toks = set(tokenize(sent))
                overlap = len(q & toks)
                if overlap == 0 or len(sent) < 25:
                    continue
                # bonus aux unités chiffrées (délais, montants, seuils), pénalité selon le rang de l'extrait
                score = overlap / (len(q) ** 0.5) + 0.15 * bool(re.search(r"\d", sent)) - 0.02 * pos - 0.1 * (ci - 1)
                scored.append((score, ci, pos, sent.strip()))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0
        # on ne garde que les unités proches de la meilleure (évite le hors-sujet)
        kept = [x for x in scored[: self.max_sentences] if x[0] >= 0.75 * best]
        picked = sorted(kept, key=lambda x: (x[1], x[2]))
        if not picked or best <= 0.3:
            text = NO_ANSWER
        else:
            text = " ".join(f"{s.rstrip('.')}. [{ci}]" for _, ci, _, s in picked)
        prompt_tokens = sum(len(c.split()) for c in contexts) + len(question.split())
        return LLMResult(text, self.name, prompt_tokens, len(text.split()), time.perf_counter() - t0)


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        self.name = settings.llm_model
        self.client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=settings.llm_timeout_s,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            transport=transport,
        )

    def generate(self, question: str, contexts: list[str]) -> LLMResult:
        payload = {
            "model": self.s.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts)},
            ],
            "temperature": self.s.llm_temperature,
            "max_tokens": self.s.llm_max_tokens,
        }
        last_err: Exception | None = None
        for attempt in range(self.s.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.post("/chat/completions", json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {})
                return LLMResult(
                    text=data["choices"][0]["message"]["content"].strip(),
                    model=data.get("model", self.name),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=time.perf_counter() - t0,
                )
            except (httpx.HTTPError, LLMError, KeyError, ValueError) as e:
                last_err = e
                if attempt < self.s.llm_max_retries:
                    time.sleep(min(0.25 * 2**attempt, 2.0))
        raise LLMError(f"LLM indisponible après {self.s.llm_max_retries + 1} tentatives : {last_err}")


class FallbackLLM:
    def __init__(self, primary, fallback: LocalExtractiveLLM):
        self.primary, self.fallback = primary, fallback
        self.name = getattr(primary, "name", "primary")

    def generate(self, question: str, contexts: list[str]) -> LLMResult:
        try:
            return self.primary.generate(question, contexts)
        except LLMError:
            res = self.fallback.generate(question, contexts)
            res.fallback_used = True
            return res


def build_llm(settings: Settings, transport: httpx.BaseTransport | None = None):
    if settings.llm_provider == "local":
        return LocalExtractiveLLM()
    if settings.llm_provider == "openai":
        primary = OpenAICompatibleLLM(settings, transport=transport)
        return FallbackLLM(primary, LocalExtractiveLLM()) if settings.llm_fallback_local else primary
    raise ValueError(f"LLM_PROVIDER inconnu : {settings.llm_provider}")
