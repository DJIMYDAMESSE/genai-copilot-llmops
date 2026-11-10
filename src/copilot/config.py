"""Configuration centralisée, pilotée par variables d'environnement (12-factor)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Données et index
    corpus_dir: Path = field(default_factory=lambda: Path(_env("COPILOT_CORPUS_DIR", str(ROOT / "data" / "corpus"))))
    index_path: Path = field(default_factory=lambda: Path(_env("COPILOT_INDEX_PATH", str(ROOT / "artifacts" / "index.pkl"))))
    chunk_size: int = int(_env("COPILOT_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(_env("COPILOT_CHUNK_OVERLAP", "120"))

    # Recherche
    top_k: int = int(_env("COPILOT_TOP_K", "4"))
    min_relevance: float = float(_env("COPILOT_MIN_RELEVANCE", "0.12"))

    # LLM : "local" (moteur extractif de secours) ou "openai" (API compatible OpenAI :
    # vLLM, Ollama, llama.cpp server, Azure OpenAI...)
    llm_provider: str = _env("LLM_PROVIDER", "local")
    llm_base_url: str = _env("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = _env("LLM_MODEL", "qwen2.5:1.5b-instruct")
    llm_api_key: str = _env("LLM_API_KEY", "not-needed")
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "60"))
    llm_max_retries: int = int(_env("LLM_MAX_RETRIES", "2"))
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "400"))
    # Si le LLM distant échoue, bascule sur le moteur local au lieu d'une erreur 5xx
    llm_fallback_local: bool = _env("LLM_FALLBACK_LOCAL", "true").lower() == "true"

    # API
    api_key: str = _env("COPILOT_API_KEY", "")  # vide = pas d'authentification (dev)
    max_question_chars: int = int(_env("COPILOT_MAX_QUESTION_CHARS", "1000"))


def get_settings() -> Settings:
    return Settings()
