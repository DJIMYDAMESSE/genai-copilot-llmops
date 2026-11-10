"""Assemblage des composants (index + LLM + agent)."""

from __future__ import annotations

from .agent import CopilotAgent
from .config import Settings
from .ingest import load_corpus
from .llm import build_llm
from .retriever import HybridRetriever


def build_index(s: Settings) -> HybridRetriever:
    retriever = HybridRetriever(load_corpus(s.corpus_dir, s.chunk_size, s.chunk_overlap))
    retriever.save(s.index_path)
    return retriever


def load_or_build_index(s: Settings) -> HybridRetriever:
    if s.index_path.exists():
        return HybridRetriever.load(s.index_path)
    return build_index(s)


def build_agent(s: Settings, retriever: HybridRetriever | None = None, transport=None) -> CopilotAgent:
    return CopilotAgent(retriever or load_or_build_index(s), build_llm(s, transport=transport), s)
