"""Agent RAG orchestré avec LangGraph.

guard ──(injection)──────────────────────────────► refuse
  │
  ▼
route ──(outil métier)──► tool ──┐
  │                              ▼
  └────────────────────────► retrieve ──(non pertinent)──► refuse
                                 │
                                 ▼
                             generate ──► verify ──► END
"""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .config import Settings
from .guardrails import check_input
from .llm import LLMResult
from .prompts import NO_ANSWER, PROMPT_VERSION
from .retriever import Hit, HybridRetriever
from .tools import ToolResult, detect_tool

INJECTION_REFUSAL = "Je ne peux pas suivre cette demande : elle tente de modifier mes règles de fonctionnement."


class AgentState(TypedDict, total=False):
    question: str
    clean_question: str
    pii: list[str]
    blocked: bool
    tool: ToolResult | None
    hits: list[Hit]
    answer: str
    llm: LLMResult | None
    grounded: bool
    steps: list[str]


class CopilotAgent:
    def __init__(self, retriever: HybridRetriever, llm: Any, settings: Settings):
        self.retriever, self.llm, self.s = retriever, llm, settings
        self.graph = self._build()

    # --- noeuds -----------------------------------------------------------
    def _guard(self, st: AgentState) -> AgentState:
        g = check_input(st["question"])
        return {"clean_question": g.text, "pii": g.pii_found, "blocked": g.injection, "steps": ["guard"]}

    def _route(self, st: AgentState) -> AgentState:
        return {"tool": detect_tool(st["clean_question"]), "steps": st["steps"] + ["route"]}

    def _tool(self, st: AgentState) -> AgentState:
        return {"steps": st["steps"] + [f"tool:{st['tool'].name}"]}

    def _retrieve(self, st: AgentState) -> AgentState:
        hits = self.retriever.search(st["clean_question"], k=self.s.top_k)
        if st.get("tool"):  # on remonte en tête le document source de l'outil
            hits.sort(key=lambda h: h.chunk.doc_id != st["tool"].source_doc)
        return {"hits": hits, "steps": st["steps"] + ["retrieve"]}

    def _generate(self, st: AgentState) -> AgentState:
        hits = st["hits"]
        if st.get("tool"):
            idx = next((i for i, h in enumerate(hits, 1) if h.chunk.doc_id == st["tool"].source_doc), 1)
            return {"answer": f"{st['tool'].output} [{idx}]", "llm": None, "steps": st["steps"] + ["generate:tool"]}
        res = self.llm.generate(st["clean_question"], [h.chunk.text for h in hits])
        return {"answer": res.text, "llm": res, "steps": st["steps"] + [f"generate:{res.model}"]}

    def _verify(self, st: AgentState) -> AgentState:
        ans, n = st["answer"], len(st["hits"])
        cited = {int(c) for c in re.findall(r"\[(\d+)\]", ans)}
        valid = bool(cited) and all(1 <= c <= n for c in cited)
        if ans.strip() == NO_ANSWER:
            return {"grounded": True, "steps": st["steps"] + ["verify:no_answer"]}
        if not valid:  # politique : pas de source = pas de réponse
            return {"answer": NO_ANSWER, "grounded": False, "steps": st["steps"] + ["verify:rejected"]}
        return {"grounded": True, "steps": st["steps"] + ["verify:ok"]}

    def _refuse(self, st: AgentState) -> AgentState:
        msg = INJECTION_REFUSAL if st.get("blocked") else NO_ANSWER
        return {"answer": msg, "grounded": True, "hits": st.get("hits", []), "steps": st["steps"] + ["refuse"]}

    # --- graphe -----------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in [
            ("guard", self._guard),
            ("route", self._route),
            ("tool", self._tool),
            ("retrieve", self._retrieve),
            ("generate", self._generate),
            ("verify", self._verify),
            ("refuse", self._refuse),
        ]:
            g.add_node(name, fn)
        g.set_entry_point("guard")
        g.add_conditional_edges("guard", lambda s: "refuse" if s["blocked"] else "route")
        g.add_conditional_edges("route", lambda s: "tool" if s.get("tool") else "retrieve")
        g.add_edge("tool", "retrieve")
        g.add_conditional_edges(
            "retrieve",
            lambda s: "generate" if s.get("tool") or (s["hits"] and s["hits"][0].relevance >= self.s.min_relevance) else "refuse",
        )
        g.add_edge("generate", "verify")
        g.add_edge("verify", END)
        g.add_edge("refuse", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        st = self.graph.invoke({"question": question, "steps": []})
        hits = st.get("hits", [])
        cited = {int(c) for c in re.findall(r"\[(\d+)\]", st["answer"])}
        llm: LLMResult | None = st.get("llm")
        return {
            "answer": st["answer"],
            "sources": [
                {"id": i, "doc_id": h.chunk.doc_id, "source": h.chunk.source, "relevance": round(h.relevance, 3), "cited": i in cited}
                for i, h in enumerate(hits, start=1)
            ],
            "grounded": st.get("grounded", False),
            "tool": st["tool"].name if st.get("tool") else None,
            "pii_masked": st.get("pii", []),
            "blocked": st.get("blocked", False),
            "model": llm.model if llm else ("tool" if st.get("tool") else None),
            "fallback_used": bool(llm and llm.fallback_used),
            "prompt_version": PROMPT_VERSION,
            "usage": {
                "prompt_tokens": llm.prompt_tokens if llm else 0,
                "completion_tokens": llm.completion_tokens if llm else 0,
                "llm_latency_ms": round(llm.latency_s * 1000, 2) if llm else 0.0,
            },
            "steps": st["steps"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
