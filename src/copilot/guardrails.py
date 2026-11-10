"""Garde-fous : masquage des données personnelles et détection d'injection de prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retriever import normalize

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "IBAN": re.compile(r"\bFR\d{2}(?:\s?[0-9A-Z]{4}){5}\s?[0-9A-Z]{3}\b", re.IGNORECASE),
    "TELEPHONE": re.compile(r"(?<!\d)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)"),
    "NIR": re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"),
}

INJECTION_PATTERNS = [
    r"ignore[rz]? (?:toutes? )?(?:les |tes |vos )?(?:instructions|regles|consignes)",
    r"ignore (?:all |the |your )?(?:previous |prior )?instructions",
    r"oublie[rz]? (?:toutes? )?(?:les |tes )?(?:instructions|regles|consignes)",
    r"(?:revele|affiche|montre|donne)[rz]? (?:moi )?(?:ton|le|tes|les) (?:prompt|instructions systeme|system prompt)",
    r"tu es (?:maintenant|desormais) ",
    r"\b(?:jailbreak|dan mode|mode developpeur)\b",
    r"system prompt",
]
_INJ = [re.compile(p) for p in INJECTION_PATTERNS]


@dataclass
class GuardResult:
    text: str
    pii_found: list[str] = field(default_factory=list)
    injection: bool = False


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in PII_PATTERNS.items():
        if pat.search(text):
            found.append(label)
            text = pat.sub(f"<{label}>", text)
    return text, found


def detect_injection(text: str) -> bool:
    t = normalize(text)
    return any(p.search(t) for p in _INJ)


def check_input(text: str) -> GuardResult:
    masked, pii = mask_pii(text)
    return GuardResult(masked, pii, detect_injection(text))
