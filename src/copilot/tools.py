"""Outils métier appelables par l'agent (calculs déterministes, jamais laissés au LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .retriever import normalize

# Données de référence issues de la fiche produit Soleil+ (corpus 09)
KIT_PRICES = {3: 7900, 6: 12900, 9: 17500}
BATTERY_PRICES = {5: 4200, 10: 7300}
YIELD_KWH_PER_KWC = (1000, 1400)
TELEWORK_DAILY, TELEWORK_MONTHLY_CAP = 2.50, 55.0


@dataclass
class ToolResult:
    name: str
    output: str
    source_doc: str


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def quote_soleil_plus(kwc: int, battery_kwh: int | None = None) -> ToolResult:
    if kwc not in KIT_PRICES:
        return ToolResult(
            "devis_soleil_plus", f"Aucun kit de {kwc} kWc : les puissances disponibles sont 3, 6 et 9 kWc.", "09_offre_autoconsommation"
        )
    total = KIT_PRICES[kwc]
    detail = f"kit {kwc} kWc {_fmt(KIT_PRICES[kwc])} €"
    if battery_kwh:
        if battery_kwh not in BATTERY_PRICES:
            return ToolResult(
                "devis_soleil_plus",
                f"Aucune batterie de {battery_kwh} kWh : les capacités disponibles sont 5 et 10 kWh.",
                "09_offre_autoconsommation",
            )
        total += BATTERY_PRICES[battery_kwh]
        detail += f" + batterie {battery_kwh} kWh {_fmt(BATTERY_PRICES[battery_kwh])} €"
    return ToolResult("devis_soleil_plus", f"Prix indicatif : {_fmt(total)} € TTC posé ({detail}).", "09_offre_autoconsommation")


def estimate_production(kwc: float) -> ToolResult:
    lo, hi = (kwc * y for y in YIELD_KWH_PER_KWC)
    return ToolResult(
        "estimation_production",
        f"Production estimée pour {kwc:g} kWc : entre {_fmt(lo)} et {_fmt(hi)} kWh par an selon la région.",
        "09_offre_autoconsommation",
    )


def telework_allowance(days: int) -> ToolResult:
    amount = min(days * TELEWORK_DAILY, TELEWORK_MONTHLY_CAP)
    capped = " (plafond mensuel de 55 € atteint)" if days * TELEWORK_DAILY > TELEWORK_MONTHLY_CAP else ""
    return ToolResult(
        "indemnite_teletravail",
        f"Indemnité pour {days} jours télétravaillés dans le mois : {amount:.2f} €{capped}.".replace(".", ",", 1),
        "05_teletravail",
    )


def detect_tool(question: str) -> ToolResult | None:
    """Routage déterministe vers un outil à partir de l'intention et des paramètres."""
    q = normalize(question)
    kwc = re.search(r"(\d+(?:[.,]\d+)?)\s*kwc", q)
    bat = re.search(r"batterie(?: de)?\s*(\d+)\s*kwh|(\d+)\s*kwh", q)
    days = re.search(r"(\d+)\s*jours?", q)
    if kwc and re.search(r"prix|cout|coute|devis|tarif|combien", q) and not re.search(r"produi|production", q):
        b = int(next(g for g in bat.groups() if g)) if bat else None
        return quote_soleil_plus(int(float(kwc.group(1).replace(",", "."))), b)
    if kwc and re.search(r"produi|production|kwh par an|kwh/an", q):
        return estimate_production(float(kwc.group(1).replace(",", ".")))
    if days and re.search(r"indemnit|teletravail", q) and re.search(r"indemnit|combien|montant", q):
        return telework_allowance(int(days.group(1)))
    return None
