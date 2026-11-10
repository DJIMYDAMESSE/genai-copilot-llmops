"""Recherche hybride : BM25 (lexical) + similarité vectorielle TF-IDF, fusion RRF.

L'encodeur vectoriel est volontairement interchangeable : TF-IDF par défaut (aucun
téléchargement de modèle), remplaçable par un modèle d'embeddings (ex. bge-m3,
multilingual-e5) via la classe `Encoder`.
"""

from __future__ import annotations

import math
import pickle
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk

STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "et",
    "eux",
    "il",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "ma",
    "mais",
    "me",
    "meme",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "c",
    "d",
    "j",
    "l",
    "m",
    "n",
    "s",
    "t",
    "y",
    "est",
    "sont",
    "etre",
    "ete",
    "fait",
    "faut",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "comment",
    "combien",
    "quoi",
    "quand",
    "dois",
    "doit",
    "peut",
    "peux",
    "puis",
    "ai",
    "as",
    "avons",
    "avez",
    "ont",
    "suis",
    "es",
    "sommes",
    "etes",
    "cette",
    "cet",
    "ils",
    "elles",
    "chez",
    "si",
    "plus",
    "tout",
    "tous",
    "toute",
    "toutes",
    "apres",
    "avant",
    "entre",
    "sous",
    "lors",
    "afin",
    "ainsi",
    "alors",
    "donc",
    "car",
    "faire",
    "cas",
    "savoir",
    "the",
    "of",
    "and",
    "to",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _stem(tok: str) -> str:
    # Racinisation légère pour le français (pluriels et suffixes fréquents)
    for suf in ("ements", "ement", "ations", "ation", "euses", "euse", "eurs", "eur", "ees", "ee", "es", "s", "x"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", normalize(text))
    return [_stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            denom_norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in query:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + denom_norm)
        return out


class Encoder:
    """Encodeur vectoriel TF-IDF (mots + n-grammes de caractères), vecteurs normalisés L2."""

    def __init__(self) -> None:
        self.word = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    def fit(self, texts: list[str]) -> Encoder:
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        w = self.word.transform(texts).toarray()
        c = self.char.transform(texts).toarray()
        v = np.hstack([w, 0.5 * c])
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms == 0, 1, norms)


@dataclass
class Hit:
    chunk: Chunk
    score: float  # score RRF fusionné
    relevance: float  # similarité cosinus (0-1), utilisée pour le seuil de refus


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [f"{c.title} {c.section} {c.text}" for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.encoder = Encoder().fit(texts)
        self.matrix = self.encoder.encode(texts)

    def search(self, query: str, k: int = 4, rrf_k: int = 60) -> list[Hit]:
        bm = self.bm25.scores(tokenize(query))
        cos = self.matrix @ self.encoder.encode([query])[0]
        rank_bm = {i: r for r, i in enumerate(np.argsort(-bm))}
        rank_cos = {i: r for r, i in enumerate(np.argsort(-cos))}
        fused = {i: 1 / (rrf_k + rank_bm[i]) + 1 / (rrf_k + rank_cos[i]) for i in range(len(self.chunks))}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], float(fused[i]), float(cos[i])) for i in top]

    # Persistance de l'index (artefact versionnable, chargé au démarrage de l'API)
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> HybridRetriever:
        with open(path, "rb") as f:
            return pickle.load(f)
