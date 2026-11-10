"""Ingestion : lecture des documents Markdown et découpage en chunks par section."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Découpe un texte trop long en fenêtres glissantes, en coupant sur les phrases."""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    parts, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > size:
            parts.append(current.strip())
            current = current[-overlap:] + " " + s if overlap else s
        else:
            current = f"{current} {s}" if current else s
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(doc_id: str, content: str, size: int = 700, overlap: int = 120) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc_id
    # Découpage par titres de niveau 2 : une section = une unité de sens
    blocks = re.split(r"^##\s+", content, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for i, block in enumerate(blocks):
        if i == 0:
            section, body = "", re.sub(r"^#\s+.+$", "", block, flags=re.MULTILINE)
        else:
            section, _, body = block.partition("\n")
        body = re.sub(r"\n\s*-\s+", " • ", body.replace("**", ""))  # conserve les puces
        body = re.sub(r"\s+", " ", body).strip(" •")
        if len(body) < 40:
            continue
        for j, part in enumerate(_split_long(body, size, overlap)):
            cid = hashlib.sha1(f"{doc_id}|{section}|{j}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, doc_id, title, section.strip(), part))
    return chunks


def load_corpus(corpus_dir: Path, size: int = 700, overlap: int = 120) -> list[Chunk]:
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"Aucun document .md trouvé dans {corpus_dir}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_markdown(f.stem, f.read_text(encoding="utf-8"), size, overlap))
    return chunks
