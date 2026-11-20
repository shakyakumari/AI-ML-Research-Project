"""Modality encoders.

The paper specifies RoBERTa/DeBERTa (text), ViT/CLIP (figures) and TUTA/TableFormer (tables).
Those need PyTorch and pretrained weights, so this reference implementation ships dependency-free
stand-ins that expose the same interfaces:

* ``HashingTextEncoder``      - signed feature hashing with IDF weights (deterministic, no training)
* ``TableEncoder``            - cell-level + row/column structural encoding built on a text encoder
* ``PlaceholderVisualEncoder``- joint-space stand-in for CLIP (see its docstring)

Swap any of them for a neural encoder by implementing ``encode(text) -> np.ndarray`` (unit norm).
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Optional, Sequence

import numpy as np

from .textutils import tokenize


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class HashingTextEncoder:
    def __init__(self, dim: int = 1024, bigrams: bool = True):
        self.dim = dim
        self.bigrams = bigrams
        self.idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "HashingTextEncoder":
        """Fit IDF weights on the document's own elements (no labels needed)."""
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(tokenize(t, drop_stop=True)))
        n = max(len(texts), 1)
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}
        return self

    def _slot(self, feat: str) -> tuple[int, float]:
        h = zlib.crc32(feat.encode("utf-8"))
        return h % self.dim, (1.0 if (h >> 16) & 1 else -1.0)

    def encode(self, text: str) -> np.ndarray:
        toks = tokenize(text, drop_stop=True)
        v = np.zeros(self.dim, dtype=np.float64)
        for t in toks:
            i, s = self._slot(t)
            v[i] += s * self.idf.get(t, 1.0)
        if self.bigrams:
            for a, b in zip(toks, toks[1:]):
                i, s = self._slot(a + "_" + b)
                v[i] += 0.5 * s
        return _unit(v)

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts]) if texts else np.zeros((0, self.dim))


def linearize_table(header: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> str:
    parts = [caption.strip(), "Columns: " + ", ".join(header) + "."]
    for r in rows:
        cells = ", ".join(f"{header[j]} {r[j]}" for j in range(1, min(len(r), len(header))))
        parts.append(f"{r[0]}: {cells}.")
    return " ".join(p for p in parts if p)


class TableEncoder:
    """Cell-level + row/column structural encoding.

    Each cell is encoded as "<row label> <column header> <value>", so a cell embedding knows its
    row and column. The table embedding mixes the mean cell embedding with a caption+header
    embedding. Cell embeddings are returned too, enabling cell-level zooming.
    """

    def __init__(self, text_encoder: HashingTextEncoder):
        self.enc = text_encoder

    def encode(self, header: Sequence[str], rows: Sequence[Sequence[str]], caption: str = ""):
        cells, cell_embs = [], []
        for ri, r in enumerate(rows):
            for ci in range(1, min(len(r), len(header))):
                cells.append((ri, ci))
                cell_embs.append(self.enc.encode(f"{r[0]} {header[ci]} {r[ci]}"))
        cell_mat = np.stack(cell_embs) if cell_embs else np.zeros((0, self.enc.dim))
        head = self.enc.encode(f"{caption} " + " ".join(header))
        body = _unit(cell_mat.mean(axis=0)) if len(cell_mat) else head
        return _unit(0.6 * body + 0.4 * head), cells, cell_mat


class PlaceholderVisualEncoder:
    """Stand-in for a CLIP/ViT image encoder.

    No pixels are read. The figure embedding is a mix of the caption embedding and a hashed
    embedding of the file name, i.e. it simulates an image encoder whose output lives in a joint
    image-text space. If real image features are supplied (``features``, unit-normalised and of the
    same dimension) they are used instead. Replace with CLIP for real experiments.
    """

    def __init__(self, text_encoder: HashingTextEncoder):
        self.enc = text_encoder

    def encode(self, path: str, alt: str = "", features: Optional[np.ndarray] = None) -> np.ndarray:
        if features is not None:
            return _unit(np.asarray(features, dtype=np.float64))
        stem = path.replace("/", " ").replace("_", " ").replace(".", " ")
        return _unit(0.6 * self.enc.encode(alt) + 0.4 * self.enc.encode(stem))
