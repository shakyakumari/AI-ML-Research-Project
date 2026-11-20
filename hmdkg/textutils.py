"""Small text helpers shared by the encoder, retriever and generator."""
from __future__ import annotations

import re
from typing import Optional

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*")

STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been by with as from that this "
    "these those it its what which who whom whose when where how does do did has have had "
    "will would can could than then there their".split()
)

# Numbers such as 1,800 / 187.9 / 2023 (comma-grouped alternative first).
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")


def stem(tok: str) -> str:
    """Very light plural stemmer (enough to match 'employs' with 'employ')."""
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, drop_stop: bool = False) -> list[str]:
    toks = [stem(t.lower()) for t in _TOKEN_RE.findall(text)]
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_number(s: str) -> Optional[float]:
    m = NUM_RE.search(s or "")
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None
