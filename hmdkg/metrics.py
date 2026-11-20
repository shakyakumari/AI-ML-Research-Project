"""Evaluation metrics: ANLS, Exact Match, token F1, ROUGE-L, evidence recall / precision."""
from __future__ import annotations

import re
from typing import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 1,800 -> 1800
    s = re.sub(r"[^\w\s.]", " ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: Sequence[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for one question (max over gold answers)."""
    p = normalize_answer(pred)
    best = 0.0
    for g in golds:
        g = normalize_answer(g)
        denom = max(len(p), len(g))
        nl = levenshtein(p, g) / denom if denom else 0.0
        best = max(best, 1.0 - nl if nl < tau else 0.0)
    return best


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds: Sequence[str]) -> float:
    def f1(a: list[str], b: list[str]) -> float:
        common = sum(min(a.count(t), b.count(t)) for t in set(a))
        if not common:
            return 0.0
        pr, rc = common / len(a), common / len(b)
        return 2 * pr * rc / (pr + rc)

    p = normalize_answer(pred).split()
    return max((f1(p, normalize_answer(g).split()) for g in golds if p), default=0.0)


def rouge_l(pred: str, ref: str) -> float:
    a, b = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if not lcs:
        return 0.0
    pr, rc = lcs / len(a), lcs / len(b)
    return 2 * pr * rc / (pr + rc)


def evidence_recall(evidence_texts: Iterable[str], gold_substrings: Sequence[str]) -> float:
    texts = [t.lower() for t in evidence_texts]
    if not gold_substrings:
        return 1.0
    return sum(any(g.lower() in t for t in texts) for g in gold_substrings) / len(gold_substrings)


def evidence_precision(evidence_texts: Sequence[str], gold_substrings: Sequence[str]) -> float:
    if not evidence_texts:
        return 0.0
    return sum(any(g.lower() in t.lower() for g in gold_substrings) for t in evidence_texts) / len(evidence_texts)
