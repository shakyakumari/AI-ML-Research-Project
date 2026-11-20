"""Evaluation harness + ablation runner (works on any QA file with the sample_qa.json schema)."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .metrics import anls, evidence_precision, evidence_recall, exact_match, rouge_l, token_f1
from .pipeline import HMDKGPipeline, PipelineConfig


def evaluate(pipe: HMDKGPipeline, qa: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in qa:
        out = pipe.answer(item["question"])
        pred, golds = out.result.answer, item["answer"]
        texts = [e.text for e in out.retrieval.evidence]
        row = {
            "id": item["id"], "type": item["type"], "pred": pred, "bucket": out.result.bucket,
            "primary": out.retrieval.primary, "n_evidence": len(texts),
            "recall": evidence_recall(texts, item.get("evidence", [])),
            "precision": evidence_precision(texts, item.get("evidence", [])),
        }
        if item["type"] == "descriptive":
            row["rougeL"] = rouge_l(pred, golds[0])
            row["em"] = row["anls"] = row["f1"] = None
        else:
            row["em"], row["anls"], row["f1"] = exact_match(pred, golds), anls(pred, golds), token_f1(pred, golds)
            row["rougeL"] = None
        rows.append(row)

    def mean(key: str, subset=None) -> Optional[float]:
        vals = [r[key] for r in (subset or rows) if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {
        "EM": mean("em"), "ANLS": mean("anls"), "F1": mean("f1"), "ROUGE-L": mean("rougeL"),
        "EvidenceRecall": mean("recall"), "EvidencePrecision": mean("precision"),
        "AvgEvidence": mean("n_evidence"),
        "per_type_EM": {t: mean("em", rs) for t, rs in by_type.items() if mean("em", rs) is not None},
        "rows": rows,
    }


def variants(base: Optional[PipelineConfig] = None) -> dict[str, PipelineConfig]:
    base = base or PipelineConfig()
    return {
        "Full (HM-DKG + QC-AER + CA-AAG)": base,
        "w/o HM-DKG + QC-AER (flat top-3)": replace(base, flat_retrieval=True),
        "w/o intent classifier (uniform traversal)": replace(base, retrieval=replace(base.retrieval, use_intent=False)),
        "w/o QG-GAT propagation": replace(base, retrieval=replace(base.retrieval, use_qggat=False)),
        "w/o cross-modal edges": replace(base, build=replace(base.build, enable_cross_modal=False)),
        "w/o semantic + cross-ref edges": replace(base, build=replace(base.build, enable_semantic=False, enable_crossref=False)),
        "w/o CA-AAG (fixed moderate control)": replace(base, adaptive_generation=False),
    }


def run_ablations(doc_text: str, qa: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, cfg in variants().items():
        pipe = HMDKGPipeline(cfg)
        pipe.index_text(doc_text)
        out[name] = evaluate(pipe, qa)
    return out


def load(doc_path: str | Path, qa_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    return Path(doc_path).read_text(encoding="utf-8"), json.loads(Path(qa_path).read_text(encoding="utf-8"))
