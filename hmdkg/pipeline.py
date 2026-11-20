"""End-to-end pipeline: document -> HM-DKG -> QC-AER -> CA-AAG, with ablation switches."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .builder import BuildConfig, build_graph, parse_markdown
from .encoders import HashingTextEncoder, PlaceholderVisualEncoder
from .generation import (CONTROL, ComplexityEstimator, EvidenceItem, ExtractiveGenerator, GenerationRequest,
                         GenerationResult, select_format)
from .graph import HMDKG
from .retrieval import QCAER, FlatRetriever, RetrievalConfig, RetrievalResult


@dataclass
class PipelineConfig:
    dim: int = 1024
    build: BuildConfig = field(default_factory=BuildConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    adaptive_generation: bool = True  # False => fixed 'moderate' control ("w/o CA-AAG")
    flat_retrieval: bool = False  # True => flat top-k baseline ("w/o HM-DKG + QC-AER")
    flat_k: int = 3


@dataclass
class PipelineOutput:
    question: str
    result: GenerationResult
    retrieval: RetrievalResult
    complexity: float
    request: GenerationRequest


class HMDKGPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None, generator=None):
        self.cfg = cfg or PipelineConfig()
        self.encoder = HashingTextEncoder(self.cfg.dim)
        self.visual = PlaceholderVisualEncoder(self.encoder)
        self.generator = generator or ExtractiveGenerator()
        self.estimator = ComplexityEstimator()
        self.graph: Optional[HMDKG] = None

    # -- indexing ------------------------------------------------------------------------------
    def index_text(self, text: str, title: str = "document") -> HMDKG:
        elements = parse_markdown(text)
        if title == "document":
            h1 = next((e.text for e in elements if e.kind == "heading"), title)
            title = h1
        self.graph = build_graph(elements, self.encoder, self.visual, self.cfg.build, title)
        if self.cfg.flat_retrieval:
            self.retriever = FlatRetriever(self.graph, self.encoder, self.cfg.flat_k)
        else:
            self.retriever = QCAER(self.graph, self.encoder, self.cfg.retrieval)
        return self.graph

    def index_file(self, path: Union[str, Path]) -> HMDKG:
        return self.index_text(Path(path).read_text(encoding="utf-8"))

    # -- answering -----------------------------------------------------------------------------
    def answer(self, question: str) -> PipelineOutput:
        assert self.graph is not None, "call index_text/index_file first"
        ret = self.retriever.retrieve(question)
        hops = ret.trace.get("hops_used", 0)
        score, bucket = self.estimator.estimate(question, ret.primary, len(ret.evidence), hops)
        if not self.cfg.adaptive_generation or self.cfg.flat_retrieval:
            bucket = "moderate"
        ctrl = CONTROL[bucket]
        evs = ret.evidence if ctrl.evidence_k is None else ret.evidence[: ctrl.evidence_k]
        items = [EvidenceItem(e.node_id, e.type, e.text, e.score, self.graph.nodes[e.node_id].page, self.graph.nodes[e.node_id].meta) for e in evs]
        ids = {e.node_id for e in evs}
        gctx = []
        if ctrl.use_graph:
            for e in evs:
                for adj in self.graph.adjacent(e.node_id):
                    if adj.nbr in ids and e.node_id < adj.nbr:
                        gctx.append(f"{e.node_id}-{adj.relation}-{adj.nbr}")
        fmt = select_format(ret.primary, {i.type for i in items}) if self.cfg.adaptive_generation else "short_span"
        req = GenerationRequest(question, ret.primary, bucket, ctrl, fmt, items, gctx)
        return PipelineOutput(question, self.generator.generate(req), ret, score, req)
