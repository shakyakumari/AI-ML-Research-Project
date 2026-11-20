"""CA-AAG: Complexity-Aware Adaptive Answer Generation.

Contains (1) the complexity estimator, (2) the control table (length / evidence / reasoning),
(3) output-format selection, (4) a prompt builder for a multimodal LLM, and (5) generators:
``ExtractiveGenerator`` (rule-based reader so the whole pipeline runs offline) and ``LLMGenerator``
(wrapper around any ``fn(prompt, max_tokens) -> str``, e.g. a fine-tuned LLaVA/Qwen-VL client).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .graph import HMDKG, NodeType
from .textutils import NUM_RE, parse_number, split_sentences, tokenize

BUCKETS = ("simple", "moderate", "complex", "highly_complex")


@dataclass(frozen=True)
class ControlConfig:
    max_tokens: int
    evidence_k: Optional[int]  # None = full retrieved subgraph
    reasoning: str  # none | single_step | multi_step | structured
    use_graph: bool = False


CONTROL: dict[str, ControlConfig] = {
    "simple": ControlConfig(32, 1, "none"),
    "moderate": ControlConfig(128, 3, "single_step"),
    "complex": ControlConfig(512, 5, "multi_step", use_graph=True),
    "highly_complex": ControlConfig(1024, None, "structured", use_graph=True),
}

INTENT_BASE = {"factoid": 0.0, "numerical": 0.8, "comparative": 1.2, "crossmodal": 1.0, "descriptive": 2.0, "multihop": 2.5}


class ComplexityEstimator:
    """C = f(intent_label, evidence_count, graph_depth, q_length) -> bucket (rule-based; to be learned)."""

    def estimate(self, question: str, primary: str, n_evidence: int, hops_used: int) -> tuple[float, str]:
        c = INTENT_BASE.get(primary, 0.0)
        c += 0.4 * (hops_used >= 2) + 0.3 * (n_evidence >= 5) + 0.3 * (len(question.split()) > 15)
        bucket = "simple" if c < 0.75 else "moderate" if c < 1.75 else "complex" if c < 2.75 else "highly_complex"
        return c, bucket


@dataclass
class EvidenceItem:
    node_id: str
    type: NodeType
    text: str
    score: float
    page: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    question: str
    primary: str
    bucket: str
    control: ControlConfig
    fmt: str
    evidence: list[EvidenceItem]
    graph_context: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    answer: str  # short answer used for EM / ANLS
    text: str  # full response (reasoning chain / citations / units) within the token budget
    fmt: str
    bucket: str
    steps: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    structured: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False


def select_format(primary: str, types: set[NodeType]) -> str:
    if primary == "numerical":
        return "cell_value" if NodeType.TABLE in types else "number_with_unit"
    if primary == "comparative":
        return "ranked_list"
    if primary == "descriptive":
        return "paragraph_with_citations"
    if primary == "multihop":
        return "reasoned_answer"
    if primary == "crossmodal" or types & {NodeType.FIG, NodeType.CAPTION}:
        return "grounded_figure"
    if NodeType.TABLE in types:
        return "cell_value"
    return "short_span"


def build_prompt(req: GenerationRequest) -> str:
    """[CLS] Q [SEP] C_token [SEP] e_1 [SEP] ... e_k [SEP]  (+ graph structure and visual-token placeholders)."""
    parts = ["[CLS]", req.question, "[SEP]", f"<COMPLEXITY={req.bucket}>", "[SEP]"]
    for i, e in enumerate(req.evidence, 1):
        parts += [f"[e{i}|{e.type.value}|p{e.page}]", e.text, "[SEP]"]
        if e.type in (NodeType.FIG, NodeType.TABLE):
            parts.append(f"<vis:{e.node_id}>")
    if req.graph_context:
        parts += ["[GRAPH]", "; ".join(req.graph_context), "[SEP]"]
    parts.append(f"[FORMAT={req.fmt}] [REASONING={req.control.reasoning}] [MAX_TOKENS={req.control.max_tokens}]")
    return " ".join(parts)


# --------------------------------------------------------------------------------------------
# Span / table helpers
# --------------------------------------------------------------------------------------------
_NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
_CAP = re.compile(r"\b[A-Z][A-Za-z]+\b")
_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}\b")
_PREP_PLACE = re.compile(r"\b(?:in|at|to|from|near)\s+([A-Z][A-Za-z]+(?:,? [A-Z][A-Za-z]+)*)")
_LEAD = {"The", "A", "An", "This", "That", "These", "Those"}
_LOWER_WORDS = ("lower", "lowest", "smaller", "smallest", "least", "fewer", "less", "minimum")


def _qtokens(question: str) -> set[str]:
    return set(tokenize(question))


def extract_span(question: str, sentence: str) -> Optional[str]:
    """Heuristic answer-span extraction keyed on the question word. Returns None if nothing fits."""
    ql, qt = question.lower(), _qtokens(question)
    first = re.search(r"\b(who|whom|when|where|which|what)\b", ql)
    wh = first.group(1) if first else ""
    if "how many" in ql or "how much" in ql:
        for m in NUM_RE.finditer(sentence):
            if m.group().replace(",", "") not in qt and not _YEAR.fullmatch(m.group()):
                return m.group()
        return None
    if wh in ("who", "whom"):
        for m in _NAME.finditer(sentence):
            text = m.group()
            if m.start() == 0 and text.split()[0] in _LEAD:
                text = " ".join(text.split()[1:])
            if len(text.split()) >= 2 and not set(tokenize(text)) <= qt:
                return text
        return None
    if wh == "when":
        for m in _YEAR.finditer(sentence):
            if m.group() not in qt:
                return m.group()
        return None
    if wh == "where":
        for m in _PREP_PLACE.finditer(sentence):
            if not set(tokenize(m.group(1))) <= qt:
                return m.group(1)
        return None
    if wh in ("which", "what"):
        for m in _CAP.finditer(sentence):
            if m.start() == 0 or m.group() in _LEAD:
                continue
            if not set(tokenize(m.group())) <= qt:
                return m.group()
    return None


def table_lookup(question: str, header: list[str], rows: list[list[str]]) -> Optional[dict[str, Any]]:
    qt = set(tokenize(question, drop_stop=True))
    col_scores = [0] + [len(set(tokenize(h, drop_stop=True)) & qt) for h in header[1:]]
    row_scores = [len(set(tokenize(r[0], drop_stop=True)) & qt) for r in rows]
    if not col_scores or max(col_scores) == 0 or not row_scores or max(row_scores) == 0:
        return None
    c, r = col_scores.index(max(col_scores)), row_scores.index(max(row_scores))
    return {"row": r, "col": c, "row_label": rows[r][0], "col_header": header[c], "value": rows[r][c]}


def rank_rows(question: str, header: list[str], rows: list[list[str]]) -> Optional[dict[str, Any]]:
    qt = set(tokenize(question, drop_stop=True))
    col_scores = [0] + [len(set(tokenize(h, drop_stop=True)) & qt) for h in header[1:]]
    if max(col_scores) == 0:
        return None
    c = col_scores.index(max(col_scores))
    matched = [i for i, r in enumerate(rows) if set(tokenize(r[0], drop_stop=True)) & qt] or list(range(len(rows)))
    vals = [(rows[i][0], parse_number(rows[i][c])) for i in matched]
    vals = [(l, v) for l, v in vals if v is not None]
    if not vals:
        return None
    low = any(w in question.lower() for w in _LOWER_WORDS)
    ranked = sorted(vals, key=lambda lv: lv[1], reverse=not low)
    return {"col_header": header[c], "ranked": ranked, "winner": ranked[0][0]}


def _unit_from_caption(caption: str) -> str:
    m = re.search(r"\(([^)]*)\)", caption or "")
    return m.group(1) if m else ""


def _truncate(text: str, max_tokens: int) -> tuple[str, bool]:
    words = text.split()
    return (text, False) if len(words) <= max_tokens else (" ".join(words[:max_tokens]), True)


def _overlap(question: str, sentence: str) -> float:
    qt = set(tokenize(question, drop_stop=True))
    st = set(tokenize(sentence, drop_stop=True))
    return len(qt & st) / (1 + 0.05 * len(st))


# --------------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------------
class ExtractiveGenerator:
    """Rule-based reader implementing the CA-AAG control logic without an LLM."""

    def generate(self, req: GenerationRequest) -> GenerationResult:
        ev = req.evidence
        cites = [f"e{i}" for i in range(1, len(ev) + 1)]
        steps: list[str] = []
        structured: dict[str, Any] = {}
        answer: Optional[str] = None
        q = req.question

        sents = [(i, s) for i, e in enumerate(ev) if e.type in (NodeType.PARA, NodeType.CAPTION) for s in split_sentences(e.text)]

        def best_sentence(pool=None):
            pool = pool if pool is not None else sents
            return max(pool, key=lambda isn: _overlap(q, isn[1]) + 0.05 * ev[isn[0]].score, default=None)

        tables = [(i, e) for i, e in enumerate(ev) if e.type == NodeType.TABLE]
        figs = [(i, e) for i, e in enumerate(ev) if e.type in (NodeType.FIG, NodeType.CAPTION)]

        if req.primary == "descriptive":
            ranked = sorted(sents, key=lambda isn: -_overlap(q, isn[1]))
            n = {"simple": 1, "moderate": 3, "complex": 5, "highly_complex": 8}[req.bucket]
            chosen = sorted(ranked[:n], key=lambda isn: (isn[0], sents.index(isn)))
            answer = " ".join(s for _, s in chosen)
            body = " ".join(f"{s} [e{i + 1}]" for i, s in chosen)
            steps = [f"Select {len(chosen)} salient sentences from {len({i for i, _ in chosen})} evidence nodes."]
            full = body
        elif req.primary == "comparative" and tables:
            i, t = tables[0]
            r = rank_rows(q, t.meta["header"], t.meta["rows"])
            if r:
                answer = r["winner"]
                structured = {"ranking": r["ranked"], "column": r["col_header"]}
                full = "Ranking by " + r["col_header"] + ": " + " > ".join(f"{l} ({v:g})" for l, v in r["ranked"]) + f" [e{i + 1}]"
                steps = [f"Read column '{r['col_header']}' of {t.meta.get('label') or 'table'} [e{i + 1}].", "Compare matched rows."]
        elif req.primary == "numerical" and tables:
            for i, t in tables:
                r = table_lookup(q, t.meta["header"], t.meta["rows"])
                if r:
                    unit = _unit_from_caption(t.meta.get("caption", ""))
                    answer = r["value"]
                    structured = {"value": parse_number(r["value"]), "unit": unit, "row": r["row_label"], "column": r["col_header"]}
                    full = f"{r['value']} {unit}".strip() + f" ({r['row_label']}, {r['col_header']}) [e{i + 1}]"
                    steps = [f"Locate row '{r['row_label']}' and column '{r['col_header']}' in {t.meta.get('label') or 'table'} [e{i + 1}]."]
                    break
        elif req.primary == "crossmodal" and figs:
            cap = next((e for _, e in figs if e.type == NodeType.CAPTION), None)
            fig = next((e for _, e in figs if e.type == NodeType.FIG), None)
            body = (cap.meta.get("body") if cap else fig.meta.get("body")) if (cap or fig) else ""
            label = (cap.meta.get("label") if cap else fig.meta.get("label")) or "figure"
            if re.match(r"\s*(what|how)\b.*\b(show|depict|illustrate|display|represent)", q, re.I):
                answer = body
            else:
                answer = extract_span(q, body) or body
            structured = {"figure": label, "region": None}
            full = f"{answer} (grounded in {label})"
            steps = [f"Match question to {label} via its caption."]
        elif req.primary == "multihop" and req.control.reasoning in ("multi_step", "structured") and sents:
            i1, s1 = best_sentence()
            bridges = [m.group() for m in _CAP.finditer(s1) if m.start() > 0 and m.group() not in _LEAD
                       and not set(tokenize(m.group())) <= _qtokens(q)]
            steps.append(f"Step 1: Identify {bridges or 'key entity'} from [e{i1 + 1}] \"{s1}\"")
            for b in bridges:
                pool = [(i, s) for i, s in sents if s != s1 and b.lower() in s.lower()]
                for i2, s2 in sorted(pool, key=lambda isn: -_overlap(q, isn[1])):
                    span = extract_span(q, s2)
                    if span:
                        steps.append(f"Step 2: Connect '{b}' to [e{i2 + 1}] \"{s2}\"")
                        steps.append(f"Step 3: Extract the requested value -> {span}")
                        answer = span
                        break
                if answer:
                    break
            if not answer:
                answer = extract_span(q, s1) or s1
            full = "\n".join(steps) + f"\nFinal Answer: {answer}"
        if answer is None:  # factoid / fallback
            bs = best_sentence()
            if bs is None:
                answer, full = "", ""
            else:
                i, s = bs
                answer = extract_span(q, s) or s
                full = f"{answer} [e{i + 1}]"
                steps = [f"Best-matching sentence in [e{i + 1}]."]

        if req.control.reasoning == "none":
            text = answer or ""
        elif req.control.reasoning == "single_step":
            text = f"{full}"
        else:
            text = (("\n".join(f"Step {k}: {s}" for k, s in enumerate(steps, 1)) + f"\nFinal Answer: {answer}") if steps and req.primary != "multihop" else full)
        text, trunc = _truncate(text, req.control.max_tokens)
        answer_t, _ = _truncate(answer or "", req.control.max_tokens)
        return GenerationResult(answer_t, text, req.fmt, req.bucket, steps, cites, structured, trunc)


class LLMGenerator:
    """Adapter for a (multimodal) LLM: ``fn(prompt, max_tokens) -> str``. The answer is the text after
    the last 'Final Answer:' marker, or the whole output if no marker is present."""

    def __init__(self, fn: Callable[[str, int], str]):
        self.fn = fn

    def generate(self, req: GenerationRequest) -> GenerationResult:
        prompt = build_prompt(req)
        out = self.fn(prompt, req.control.max_tokens).strip()
        ans = out.rsplit("Final Answer:", 1)[-1].strip() if "Final Answer:" in out else out
        text, trunc = _truncate(out, req.control.max_tokens)
        return GenerationResult(ans, text, req.fmt, req.bucket, truncated=trunc)
