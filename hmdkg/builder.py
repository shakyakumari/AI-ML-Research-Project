"""HM-DKG construction: parse -> extract elements -> encode -> build typed edges.

Input is Markdown-like text (headings, paragraphs, pipe tables, ``![Figure N: caption](path)``,
``<!-- page N -->`` markers). A production system would obtain the same ``Element`` list from a
layout parser (LayoutLMv3 / DocLayNet) run over PDF pages; everything downstream is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .encoders import HashingTextEncoder, PlaceholderVisualEncoder, TableEncoder, linearize_table, _unit
from .graph import CONTENT_TYPES, EdgeKind, HMDKG, Node, NodeType

PAGE_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->$", re.I)
HEAD_RE = re.compile(r"^(#{1,4})\s+(.*)$")
FIG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
LABEL_RE = re.compile(r"^(Table|Figure|Fig\.)\s*(\d+)\s*[:.]?\s*(.*)$", re.I)
REF_RE = re.compile(r"\b(Table|Figure|Fig\.)\s*(\d+)\b", re.I)


@dataclass
class Element:
    kind: str  # heading | paragraph | table | figure
    text: str = ""
    level: int = 0
    page: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildConfig:
    # NOTE: 0.75 is a sensible value for neural sentence encoders. Hashed TF-IDF vectors have much
    # lower cosines, so the dependency-free default is lower.
    sim_threshold: float = 0.12
    cross_sim_threshold: float = 0.10
    max_sim_edges: int = 4
    enable_semantic: bool = True
    enable_crossref: bool = True
    enable_cross_modal: bool = True  # figure/caption/table <-> paragraph semantic + REFERENCES edges


def norm_label(kind: str, num: str) -> str:
    return ("Table" if kind.lower().startswith("table") else "Figure") + f" {int(num)}"


def parse_markdown(text: str) -> list[Element]:
    lines = text.splitlines()
    elements: list[Element] = []
    buf: list[str] = []
    page, i = 1, 0

    def flush() -> None:
        if buf:
            elements.append(Element("paragraph", " ".join(buf).strip(), page=page))
            buf.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if (m := PAGE_RE.match(s)):
            flush()
            page = int(m.group(1))
            i += 1
        elif not s:
            flush()
            i += 1
        elif (m := HEAD_RE.match(s)):
            flush()
            elements.append(Element("heading", m.group(2).strip(), level=len(m.group(1)), page=page))
            i += 1
        elif (m := FIG_RE.match(s)):
            flush()
            alt, path = m.group(1).strip(), m.group(2).strip()
            lm = LABEL_RE.match(alt)
            label = norm_label(lm.group(1), lm.group(2)) if lm else None
            body = lm.group(3).strip() if lm else alt
            elements.append(Element("figure", alt, page=page, extra={"path": path, "label": label, "body": body}))
            i += 1
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            caption, label = "", None
            if elements and elements[-1].kind == "paragraph":
                lm = LABEL_RE.match(elements[-1].text)
                if lm and lm.group(1).lower().startswith("table"):
                    caption, label = elements.pop().text, norm_label(lm.group(1), lm.group(2))
            if rows:
                width = len(rows[0])
                body = [r + [""] * (width - len(r)) for r in rows[1:]]
                elements.append(
                    Element("table", caption, page=page, extra={"header": rows[0], "rows": body, "label": label})
                )
        else:
            buf.append(s)
            i += 1
    flush()
    return elements


def build_graph(
    elements: list[Element],
    encoder: HashingTextEncoder,
    visual_encoder: Optional[PlaceholderVisualEncoder] = None,
    cfg: Optional[BuildConfig] = None,
    title: str = "document",
) -> HMDKG:
    cfg = cfg or BuildConfig()
    table_enc = TableEncoder(encoder)
    visual_encoder = visual_encoder or PlaceholderVisualEncoder(encoder)

    # fit IDF on the document's own elements
    fit_texts = []
    for e in elements:
        if e.kind == "table":
            fit_texts.append(linearize_table(e.extra["header"], e.extra["rows"], e.text))
        else:
            fit_texts.append(e.text)
    encoder.fit(fit_texts)

    G = HMDKG()
    G.add_node(Node(G.ROOT, NodeType.SECTION, title, level=0, emb=encoder.encode(title)))
    stack: list[tuple[int, str]] = [(0, G.ROOT)]
    section_ids: list[str] = []
    labels: dict[str, str] = {}
    counters = {"p": 0, "s": 0, "t": 0, "f": 0, "c": 0}

    def nid(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}{counters[prefix]}"

    for el in elements:
        if el.kind == "heading":
            while stack[-1][0] >= el.level:
                stack.pop()
            sid = nid("s")
            G.add_node(Node(sid, NodeType.SECTION, el.text, page=el.page, level=el.level, emb=encoder.encode(el.text)))
            G.add_edge(sid, stack[-1][1], EdgeKind.STRUCTURAL, "IS_PART_OF")
            stack.append((el.level, sid))
            section_ids.append(sid)
            continue
        parent = stack[-1][1]
        if el.kind == "paragraph":
            pid = nid("p")
            G.add_node(Node(pid, NodeType.PARA, el.text, page=el.page, emb=encoder.encode(el.text)))
            G.add_edge(pid, parent, EdgeKind.STRUCTURAL, "IS_PART_OF")
        elif el.kind == "table":
            header, rows = el.extra["header"], el.extra["rows"]
            lin = linearize_table(header, rows, el.text)
            emb, cells, cell_mat = table_enc.encode(header, rows, el.text)
            tid = nid("t")
            G.add_node(
                Node(
                    tid, NodeType.TABLE, lin, page=el.page, emb=emb,
                    meta={"header": header, "rows": rows, "label": el.extra["label"], "caption": el.text,
                          "cells": cells, "cell_embs": cell_mat},
                )
            )
            G.add_edge(tid, parent, EdgeKind.STRUCTURAL, "BELONGS_TO")
            if el.extra["label"]:
                labels[el.extra["label"]] = tid
        elif el.kind == "figure":
            fid, cid = nid("f"), nid("c")
            body, path = el.extra["body"], el.extra["path"]
            G.add_node(
                Node(fid, NodeType.FIG, f"[image: {path}]", page=el.page,
                     emb=visual_encoder.encode(path, el.text),
                     meta={"path": path, "label": el.extra["label"], "alt": el.text, "body": body, "caption_id": cid})
            )
            G.add_node(
                Node(cid, NodeType.CAPTION, el.text, page=el.page, emb=encoder.encode(el.text),
                     meta={"label": el.extra["label"], "body": body, "figure_id": fid})
            )
            G.add_edge(fid, parent, EdgeKind.STRUCTURAL, "BELONGS_TO")
            G.add_edge(fid, cid, EdgeKind.STRUCTURAL, "HAS_CAPTION")
            if el.extra["label"]:
                labels[el.extra["label"]] = fid

    # ---- section embeddings: hierarchical pooling (children first) ---------------------------
    for sid in reversed(section_ids):
        kids = [G.nodes[c].emb for c in G.children(sid) if G.nodes[c].type != NodeType.CAPTION]
        if kids:
            G.nodes[sid].emb = _unit(0.8 * _unit(np.mean(kids, axis=0)) + 0.2 * G.nodes[sid].emb)

    # ---- semantic edges --------------------------------------------------------------------------
    if cfg.enable_semantic:
        content = [n.id for n in G.nodes.values() if n.type in CONTENT_TYPES]
        E = np.stack([G.nodes[i].emb for i in content])
        S = E @ E.T
        added: set[frozenset[str]] = set()
        for a_idx, a in enumerate(content):
            order = np.argsort(-S[a_idx])
            taken = 0
            for b_idx in order:
                b = content[b_idx]
                if b == a:
                    continue
                ta, tb = G.nodes[a].type, G.nodes[b].type
                if {ta, tb} == {NodeType.FIG, NodeType.CAPTION}:
                    continue
                cross = ta != NodeType.PARA or tb != NodeType.PARA
                if cross and not cfg.enable_cross_modal:
                    continue
                thr = cfg.cross_sim_threshold if cross else cfg.sim_threshold
                if taken >= cfg.max_sim_edges or S[a_idx, b_idx] < min(cfg.sim_threshold, cfg.cross_sim_threshold):
                    break
                if S[a_idx, b_idx] < thr:
                    continue
                key = frozenset((a, b))
                taken += 1
                if key in added:
                    continue
                added.add(key)
                w = float(S[a_idx, b_idx])
                kinds = {ta, tb}
                if NodeType.PARA in kinds and kinds & {NodeType.FIG, NodeType.CAPTION}:
                    para, fig = (a, b) if ta == NodeType.PARA else (b, a)
                    G.add_edge(para, fig, EdgeKind.SEMANTIC, "DESCRIBES", w)
                elif NodeType.PARA in kinds and NodeType.TABLE in kinds:
                    para, tab = (a, b) if ta == NodeType.PARA else (b, a)
                    G.add_edge(tab, para, EdgeKind.SEMANTIC, "SUPPORTS", w)
                else:
                    G.add_edge(a, b, EdgeKind.SEMANTIC, "SEMANTIC_SIM", w)

    # ---- cross-reference edges ("see Figure 3", "as shown in Table 2") -------------------------
    if cfg.enable_crossref and cfg.enable_cross_modal:
        for n in G.of_type(NodeType.PARA):
            for m in REF_RE.finditer(n.text):
                target = labels.get(norm_label(m.group(1), m.group(2)))
                if target:
                    G.add_edge(n.id, target, EdgeKind.CROSSREF, "REFERENCES", 1.0)
    return G
