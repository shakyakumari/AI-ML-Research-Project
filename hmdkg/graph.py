"""Hierarchical Multimodal Document Knowledge Graph  G = (V, E, phi, psi)."""
from __future__ import annotations

from collections import Counter, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

import networkx as nx
import numpy as np


class NodeType(str, Enum):
    PARA = "para"
    SECTION = "section"
    FIG = "fig"
    CAPTION = "caption"
    TABLE = "table"
    REF = "ref"


class EdgeKind(str, Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    CROSSREF = "crossref"


CONTENT_TYPES = (NodeType.PARA, NodeType.TABLE, NodeType.FIG, NodeType.CAPTION)
HIERARCHY_RELATIONS = frozenset({"IS_PART_OF", "BELONGS_TO"})
Adj = namedtuple("Adj", "nbr kind relation weight")


@dataclass(eq=False)
class Node:
    id: str
    type: NodeType
    text: str = ""
    page: int = 1
    level: int = 0
    emb: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)


class HMDKG:
    ROOT = "doc"

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.nodes: dict[str, Node] = {}

    # ---- construction -------------------------------------------------
    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.g.add_node(node.id)
        return node

    def add_edge(self, src: str, dst: str, kind: EdgeKind, relation: str, weight: float = 1.0) -> None:
        self.g.add_edge(src, dst, kind=kind, relation=relation, weight=float(weight))

    # ---- access -------------------------------------------------------
    def emb(self, nid: str) -> np.ndarray:
        e = self.nodes[nid].emb
        if e is None:
            raise ValueError(f"node {nid} has no embedding")
        return e

    def of_type(self, *types: NodeType) -> list[Node]:
        return [n for n in self.nodes.values() if n.type in types]

    def adjacent(self, nid: str) -> Iterator[Adj]:
        """All neighbours (edges are traversed in both directions)."""
        for _, v, d in self.g.out_edges(nid, data=True):
            yield Adj(v, d["kind"], d["relation"], d["weight"])
        for u, _, d in self.g.in_edges(nid, data=True):
            yield Adj(u, d["kind"], d["relation"], d["weight"])

    def edges_between(self, a: str, b: str) -> list[str]:
        rels: list[str] = []
        for _, v, d in self.g.out_edges(a, data=True):
            if v == b:
                rels.append(d["relation"])
        for u, _, d in self.g.in_edges(a, data=True):
            if u == b:
                rels.append(d["relation"])
        return rels

    # ---- hierarchy / zooming -----------------------------------------
    def parent(self, nid: str) -> Optional[str]:
        node = self.nodes[nid]
        if node.type == NodeType.CAPTION:
            for u, _, d in self.g.in_edges(nid, data=True):
                if d["relation"] == "HAS_CAPTION":
                    return self.parent(u)
            return None
        for _, v, d in self.g.out_edges(nid, data=True):
            if d["relation"] in HIERARCHY_RELATIONS:
                return v
        return None

    def children(self, nid: str) -> list[str]:
        kids = [u for u, _, d in self.g.in_edges(nid, data=True) if d["relation"] in HIERARCHY_RELATIONS]
        if self.nodes[nid].type == NodeType.FIG:
            kids += [v for _, v, d in self.g.out_edges(nid, data=True) if d["relation"] == "HAS_CAPTION"]
        return kids

    def zoom_out(self, nid: str) -> list[str]:
        """Ancestors from the immediate parent up to the document root."""
        chain, cur = [], self.parent(nid)
        while cur is not None:
            chain.append(cur)
            cur = self.parent(cur)
        return chain

    def zoom_in(self, nid: str, depth: int = 1) -> list[str]:
        """Descendants down to ``depth`` levels (use a large depth for all content)."""
        out, layer = [], [nid]
        for _ in range(depth):
            nxt = [c for n in layer for c in self.children(n)]
            out += nxt
            layer = nxt
            if not layer:
                break
        return out

    def content_descendants(self, nid: str) -> list[str]:
        return [c for c in self.zoom_in(nid, depth=50) if self.nodes[c].type in CONTENT_TYPES]

    def table_cells(self, nid: str) -> list[dict[str, Any]]:
        """Cell-level view of a table node (finest granularity of the hierarchy)."""
        m = self.nodes[nid].meta
        out = []
        for k, (ri, ci) in enumerate(m["cells"]):
            out.append(
                {
                    "row": ri,
                    "col": ci,
                    "row_label": m["rows"][ri][0],
                    "col_header": m["header"][ci],
                    "value": m["rows"][ri][ci],
                    "emb": m["cell_embs"][k],
                }
            )
        return out

    # ---- inspection ---------------------------------------------------
    def stats(self) -> dict[str, Any]:
        nt = Counter(n.type.value for n in self.nodes.values())
        ek, er = Counter(), Counter()
        for _, _, d in self.g.edges(data=True):
            ek[d["kind"].value] += 1
            er[d["relation"]] += 1
        return {"nodes": dict(nt), "edge_kinds": dict(ek), "edge_relations": dict(er)}

    def to_dict(self) -> dict[str, Any]:
        nodes = [
            {"id": n.id, "type": n.type.value, "page": n.page, "level": n.level, "text": n.text[:160]}
            for n in self.nodes.values()
        ]
        edges = [
            {"src": u, "dst": v, "kind": d["kind"].value, "relation": d["relation"], "weight": round(d["weight"], 4)}
            for u, v, d in self.g.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
