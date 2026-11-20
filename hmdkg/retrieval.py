"""QC-AER: Question-Conditioned Adaptive Evidence Retrieval (pure-NumPy reference).

Pipeline: intent classification -> seed selection -> question-gated graph attention (QG-GAT)
-> adaptive traversal with intent-dependent depth/breadth/stopping -> cross-attention reranking.

The learnable parts (seed projections, attention vector, reranker projections) are present and
zero/identity-initialised, so the untrained system is driven by interpretable priors. Training them
(PyTorch port) is the next step of the project plan.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .encoders import HashingTextEncoder
from .graph import CONTENT_TYPES, HMDKG, NodeType

INTENTS = ("factoid", "numerical", "comparative", "multihop", "descriptive", "crossmodal")
PRIORITY = ("multihop", "comparative", "crossmodal", "numerical", "descriptive", "factoid")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------------------------
# Intent classification (multi-label, rule-based placeholder for a learned BERT/T5 head)
# --------------------------------------------------------------------------------------------
_P = re.compile
_PATTERNS = {
    "numerical": [
        (_P(r"\bhow (?:many|much)\b|\b(?:total|sum|average|mean|percentage|percent|calculate|difference|number of|count)\b", re.I), 0.9),
        (_P(r"\b(?:revenue|sales|profit|cost|price|amount|headcount|salary|income|expense|rate|margin|growth)\b", re.I), 0.5),
    ],
    "comparative": [
        (_P(r"\b(?:compare|compared|versus|vs\.?|than|higher|lower|highest|lowest|larger|largest|smaller|smallest|greater|greatest|more|less|most|least|bigger|better|worse|outperform|rank)\b", re.I), 0.9),
    ],
    "multihop": [
        (_P(r"\b(?:the|a|an)\s+\w+\s+(?:that|which|who|whose)\b", re.I), 0.7),
        (_P(r"\bwhose\b", re.I), 0.7),
        (_P(r"\b(?:of|by|in)\s+the\s+\w+\s+(?:of|in|by)\s+the\b", re.I), 0.5),
    ],
    "descriptive": [
        (_P(r"\b(?:explain|describe|summari[sz]e|summary|discuss|overview|why|elaborate|outline)\b|\bhow (?:does|do|did)\b", re.I), 0.9),
        (_P(r"\bwhat are the (?:main|key|major)\b", re.I), 0.6),
    ],
    "crossmodal": [
        (_P(r"\b(?:figure|fig\.?|chart|graph|plot|diagram|image|picture|infographic|illustrat\w*|depict\w*)\b", re.I), 0.9),
    ],
    "factoid": [(_P(r"^\s*(?:who|whom|what|when|where|which)\b", re.I), 0.4)],
}


class IntentClassifier:
    threshold = 0.5

    def scores(self, question: str) -> dict[str, float]:
        out = {k: 0.0 for k in INTENTS}
        for intent, pats in _PATTERNS.items():
            for rx, w in pats:
                if rx.search(question):
                    out[intent] = max(out[intent], w)
        if len(question.split()) >= 18:
            out["multihop"] = max(out["multihop"], 0.3) + 0.2
        if max(out.values()) == 0:
            out["factoid"] = 0.4
        return out

    def classify(self, question: str) -> tuple[dict[str, float], list[str], str]:
        sc = self.scores(question)
        labels = [k for k in INTENTS if sc[k] >= self.threshold] or ["factoid"]
        best = max(sc.values())
        primary = next(k for k in PRIORITY if sc[k] == best)
        return sc, labels, primary


# --------------------------------------------------------------------------------------------
# Intent -> retrieval policy (depth, breadth, stopping, type priors)
# --------------------------------------------------------------------------------------------
@dataclass
class IntentPolicy:
    k_seed: int = 4
    max_hops: int = 1
    final_k: int = 4
    type_prior: dict[NodeType, float] = field(default_factory=dict)
    iterative: bool = False
    cluster: bool = False
    follow_refs: bool = False

    def prior(self, t: NodeType) -> float:
        return self.type_prior.get(t, 1.0)


BASE_POLICIES: dict[str, IntentPolicy] = {
    "factoid": IntentPolicy(3, 2, 3),
    "numerical": IntentPolicy(4, 1, 4, {NodeType.TABLE: 1.6}, follow_refs=True),
    "comparative": IntentPolicy(5, 1, 5, {NodeType.TABLE: 1.5}, follow_refs=True),
    "multihop": IntentPolicy(5, 4, 6, iterative=True),
    "descriptive": IntentPolicy(6, 2, 6, {NodeType.SECTION: 1.3}),
    "crossmodal": IntentPolicy(4, 1, 5, {NodeType.FIG: 1.3, NodeType.CAPTION: 1.3}, cluster=True, follow_refs=True),
}
UNIFORM_POLICY = IntentPolicy(4, 1, 4)

# relation-level routing bias per intent (added to attention logits, and used to gate traversal)
EDGE_ROUTING: dict[str, dict[str, float]] = {
    "factoid": {},
    "numerical": {"SUPPORTS": 1.0, "REFERENCES": 0.6},
    "comparative": {"SUPPORTS": 0.8, "REFERENCES": 0.5},
    "multihop": {"REFERENCES": 0.7, "SEMANTIC_SIM": 0.7},
    "descriptive": {"IS_PART_OF": 0.3, "SEMANTIC_SIM": 0.4},
    "crossmodal": {"DESCRIBES": 1.0, "HAS_CAPTION": 1.0, "REFERENCES": 1.0},
}


def merge_policy(labels: list[str]) -> IntentPolicy:
    ps = [BASE_POLICIES[l] for l in labels]
    prior: dict[NodeType, float] = {}
    for p in ps:
        for t, v in p.type_prior.items():
            prior[t] = max(prior.get(t, 1.0), v)
    return IntentPolicy(
        k_seed=max(p.k_seed for p in ps),
        max_hops=max(p.max_hops for p in ps),
        final_k=max(p.final_k for p in ps),
        type_prior=prior,
        iterative=any(p.iterative for p in ps),
        cluster=any(p.cluster for p in ps),
        follow_refs=any(p.follow_refs for p in ps),
    )


def routing_bias(intent_scores: dict[str, float], labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for l in labels:
        for rel, b in EDGE_ROUTING[l].items():
            out[rel] = out.get(rel, 0.0) + b * intent_scores.get(l, 1.0)
    return out


# --------------------------------------------------------------------------------------------
# Question-gated graph attention (forward pass)
# --------------------------------------------------------------------------------------------
class QGGAT:
    """Question-gated GAT layer stack.

    logit_ij = a^T LeakyReLU(W_s h_i + W_d h_j + W_q q)      (learned, GATv2-style, dynamic)
             + lam * <q, h_j> + b[relation_ij] + log w_ij     (question relevance, routing, edge weight)
    alpha_ij = softmax_j(logit_ij);  h_i' = normalize((1 - beta) h_i + beta * sum_j alpha_ij h_j)

    Design note: with the additive form  a^T [W h_i || W h_j || W_q q]  the q-term is constant across
    neighbours j and cancels in the softmax (up to the LeakyReLU kink), so q could not route evidence.
    Here q enters multiplicatively (GATv2 hidden layer and the explicit <q, h_j> term). ``a`` is
    zero-initialised, so the untrained model uses only the interpretable second line.
    """

    def __init__(self, dim: int, layers: int = 2, beta: float = 0.5, d_att: int = 32, lam: float = 4.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(dim)
        self.Ws, self.Wd, self.Wq = (rng.normal(0, s, (dim, d_att)) for _ in range(3))
        self.a = np.zeros(d_att)
        self.layers, self.beta, self.lam = layers, beta, lam

    @staticmethod
    def _leaky(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.2 * x)

    def propagate(self, graph: HMDKG, q: np.ndarray, active: set[str], bias: dict[str, float]) -> dict[str, np.ndarray]:
        H = {n: graph.emb(n) for n in active}
        qproj = q @ self.Wq
        for _ in range(self.layers):
            new: dict[str, np.ndarray] = {}
            for u in active:
                nbrs = [(u, "SELF", 1.0)]
                for adj in graph.adjacent(u):
                    if adj.nbr != graph.ROOT:
                        nbrs.append((adj.nbr, adj.relation, adj.weight))
                Hn = np.stack([H[v] if v in H else graph.emb(v) for v, _, _ in nbrs])
                learned = self._leaky(H[u] @ self.Ws + Hn @ self.Wd + qproj) @ self.a
                prior = self.lam * (Hn @ q) + np.array([bias.get(r, 0.0) for _, r, _ in nbrs]) + np.log(
                    np.array([w for _, _, w in nbrs]) + 1e-6
                )
                logits = learned + prior
                alpha = np.exp(logits - logits.max())
                alpha /= alpha.sum()
                new[u] = _unit((1 - self.beta) * H[u] + self.beta * (alpha @ Hn))
            H = new
        return H


class CrossAttnReranker:
    """Relevance(v | Q) = scaled dot-product between projected q and h_v (identity projections by default)."""

    def __init__(self, Wq: Optional[np.ndarray] = None, Wk: Optional[np.ndarray] = None, alpha: float = 0.5):
        self.Wq, self.Wk, self.alpha = Wq, Wk, alpha

    def score(self, q: np.ndarray, h_final: np.ndarray, h_init: np.ndarray) -> float:
        qq = q if self.Wq is None else q @ self.Wq
        kk = h_final if self.Wk is None else h_final @ self.Wk
        return float(self.alpha * (qq @ kk) + (1 - self.alpha) * (q @ h_init))


# --------------------------------------------------------------------------------------------
# QC-AER
# --------------------------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    tau: float = 0.1  # seed softmax temperature
    gamma: float = 0.6  # relative admission threshold for traversal candidates
    min_abs: float = 0.05
    max_add_per_hop: int = 3
    max_nodes: int = 12
    mu: float = 0.5  # query expansion strength for multi-hop
    hop_decay: float = 0.9
    zoom_children: int = 3  # content children taken from a selected section seed
    use_intent: bool = True
    use_qggat: bool = True
    use_hierarchy_zoom: bool = True


@dataclass
class Evidence:
    node_id: str
    score: float
    hop: int
    path: list[str]
    type: NodeType
    text: str
    via: str = "seed"


@dataclass
class RetrievalResult:
    question: str
    intent_scores: dict[str, float]
    labels: list[str]
    primary: str
    policy: IntentPolicy
    evidence: list[Evidence]
    trace: dict


class QCAER:
    def __init__(self, graph: HMDKG, encoder: HashingTextEncoder, cfg: Optional[RetrievalConfig] = None,
                 classifier: Optional[IntentClassifier] = None):
        self.G, self.enc = graph, encoder
        self.cfg = cfg or RetrievalConfig()
        self.clf = classifier or IntentClassifier()
        self.qggat = QGGAT(encoder.dim)
        self.reranker = CrossAttnReranker()
        ids = [n for n in graph.nodes if n != graph.ROOT]
        self._ids = ids
        self._E = np.stack([graph.emb(i) for i in ids])

    # -- phase 1 ------------------------------------------------------------------------------
    def _seed_probs(self, q: np.ndarray, policy: IntentPolicy, Wq=None, Wv=None) -> np.ndarray:
        qv = q if Wq is None else q @ Wq
        Hv = self._E if Wv is None else self._E @ Wv
        prior = np.array([policy.prior(self.G.nodes[i].type) for i in self._ids])
        # sections are summaries: down-weight them unless the policy raises them (descriptive)
        sec = np.array([self.G.nodes[i].type == NodeType.SECTION for i in self._ids])
        prior = np.where(sec & (prior <= 1.0), 0.6, prior)
        logits = Hv @ qv / self.cfg.tau + np.log(prior)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def select_seeds(self, q: np.ndarray, policy: IntentPolicy) -> list[tuple[str, float]]:
        p = self._seed_probs(q, policy)
        order = np.argsort(-p)[: policy.k_seed]
        seeds: dict[str, float] = {}
        for k in order:
            nid, prob = self._ids[k], float(p[k])
            node = self.G.nodes[nid]
            if node.type in CONTENT_TYPES:
                seeds[nid] = max(seeds.get(nid, 0.0), prob)
            elif node.type == NodeType.SECTION and self.cfg.use_hierarchy_zoom:  # zoom in
                kids = self.G.content_descendants(nid)
                kids.sort(key=lambda c: -float(q @ self.G.emb(c)))
                for c in kids[: self.cfg.zoom_children]:
                    seeds[c] = max(seeds.get(c, 0.0), prob * 0.9)
        return sorted(seeds.items(), key=lambda kv: -kv[1])

    # -- full retrieval -----------------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        G, cfg = self.G, self.cfg
        q = self.enc.encode(question)
        scores, labels, primary = self.clf.classify(question)
        if cfg.use_intent:
            policy, bias = merge_policy(labels), routing_bias(scores, labels)
        else:
            policy, bias = UNIFORM_POLICY, {}
        seeds = self.select_seeds(q, policy)

        # active subgraph (content nodes within max(L, hops) of the seeds) -> scalable local propagation
        depth = max(self.qggat.layers, policy.max_hops)
        active, layer = {s for s, _ in seeds}, {s for s, _ in seeds}
        for _ in range(depth):
            nxt = {a.nbr for u in layer for a in G.adjacent(u) if a.nbr != G.ROOT and G.nodes[a.nbr].type in CONTENT_TYPES}
            layer = nxt - active
            active |= nxt
        H0 = {n: G.emb(n) for n in active}
        HL = self.qggat.propagate(G, q, active, bias) if cfg.use_qggat else H0

        def rel(v: str, qt: np.ndarray) -> float:
            return float(qt @ HL.get(v, G.emb(v)))

        selected: dict[str, Evidence] = {}
        for s, _ in seeds:
            n = G.nodes[s]
            selected[s] = Evidence(s, rel(s, q), 0, [s], n.type, n.text)
        seed_rels = [e.score for e in selected.values()]
        thresh = max(cfg.min_abs, cfg.gamma * (min(seed_rels) if seed_rels else 0.0))

        frontier, qt, stop = list(selected), q.copy(), "max_hops"
        hops_used = 0
        for hop in range(1, policy.max_hops + 1):
            cands: dict[str, Evidence] = {}
            for u in frontier:
                for adj in G.adjacent(u):
                    v = adj.nbr
                    if v in selected or v == G.ROOT or G.nodes[v].type not in CONTENT_TYPES:
                        continue
                    gate = 1.0 + bias.get(adj.relation, 0.0)
                    r = rel(v, qt) * gate * (policy.prior(G.nodes[v].type) ** 0.5) * (cfg.hop_decay ** hop)
                    if r >= thresh and (v not in cands or r > cands[v].score):
                        cands[v] = Evidence(v, r, hop, selected[u].path + [v], G.nodes[v].type, G.nodes[v].text, adj.relation)
            if not cands:
                stop = "no_candidates"
                break
            new = sorted(cands.values(), key=lambda e: -e.score)[: cfg.max_add_per_hop]
            for e in new:
                selected[e.node_id] = e
            hops_used = hop
            if len(selected) >= cfg.max_nodes:
                stop = "budget"
                break
            if policy.iterative:  # multi-hop: fold what was found into the query, then continue
                qt = _unit(q + cfg.mu * np.mean([H0.get(e.node_id, G.emb(e.node_id)) for e in new], axis=0))
            frontier = [e.node_id for e in new]

        # rerank (cross-attention relevance, hop penalty, type prior)
        ranked = []
        for e in selected.values():
            h_f, h_0 = HL.get(e.node_id, G.emb(e.node_id)), G.emb(e.node_id)
            s = self.reranker.score(q, h_f, h_0) * (policy.prior(e.type) ** 0.5) * (cfg.hop_decay ** e.hop)
            ranked.append(Evidence(e.node_id, s, e.hop, e.path, e.type, e.text, e.via))
        ranked.sort(key=lambda e: -e.score)
        final = ranked[: policy.final_k]
        have = {e.node_id for e in final}

        # cluster completion: figure<->caption, and referenced tables/figures
        extra: list[Evidence] = []
        for e in list(final):
            partners: list[tuple[str, str]] = []
            if policy.cluster or e.type in (NodeType.FIG, NodeType.CAPTION):
                n = G.nodes[e.node_id]
                partner = n.meta.get("caption_id") or n.meta.get("figure_id")
                if partner:
                    partners.append((partner, "HAS_CAPTION"))
            if policy.follow_refs and e.type == NodeType.PARA:
                partners += [(a.nbr, "REFERENCES") for a in G.adjacent(e.node_id) if a.relation == "REFERENCES" and a.nbr != e.node_id]
            for pid, via in partners:
                if pid not in have and len(extra) < 2:
                    pn = G.nodes[pid]
                    extra.append(Evidence(pid, e.score * 0.9, e.hop + 1, e.path + [pid], pn.type, pn.text, via))
                    have.add(pid)
        final += extra
        final.sort(key=lambda e: -e.score)

        trace = {
            "seeds": [(s, round(p, 4)) for s, p in seeds],
            "hops_used": hops_used,
            "stop_reason": stop,
            "active_nodes": len(active),
            "total_nodes": len(G.nodes),
            "candidates_admitted": len(selected),
        }
        return RetrievalResult(question, scores, labels, primary, policy, final, trace)


class FlatRetriever:
    """Baseline: fixed top-k cosine over flat chunks (paragraphs, tables, captions); no graph."""

    def __init__(self, graph: HMDKG, encoder: HashingTextEncoder, k: int = 3):
        self.G, self.enc, self.k = graph, encoder, k
        self.ids = [n.id for n in graph.of_type(NodeType.PARA, NodeType.TABLE, NodeType.CAPTION)]
        self.E = np.stack([graph.emb(i) for i in self.ids])
        self.clf = IntentClassifier()

    def retrieve(self, question: str) -> RetrievalResult:
        q = self.enc.encode(question)
        sims = self.E @ q
        order = np.argsort(-sims)[: self.k]
        ev = [Evidence(self.ids[i], float(sims[i]), 0, [self.ids[i]], self.G.nodes[self.ids[i]].type, self.G.nodes[self.ids[i]].text) for i in order]
        sc, labels, primary = self.clf.classify(question)
        return RetrievalResult(question, sc, labels, primary, UNIFORM_POLICY, ev, {"mode": "flat_topk", "k": self.k})
