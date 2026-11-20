import numpy as np

from hmdkg.graph import NodeType
from hmdkg.retrieval import QCAER, FlatRetriever, IntentClassifier, QGGAT, merge_policy


def test_intent_classification():
    clf = IntentClassifier()
    cases = {
        "Who is the Chief Executive Officer of Acme Robotics?": "factoid",
        "How many people does the Logistics division employ?": "numerical",
        "Which region had higher revenue in 2024, APAC or EMEA?": "comparative",
        "What does Figure 1 show?": "crossmodal",
        "Who leads the division that opened a plant in Chennai?": "multihop",
        "Describe the main risks facing Acme Robotics.": "descriptive",
    }
    for q, want in cases.items():
        assert clf.classify(q)[2] == want, q


def test_multilabel_and_policy_merge():
    _, labels, _ = IntentClassifier().classify("Which region had higher revenue in 2024, APAC or EMEA?")
    assert {"comparative", "numerical"} <= set(labels)
    p = merge_policy(labels)
    assert p.type_prior[NodeType.TABLE] >= 1.5 and p.final_k >= 5


def test_numerical_question_prioritises_table(pipe):
    r = pipe.retriever.retrieve("What was the revenue of EMEA in 2023 in USD millions?")
    assert r.evidence[0].type == NodeType.TABLE


def test_crossmodal_cluster_has_figure_and_caption(pipe):
    r = pipe.retriever.retrieve("What does Figure 1 show?")
    types = {e.type for e in r.evidence}
    assert NodeType.FIG in types and NodeType.CAPTION in types


def test_multihop_reaches_second_hop_evidence(pipe):
    r = pipe.retriever.retrieve("Who leads the division that opened a plant in Chennai?")
    texts = " ".join(e.text for e in r.evidence)
    assert "Chennai" in texts and "Rahul Verma" in texts


def test_adaptive_depth_differs_by_intent(pipe):
    fact = pipe.retriever.retrieve("When was Acme Robotics founded?")
    multi = pipe.retriever.retrieve("Who leads the division that opened a plant in Chennai?")
    assert fact.policy.max_hops < multi.policy.max_hops
    assert fact.policy.final_k < multi.policy.final_k


def test_section_seed_zooms_to_children(pipe):
    r = pipe.retriever.retrieve("Describe the main risks facing Acme Robotics.")
    assert any("Supply chain" in e.text for e in r.evidence)
    assert all(e.type != NodeType.SECTION for e in r.evidence)


def test_qggat_output_is_unit_norm_and_deterministic(pipe):
    G, enc = pipe.graph, pipe.encoder
    q = enc.encode("revenue by region")
    active = {n.id for n in G.of_type(NodeType.PARA, NodeType.TABLE)}
    a = QGGAT(enc.dim).propagate(G, q, active, {})
    b = QGGAT(enc.dim).propagate(G, q, active, {})
    for k in active:
        assert abs(np.linalg.norm(a[k]) - 1) < 1e-6
        assert np.allclose(a[k], b[k])


def test_qggat_question_changes_routing(pipe):
    """Different questions must yield different propagated features (question-conditioned)."""
    G, enc = pipe.graph, pipe.encoder
    active = {n.id for n in G.of_type(NodeType.PARA, NodeType.TABLE)}
    net = QGGAT(enc.dim)
    a = net.propagate(G, enc.encode("revenue by region"), active, {})
    b = net.propagate(G, enc.encode("who leads the logistics division"), active, {})
    assert any(not np.allclose(a[k], b[k]) for k in active)


def test_flat_baseline_returns_k(doc_text):
    from hmdkg import HMDKGPipeline, PipelineConfig

    p = HMDKGPipeline(PipelineConfig(flat_retrieval=True, flat_k=3))
    p.index_text(doc_text)
    assert len(p.retriever.retrieve("When was Acme Robotics founded?").evidence) == 3
    assert isinstance(p.retriever, FlatRetriever)
