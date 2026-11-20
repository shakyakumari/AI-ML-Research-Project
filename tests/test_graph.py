import numpy as np

from hmdkg.builder import BuildConfig, build_graph, parse_markdown
from hmdkg.encoders import HashingTextEncoder
from hmdkg.graph import EdgeKind, NodeType


def test_parse_elements(doc_text):
    els = parse_markdown(doc_text)
    kinds = [e.kind for e in els]
    assert kinds.count("table") == 1 and kinds.count("figure") == 1
    table = next(e for e in els if e.kind == "table")
    assert table.extra["label"] == "Table 1"
    assert table.extra["header"] == ["Region", "2022", "2023", "2024"]
    assert len(table.extra["rows"]) == 3
    assert next(e for e in els if e.kind == "figure").extra["label"] == "Figure 1"
    assert {e.page for e in els} == {1, 2}


def test_node_and_edge_types(pipe):
    G = pipe.graph
    s = G.stats()
    assert s["nodes"]["table"] == 1 and s["nodes"]["fig"] == 1 and s["nodes"]["caption"] == 1
    for rel in ("IS_PART_OF", "BELONGS_TO", "HAS_CAPTION", "REFERENCES", "SUPPORTS", "DESCRIBES"):
        assert rel in s["edge_relations"], rel
    assert all(n.emb is not None and abs(np.linalg.norm(n.emb) - 1) < 1e-6 for n in G.nodes.values())


def test_crossref_edges_point_to_table_and_figure(pipe):
    G = pipe.graph
    refs = [(u, v) for u, v, d in G.g.edges(data=True) if d["relation"] == "REFERENCES"]
    assert {G.nodes[v].type for _, v in refs} == {NodeType.TABLE, NodeType.FIG}
    assert all(G.nodes[u].type == NodeType.PARA for u, _ in refs)


def test_zoom_in_and_out(pipe):
    G = pipe.graph
    fin = next(n for n in G.of_type(NodeType.SECTION) if "Financial" in n.text)
    kids = G.zoom_in(fin.id)
    types = {G.nodes[k].type for k in kids}
    assert NodeType.TABLE in types and NodeType.FIG in types
    table = G.of_type(NodeType.TABLE)[0]
    chain = G.zoom_out(table.id)
    assert chain[0] == fin.id and chain[-1] == G.ROOT
    # caption zooms out through its figure
    cap = G.of_type(NodeType.CAPTION)[0]
    assert G.parent(cap.id) == fin.id


def test_cell_level_zoom(pipe):
    G = pipe.graph
    cells = G.table_cells(G.of_type(NodeType.TABLE)[0].id)
    assert len(cells) == 9
    emea_2023 = next(c for c in cells if c["row_label"] == "EMEA" and c["col_header"] == "2023")
    assert emea_2023["value"] == "110.7"


def test_ablation_flags_remove_edges(doc_text):
    enc = HashingTextEncoder(1024)
    els = parse_markdown(doc_text)
    g_all = build_graph(els, enc, cfg=BuildConfig())
    g_nocm = build_graph(els, enc, cfg=BuildConfig(enable_cross_modal=False))
    g_none = build_graph(els, enc, cfg=BuildConfig(enable_semantic=False, enable_crossref=False))
    rel = lambda g: g.stats()["edge_relations"]
    assert "REFERENCES" in rel(g_all) and "REFERENCES" not in rel(g_nocm)
    assert "DESCRIBES" not in rel(g_nocm) and "SEMANTIC_SIM" in rel(g_nocm)
    assert EdgeKind.SEMANTIC.value not in g_none.stats()["edge_kinds"]
