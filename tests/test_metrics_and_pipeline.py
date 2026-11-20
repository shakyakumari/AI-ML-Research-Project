from hmdkg.evaluation import evaluate, run_ablations
from hmdkg.metrics import anls, evidence_recall, exact_match, levenshtein, normalize_answer, rouge_l, token_f1


def test_normalisation_and_em():
    assert normalize_answer("The 1,800 Units.") == "1800 units"
    assert exact_match("the Priya Sharma", ["Priya Sharma"]) == 1.0
    assert exact_match("Sharma", ["Priya Sharma"]) == 0.0


def test_levenshtein_and_anls():
    assert levenshtein("kitten", "sitting") == 3
    assert anls("Bengaluru", ["Bengaluru"]) == 1.0
    assert 0.0 < anls("Bengalur", ["Bengaluru"]) < 1.0
    assert anls("xyz", ["Bengaluru"]) == 0.0  # NL >= 0.5 threshold


def test_f1_rouge_recall():
    assert token_f1("Priya Sharma", ["Priya Sharma"]) == 1.0
    assert 0 < token_f1("Priya", ["Priya Sharma"]) < 1.0
    assert rouge_l("a b c d", "a b c d") == 1.0 and rouge_l("a b", "c d") == 0.0
    assert evidence_recall(["Priya Sharma has served"], ["priya sharma", "missing"]) == 0.5


def test_end_to_end_toy_accuracy(pipe, qa):
    res = evaluate(pipe, qa)
    assert res["EM"] >= 0.9
    assert res["EvidenceRecall"] >= 0.95
    for t in ("factoid", "numerical", "comparative", "multihop"):
        assert res["per_type_EM"][t] >= 0.99, t


def test_ablation_runner_executes_all_variants(doc_text, qa):
    out = run_ablations(doc_text, qa)
    assert len(out) == 7
    full = out["Full (HM-DKG + QC-AER + CA-AAG)"]
    flat = out["w/o HM-DKG + QC-AER (flat top-3)"]
    assert full["EM"] >= flat["EM"]


def test_index_file_roundtrip(tmp_path, doc_text):
    from hmdkg import HMDKGPipeline

    f = tmp_path / "doc.md"
    f.write_text(doc_text, encoding="utf-8")
    p = HMDKGPipeline()
    p.index_file(f)
    assert p.answer("When was Acme Robotics founded?").result.answer == "2009"
