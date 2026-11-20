"""Run the full pipeline and all ablations on the bundled toy document; write results/toy_eval.json.

This is a *sanity check* on one hand-written document with 15 questions - NOT a benchmark result.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmdkg.evaluation import load, run_ablations  # noqa: E402


def main() -> None:
    doc, qa = load(ROOT / "data/sample_report.md", ROOT / "data/sample_qa.json")
    results = run_ablations(doc, qa)
    hdr = f"{'Variant':46s} {'EM':>6s} {'ANLS':>6s} {'ROUGE-L':>8s} {'Recall':>7s} {'Prec':>6s} {'#Ev':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        f = lambda x: f"{x:6.3f}" if x is not None else "   n/a"
        print(f"{name:46s} {f(r['EM'])} {f(r['ANLS'])} {f(r['ROUGE-L']):>8s} {f(r['EvidenceRecall']):>7s} {f(r['EvidencePrecision'])} {r['AvgEvidence']:5.1f}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "toy_eval.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out / 'toy_eval.json'}")


if __name__ == "__main__":
    main()
