"""Interactive demo:  python demo.py [doc.md] ["your question"]

Without arguments it indexes data/sample_report.md and walks through one question per intent type,
printing the graph statistics, retrieval trace, complexity bucket and the generated answer.
"""
import sys
from pathlib import Path

from hmdkg import HMDKGPipeline

ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    "When was Acme Robotics founded?",
    "What was the revenue of EMEA in 2023 in USD millions?",
    "Which region had higher revenue in 2024, APAC or EMEA?",
    "What does Figure 1 show?",
    "Who leads the division that opened a plant in Chennai?",
    "Describe the main risks facing Acme Robotics.",
]


def show(pipe: HMDKGPipeline, q: str) -> None:
    out = pipe.answer(q)
    r = out.retrieval
    print(f"\nQ: {q}")
    print(f"   intent={r.primary} (labels={r.labels})  complexity={out.complexity:.2f} -> {out.result.bucket}")
    print(f"   retrieval: seeds={len(r.trace.get('seeds', []))} hops={r.trace.get('hops_used')} "
          f"stop={r.trace.get('stop_reason')} active={r.trace.get('active_nodes')}/{r.trace.get('total_nodes')}")
    for e in out.request.evidence:
        print(f"     - [{e.node_id}:{e.type.value}] {e.text[:72]}")
    print(f"   format={out.result.fmt}")
    print(f"   ANSWER: {out.result.answer}")
    if out.result.bucket in ("complex", "highly_complex"):
        print("   TEXT:\n      " + out.result.text.replace("\n", "\n      "))


def main() -> None:
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/sample_report.md"
    pipe = HMDKGPipeline()
    g = pipe.index_file(doc)
    print("HM-DKG:", g.stats())
    for q in (sys.argv[2:] or QUESTIONS):
        show(pipe, q)


if __name__ == "__main__":
    main()
