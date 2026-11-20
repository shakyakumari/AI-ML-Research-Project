# HM-DKG DocQA: question-aware hierarchical multimodal graph reasoning

Reference implementation of the three contributions in the project proposal:

| Contribution | Module | What is implemented |
|---|---|---|
| **HM-DKG** (graph construction) | `hmdkg/builder.py`, `graph.py`, `encoders.py` | Markdown/layout elements -> typed heterogeneous graph; structural, semantic (cosine threshold + top-m cap) and cross-reference edges; hierarchical pooling for sections; cell-level table encoding; zoom in/out; ablation flags |
| **QC-AER** (retrieval) | `hmdkg/retrieval.py` | Multi-label intent classifier; intent-conditioned policy (seeds, hops, final-k, type priors); seed scoring `softmax(W_q q ⊙ W_v h)`; question-gated GAT forward pass; adaptive traversal with relative-threshold stopping; multi-hop query expansion; figure/caption/table cluster completion; cross-attention reranker |
| **CA-AAG** (generation) | `hmdkg/generation.py` | Complexity score -> 4 buckets; control table (tokens / evidence / reasoning); format selection; LLM prompt builder; rule-based `ExtractiveGenerator` (runs offline) and `LLMGenerator` adapter for a real multimodal LLM |

Also: `metrics.py` (ANLS, EM, F1, ROUGE-L, evidence recall/precision), `evaluation.py` (harness + 7 ablation variants), `pipeline.py` (end-to-end).

## Quick start

```bash
pip install -r requirements.txt     # numpy, networkx, pytest only
python -m pytest -q                 # 32 tests
python demo.py                      # walk-through, one question per intent type
python demo.py data/sample_report.md "Who leads the Healthcare division?"
python scripts/run_toy_eval.py      # ablation table on the toy document
```

```python
from hmdkg import HMDKGPipeline
pipe = HMDKGPipeline()
pipe.index_file("data/sample_report.md")
out = pipe.answer("What was the revenue of EMEA in 2023 in USD millions?")
out.result.answer        # '110.7'
out.result.structured    # {'value': 110.7, 'unit': 'USD millions', 'row': 'EMEA', 'column': '2023'}
out.retrieval.trace      # seeds, hops used, stop reason, active-subgraph size
```

## Input format

Markdown-like text: `#`/`##` headings, paragraphs, pipe tables (optionally preceded by `Table N: caption`),
`![Figure N: caption](path.png)`, and `<!-- page N -->` markers. A PDF front end (LayoutLMv3 / DocLayNet) only has to
emit the same `Element` list from `builder.py`; nothing downstream changes.

## What is a stand-in, and what to replace

This repository runs without PyTorch or model weights, so several components are deliberately lightweight:

| Paper component | Here | Replace with |
|---|---|---|
| RoBERTa/DeBERTa text encoder | signed hashing + IDF (`HashingTextEncoder`) | sentence/DeBERTa encoder implementing `encode(text) -> unit vector` |
| ViT/CLIP figure encoder | `PlaceholderVisualEncoder` (reads no pixels) | CLIP image tower, or pass `features=` |
| TUTA/TableFormer | cell-text encoder (`TableEncoder`) | TableFormer-style encoder |
| BERT/T5 intent head | regex rules (`IntentClassifier`) | fine-tuned multi-label classifier |
| Learned QG-GAT / reranker / seed projections | forward pass only; `a` is zero-initialised so behaviour is driven by interpretable priors | PyTorch/PyG port + training |
| Fine-tuned multimodal LLM | `ExtractiveGenerator` (rules) | `LLMGenerator(fn)` around LLaVA / Qwen-VL / InternVL |
| Cross-reference model | regex on "Table N" / "Figure N" | learned alignment (open question Q4) |

The default semantic-edge threshold is 0.12 rather than the proposal's 0.75, because hashed TF-IDF cosines are much
lower than neural-encoder cosines; use 0.75 with a neural encoder.

## Design note on QG-GAT

With the additive attention `a^T [W h_i || W h_j || W_q q]`, the question term is the same for every neighbour `j`, so it
cancels in the softmax (apart from the LeakyReLU kink) and cannot route evidence. `QGGAT` therefore injects `q`
multiplicatively (GATv2-style hidden layer plus an explicit `<q, h_j>` term) and adds per-relation routing biases.

## Status and honesty

`scripts/run_toy_eval.py` runs on one hand-written 2-page document with 15 questions. It checks that the components fit
together and that the ablation switches work. It is **not** a benchmark: most ablations tie on it, and the numbers say
nothing about DocVQA / MP-DocVQA / DUDE / InfographicVQA / TAT-QA performance. Running those is the next project phase.

## Layout

```
hmdkg/        library (graph, builder, encoders, retrieval, generation, pipeline, metrics, evaluation)
tests/        32 unit + end-to-end tests
data/         toy document and QA pairs
scripts/      toy evaluation / ablations
```
