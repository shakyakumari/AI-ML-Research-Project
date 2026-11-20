from hmdkg.generation import (CONTROL, ComplexityEstimator, LLMGenerator, build_prompt, extract_span, rank_rows,
                              select_format, table_lookup)
from hmdkg.graph import NodeType

HEADER = ["Region", "2022", "2023", "2024"]
ROWS = [["APAC", "120.5", "148.2", "187.9"], ["EMEA", "98.3", "110.7", "125.4"], ["Americas", "140.0", "152.6", "161.3"]]


def test_complexity_buckets_are_ordered():
    est = ComplexityEstimator()
    b = lambda q, p, n=3, h=1: est.estimate(q, p, n, h)[1]
    assert b("When was it founded?", "factoid") == "simple"
    assert b("What was the revenue in 2023?", "numerical") == "moderate"
    assert b("Describe the risks.", "descriptive") == "complex"
    assert b("Who leads the division that opened a plant in Chennai?", "multihop", 6, 2) == "highly_complex"


def test_control_table_matches_design():
    assert [CONTROL[k].max_tokens for k in ("simple", "moderate", "complex", "highly_complex")] == [32, 128, 512, 1024]
    assert CONTROL["simple"].evidence_k == 1 and CONTROL["highly_complex"].evidence_k is None


def test_table_lookup_and_ranking():
    r = table_lookup("What was the revenue of EMEA in 2023?", HEADER, ROWS)
    assert r["value"] == "110.7"
    assert rank_rows("Which region had higher revenue in 2024, APAC or EMEA?", HEADER, ROWS)["winner"] == "APAC"
    assert rank_rows("Which region had the lowest revenue in 2022?", HEADER, ROWS)["winner"] == "EMEA"
    assert table_lookup("What colour is the sky?", HEADER, ROWS) is None


def test_extract_span():
    assert extract_span("Who is the CEO of Acme?", "Priya Sharma has served as CEO since 2018.") == "Priya Sharma"
    assert extract_span("When was Acme founded?", "Acme was founded in 2009 in Pune, India.") == "2009"
    assert extract_span("Where did the headquarters move?", "The headquarters moved to Bengaluru in 2015.") == "Bengaluru"
    assert extract_span("How many people does it employ?", "It employs 420 people.") == "420"


def test_format_selection():
    assert select_format("numerical", {NodeType.TABLE}) == "cell_value"
    assert select_format("comparative", {NodeType.TABLE}) == "ranked_list"
    assert select_format("descriptive", {NodeType.PARA}) == "paragraph_with_citations"
    assert select_format("factoid", {NodeType.FIG}) == "grounded_figure"
    assert select_format("factoid", {NodeType.PARA}) == "short_span"


def test_answer_length_respects_budget(pipe):
    out = pipe.answer("When was Acme Robotics founded?")
    assert out.result.bucket == "simple"
    assert len(out.result.text.split()) <= CONTROL["simple"].max_tokens
    assert len(out.request.evidence) == 1  # simple => top-1 evidence only


def test_numerical_answer_has_structured_unit(pipe):
    out = pipe.answer("What was the revenue of EMEA in 2023 in USD millions?")
    assert out.result.answer == "110.7"
    assert out.result.structured["unit"] == "USD millions"
    assert out.result.fmt == "cell_value"


def test_multihop_emits_reasoning_chain(pipe):
    out = pipe.answer("Who leads the division that opened a plant in Chennai?")
    assert out.result.answer == "Rahul Verma"
    assert "Step 1" in out.result.text and "Final Answer: Rahul Verma" in out.result.text


def test_prompt_contains_all_parts(pipe):
    out = pipe.answer("Who leads the division that opened a plant in Chennai?")
    prompt = build_prompt(out.request)
    assert prompt.startswith("[CLS]") and "<COMPLEXITY=highly_complex>" in prompt and "[REASONING=structured]" in prompt


def test_llm_generator_adapter(pipe):
    pipe.generator = LLMGenerator(lambda prompt, n: "Step 1: ...\nFinal Answer: Priya Sharma")
    assert pipe.answer("Who is the Chief Executive Officer of Acme Robotics?").result.answer == "Priya Sharma"
