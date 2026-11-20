import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmdkg import HMDKGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def doc_text():
    return (ROOT / "data/sample_report.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def qa():
    return json.loads((ROOT / "data/sample_qa.json").read_text(encoding="utf-8"))


@pytest.fixture()
def pipe(doc_text):
    p = HMDKGPipeline()
    p.index_text(doc_text)
    return p
