import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import Settings
from copilot.ingest import load_corpus
from copilot.retriever import HybridRetriever


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    import os

    os.environ["COPILOT_INDEX_PATH"] = str(tmp_path_factory.mktemp("idx") / "index.pkl")
    os.environ["LLM_PROVIDER"] = "local"
    return Settings()


@pytest.fixture(scope="session")
def retriever(settings):
    return HybridRetriever(load_corpus(settings.corpus_dir))
