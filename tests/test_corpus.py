import pytest
from src.config import Config
from src.corpus import ProvenanceKB

def test_corpus_initialization():
    config = Config()
    config.initial_real_facts = 20
    config.initial_fake_facts = 3
    kb = ProvenanceKB(config)
    assert kb is not None
