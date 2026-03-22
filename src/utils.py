"""Utility functions."""

import json
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_api_keys(filepath: str) -> list:
    """Load API keys from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('keys', [])
```

tests/test_corpus.py content

```python
import pytest
from src.config import Config
from src.corpus import ProvenanceKB

def test_corpus_initialization():
    config = Config()
    config.initial_real_facts = 20
    config.initial_fake_facts = 3
    kb = ProvenanceKB(config)
    assert kb is not None
