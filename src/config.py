```python
"""Configuration module for simulation parameters."""

from dataclasses import dataclass
import os
import numpy as np
import torch

@dataclass
class Config:
    """Simulation configuration with reproducibility settings."""
    
    # Reproducibility
    seed: int = 42
    
    # Experimental parameters
    initial_real_facts: int = 20
    initial_fake_facts: int = 3
    total_iterations: int = 50
    retrieval_top_k: int = 3
    
    # Model parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"  # or "gemini-1.5-flash"
    generation_temperature: float = 0.7
    
    # Correction parameters
    provenance_weighting: bool = False
    weighting_gamma: float = 0.8
    
    # Paths
    results_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    
    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
