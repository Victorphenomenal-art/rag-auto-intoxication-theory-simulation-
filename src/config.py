"""Configuration module for simulation parameters."""

import numpy as np
import torch
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Simulation configuration with settings to make sure we get the same results every time."""

    # 1. Reproducibility (Keeping results the same every time we run it)
    seed: int = 42

    # 2. Experimental parameters (The numbers from the research paper)
    initial_real_facts: int = 20
    initial_fake_facts: int = 3
    total_iterations: int = 50
    retrieval_top_k: int = 3

    # 3. AI Model parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"  # Or we can use gemini-1.5-flash later
    generation_temperature: float = 0.7

    # 4. Correction parameters (For fixing the AI's mistakes)
    provenance_weighting: bool = False
    weighting_gamma: float = 0.8

    # 5. Paths (Where to save our files)
    results_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"

    def __post_init__(self):
        """Set random seeds and create folders automatically."""
        # Make sure our random numbers are exactly the same every time
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create the folders for our results if they don't exist yet
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)