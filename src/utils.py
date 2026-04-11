"""Helper functions for the simulation."""
import numpy as np
import torch

def set_global_seeds(seed: int = 42):
    """Ensure our random numbers are exactly the same every time."""
    np.random.seed(seed)
    torch.manual_seed(seed)