"""Tests for helper functions."""
import numpy as np
from src.utils import set_global_seeds


def test_set_global_seeds():
    """Test if the seed locker works perfectly."""
    set_global_seeds(42)
    val1 = np.random.rand()  # Get a random number

    set_global_seeds(42)  # Reset the locker
    val2 = np.random.rand()  # Get another random number

    # Check if the two numbers are exactly the same
    assert val1 == val2