"""Shared test fixtures for Quantum-Neural-Search."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
