import pytest
import math
from src.validator import _is_valid_prob

def test_validator_probability_hygiene():
    """Validates the strict vector hygiene required by Phase 0."""
    # Good probabilities
    assert _is_valid_prob(0.5) is True
    assert _is_valid_prob(1.0) is True
    assert _is_valid_prob(0.0) is True
    
    # Boundary / type breaks
    assert _is_valid_prob(-0.1) is False
    assert _is_valid_prob(1.1) is False
    assert _is_valid_prob(None) is False
    assert _is_valid_prob("0.5") is False
    assert _is_valid_prob(float("nan")) is False
    assert _is_valid_prob(float("inf")) is False