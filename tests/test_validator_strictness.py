import pytest
import math
from src.validator import _is_valid_prob

def test_probability_vectors():
    assert _is_valid_prob(0.5)
    assert not _is_valid_prob(-0.1)
    assert not _is_valid_prob(None)
    assert not _is_valid_prob(float("nan"))