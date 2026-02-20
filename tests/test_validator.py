from src.models import predicted_class

def test_predicted_class_determinism():
    """Asserts stable tie-breaking and NaN fallbacks for predictions."""
    # Tie rule must strictly favor FLAT > UP > DOWN
    assert predicted_class(0.33, 0.33, 0.33) == "FLAT"
    assert predicted_class(0.40, 0.40, 0.20) == "UP"
    assert predicted_class(0.30, 0.40, 0.30) == "DOWN"
    
    # Graceful degradation of broken maths (Nones / NaNs)
    assert predicted_class(float("nan"), 0.5, 0.5) == "FLAT"
    assert predicted_class(None, None, None) == "FLAT"