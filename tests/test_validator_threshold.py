from src.validator import realized_label

def test_realized_label_threshold():
    # threshold=0.001 => 0.10%
    assert realized_label(100.0, 100.05, 0.001) == "FLAT"  # +0.05%
    assert realized_label(100.0, 100.11, 0.001) == "UP"    # +0.11%
    assert realized_label(100.0, 99.89, 0.001) == "DOWN"   # -0.11%
