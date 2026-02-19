from src.validator import realized_label

def test_realized_label():
    start = 100.0
    thr = 0.001  # 0.10%
    assert realized_label(start, 100.05, thr) == "FLAT"
    assert realized_label(start, 100.11, thr) == "UP"
    assert realized_label(start, 99.89, thr) == "DOWN"
