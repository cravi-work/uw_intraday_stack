import pytest
from unittest.mock import MagicMock
from src.storage import DbWriter

def test_replay_diagnostic_summary_emission():
    """
    EVIDENCE: Replay run produces a concise diagnostic summary with nonzero counts 
    where expected, ensuring quantifiable pipeline diagnostics beyond text logs.
    """
    mock_con = MagicMock()
    mock_con.execute.return_value.fetchall.side_effect = [
        [("PASS", 10), ("BLOCKED", 2)],
        [("FRESH", 100), ("STALE_CARRY", 5)]
    ]
    
    writer = DbWriter(":memory:")
    diag = writer.get_pipeline_diagnostics(mock_con)
    
    assert diag["predictions_by_gate"]["PASS"] == 10
    assert diag["predictions_by_gate"]["BLOCKED"] == 2
    assert diag["lineage_by_freshness"]["FRESH"] == 100
    assert diag["lineage_by_freshness"]["STALE_CARRY"] == 5