from __future__ import annotations

import datetime

from src.endpoint_truth import infer_source_time_hints


UTC = datetime.timezone.utc


def test_infer_source_time_hints_parses_epoch_millis_string() -> None:
    payload = {
        "data": [
            {"t": "1700000000000", "c": 100.0},
            {"t": "1700003600000", "c": 101.0},
        ]
    }

    hints = infer_source_time_hints(
        payload_json=payload,
        response_headers={},
        reference_utc=datetime.datetime(2026, 3, 6, 15, 16, tzinfo=UTC),
        provider_naive_tz="UTC",
    )

    assert hints.event_time_utc is not None
    # 1,700,003,600,000 ms -> 1,700,003,600 s
    expected = datetime.datetime.fromtimestamp(1700003600, tz=UTC)
    assert hints.event_time_utc == expected


def test_infer_source_time_hints_parses_epoch_seconds_string() -> None:
    payload = {"timestamp": "1700003600"}

    hints = infer_source_time_hints(
        payload_json=payload,
        response_headers={},
        reference_utc=datetime.datetime(2026, 3, 6, 15, 16, tzinfo=UTC),
        provider_naive_tz="UTC",
    )

    assert hints.event_time_utc is not None
    expected = datetime.datetime.fromtimestamp(1700003600, tz=UTC)
    assert hints.event_time_utc == expected
