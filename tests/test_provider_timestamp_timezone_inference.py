import datetime as dt

from src.endpoint_truth import infer_source_time_hints


def test_provider_naive_timestamp_assumes_exchange_tz_when_closer_to_reference():
    # 2026-03-05 09:30 ET == 14:30 UTC (EST; DST not started yet)
    asof_utc = dt.datetime(2026, 3, 5, 14, 30, tzinfo=dt.timezone.utc)

    hints = infer_source_time_hints(
        payload_json={"timestamp": "2026-03-05T09:30:00"},
        response_headers={},
        reference_utc=asof_utc,
    )

    assert hints.event_time_utc == asof_utc


def test_provider_naive_timestamp_keeps_utc_when_closer_to_reference():
    asof_utc = dt.datetime(2026, 3, 5, 14, 30, tzinfo=dt.timezone.utc)

    hints = infer_source_time_hints(
        payload_json={"timestamp": "2026-03-05T14:30:00"},
        response_headers={},
        reference_utc=asof_utc,
    )

    assert hints.event_time_utc == asof_utc


def test_provider_candle_start_end_time_is_inferred_as_event_time():
    """UW OHLC Candle objects commonly surface timestamps as start_time/end_time (UTC)."""
    asof_utc = dt.datetime(2026, 3, 5, 14, 32, tzinfo=dt.timezone.utc)

    hints = infer_source_time_hints(
        payload_json={
            "data": [
                {
                    "start_time": "2026-03-05T14:31:00Z",
                    "end_time": "2026-03-05T14:32:00Z",
                    "open": 149.5,
                    "high": 150.2,
                    "low": 149.1,
                    "close": 150.0,
                    "volume": 12345,
                }
            ]
        },
        response_headers={},
        reference_utc=asof_utc,
    )

    assert hints.event_time_utc == asof_utc
    assert hints.effective_time_utc == asof_utc
