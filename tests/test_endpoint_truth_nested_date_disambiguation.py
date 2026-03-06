import datetime

from src.endpoint_truth import infer_source_time_hints


def test_infer_source_time_hints_ignores_nested_date_only_date_key() -> None:
    """Nested date-only "date" fields are commonly option expiry dates.

    They must not be interpreted as the payload as-of timestamp.
    """

    reference_utc = datetime.datetime(2026, 3, 6, 12, 16, 11, tzinfo=datetime.timezone.utc)

    payload = {
        "data": [
            {"date": "2025-04-22", "vanna": 1.0},
            {"date": "2025-05-22", "vanna": 2.0},
        ]
    }

    hints = infer_source_time_hints(payload_json=payload, response_headers={}, reference_utc=reference_utc)

    assert hints.event_time_utc is None
    assert hints.effective_time_utc is None


def test_infer_source_time_hints_allows_top_level_date_only_date_key() -> None:
    reference_utc = datetime.datetime(2026, 3, 6, 12, 16, 11, tzinfo=datetime.timezone.utc)

    payload = {"date": "2026-03-06"}

    hints = infer_source_time_hints(payload_json=payload, response_headers={}, reference_utc=reference_utc)

    # Date-only strings are interpreted as midnight in the provider's local timezone.
    # (America/New_York midnight -> 05:00 UTC on 2026-03-06)
    assert hints.event_time_utc == datetime.datetime(2026, 3, 6, 5, 0, 0, tzinfo=datetime.timezone.utc)


def test_infer_source_time_hints_allows_nested_date_with_time_component() -> None:
    reference_utc = datetime.datetime(2026, 3, 6, 12, 16, 11, tzinfo=datetime.timezone.utc)

    payload = {"data": [{"date": "2026-03-06T12:15:00Z"}]}

    hints = infer_source_time_hints(payload_json=payload, response_headers={}, reference_utc=reference_utc)

    assert hints.event_time_utc == datetime.datetime(2026, 3, 6, 12, 15, 0, tzinfo=datetime.timezone.utc)
