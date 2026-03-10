from __future__ import annotations

from datetime import datetime, timezone

from src.features import _infer_daily_snapshot_effective_ts_utc


def test_infer_daily_snapshot_ts_from_rows_newest_first() -> None:
    payload = {"data": [{"date": "2026-03-06"}, {"date": "2026-03-05"}]}
    inferred = _infer_daily_snapshot_effective_ts_utc(payload)
    assert inferred == datetime(2026, 3, 6, 0, 0, 0, tzinfo=timezone.utc)


def test_infer_daily_snapshot_ts_from_rows_oldest_first() -> None:
    payload = {"data": [{"date": "2026-03-05"}, {"date": "2026-03-06"}]}
    inferred = _infer_daily_snapshot_effective_ts_utc(payload)
    assert inferred == datetime(2026, 3, 6, 0, 0, 0, tzinfo=timezone.utc)


def test_infer_daily_snapshot_ts_from_rows_as_of_key() -> None:
    payload = {"data": [{"as_of": "2026-03-06"}]}
    inferred = _infer_daily_snapshot_effective_ts_utc(payload)
    assert inferred == datetime(2026, 3, 6, 0, 0, 0, tzinfo=timezone.utc)


def test_infer_daily_snapshot_ts_rejects_timestamp_strings() -> None:
    payload = {"data": [{"date": "2026-03-06T12:00:00Z"}]}
    inferred = _infer_daily_snapshot_effective_ts_utc(payload)
    assert inferred is None


def test_infer_daily_snapshot_ts_from_top_level_list() -> None:
    payload = [{"date": "2026-03-06"}]
    inferred = _infer_daily_snapshot_effective_ts_utc(payload)
    assert inferred == datetime(2026, 3, 6, 0, 0, 0, tzinfo=timezone.utc)
