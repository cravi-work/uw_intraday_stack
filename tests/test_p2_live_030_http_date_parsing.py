from src.uw_client import _parse_http_date_to_utc_iso


def test_parse_http_date_to_utc_iso_valid_rfc1123():
    # RFC 1123 date format used in HTTP Date header
    iso = _parse_http_date_to_utc_iso("Mon, 09 Mar 2026 20:30:00 GMT")
    assert iso == "2026-03-09T20:30:00+00:00"


def test_parse_http_date_to_utc_iso_invalid_returns_none():
    assert _parse_http_date_to_utc_iso("not a date") is None
