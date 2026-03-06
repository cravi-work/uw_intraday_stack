from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .na import is_na, safe_float

CALL_ALIASES = {"C", "CALL", "CALLS"}
PUT_ALIASES = {"P", "PUT", "PUTS"}

OPTION_SYMBOL_FIELDS: Tuple[str, ...] = (
    "option_symbol",
    "occ_symbol",
    "contract_symbol",
    "display_symbol",
    "option_contract",
    "contract",
    "option",
)
UNDERLYING_FIELDS: Tuple[str, ...] = (
    "underlying",
    "underlying_symbol",
    "underlier",
    "underlier_symbol",
    "underlying_ticker",
    "root",
)
EXPIRATION_FIELDS: Tuple[str, ...] = (
    "expiration",
    "expiry",
    "expiration_date",
    "expiry_date",
    "exp_date",
    "exp",
    "maturity_date",
)
STRIKE_FIELDS: Tuple[str, ...] = (
    "strike",
    "strike_price",
    "exercise_price",
)
PUT_CALL_FIELDS: Tuple[str, ...] = (
    "put_call",
    "option_type",
    "type",
    "pc",
    "right",
)
MULTIPLIER_FIELDS: Tuple[str, ...] = (
    "multiplier",
    "contract_multiplier",
    "contract_size",
    "shares_per_contract",
    "share_multiplier",
)
DELIVERABLE_SHARES_FIELDS: Tuple[str, ...] = (
    "deliverable_shares",
    "deliverable_share_count",
    "shares_deliverable",
    "deliverable_quantity",
)
DELIVERABLE_FIELDS: Tuple[str, ...] = (
    "deliverable",
    "deliverables",
    "deliverable_desc",
)
ADJUSTMENT_FLAG_FIELDS: Tuple[str, ...] = (
    "adjusted",
    "is_adjusted",
    "nonstandard",
    "adjustment_flag",
    "has_adjustment",
)
CONTRACT_ID_FIELDS: Tuple[str, ...] = (
    "contract_id",
    "option_id",
    "instrument_id",
)

_OCC_REGEX = re.compile(r"^([A-Z0-9\.]{1,6})(\d{6})([CP])(\d{8})$")
_DATE_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y%m%d",
    "%y%m%d",
)


@dataclass(frozen=True)
class OptionContractIdentity:
    underlying: str
    expiration: str
    strike: float
    put_call: str
    multiplier: float
    deliverable_shares: float
    adjustment_flag: bool
    deliverable_repr: str
    canonical_contract_key: str
    canonical_series_key: str
    source_symbol: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "underlying": self.underlying,
            "expiration": self.expiration,
            "strike": self.strike,
            "put_call": self.put_call,
            "multiplier": self.multiplier,
            "deliverable_shares": self.deliverable_shares,
            "adjustment_flag": self.adjustment_flag,
            "deliverable": self.deliverable_repr,
            "canonical_contract_key": self.canonical_contract_key,
            "canonical_series_key": self.canonical_series_key,
            "source_symbol": self.source_symbol,
        }


@dataclass(frozen=True)
class ContractNormalizationResult:
    status: str
    identity: Optional[OptionContractIdentity]
    failure_reason: Optional[str]
    row_index: int
    relevant: bool
    source_fields: Tuple[str, ...]
    details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "row_index": self.row_index,
            "relevant": self.relevant,
            "source_fields": list(self.source_fields),
            "details": self.details,
            "identity": self.identity.as_dict() if self.identity else None,
        }


@dataclass(frozen=True)
class NormalizedOptionRow:
    row_index: int
    row: Mapping[str, Any]
    identity: OptionContractIdentity


@dataclass
class ContractNormalizationSummary:
    status: str
    results: List[ContractNormalizationResult]
    normalized_rows: List[NormalizedOptionRow]
    required_row_count: int
    invalid_row_count: int
    adjusted_contract_count: int
    duplicate_contract_keys: int
    failure_reason: Optional[str] = None
    series_conflicts: Optional[List[Dict[str, Any]]] = None

    def as_dict(self) -> Dict[str, Any]:
        sample_keys = [r.identity.canonical_contract_key for r in self.normalized_rows[:5]]
        inferred_standard_contract_count = sum(
            1
            for r in self.results
            if r.status == "NORMALIZED" and bool((r.details or {}).get("inferred_standard_contract"))
        )
        return {
            "status": self.status,
            "required_row_count": self.required_row_count,
            "normalized_row_count": len(self.normalized_rows),
            "invalid_row_count": self.invalid_row_count,
            "adjusted_contract_count": self.adjusted_contract_count,
            "duplicate_contract_keys": self.duplicate_contract_keys,
            "inferred_standard_contract_count": inferred_standard_contract_count,
            "failure_reason": self.failure_reason,
            "series_conflicts": list(self.series_conflicts or []),
            "sample_contract_keys": sample_keys,
            "failures": [
                {
                    "row_index": r.row_index,
                    "failure_reason": r.failure_reason,
                    "details": r.details,
                }
                for r in self.results
                if r.status == "INVALID"
            ][:3],
        }


def normalize_option_rows(rows: Sequence[Mapping[str, Any]], *, default_underlying: Optional[str] = None) -> ContractNormalizationSummary:
    results: List[ContractNormalizationResult] = []
    normalized_rows: List[NormalizedOptionRow] = []
    required_row_count = 0
    invalid_row_count = 0
    adjusted_contract_count = 0

    for idx, row in enumerate(rows):
        res = normalize_option_contract(row, row_index=idx, default_underlying=default_underlying)
        results.append(res)
        if res.relevant:
            required_row_count += 1
        if res.status == "INVALID":
            invalid_row_count += 1
        elif res.status == "NORMALIZED" and res.identity is not None:
            normalized_rows.append(NormalizedOptionRow(idx, row, res.identity))
            if res.identity.adjustment_flag:
                adjusted_contract_count += 1

    duplicate_contract_keys = 0
    full_key_counts: Dict[str, int] = {}
    series_shapes: Dict[str, set[Tuple[float, float, str]]] = {}
    for nrow in normalized_rows:
        ident = nrow.identity
        full_key_counts[ident.canonical_contract_key] = full_key_counts.get(ident.canonical_contract_key, 0) + 1
        series_shapes.setdefault(ident.canonical_series_key, set()).add(
            (round(ident.multiplier, 8), round(ident.deliverable_shares, 8), ident.deliverable_repr)
        )
    duplicate_contract_keys = sum(max(0, count - 1) for count in full_key_counts.values())

    series_conflicts: List[Dict[str, Any]] = []
    for series_key, shapes in sorted(series_shapes.items()):
        if len(shapes) <= 1:
            continue
        series_conflicts.append(
            {
                "series_key": series_key,
                "conflicting_shapes": [
                    {
                        "multiplier": shape[0],
                        "deliverable_shares": shape[1],
                        "deliverable": shape[2],
                    }
                    for shape in sorted(shapes)
                ],
            }
        )

    status = "NOT_APPLICABLE"
    failure_reason: Optional[str] = None
    if required_row_count > 0:
        status = "NORMALIZED"
        if invalid_row_count > 0:
            status = "INVALID"
            failure_reason = next((r.failure_reason for r in results if r.status == "INVALID" and r.failure_reason), "contract_normalization_invalid")
        elif series_conflicts:
            status = "INVALID"
            failure_reason = "contract_multiplier_conflict"

    return ContractNormalizationSummary(
        status=status,
        results=results,
        normalized_rows=normalized_rows,
        required_row_count=required_row_count,
        invalid_row_count=invalid_row_count,
        adjusted_contract_count=adjusted_contract_count,
        duplicate_contract_keys=duplicate_contract_keys,
        failure_reason=failure_reason,
        series_conflicts=series_conflicts,
    )


def normalize_option_contract(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
    default_underlying: Optional[str] = None,
) -> ContractNormalizationResult:
    relevant, source_fields = _row_requires_normalization(row)
    if not relevant:
        return ContractNormalizationResult(
            status="NOT_APPLICABLE",
            identity=None,
            failure_reason=None,
            row_index=row_index,
            relevant=False,
            source_fields=tuple(source_fields),
            details={},
        )

    source_symbol, occ_data = _extract_contract_symbol_and_occ(row)
    explicit_underlying = _extract_underlying(row) or _normalize_underlying(default_underlying)
    explicit_expiration = _extract_expiration(row)
    explicit_strike = _extract_strike(row)
    explicit_put_call = _extract_put_call(row)

    occ_underlying = _normalize_underlying(occ_data.get("underlying")) if occ_data else None
    occ_expiration = occ_data.get("expiration") if occ_data else None
    occ_strike = occ_data.get("strike") if occ_data else None
    occ_put_call = occ_data.get("put_call") if occ_data else None

    failure = _first_conflict(
        ("underlying", explicit_underlying, occ_underlying, "contract_symbol_underlying_conflict"),
        ("expiration", explicit_expiration, occ_expiration, "contract_symbol_expiration_conflict"),
        ("strike", explicit_strike, occ_strike, "contract_symbol_strike_conflict"),
        ("put_call", explicit_put_call, occ_put_call, "contract_symbol_put_call_conflict"),
    )
    if failure is not None:
        return _invalid_result(row_index, source_fields, failure, {"source_symbol": source_symbol})

    underlying = explicit_underlying or occ_underlying
    expiration = explicit_expiration or occ_expiration
    strike = explicit_strike if explicit_strike is not None else occ_strike
    put_call = explicit_put_call or occ_put_call

    if source_symbol and occ_data is None and not all(v is not None for v in (underlying, expiration, strike, put_call)):
        return _invalid_result(
            row_index,
            source_fields,
            "display_symbol_only_identity",
            {"source_symbol": source_symbol},
        )

    if underlying is None:
        return _invalid_result(row_index, source_fields, "missing_underlying", {"source_symbol": source_symbol})
    if expiration is None:
        return _invalid_result(row_index, source_fields, "missing_expiration", {"source_symbol": source_symbol})
    if strike is None:
        return _invalid_result(row_index, source_fields, "missing_strike", {"source_symbol": source_symbol})
    if put_call is None:
        return _invalid_result(row_index, source_fields, "missing_put_call", {"source_symbol": source_symbol})

    multiplier = _extract_multiplier(row)
    deliverable_shares, deliverable_repr = _extract_deliverable(row)
    deliverable_present = any(field in row and not is_na(row.get(field)) for field in DELIVERABLE_FIELDS + DELIVERABLE_SHARES_FIELDS)
    if multiplier is not None and deliverable_shares is not None and not math.isclose(multiplier, deliverable_shares, rel_tol=0.0, abs_tol=1e-9):
        return _invalid_result(
            row_index,
            source_fields,
            "multiplier_deliverable_mismatch",
            {
                "multiplier": multiplier,
                "deliverable_shares": deliverable_shares,
                "source_symbol": source_symbol,
            },
        )
    if deliverable_present and deliverable_shares is None:
        return _invalid_result(
            row_index,
            source_fields,
            "unparsed_deliverable",
            {"deliverable": deliverable_repr, "source_symbol": source_symbol},
        )

    normalized_multiplier = multiplier if multiplier is not None else deliverable_shares
    normalized_deliverable = deliverable_shares if deliverable_shares is not None else multiplier
    if normalized_deliverable is not None and (not deliverable_repr or deliverable_repr == _default_deliverable_repr(100.0)):
        deliverable_repr = _default_deliverable_repr(normalized_deliverable)

    explicit_adjusted = _extract_adjustment_flag(row)
    inferred_standard_contract = False

    # Live provider payloads sometimes omit both multiplier and deliverable information for
    # *standard* contracts. If the row is not explicitly adjusted, infer a standard
    # 100-share contract shape so decision-eligible features remain operational.
    #
    # If the row is explicitly adjusted, we must reject (cannot safely infer).
    if normalized_multiplier is None or normalized_deliverable is None:
        if explicit_adjusted:
            return _invalid_result(
                row_index,
                source_fields,
                "missing_multiplier",
                {"source_symbol": source_symbol, "explicit_adjusted": True},
            )

        inferred_standard_contract = True
        normalized_multiplier = normalized_multiplier or 100.0
        normalized_deliverable = normalized_deliverable or 100.0
        if not deliverable_repr:
            deliverable_repr = _default_deliverable_repr(float(normalized_deliverable))
    if normalized_multiplier <= 0 or normalized_deliverable <= 0:
        return _invalid_result(
            row_index,
            source_fields,
            "invalid_multiplier",
            {
                "multiplier": normalized_multiplier,
                "deliverable_shares": normalized_deliverable,
                "source_symbol": source_symbol,
            },
        )

    adjustment_flag = bool(
        explicit_adjusted
        or not math.isclose(normalized_multiplier, 100.0, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(normalized_deliverable, 100.0, rel_tol=0.0, abs_tol=1e-9)
        or deliverable_repr != _default_deliverable_repr(100.0)
    )

    canonical_series_key = f"{underlying}|{expiration}|{put_call}|{_format_strike(strike)}"
    canonical_contract_key = (
        f"{canonical_series_key}|mult={_format_number(normalized_multiplier)}|"
        f"deliverable={deliverable_repr}|adj={1 if adjustment_flag else 0}"
    )

    identity = OptionContractIdentity(
        underlying=underlying,
        expiration=expiration,
        strike=strike,
        put_call=put_call,
        multiplier=normalized_multiplier,
        deliverable_shares=normalized_deliverable,
        adjustment_flag=adjustment_flag,
        deliverable_repr=deliverable_repr,
        canonical_contract_key=canonical_contract_key,
        canonical_series_key=canonical_series_key,
        source_symbol=source_symbol,
    )
    return ContractNormalizationResult(
        status="NORMALIZED",
        identity=identity,
        failure_reason=None,
        row_index=row_index,
        relevant=True,
        source_fields=tuple(source_fields),
        details={
            **({"source_symbol": source_symbol} if source_symbol else {}),
            "explicit_adjusted": explicit_adjusted,
            **({"inferred_standard_contract": True} if inferred_standard_contract else {}),
        },
    )


def contract_scale(identity: OptionContractIdentity) -> float:
    return identity.deliverable_shares


def normalized_contract_map(summary: ContractNormalizationSummary) -> Dict[int, OptionContractIdentity]:
    return {n.row_index: n.identity for n in summary.normalized_rows}


def _invalid_result(row_index: int, source_fields: Iterable[str], failure_reason: str, details: Optional[Dict[str, Any]] = None) -> ContractNormalizationResult:
    return ContractNormalizationResult(
        status="INVALID",
        identity=None,
        failure_reason=failure_reason,
        row_index=row_index,
        relevant=True,
        source_fields=tuple(source_fields),
        details=details or {},
    )


def _row_requires_normalization(row: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    present: List[str] = []
    for field in OPTION_SYMBOL_FIELDS + MULTIPLIER_FIELDS + DELIVERABLE_SHARES_FIELDS + DELIVERABLE_FIELDS + ADJUSTMENT_FLAG_FIELDS + CONTRACT_ID_FIELDS:
        if field in row and not is_na(row.get(field)):
            present.append(field)
    has_expiration = any(field in row and not is_na(row.get(field)) for field in EXPIRATION_FIELDS)
    has_strike = any(field in row and not is_na(row.get(field)) for field in STRIKE_FIELDS)
    has_put_call = any(field in row and not is_na(row.get(field)) for field in PUT_CALL_FIELDS)
    if has_expiration and has_strike and has_put_call:
        present.extend([field for field in ("expiration_identity",) if field not in present])
    return (len(present) > 0), present


def _extract_contract_symbol_and_occ(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    symbol: Optional[str] = None
    for field in OPTION_SYMBOL_FIELDS:
        raw = row.get(field)
        if is_na(raw):
            continue
        symbol = str(raw).strip()
        if symbol:
            break
    if symbol is None and "symbol" in row and not is_na(row.get("symbol")):
        candidate = str(row.get("symbol")).strip()
        if _parse_occ_symbol(candidate) is not None:
            symbol = candidate
    occ_data = _parse_occ_symbol(symbol) if symbol else None
    return symbol, occ_data


def _parse_occ_symbol(symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    if not symbol:
        return None
    compact = str(symbol).upper().replace(" ", "")
    match = _OCC_REGEX.match(compact)
    if not match:
        return None
    root, yyMMdd, cp, strike_digits = match.groups()
    try:
        expiration = dt.datetime.strptime(yyMMdd, "%y%m%d").date().isoformat()
    except ValueError:
        return None
    strike = safe_float(int(strike_digits) / 1000.0)
    if strike is None:
        return None
    return {
        "underlying": _normalize_underlying(root),
        "expiration": expiration,
        "strike": strike,
        "put_call": "CALL" if cp == "C" else "PUT",
    }


def _extract_underlying(row: Mapping[str, Any]) -> Optional[str]:
    for field in UNDERLYING_FIELDS:
        value = row.get(field)
        norm = _normalize_underlying(value)
        if norm is not None:
            return norm
    if "ticker" in row:
        norm = _normalize_underlying(row.get("ticker"))
        if norm is not None:
            return norm
    return None


def _normalize_underlying(value: Any) -> Optional[str]:
    if is_na(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text


def _extract_expiration(row: Mapping[str, Any]) -> Optional[str]:
    for field in EXPIRATION_FIELDS:
        if field not in row:
            continue
        parsed = _normalize_expiration(row.get(field))
        if parsed is not None:
            return parsed
    return None


def _normalize_expiration(value: Any) -> Optional[str]:
    if value is None or is_na(value):
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    try:
        return dt.date.fromisoformat(text.replace("/", "-")).isoformat()
    except ValueError:
        return None


def _extract_strike(row: Mapping[str, Any]) -> Optional[float]:
    for field in STRIKE_FIELDS:
        if field not in row:
            continue
        strike = safe_float(row.get(field))
        if strike is not None and math.isfinite(strike):
            return strike
    return None


def _extract_put_call(row: Mapping[str, Any]) -> Optional[str]:
    for field in PUT_CALL_FIELDS:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        norm = str(value).upper().strip()
        if norm in CALL_ALIASES:
            return "CALL"
        if norm in PUT_ALIASES:
            return "PUT"
    return None


def _extract_multiplier(row: Mapping[str, Any]) -> Optional[float]:
    for field in MULTIPLIER_FIELDS:
        if field not in row:
            continue
        value = safe_float(row.get(field))
        if value is not None and math.isfinite(value):
            return value
    return None


def _extract_deliverable(row: Mapping[str, Any]) -> Tuple[Optional[float], str]:
    for field in DELIVERABLE_SHARES_FIELDS:
        if field not in row:
            continue
        value = safe_float(row.get(field))
        if value is not None and math.isfinite(value):
            return value, _default_deliverable_repr(value)

    for field in DELIVERABLE_FIELDS:
        if field not in row:
            continue
        value = row.get(field)
        shares, repr_str = _parse_deliverable_value(value)
        if shares is not None:
            return shares, repr_str
        if repr_str:
            return None, repr_str

    return None, _default_deliverable_repr(100.0)


def _parse_deliverable_value(value: Any) -> Tuple[Optional[float], str]:
    if value is None or is_na(value):
        return None, _default_deliverable_repr(100.0)
    if isinstance(value, (int, float)):
        num = safe_float(value)
        if num is not None:
            return num, _default_deliverable_repr(num)
        return None, _default_deliverable_repr(100.0)
    if isinstance(value, str):
        num = safe_float(value)
        if num is not None:
            return num, _default_deliverable_repr(num)
        return None, value.strip()
    if isinstance(value, Mapping):
        share_keys = (
            "shares",
            "share_count",
            "quantity",
            "qty",
            "amount",
            "units",
        )
        for key in share_keys:
            num = safe_float(value.get(key))
            if num is not None and math.isfinite(num):
                return num, _default_deliverable_repr(num)
        return None, json.dumps(dict(value), sort_keys=True, default=str)
    if isinstance(value, list):
        share_total = 0.0
        found_share = False
        serializable: List[Any] = []
        for item in value:
            serializable.append(item)
            if isinstance(item, Mapping):
                asset_type = str(item.get("asset_type") or item.get("type") or item.get("kind") or "").strip().lower()
                if asset_type in {"share", "shares", "stock", "equity"}:
                    qty = safe_float(item.get("quantity") or item.get("qty") or item.get("shares") or item.get("amount"))
                    if qty is not None and math.isfinite(qty):
                        share_total += qty
                        found_share = True
        if found_share:
            return share_total, _default_deliverable_repr(share_total)
        return None, json.dumps(serializable, sort_keys=True, default=str)
    return None, json.dumps(value, sort_keys=True, default=str)


def _extract_adjustment_flag(row: Mapping[str, Any]) -> bool:
    for field in ADJUSTMENT_FLAG_FIELDS:
        value = row.get(field)
        if value is None or is_na(value):
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return False


def _first_conflict(*items: Tuple[str, Any, Any, str]) -> Optional[str]:
    for _name, explicit, parsed, reason in items:
        if explicit is None or parsed is None:
            continue
        if isinstance(explicit, float) or isinstance(parsed, float):
            if not math.isclose(float(explicit), float(parsed), rel_tol=0.0, abs_tol=1e-9):
                return reason
        elif explicit != parsed:
            return reason
    return None


def _default_deliverable_repr(shares: float) -> str:
    return f"shares:{_format_number(shares)}"


def _format_strike(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")
