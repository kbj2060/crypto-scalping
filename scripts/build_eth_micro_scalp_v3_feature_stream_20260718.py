"""Build the exact one-minute feature stream required by the v3 observer.

Only public USD-M futures market-data GET endpoints are used.  The builder
reuses the frozen FeatureEngineer and micro/order-book alignment code, then
requires scaled feature parity against the frozen cache on their overlapping
interval before atomically publishing any post-freeze stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import duckdb


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from features.engineering import FeatureEngineer  # noqa: E402
import train_eval_deepscalp_pnl_20260717 as deep  # noqa: E402
import train_eval_eth_micro_scalp_opportunity_moe_20260718 as v3  # noqa: E402
import run_eth_micro_scalp_v3_fresh_forward_observer_20260718 as observer  # noqa: E402


BASE_URL = "https://fapi.binance.com"
PUBLIC_ENDPOINTS = {
    "klines": "/fapi/v1/klines",
    "funding": "/fapi/v1/fundingRate",
    "open_interest": "/futures/data/openInterestHist",
    "top_position": "/futures/data/topLongShortPositionRatio",
    "top_account": "/futures/data/topLongShortAccountRatio",
    "global_account": "/futures/data/globalLongShortAccountRatio",
    "taker": "/futures/data/takerlongshortRatio",
}
DEFAULT_OUTPUT = observer.DEFAULT_FEATURE_STREAM
DEFAULT_REPORT = observer.OBSERVER_DIR / "feature_stream_build.json"
PARITY_ROWS = 360
MAX_SCALED_ERROR = 0.10
P99_SCALED_ERROR = 0.02
CAUSAL_CONTEXT = pd.Timedelta(hours=12)
CANONICAL_PARITY_END_UTC = pd.Timestamp("2026-06-30 16:00:00")

KLINE_COLUMNS = (
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _publish_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _request_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    retries: int = 4,
) -> list[Any]:
    if endpoint not in PUBLIC_ENDPOINTS.values():
        raise RuntimeError(f"endpoint is outside the public allowlist: {endpoint}")
    for attempt in range(retries):
        response = session.get(BASE_URL + endpoint, params=params, timeout=30)
        if response.status_code in (418, 429):
            retry_after = float(response.headers.get("Retry-After", 1.0))
            time.sleep(max(retry_after, 1.0) * (attempt + 1))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected public market-data response: {payload}")
        return payload
    raise RuntimeError(f"public market-data request exhausted retries: {endpoint}")


def fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = _request_json(
            session, PUBLIC_ENDPOINTS["klines"],
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("kline pagination did not advance")
        cursor = next_cursor
        if len(batch) < 1500:
            break
        time.sleep(0.05)
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"no {symbol} {interval} klines returned")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    for name in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    differences = frame["timestamp"].diff().dropna()
    expected = pd.Timedelta(minutes=1 if interval == "1m" else 5)
    if len(differences) and (differences != expected).any():
        raise RuntimeError(f"{symbol} {interval} kline cadence gap")
    return frame


def _fetch_recent_series_backward(
    session: requests.Session,
    endpoint: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    cursor_end = int(end.tz_localize("UTC").timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    while cursor_end >= start_ms:
        batch = _request_json(
            session, endpoint,
            {"symbol": "ETHUSDT", "period": "5m", "startTime": start_ms, "endTime": cursor_end, "limit": 500},
        )
        if not batch:
            break
        rows.extend(batch)
        first_timestamp = min(int(row["timestamp"]) for row in batch)
        if first_timestamp <= start_ms:
            break
        next_end = first_timestamp - 1
        if next_end >= cursor_end:
            raise RuntimeError(f"metric pagination did not retreat: {endpoint}")
        cursor_end = next_end
        time.sleep(0.05)
    unique = {int(row["timestamp"]): row for row in rows if start_ms <= int(row["timestamp"]) <= int(end.tz_localize("UTC").timestamp() * 1000)}
    return [unique[key] for key in sorted(unique)]


def fetch_metrics(
    session: requests.Session, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    series = {
        name: _fetch_recent_series_backward(session, PUBLIC_ENDPOINTS[name], start, end)
        for name in ("open_interest", "top_position", "top_account", "global_account", "taker")
    }
    oi = pd.DataFrame(series["open_interest"])
    if oi.empty:
        raise RuntimeError("open-interest history is empty")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(oi["timestamp"], unit="ms"),
            "sum_open_interest": pd.to_numeric(oi["sumOpenInterest"]),
            "sum_open_interest_value": pd.to_numeric(oi["sumOpenInterestValue"]),
        }
    ).sort_values("timestamp")
    mappings = (
        ("top_position", "sum_toptrader_long_short_ratio", "longShortRatio"),
        ("top_account", "count_toptrader_long_short_ratio", "longShortRatio"),
        ("global_account", "count_long_short_ratio", "longShortRatio"),
        ("taker", "sum_taker_long_short_vol_ratio", "buySellRatio"),
    )
    for source, target, value_name in mappings:
        raw = pd.DataFrame(series[source])
        aligned = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["timestamp"], unit="ms"),
                target: pd.to_numeric(raw[value_name]),
            }
        ).sort_values("timestamp")
        frame = pd.merge_asof(
            frame.sort_values("timestamp"), aligned, on="timestamp",
            direction="backward", tolerance=pd.Timedelta(minutes=5),
        )
    required = [target for _, target, _ in mappings]
    if frame[required].isna().any().any():
        raise RuntimeError("recent metric series do not align causally")
    return frame


def fetch_funding(
    session: requests.Session, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = _request_json(
            session, PUBLIC_ENDPOINTS["funding"],
            {"symbol": "ETHUSDT", "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("funding pagination did not advance")
        cursor = next_cursor
        if len(batch) < 1000:
            break
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("funding history is empty")
    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["fundingTime"], unit="ms"),
            "last_funding_rate": pd.to_numeric(frame["fundingRate"]),
        }
    )
    return result.drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def build_engineered_frame(
    session: requests.Session,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eth = fetch_klines(session, "ETHUSDT", "1m", start, end)
    btc = fetch_klines(session, "BTCUSDT", "5m", start.floor("5min"), end.floor("5min"))
    context_start = start - CAUSAL_CONTEXT
    metrics = fetch_metrics(session, context_start, end)
    funding = fetch_funding(session, context_start, end)
    eth = pd.merge_asof(
        eth.sort_values("timestamp"), metrics.sort_values("timestamp"), on="timestamp",
        direction="backward", tolerance=pd.Timedelta(hours=9),
    )
    eth = pd.merge_asof(
        eth.sort_values("timestamp"), funding.sort_values("timestamp"), on="timestamp",
        direction="backward",
    )
    metric_columns = (
        "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
    )
    if eth[list(metric_columns) + ["last_funding_rate"]].isna().any().any():
        raise RuntimeError("causal metric/funding join contains missing values")
    engineer = FeatureEngineer(candle_minutes=1, keep_only_active=True, include_entry_price=False)
    engineered = engineer.process(
        eth,
        btc[["timestamp", "close", "volume", "quote_volume"]],
    )
    if engineered["timestamp"].duplicated().any():
        raise RuntimeError("engineered feature rows contain duplicate timestamps")
    source = {
        "eth_kline_rows": len(eth),
        "btc_kline_rows": len(btc),
        "metric_rows": len(metrics),
        "funding_rows": len(funding),
        "source_start_utc": str(start),
        "causal_context_start_utc": str(context_start),
        "source_end_utc": str(end),
    }
    return engineered, source


def assemble_model_stream(engineered: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    base, base_names = deep.make_base_features(engineered)
    micro_start = pd.Timestamp(engineered["timestamp"].min())
    micro_end = pd.Timestamp(engineered["timestamp"].max())
    for attempt in range(5):
        try:
            micro, book = deep.read_micro_frames(micro_start, micro_end)
            break
        except duckdb.IOException:
            if attempt == 4:
                raise
            time.sleep(2.0)
    micro_values, micro_names, _ = deep.make_micro_features(engineered["timestamp"], micro, book)
    base_index = [base_names.index(name) for name in v3.core.BASE_FEATURES]
    micro_index = [micro_names.index(name) for name in v3.core.MICRO_FEATURES]
    selected_base = base[:, base_index]
    selected_micro = micro_values[:, micro_index]
    stream = pd.DataFrame(
        selected_base, columns=v3.core.BASE_FEATURES
    )
    for index, name in enumerate(v3.core.MICRO_FEATURES):
        stream[name] = selected_micro[:, index]
    stream.insert(0, "close", pd.to_numeric(engineered["close"], errors="coerce").to_numpy())
    stream.insert(0, "timestamp", pd.to_datetime(engineered["timestamp"]).to_numpy())
    return stream, selected_base, selected_micro


def parity_audit(
    timestamps: pd.Series,
    selected_base: np.ndarray,
    selected_micro: np.ndarray,
    checkpoint: dict[str, Any],
    base_feature_names: tuple[str, ...] = v3.core.BASE_FEATURES,
    micro_feature_names: tuple[str, ...] = v3.core.MICRO_FEATURES,
) -> dict[str, Any]:
    arrays, metadata = v3.core.load_frozen_cache()
    cache_timestamps = np.asarray(arrays["timestamp_ns"], dtype=np.int64)
    rebuilt_ns = pd.to_datetime(timestamps).astype("datetime64[ns]").astype("int64").to_numpy()
    positions = np.searchsorted(cache_timestamps, rebuilt_ns)
    valid = (positions < len(cache_timestamps))
    valid &= cache_timestamps[np.minimum(positions, len(cache_timestamps) - 1)] == rebuilt_ns
    valid &= rebuilt_ns <= CANONICAL_PARITY_END_UTC.value
    overlap_indices = np.flatnonzero(valid)[-PARITY_ROWS:]
    if len(overlap_indices) < PARITY_ROWS:
        raise RuntimeError(f"insufficient frozen parity overlap: {len(overlap_indices)} rows")
    cache_positions = positions[overlap_indices]
    base_cache_index = [metadata["base_feature_names"].index(name) for name in base_feature_names]
    micro_cache_index = [metadata["micro_feature_names"].index(name) for name in micro_feature_names]
    cached_base = np.asarray(arrays["base"][cache_positions][:, base_cache_index], dtype=np.float64)
    cached_micro = np.asarray(arrays["micro"][cache_positions][:, micro_cache_index], dtype=np.float64)
    rebuilt_base = np.asarray(selected_base[overlap_indices], dtype=np.float64)
    rebuilt_micro = np.asarray(selected_micro[overlap_indices], dtype=np.float64)
    base_scale = np.asarray(checkpoint["scalers"]["base_scale"], dtype=np.float64)
    micro_scale = np.asarray(checkpoint["scalers"]["micro_scale"], dtype=np.float64)
    base_error = np.abs(rebuilt_base - cached_base) / np.maximum(base_scale, 1e-8)
    micro_error = np.abs(rebuilt_micro - cached_micro) / np.maximum(micro_scale, 1e-8)
    errors = np.column_stack([base_error, micro_error])
    names = [*base_feature_names, *micro_feature_names]
    per_feature = {
        name: {
            "max_scaled_error": float(np.nanmax(errors[:, index])),
            "p99_scaled_error": float(np.nanquantile(errors[:, index], 0.99)),
        }
        for index, name in enumerate(names)
    }
    maximum = float(np.nanmax(errors))
    p99 = float(np.nanquantile(errors, 0.99))
    passed = bool(
        np.isfinite(errors).all()
        and maximum <= MAX_SCALED_ERROR
        and p99 <= P99_SCALED_ERROR
    )
    worst = sorted(
        per_feature.items(), key=lambda item: item[1]["max_scaled_error"], reverse=True
    )[:15]
    return {
        "pass": passed,
        "rows": len(overlap_indices),
        "start_utc": str(pd.to_datetime(timestamps.iloc[overlap_indices[0]])),
        "end_utc": str(pd.to_datetime(timestamps.iloc[overlap_indices[-1]])),
        "max_scaled_error": maximum,
        "p99_scaled_error": p99,
        "thresholds": {
            "max_scaled_error": MAX_SCALED_ERROR,
            "p99_scaled_error": P99_SCALED_ERROR,
        },
        "canonical_source_coverage_end_utc": str(CANONICAL_PARITY_END_UTC),
        "worst_features": [{"name": name, **metrics} for name, metrics in worst],
    }


def build(
    output: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    lookback_days: int = 21,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if lookback_days < 7 or lookback_days > 29:
        raise ValueError("lookback_days must be between 7 and 29")
    end = pd.Timestamp.utcnow().tz_localize(None).floor("min") - pd.Timedelta(minutes=1) if end is None else pd.Timestamp(end)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    start = end - pd.Timedelta(days=lookback_days)
    checkpoint = torch.load(v3.MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("model_id") != v3.MODEL_ID or checkpoint.get("activation_allowed") is not False:
        raise RuntimeError("v3 artifact safety contract mismatch")
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-scalping-v3-public-feature-observer/1.0"})
    engineered, source = build_engineered_frame(session, start, end)
    stream, selected_base, selected_micro = assemble_model_stream(engineered)
    parity = parity_audit(stream["timestamp"], selected_base, selected_micro, checkpoint)
    warmup_start = observer.FRESH_START_UTC - pd.Timedelta(minutes=checkpoint["config"]["window"] - 1)
    publish = stream[(stream["timestamp"] >= warmup_start) & (stream["timestamp"] <= end)].copy()
    expected_columns = ["timestamp", "close", *v3.core.BASE_FEATURES, *v3.core.MICRO_FEATURES]
    publish = publish[expected_columns]
    differences = publish["timestamp"].diff().dropna()
    numeric_publish_frame = publish.drop(columns="timestamp")
    numeric_publish = numeric_publish_frame.to_numpy(dtype=np.float64)
    always_required = ["close", *v3.core.BASE_FEATURES, "micro_available", "book_available"]
    required_finite = bool(
        np.isfinite(numeric_publish_frame[always_required].to_numpy(dtype=np.float64)).all()
    )
    non_finite_required_columns = [
        name for name in always_required
        if not np.isfinite(numeric_publish_frame[name].to_numpy(dtype=np.float64)).all()
    ]
    no_infinities = bool(not np.isinf(numeric_publish).any())
    micro_payload = [name for name in v3.core.MICRO_FEATURES if name.startswith("micro_")]
    book_payload = [name for name in v3.core.MICRO_FEATURES if name.startswith("book_")]
    micro_present = numeric_publish_frame["micro_available"] > 0.5
    book_present = numeric_publish_frame["book_available"] > 0.5
    available_micro_finite = bool(
        not micro_present.any()
        or np.isfinite(
            numeric_publish_frame.loc[micro_present, micro_payload].to_numpy(dtype=np.float64)
        ).all()
    )
    available_book_finite = bool(
        not book_present.any()
        or np.isfinite(
            numeric_publish_frame.loc[book_present, book_payload].to_numpy(dtype=np.float64)
        ).all()
    )
    contract_pass = bool(
        len(publish) >= checkpoint["config"]["window"]
        and (differences == pd.Timedelta(minutes=1)).all()
        and required_finite
        and no_infinities
        and available_micro_finite
        and available_book_finite
        and publish["timestamp"].iloc[-1] >= observer.FRESH_START_UTC
    )
    contract_diagnostics = {
        "minimum_required_rows": int(checkpoint["config"]["window"]),
        "actual_rows": len(publish),
        "one_minute_cadence": bool((differences == pd.Timedelta(minutes=1)).all()),
        "required_values_finite": required_finite,
        "non_finite_required_columns": non_finite_required_columns,
        "no_infinite_values": no_infinities,
        "available_micro_values_finite": available_micro_finite,
        "available_book_values_finite": available_book_finite,
        "non_finite_values": int((~np.isfinite(numeric_publish)).sum()),
        "covers_fresh_start": bool(
            len(publish) and publish["timestamp"].iloc[-1] >= observer.FRESH_START_UTC
        ),
    }
    published = bool(parity["pass"] and contract_pass)
    if published:
        _publish_csv_atomic(output, publish)
        observer.load_feature_stream(output)
    report = {
        "schema_version": "eth_micro_scalp_v3.feature_stream_build.v1",
        "model_id": v3.MODEL_ID,
        "model_sha256": _sha256(v3.MODEL_PATH),
        "public_market_data_only": True,
        "account_credentials_used": False,
        "order_endpoints_used": False,
        "endpoint_allowlist": list(PUBLIC_ENDPOINTS.values()),
        "source": source,
        "parity": parity,
        "stream_contract_pass": contract_pass,
        "stream_contract": contract_diagnostics,
        "published": published,
        "output": str(output),
        "output_rows": len(publish),
        "output_start_utc": str(publish["timestamp"].min()) if len(publish) else None,
        "output_end_utc": str(publish["timestamp"].max()) if len(publish) else None,
        "output_sha256": _sha256(output) if published else None,
        "failure_reason": None if published else "feature parity or one-minute stream contract failed",
    }
    _write_json_atomic(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--end", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(
        args.output, args.report, args.lookback_days,
        pd.Timestamp(args.end) if args.end else None,
    )
    print(json.dumps(report, indent=2, default=_json_default))
    return 0 if report["published"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
