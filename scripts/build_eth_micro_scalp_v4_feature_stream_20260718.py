"""Build and publish the exact source-stable v4 one-minute feature stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_eth_micro_scalp_v3_feature_stream_20260718 as base  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402


FRESH_START_UTC = pd.Timestamp("2026-07-18 02:45:00")
OBSERVER_DIR = v4.ARTIFACT_DIR / "fresh_forward_observer"
DEFAULT_OUTPUT = ROOT / "data/live/eth_micro_scalp_v4_features_1m.csv"
DEFAULT_REPORT = OBSERVER_DIR / "feature_stream_build.json"


def _stream_contract(
    stream: pd.DataFrame,
    checkpoint: dict[str, Any],
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    warmup_start = FRESH_START_UTC - pd.Timedelta(
        minutes=checkpoint["config"]["window"] - 1
    )
    names = [
        "timestamp", "close", *v4.SOURCE_STABLE_FEATURES,
        *v4.v3.core.MICRO_FEATURES,
    ]
    publish = stream[
        (stream["timestamp"] >= warmup_start) & (stream["timestamp"] <= end)
    ][names].copy()
    differences = publish["timestamp"].diff().dropna()
    numeric = publish.drop(columns="timestamp")
    values = numeric.to_numpy(dtype=np.float64)
    required = [
        "close", *v4.SOURCE_STABLE_FEATURES, "micro_available", "book_available",
    ]
    required_finite = bool(
        np.isfinite(numeric[required].to_numpy(dtype=np.float64)).all()
    )
    micro_names = [
        name for name in v4.v3.core.MICRO_FEATURES if name.startswith("micro_")
    ]
    book_names = [
        name for name in v4.v3.core.MICRO_FEATURES if name.startswith("book_")
    ]
    micro_present = numeric["micro_available"] > 0.5
    book_present = numeric["book_available"] > 0.5
    micro_finite = bool(
        not micro_present.any()
        or np.isfinite(numeric.loc[micro_present, micro_names].to_numpy(dtype=np.float64)).all()
    )
    book_finite = bool(
        not book_present.any()
        or np.isfinite(numeric.loc[book_present, book_names].to_numpy(dtype=np.float64)).all()
    )
    diagnostics = {
        "pass": bool(
            len(publish) >= checkpoint["config"]["window"]
            and (differences == pd.Timedelta(minutes=1)).all()
            and required_finite
            and not np.isinf(values).any()
            and micro_finite
            and book_finite
            and len(publish)
            and publish["timestamp"].iloc[-1] >= FRESH_START_UTC
        ),
        "minimum_required_rows": int(checkpoint["config"]["window"]),
        "actual_rows": len(publish),
        "one_minute_cadence": bool((differences == pd.Timedelta(minutes=1)).all()),
        "required_values_finite": required_finite,
        "no_infinite_values": bool(not np.isinf(values).any()),
        "available_micro_values_finite": micro_finite,
        "available_book_values_finite": book_finite,
        "covers_fresh_start": bool(
            len(publish) and publish["timestamp"].iloc[-1] >= FRESH_START_UTC
        ),
    }
    return publish, diagnostics


def build(
    output: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    lookback_days: int = 21,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if lookback_days < 7 or lookback_days > 29:
        raise ValueError("lookback_days must be between 7 and 29")
    end = (
        pd.Timestamp.utcnow().tz_localize(None).floor("min") - pd.Timedelta(minutes=1)
        if end is None else pd.Timestamp(end)
    )
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    start = end - pd.Timedelta(days=lookback_days)
    checkpoint = torch.load(v4.MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("model_id") != v4.MODEL_ID:
        raise RuntimeError("v4 model id mismatch")
    if checkpoint.get("activation_allowed") is not False or checkpoint["policy"]["enabled"]:
        raise RuntimeError("v4 artifact safety contract mismatch")
    if tuple(checkpoint["base_feature_names"]) != tuple(v4.SOURCE_STABLE_FEATURES):
        raise RuntimeError("v4 source-stable feature contract mismatch")
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-scalping-v4-public-feature-observer/1.0"})
    engineered, source = base.build_engineered_frame(session, start, end)
    full_stream, full_base, selected_micro = base.assemble_model_stream(engineered)
    stable_indices = [
        v4.v3.core.BASE_FEATURES.index(name) for name in v4.SOURCE_STABLE_FEATURES
    ]
    stable_base = full_base[:, stable_indices]
    stream = full_stream[
        ["timestamp", "close", *v4.SOURCE_STABLE_FEATURES, *v4.v3.core.MICRO_FEATURES]
    ].copy()
    parity = base.parity_audit(
        stream["timestamp"], stable_base, selected_micro, checkpoint,
        v4.SOURCE_STABLE_FEATURES, v4.v3.core.MICRO_FEATURES,
    )
    publish, stream_contract = _stream_contract(stream, checkpoint, end)
    published = bool(parity["pass"] and stream_contract["pass"])
    if published:
        base._publish_csv_atomic(output, publish)
    report = {
        "schema_version": "eth_micro_scalp_v4.feature_stream_build.v1",
        "model_id": v4.MODEL_ID,
        "model_sha256": base._sha256(v4.MODEL_PATH),
        "fresh_start_utc": str(FRESH_START_UTC),
        "public_market_data_only": True,
        "account_credentials_used": False,
        "order_endpoints_used": False,
        "endpoint_allowlist": list(base.PUBLIC_ENDPOINTS.values()),
        "source": source,
        "parity": parity,
        "stream_contract": stream_contract,
        "published": published,
        "output": str(output),
        "output_rows": len(publish),
        "output_start_utc": str(publish["timestamp"].min()) if len(publish) else None,
        "output_end_utc": str(publish["timestamp"].max()) if len(publish) else None,
        "output_sha256": base._sha256(output) if published else None,
        "failure_reason": None if published else "v4 parity or stream contract failed",
    }
    base._write_json_atomic(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--end", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = build(
        args.output, args.report, args.lookback_days,
        pd.Timestamp(args.end) if args.end else None,
    )
    print(json.dumps(result, indent=2, default=base._json_default))
    raise SystemExit(0 if result["published"] else 2)
