#!/usr/bin/env python3
"""Adapt the selected hourly BTC trend-scan target to BTC v1's 5-minute rows.

The mapping is deliberately forward-only: a 5-minute feature row receives the
first hourly target timestamp that is greater than or equal to the row time.
This avoids assigning an hourly target whose trend window started before the
decision row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "tmp/causal_regen_20260516/btc_best_mean_pnl_trendscan_labels_20260715"
)
DEFAULT_OUTPUT = (
    ROOT
    / "tmp/causal_regen_20260516/btc_v1_trendscan_t2_5m_labels_20260715"
)


def _load_hourly(source_dir: Path) -> pd.DataFrame:
    frames = []
    for year in (2024, 2025, 2026):
        path = source_dir / f"btc_1h_trendscan_t2_labels_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        required = {
            "timestamp",
            "action_id",
            "trend_t_value",
            "trend_beta",
            "trend_horizon_hours",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        frames.append(frame[list(required)])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if out["timestamp"].duplicated().any():
        raise RuntimeError("hourly source contains duplicate timestamps")
    invalid = sorted(set(out["action_id"].astype(int).unique()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"invalid trend-scan action classes: {invalid}")
    return out.reset_index(drop=True)


def _adapt_year(hourly: pd.DataFrame, feature_path: Path) -> tuple[pd.DataFrame, dict]:
    features = pd.read_csv(feature_path, usecols=["timestamp"], parse_dates=["timestamp"])
    if features["timestamp"].duplicated().any():
        raise RuntimeError(f"{feature_path} contains duplicate timestamps")
    target_ts = features["timestamp"].dt.ceil("h")
    lookup = hourly.set_index("timestamp")
    missing = ~target_ts.isin(lookup.index)
    if missing.any():
        sample = target_ts.loc[missing].head(10).astype(str).tolist()
        raise RuntimeError(
            f"{feature_path} has {int(missing.sum())} rows without a forward hourly label: {sample}"
        )
    selected = lookup.loc[target_ts].reset_index()
    out = pd.DataFrame(
        {
            "timestamp": features["timestamp"].to_numpy(),
            "zigzag_action": selected["action_id"].astype("int64").to_numpy(),
            "trendscan_target_timestamp": selected["timestamp"].to_numpy(),
            "trendscan_t_value": selected["trend_t_value"].to_numpy(),
            "trendscan_beta": selected["trend_beta"].to_numpy(),
            "trendscan_horizon_hours": selected["trend_horizon_hours"].astype("int64").to_numpy(),
        }
    )
    delay = out["trendscan_target_timestamp"] - out["timestamp"]
    if (delay < pd.Timedelta(0)).any() or (delay >= pd.Timedelta(hours=1)).any():
        raise RuntimeError("non-causal or out-of-contract 5m-to-1h target mapping")
    return out, {
        "rows": int(len(out)),
        "first_timestamp": str(out["timestamp"].iloc[0]),
        "last_timestamp": str(out["timestamp"].iloc[-1]),
        "max_target_delay_minutes": float(delay.dt.total_seconds().max() / 60.0),
        "class_counts": {
            str(k): int(v)
            for k, v in out["zigzag_action"].value_counts().sort_index().items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    hourly = _load_hourly(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = {}
    for year in (2025, 2026):
        feature_path = ROOT / f"data/splits/year_oos/btc_features_{year}.csv"
        labels, summary = _adapt_year(hourly, feature_path)
        output_path = args.output_dir / f"zigzag_action_labels_{year}.csv"
        labels.to_csv(output_path, index=False)
        years[str(year)] = {**summary, "path": str(output_path)}

    manifest = {
        "label_family": "btc_1h_trend_scanning_t_abs_ge_2",
        "source_dir": str(args.source_dir),
        "mapping": "target_timestamp = ceil(feature_timestamp, 1h)",
        "causal_target_contract": {
            "target_timestamp_not_before_feature_timestamp": True,
            "target_delay_less_than_one_hour": True,
            "labels_are_offline_training_targets_only": True,
        },
        "years": years,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "years": years}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
