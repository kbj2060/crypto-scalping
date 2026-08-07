#!/usr/bin/env python3
"""Build a full plus_t12 live-candidate artifact from audited components."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_H48_REPORT = ROOT / (
    "tmp/causal_regen_20260516/"
    "omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/"
    "report.json"
)
DEFAULT_ZIG_REPORT = ROOT / (
    "tmp/causal_regen_20260516/"
    "omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/"
    "report.json"
)
DEFAULT_SOURCE_REPORT = ROOT / (
    "tmp/causal_regen_20260516/omega_creative_until_10am_20260630/"
    "nested_router_scale_robust_family_oos_blind_20260630/selected_readouts/"
    "plus_t12_target_guard_03/report.json"
)
DEFAULT_OUT_DIR = ROOT / (
    "tmp/causal_regen_20260516/omega_creative_until_10am_20260630/"
    "plus_t12_livepass_rebuild_20260630"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def report_dir(report_path: Path) -> Path:
    return report_path.parent


def ledger_path(component_dir: Path, split: str) -> Path:
    preferred = component_dir / f"{split}_selected_risk_replayed_trade_ledger.csv"
    if preferred.exists():
        return preferred
    fallback = component_dir / f"{split}_selected_risk_trade_ledger.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"missing {split} ledger in {component_dir}")


def load_component_ledger(report_path: Path, alias: str, split: str) -> pd.DataFrame:
    path = ledger_path(report_dir(report_path), split)
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["source_alias"] = alias
    df["source_report"] = str(report_path)
    df["source_ledger"] = str(path)
    return df


def load_market_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def priority_route(candidates: pd.DataFrame, priority: list[str]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    rank = {alias: idx for idx, alias in enumerate(priority)}
    missing = sorted(set(candidates["source_alias"]) - set(rank))
    if missing:
        raise ValueError(f"source aliases missing from priority order: {missing}")
    work = candidates.copy()
    work["_priority_rank"] = work["source_alias"].map(rank).astype(int)
    work["_source_row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["entry_i", "_priority_rank", "exit_i", "_source_row"]).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    active_until = -1
    for _, row in work.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= active_until:
            continue
        selected_rows.append(row)
        active_until = max(active_until, exit_i)

    if not selected_rows:
        return work.iloc[0:0].drop(columns=["_priority_rank", "_source_row"])
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    return out.drop(columns=["_priority_rank", "_source_row"])


def apply_max_hold_time_stop(ledger: pd.DataFrame, market: pd.DataFrame, max_hold_hours: float) -> pd.DataFrame:
    if ledger.empty or max_hold_hours <= 0.0:
        return ledger.copy()
    out = ledger.copy()
    market_ts = market["timestamp"].to_numpy()
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    max_hold = pd.Timedelta(hours=float(max_hold_hours))

    for idx, row in out.iterrows():
        entry_ts = pd.Timestamp(row["entry_timestamp"])
        old_exit_ts = pd.Timestamp(row["exit_timestamp"])
        if old_exit_ts - entry_ts <= max_hold:
            continue
        pos = int(np.searchsorted(market_ts, np.datetime64(entry_ts + max_hold), side="left"))
        if pos >= len(market_ts):
            continue
        cap_ts = pd.Timestamp(market_ts[pos])
        if cap_ts >= old_exit_ts:
            continue
        if entry_ts not in close_by_ts:
            raise ValueError(f"entry timestamp not found in market frame: {entry_ts}")

        entry_price = float(close_by_ts[entry_ts])
        exit_price = float(market.loc[pos, "close"])
        side = int(row["side"])
        raw_move = float(side) * (exit_price / entry_price - 1.0)
        old_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        net_per_notional = raw_move - old_cost
        notional = float(row["notional"])
        window = market[(market["timestamp"] >= entry_ts) & (market["timestamp"] <= cap_ts)]
        if window.empty:
            raise ValueError(f"empty market window for {entry_ts}..{cap_ts}")
        if side > 0:
            mfe = float(window["high"].max() / entry_price - 1.0)
            mae = float(window["low"].min() / entry_price - 1.0)
        else:
            mfe = float(entry_price / window["low"].min() - 1.0)
            mae = float(entry_price / window["high"].max() - 1.0)
        hold_bars = int(round((cap_ts - entry_ts).total_seconds() / 300.0))

        out.at[idx, "exit_i"] = int(row["entry_i"]) + hold_bars
        out.at[idx, "exit_timestamp"] = cap_ts.strftime("%Y-%m-%d %H:%M:%S")
        out.at[idx, "reason"] = f"time_stop_{int(max_hold_hours)}h"
        out.at[idx, "raw_exit_price_move"] = raw_move
        out.at[idx, "mfe_price_move"] = mfe
        out.at[idx, "mae_price_move"] = mae
        out.at[idx, "net_per_notional"] = net_per_notional
        out.at[idx, "trade_return"] = net_per_notional * notional
        out.at[idx, "win"] = int(net_per_notional > 0.0)

    old_exit = pd.to_datetime(ledger["exit_timestamp"], errors="raise")
    new_exit = pd.to_datetime(out["exit_timestamp"], errors="raise")
    out["max_hold_time_stop_applied"] = new_exit < old_exit
    return out


def scale_ledger(
    ledger: pd.DataFrame,
    scale_map: dict[str, float],
    leverage_cap: float,
    notional_cap: float,
    live_risk_scale: float,
) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    required = {"source_alias", "side", "notional", "margin_fraction", "leverage", "net_per_notional"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")

    out = ledger.copy()
    side_key = np.where(out["side"].astype(int) > 0, "L", "S")
    out["side_key"] = side_key
    out["scale_group"] = out["source_alias"].astype(str) + "_" + out["side_key"].astype(str)
    missing_scales = sorted(set(out["scale_group"]) - set(scale_map))
    if missing_scales:
        raise ValueError(f"scale_map missing groups: {missing_scales}")

    original_notional = out["notional"].astype(float).to_numpy()
    original_leverage = out["leverage"].astype(float).to_numpy()
    original_margin = out["margin_fraction"].astype(float).to_numpy()
    raw_scale = out["scale_group"].map(scale_map).astype(float).to_numpy()
    scaled_leverage = np.minimum(original_leverage * raw_scale, float(leverage_cap)) * float(live_risk_scale)
    scaled_notional = np.minimum(original_margin * scaled_leverage, float(notional_cap))
    scaled_leverage = np.divide(
        scaled_notional,
        original_margin,
        out=np.zeros_like(scaled_notional, dtype=float),
        where=np.abs(original_margin) > 1.0e-12,
    )

    out["original_notional"] = original_notional
    out["original_leverage"] = original_leverage
    out["original_margin_fraction"] = original_margin
    out["raw_source_side_scale"] = raw_scale
    out["effective_source_side_scale"] = np.divide(
        scaled_notional,
        original_notional,
        out=np.zeros_like(scaled_notional, dtype=float),
        where=np.abs(original_notional) > 1.0e-12,
    )
    out["leverage"] = scaled_leverage
    out["notional"] = scaled_notional
    out["margin_fraction"] = original_margin
    out["risk_notional"] = scaled_notional
    out["risk_leverage"] = scaled_leverage
    out["risk_margin_fraction"] = original_margin
    if "exit_input_notional" in out.columns:
        out["exit_input_notional"] = scaled_notional
    if "exit_input_leverage" in out.columns:
        out["exit_input_leverage"] = scaled_leverage
    if "exit_input_exposure" in out.columns:
        out["exit_input_exposure"] = scaled_notional * scaled_leverage
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    return out


def metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "max_leverage": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "min_effective_scale": 0.0,
            "max_margin_fraction": 0.0,
            "overlap_count": 0,
            "accounting_error_max_abs": 0.0,
            "notional_contract_error_max_abs": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "time_stop_count": 0,
        }
    returns = ledger["trade_return"].astype(float).to_numpy()
    equity = np.cumprod(1.0 + returns)
    curve = np.concatenate([[1.0], equity])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, 1.0e-12) - 1.0
    accounting_error = (
        ledger["trade_return"].astype(float)
        - ledger["net_per_notional"].astype(float) * ledger["notional"].astype(float)
    ).abs()
    notional_contract_error = (
        ledger["notional"].astype(float)
        - ledger["margin_fraction"].astype(float) * ledger["leverage"].astype(float)
    ).abs()
    hold_hours = (
        pd.to_datetime(ledger["exit_timestamp"], errors="raise")
        - pd.to_datetime(ledger["entry_timestamp"], errors="raise")
    ).dt.total_seconds() / 3600.0
    source_counts = ledger["source_alias"].value_counts().sort_index().to_dict()
    side_counts = ledger["side_key"].value_counts().sort_index().to_dict() if "side_key" in ledger.columns else {}
    reason_counts = ledger["reason"].value_counts().sort_index().to_dict() if "reason" in ledger.columns else {}
    time_stop_col = ledger.get("max_hold_time_stop_applied", pd.Series(False, index=ledger.index))
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(ledger)),
        "wr": float(ledger["win"].astype(float).mean()) if "win" in ledger.columns else float((returns > 0).mean()),
        "max_leverage": float(ledger["leverage"].astype(float).max()),
        "avg_notional": float(ledger["notional"].astype(float).mean()),
        "max_notional": float(ledger["notional"].astype(float).max()),
        "min_effective_scale": float(ledger["effective_source_side_scale"].astype(float).min())
        if "effective_source_side_scale" in ledger.columns
        else 0.0,
        "max_margin_fraction": float(ledger["margin_fraction"].astype(float).max()),
        "long_trades": int((ledger["side"].astype(int) > 0).sum()),
        "short_trades": int((ledger["side"].astype(int) < 0).sum()),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "side_counts": {str(k): int(v) for k, v in side_counts.items()},
        "overlap_count": int(overlap_count(ledger)),
        "accounting_error_max_abs": float(accounting_error.max()),
        "notional_contract_error_max_abs": float(notional_contract_error.max()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "time_stop_count": int(time_stop_col.astype(bool).sum()),
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
    }


def overlap_count(ledger: pd.DataFrame) -> int:
    if len(ledger) <= 1:
        return 0
    ordered = ledger.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, exit_i)
    return overlaps


def build_split(
    split: str,
    h48_report: Path,
    zig_report: Path,
    out_dir: Path,
    scale_map: dict[str, float],
    priority: list[str],
    leverage_cap: float,
    notional_cap: float,
    live_risk_scale: float,
    market: pd.DataFrame,
    max_hold_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    raw = pd.concat(
        [
            load_component_ledger(h48_report, "h48qual", split),
            load_component_ledger(zig_report, "zig075", split),
        ],
        ignore_index=True,
    )
    routed = priority_route(raw, priority)
    scaled = scale_ledger(routed, scale_map, leverage_cap, notional_cap, live_risk_scale)
    scaled = apply_max_hold_time_stop(scaled, market, max_hold_hours)
    out_path = out_dir / f"{split}_scaled_trade_ledger.csv"
    scaled.to_csv(out_path, index=False)
    return scaled, metrics(scaled), out_path


def non_pnl_live_gate(validation: dict[str, Any], oos: dict[str, Any], artifact_pass: bool) -> dict[str, Any]:
    checks = {
        "artifact_integrity_pass": bool(artifact_pass),
        "validation_mdd_lte_20_abs": abs(float(validation["mdd"])) <= 20.0,
        "oos_mdd_lte_20_abs": abs(float(oos["mdd"])) <= 20.0,
        "validation_leverage_lte_5": float(validation["max_leverage"]) <= 5.0 + 1.0e-9,
        "oos_leverage_lte_5": float(oos["max_leverage"]) <= 5.0 + 1.0e-9,
        "validation_notional_lte_1p8": float(validation["max_notional"]) <= 1.8 + 1.0e-9,
        "oos_notional_lte_1p8": float(oos["max_notional"]) <= 1.8 + 1.0e-9,
        "validation_max_hold_lte_24h": float(validation["max_hold_hours"]) <= 24.0 + 1.0e-9,
        "oos_max_hold_lte_24h": float(oos["max_hold_hours"]) <= 24.0 + 1.0e-9,
        "validation_no_overlaps": int(validation["overlap_count"]) == 0,
        "oos_no_overlaps": int(oos["overlap_count"]) == 0,
        "validation_accounting_consistent": float(validation["accounting_error_max_abs"]) <= 1.0e-10,
        "oos_accounting_consistent": float(oos["accounting_error_max_abs"]) <= 1.0e-10,
        "validation_notional_contract_consistent": float(validation["notional_contract_error_max_abs"]) <= 1.0e-10,
        "oos_notional_contract_consistent": float(oos["notional_contract_error_max_abs"]) <= 1.0e-10,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "scope": "PnL target excluded; artifact, risk, overlap, and accounting gates included",
        "pass": not failures,
        "checks": checks,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h48-report", type=Path, default=DEFAULT_H48_REPORT)
    ap.add_argument("--zig-report", type=Path, default=DEFAULT_ZIG_REPORT)
    ap.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--leverage-cap", type=float, default=5.0)
    ap.add_argument("--notional-cap", type=float, default=1.8)
    ap.add_argument("--live-risk-scale", type=float, default=0.8)
    ap.add_argument("--max-hold-hours", type=float, default=24.0)
    ap.add_argument("--artifact-pass", action="store_true")
    args = ap.parse_args()

    h48_report = resolve_path(args.h48_report)
    zig_report = resolve_path(args.zig_report)
    source_report = resolve_path(args.source_report)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_map = {
        "h48qual_L": 0.38,
        "h48qual_S": 2.499,
        "zig075_L": 2.446,
        "zig075_S": 2.478,
    }
    priority = ["h48qual", "zig075"]
    h48 = read_json(h48_report)
    zig = read_json(zig_report)
    validation_market = load_market_frame(resolve_path(h48["risk_model"]["train_csv"]))
    oos_market = load_market_frame(resolve_path(h48["risk_model"]["eval_csv"]))

    validation_ledger, validation, validation_path = build_split(
        "validation",
        h48_report,
        zig_report,
        out_dir,
        scale_map,
        priority,
        args.leverage_cap,
        args.notional_cap,
        args.live_risk_scale,
        validation_market,
        args.max_hold_hours,
    )
    oos_ledger, oos, oos_path = build_split(
        "oos",
        h48_report,
        zig_report,
        out_dir,
        scale_map,
        priority,
        args.leverage_cap,
        args.notional_cap,
        args.live_risk_scale,
        oos_market,
        args.max_hold_hours,
    )

    source = read_json(source_report) if source_report.exists() else {}
    report = {
        "model_id": "omega_creative_robust_family_plus_t12_livepass_rebuild_20260630",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": "plus_t12",
        "router_order": "h48qual > zig075",
        "policy": "non_pnl_livepass_rebuild",
        "selection": {
            "selection_oos_independent": True,
            "selection_basis": "original plus_t12_target_guard_03 scale/router contract, rebuilt from exact-threshold precomputed component artifacts",
            "pnl_target_excluded_from_live_gate": True,
        },
        "scale_map": scale_map,
        "leverage_cap": float(args.leverage_cap),
        "notional_cap": float(args.notional_cap),
        "live_risk_scale": float(args.live_risk_scale),
        "max_hold_hours": float(args.max_hold_hours),
        "components": {
            "h48qual": {
                "out_dir": str(report_dir(h48_report)),
                "report": str(h48_report),
                "quality_threshold": h48.get("contract", {}).get("quality_threshold"),
                "precomputed_prediction_dir": h48.get("risk_model", {}).get("precomputed_prediction_dir"),
                "precomputed_prediction_tag": h48.get("risk_model", {}).get("precomputed_prediction_tag"),
            },
            "zig075": {
                "out_dir": str(report_dir(zig_report)),
                "report": str(zig_report),
                "quality_threshold": zig.get("contract", {}).get("quality_threshold"),
                "precomputed_prediction_dir": zig.get("risk_model", {}).get("precomputed_prediction_dir"),
                "precomputed_prediction_tag": zig.get("risk_model", {}).get("precomputed_prediction_tag"),
            },
        },
        "source_report": str(source_report),
        "source_report_metrics": {
            "validation": source.get("validation"),
            "oos": source.get("oos"),
        },
        "validation": validation,
        "oos": oos,
        "non_pnl_live_gate": non_pnl_live_gate(validation, oos, artifact_pass=bool(args.artifact_pass)),
        "artifacts": {
            "validation_scaled_trade_ledger": str(validation_path),
            "oos_scaled_trade_ledger": str(oos_path),
            "report": str(out_dir / "report.json"),
        },
        "notes": [
            "This rebuild intentionally uses explicit component prediction artifacts instead of historical readout ledgers.",
            "PnL target is not part of the non_pnl_live_gate, but reported metrics remain diagnostic.",
        ],
    }
    report["diagnostics"] = {
        "validation_first_entry": validation_ledger["entry_timestamp"].iloc[0] if len(validation_ledger) else None,
        "oos_first_entry": oos_ledger["entry_timestamp"].iloc[0] if len(oos_ledger) else None,
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps({"report": str(out_dir / "report.json"), "validation": validation, "oos": oos}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
