#!/usr/bin/env python3
"""Paper-inspired optimal-stopping exit overlay for Omega 4.6.2 cap220.

The experiment is validation-selected and OOS-readout only. It keeps the
Omega4.6.2 entry/exposure contract fixed and changes only the lifecycle exit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega4_6_2_cap220_paper_optstop_exit_overlay_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
BASE_VARIANT = "short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h"
DEFAULT_RUNTIME = ROOT / "tmp/causal_regen_20260516" / BASE_MODEL_ID / "runtime_contract.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
EPS = 1.0e-12


@dataclass(frozen=True)
class StopSpec:
    name: str
    hard_stop_hours: float
    loss_after_hours: float
    loss_stop_move: float
    trail_after_hours: float
    trail_arm_move: float
    trail_giveback_move: float
    trail_floor_move: float
    stall_after_hours: float
    stall_lookback_hours: float
    stall_min_profit_move: float
    stall_slope_max: float


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def component_train_eval_paths(runtime: dict[str, Any]) -> tuple[Path, Path]:
    risk_report = read_json(resolve_path(runtime["components"]["h48qual"]["report"]))
    risk_model = risk_report["risk_model"]
    return resolve_path(risk_model["train_csv"]), resolve_path(risk_model["eval_csv"])


def load_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "entry_timestamp_dt" not in out.columns or not pd.api.types.is_datetime64_any_dtype(out["entry_timestamp_dt"]):
        out["entry_timestamp_dt"] = pd.to_datetime(out["entry_timestamp"], errors="raise")
    if "exit_timestamp_dt" not in out.columns or not pd.api.types.is_datetime64_any_dtype(out["exit_timestamp_dt"]):
        out["exit_timestamp_dt"] = pd.to_datetime(out["exit_timestamp"], errors="raise")
    out["hold_hours"] = (out["exit_timestamp_dt"] - out["entry_timestamp_dt"]).dt.total_seconds() / 3600.0
    out["entry_month"] = out["entry_timestamp_dt"].dt.to_period("M").astype(str)
    return out


def active_ledger(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["notional"].astype(float) > EPS].copy()


def overlap_count(df: pd.DataFrame) -> int:
    active = active_ledger(df)
    if len(active) <= 1:
        return 0
    ordered = active.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, exit_i)
    return overlaps


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    df = ensure_time_columns(df)
    active = active_ledger(df)
    if active.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "avg_hold_hours": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "max_leverage": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "skipped": int(len(df)),
            "overlap_count": 0,
            "accounting_error_max_abs": 0.0,
            "notional_contract_error_max_abs": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "reason_counts": {},
        }
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, EPS) - 1.0
    hold_hours = active["hold_hours"].astype(float)
    accounting_error = (
        active["trade_return"].astype(float)
        - active["net_per_notional"].astype(float) * active["notional"].astype(float)
    ).abs()
    notional_contract_error = (
        active["notional"].astype(float)
        - active["margin_fraction"].astype(float) * active["leverage"].astype(float)
    ).abs()
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(active)),
        "wr": float((active["trade_return"].astype(float) > 0.0).mean()),
        "avg_hold_hours": float(hold_hours.mean()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "avg_notional": float(active["notional"].astype(float).mean()),
        "max_notional": float(active["notional"].astype(float).max()),
        "skipped": int((df["notional"].astype(float) <= EPS).sum()),
        "overlap_count": int(overlap_count(df)),
        "accounting_error_max_abs": float(accounting_error.max()),
        "notional_contract_error_max_abs": float(notional_contract_error.max()),
        "long_trades": int((active["side"].astype(int) > 0).sum()),
        "short_trades": int((active["side"].astype(int) < 0).sum()),
        "reason_counts": {str(k): int(v) for k, v in active["reason"].value_counts().sort_index().to_dict().items()},
    }


def monthly_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for month, group in ensure_time_columns(df).groupby("entry_month", sort=True):
        row = metrics(group)
        row["month"] = str(month)
        rows.append(row)
    if not rows:
        return {"monthly_min_pnl": 0.0, "monthly_worst_mdd": 0.0, "monthly_positive_count": 0, "monthly_count": 0, "monthly_rows": []}
    return {
        "monthly_min_pnl": float(min(row["pnl"] for row in rows)),
        "monthly_worst_mdd": float(min(row["mdd"] for row in rows)),
        "monthly_positive_count": int(sum(float(row["pnl"]) > 0.0 for row in rows)),
        "monthly_count": int(len(rows)),
        "monthly_rows": rows,
    }


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def market_window(market: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> pd.DataFrame:
    window = market[(market["timestamp"] >= entry_ts) & (market["timestamp"] <= exit_ts)].copy()
    if window.empty:
        raise RuntimeError(f"empty market window for {entry_ts}..{exit_ts}")
    return window.reset_index(drop=True)


def move_series(window: pd.DataFrame, side: int, entry_price: float) -> np.ndarray:
    close = window["close"].astype(float).to_numpy()
    return float(side) * (close / float(entry_price) - 1.0)


def recompute_mae_mfe(window: pd.DataFrame, side: int, entry_price: float) -> tuple[float, float]:
    if side > 0:
        mfe = float(window["high"].astype(float).max() / entry_price - 1.0)
        mae = float(window["low"].astype(float).min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / window["low"].astype(float).min() - 1.0)
        mae = float(entry_price / window["high"].astype(float).max() - 1.0)
    return mfe, mae


def apply_stop_spec(df: pd.DataFrame, market: pd.DataFrame, spec: StopSpec) -> pd.DataFrame:
    out = ensure_time_columns(df)
    out["paper_optstop_hit"] = False
    out["paper_optstop_spec"] = spec.name
    market_ts = market["timestamp"].to_numpy()
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    for idx, row in out.iterrows():
        if float(row["notional"]) <= EPS:
            continue
        side = int(row["side"])
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        orig_exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        entry_price = close_by_ts.get(entry_ts)
        if entry_price is None:
            raise RuntimeError(f"missing entry close: {entry_ts}")
        window = market_window(market, entry_ts, orig_exit_ts)
        moves = move_series(window, side, float(entry_price))
        hours = (window["timestamp"] - entry_ts).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
        mfe_so_far = np.maximum.accumulate(moves)
        exit_pos: int | None = None
        exit_reason = ""
        for pos in range(1, len(window)):
            hold_h = float(hours[pos])
            raw_move = float(moves[pos])
            mfe = float(mfe_so_far[pos])
            if hold_h >= float(spec.loss_after_hours) and raw_move <= float(spec.loss_stop_move):
                exit_pos = pos
                exit_reason = "paper_optstop_loss_exit"
                break
            if (
                hold_h >= float(spec.trail_after_hours)
                and mfe >= float(spec.trail_arm_move)
                and raw_move >= float(spec.trail_floor_move)
                and raw_move <= mfe - float(spec.trail_giveback_move)
            ):
                exit_pos = pos
                exit_reason = "paper_optstop_trail_exit"
                break
            if hold_h >= float(spec.stall_after_hours) and raw_move >= float(spec.stall_min_profit_move):
                lookback_ts = pd.Timestamp(window.loc[pos, "timestamp"]) - pd.Timedelta(hours=float(spec.stall_lookback_hours))
                lb_pos = int(np.searchsorted(market_ts, np.datetime64(lookback_ts), side="left"))
                entry_pos_global = int(np.searchsorted(market_ts, np.datetime64(entry_ts), side="left"))
                rel_lb = max(0, min(pos, lb_pos - entry_pos_global))
                slope = raw_move - float(moves[rel_lb])
                if slope <= float(spec.stall_slope_max):
                    exit_pos = pos
                    exit_reason = "paper_optstop_stall_exit"
                    break
            if hold_h >= float(spec.hard_stop_hours):
                exit_pos = pos
                exit_reason = "paper_optstop_time_exit"
                break
        if exit_pos is None:
            continue
        new_exit_ts = pd.Timestamp(window.loc[exit_pos, "timestamp"])
        if new_exit_ts >= orig_exit_ts:
            continue
        raw_move = float(moves[exit_pos])
        old_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        net_per_notional = raw_move - old_cost
        new_window = window.iloc[: exit_pos + 1]
        mfe, mae = recompute_mae_mfe(new_window, side, float(entry_price))
        hold_bars = int(round((new_exit_ts - entry_ts).total_seconds() / 300.0))
        out.at[idx, "exit_i"] = int(row["entry_i"]) + hold_bars
        out.at[idx, "exit_timestamp"] = new_exit_ts.strftime("%Y-%m-%d %H:%M:%S")
        out.at[idx, "exit_timestamp_dt"] = new_exit_ts
        out.at[idx, "reason"] = exit_reason
        out.at[idx, "raw_exit_price_move"] = raw_move
        out.at[idx, "mfe_price_move"] = mfe
        out.at[idx, "mae_price_move"] = mae
        out.at[idx, "net_per_notional"] = net_per_notional
        out.at[idx, "trade_return"] = net_per_notional * float(row["notional"])
        out.at[idx, "win"] = int(net_per_notional > 0.0)
        out.at[idx, "hold_hours"] = float((new_exit_ts - entry_ts).total_seconds() / 3600.0)
        out.at[idx, "paper_optstop_hit"] = True
        out.at[idx, "paper_optstop_spec"] = spec.name
    return ensure_time_columns(out)


def stop_specs() -> list[StopSpec]:
    loss_specs = [
        ("loss12_2p5", 12.0, -0.025),
        ("loss24_2p5", 24.0, -0.025),
        ("loss24_3p5", 24.0, -0.035),
        ("loss36_3p5", 36.0, -0.035),
        ("loss48_4p5", 48.0, -0.045),
    ]
    trail_specs = [
        ("trail12_3p5_gap1p2", 12.0, 0.035, 0.012, 0.005),
        ("trail24_4p5_gap1p5", 24.0, 0.045, 0.015, 0.010),
        ("trail36_5p5_gap1p8", 36.0, 0.055, 0.018, 0.015),
        ("trail48_6p5_gap2p0", 48.0, 0.065, 0.020, 0.020),
        ("trail72_7p0_gap2p5", 72.0, 0.070, 0.025, 0.025),
    ]
    stall_specs = [
        ("stall24_lb6_min2p5", 24.0, 6.0, 0.025, 0.0000),
        ("stall36_lb12_min3p5", 36.0, 12.0, 0.035, 0.0010),
        ("stall48_lb24_min4p5", 48.0, 24.0, 0.045, 0.0015),
        ("stall72_lb24_min5p5", 72.0, 24.0, 0.055, 0.0020),
        ("stall96_lb24_min6p5", 96.0, 24.0, 0.065, 0.0025),
    ]
    specs: list[StopSpec] = []
    for hard in [36.0, 48.0, 72.0, 96.0, 120.0]:
        for loss_name, loss_after, loss_stop in loss_specs:
            for trail_name, trail_after, trail_arm, trail_gap, trail_floor in trail_specs:
                for stall_name, stall_after, stall_lb, stall_min, stall_slope in stall_specs:
                    name = f"hard{int(hard)}__{loss_name}__{trail_name}__{stall_name}"
                    specs.append(
                        StopSpec(
                            name=name,
                            hard_stop_hours=hard,
                            loss_after_hours=loss_after,
                            loss_stop_move=loss_stop,
                            trail_after_hours=trail_after,
                            trail_arm_move=trail_arm,
                            trail_giveback_move=trail_gap,
                            trail_floor_move=trail_floor,
                            stall_after_hours=stall_after,
                            stall_lookback_hours=stall_lb,
                            stall_min_profit_move=stall_min,
                            stall_slope_max=stall_slope,
                        )
                    )
    return specs


def gate_and_score(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, float]:
    val_pnl = float(row["validation_pnl"])
    val_mdd = float(row["validation_mdd"])
    val_avg_hold = float(row["validation_avg_hold_hours"])
    val_max_hold = float(row["validation_max_hold_hours"])
    base_pnl = float(baseline["pnl"])
    base_avg_hold = float(baseline["avg_hold_hours"])
    base_max_hold = float(baseline["max_hold_hours"])
    pnl_gain = val_pnl - base_pnl
    avg_hold_drop = base_avg_hold - val_avg_hold
    max_hold_drop = base_max_hold - val_max_hold
    gate = bool(
        pnl_gain > 0.0
        and avg_hold_drop > 0.0
        and max_hold_drop > 0.0
        and abs(val_mdd) <= 20.0
        and int(row["validation_overlap_count"]) == 0
        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
    )
    # Validation-only score: prioritize PnL, then reward shorter holding time.
    score = pnl_gain + 0.35 * avg_hold_drop + 0.05 * max_hold_drop
    if abs(val_mdd) > 20.0:
        score -= 20.0 * (abs(val_mdd) - 20.0)
    return gate, float(score)


def source_variant_ledgers(source_dir: Path) -> tuple[Path, Path]:
    safe = BASE_VARIANT.replace(".", "p").replace("/", "_")
    return source_dir / f"validation_{safe}_ledger.csv", source_dir / f"oos_{safe}_ledger.csv"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    baseline = report["baseline"]
    text = f"""# Omega 4.6.2 Paper-Inspired Optimal-Stopping Exit Overlay - 2026-07-01

## Paper Rationale

- HF paper `2302.07320`: stochastic control with exit time; model-free policy/value learning can incorporate transaction costs.
- HF paper `2003.03051`: cost-sensitive reward uses log-growth minus risk and transaction-cost penalties.
- HF paper `2505.04553`: risk-sensitive RL frames variance/expected-shortfall style objectives through augmented state and actor-critic optimization.

Applied interpretation: keep Omega4.6.2 entry/exposure fixed, and replace blunt max-hold-only lifecycle with a validation-selected stopping overlay using loss cut, trailing giveback, profit-stall, and hard time stop.

## Selected Variant

- Spec: `{selected["spec"]}`
- Selection rule: validation-only; OOS is readout only.

| Metric | Baseline Val | Selected Val | Baseline OOS | Selected OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{baseline["validation"]["pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{baseline["oos"]["pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{baseline["validation"]["mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{baseline["oos"]["mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{baseline["validation"]["avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{baseline["oos"]["avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{baseline["validation"]["max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{baseline["oos"]["max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
- Report: `{report["artifacts"]["report"]}`

## Status

`{report["status"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    runtime = read_json(resolve_path(args.runtime_contract))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = component_train_eval_paths(runtime)
    train_market = load_market(train_csv)
    eval_market = load_market(eval_csv)
    source_dir = resolve_path(runtime["source_report"]).parent
    val_path, oos_path = source_variant_ledgers(source_dir)
    val = ensure_time_columns(pd.read_csv(val_path))
    oos = ensure_time_columns(pd.read_csv(oos_path))
    baseline_val = metrics(val)
    baseline_oos = metrics(oos)

    rows: list[dict[str, Any]] = []
    selected_ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for spec in stop_specs():
        val_work = apply_stop_spec(val, train_market, spec)
        oos_work = apply_stop_spec(oos, eval_market, spec)
        val_metrics = metrics(val_work)
        oos_metrics = metrics(oos_work)
        val_month = monthly_summary(val_work)
        gate, score = gate_and_score({**flatten("validation", val_metrics)}, baseline_val)
        row = {
            "spec": spec.name,
            **asdict(spec),
            "validation_gate_pass": gate,
            "selection_score": score,
            "validation_monthly_min_pnl": val_month["monthly_min_pnl"],
            "validation_monthly_worst_mdd": val_month["monthly_worst_mdd"],
            "validation_monthly_positive_count": val_month["monthly_positive_count"],
            "validation_monthly_count": val_month["monthly_count"],
            **flatten("validation", val_metrics),
            **flatten("oos", oos_metrics),
        }
        rows.append(row)
        if gate:
            selected_ledgers[spec.name] = (val_work, oos_work)

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_gate_pass", "selection_score", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking_path = out_dir / "paper_optstop_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    if not bool(ranking.iloc[0]["validation_gate_pass"]):
        selected_name = str(ranking.iloc[0]["spec"])
        selected_spec = next(spec for spec in stop_specs() if spec.name == selected_name)
        selected_val = apply_stop_spec(val, train_market, selected_spec)
        selected_oos = apply_stop_spec(oos, eval_market, selected_spec)
        status = "NO_VALIDATION_CANDIDATE_IMPROVED_BOTH_PNL_AND_HOLD_TIME"
    else:
        selected_name = str(ranking.iloc[0]["spec"])
        selected_val, selected_oos = selected_ledgers[selected_name]
        status = "VALIDATION_SELECTED_CANDIDATE_IMPROVES_PNL_AND_HOLD_TIME"

    selected_safe = selected_name.replace(".", "p").replace("/", "_")
    selected_val_path = out_dir / f"validation_{selected_safe}_ledger.csv"
    selected_oos_path = out_dir / f"oos_{selected_safe}_ledger.csv"
    selected_val.to_csv(selected_val_path, index=False)
    selected_oos.to_csv(selected_oos_path, index=False)

    top10_path = out_dir / "paper_optstop_top10.csv"
    ranking.head(10).to_csv(top10_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "base_variant": BASE_VARIANT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only; OOS readout only",
        "paper_sources": [
            {
                "paper_id": "2302.07320",
                "url": "https://hf.co/papers/2302.07320",
                "use": "exit-time stochastic control; model-free stopping policy framing",
            },
            {
                "paper_id": "2003.03051",
                "url": "https://hf.co/papers/2003.03051",
                "use": "cost-sensitive log-growth reward with risk and transaction penalties",
            },
            {
                "paper_id": "2505.04553",
                "url": "https://hf.co/papers/2505.04553",
                "use": "risk-sensitive RL with variance/ES style objectives",
            },
        ],
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "variants_evaluated": int(len(ranking)),
        "selected_variant": ranking.iloc[0].to_dict(),
        "top10": ranking.head(10).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(ranking_path),
            "top10": str(top10_path),
            "selected_validation_ledger": str(selected_val_path),
            "selected_oos_ledger": str(selected_oos_path),
            "report": str(out_dir / "report.json"),
            "audit_md": str(ROOT / "docs/audits/omega4_6_2_paper_optstop_exit_overlay_20260701.md"),
        },
        "live_promotion_note": (
            "This is still ledger-level lifecycle overlay research. Full promotion requires "
            "FinalGovernorRuntime.decide native replay and fresh post-OOS holdout."
        ),
    }
    write_json(out_dir / "report.json", report)
    write_json(out_dir / "selected_spec.json", {"selected_spec": report["selected_variant"]})
    write_markdown(ROOT / "docs/audits/omega4_6_2_paper_optstop_exit_overlay_20260701.md", report)
    print(
        json.dumps(
            {
                "report": str(out_dir / "report.json"),
                "status": status,
                "selected_spec": selected_name,
                "baseline_validation": baseline_val,
                "selected_validation": {
                    "pnl": report["selected_variant"]["validation_pnl"],
                    "mdd": report["selected_variant"]["validation_mdd"],
                    "avg_hold_hours": report["selected_variant"]["validation_avg_hold_hours"],
                    "max_hold_hours": report["selected_variant"]["validation_max_hold_hours"],
                },
                "baseline_oos": baseline_oos,
                "selected_oos": {
                    "pnl": report["selected_variant"]["oos_pnl"],
                    "mdd": report["selected_variant"]["oos_mdd"],
                    "avg_hold_hours": report["selected_variant"]["oos_avg_hold_hours"],
                    "max_hold_hours": report["selected_variant"]["oos_max_hold_hours"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
