#!/usr/bin/env python3
"""Roll24 daytrade sweep with segment-level loss governor and exposure tuning."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_loss_cluster_governor_20260701.py"
MODEL_ID = "omega4_6_2_roll24_segment_governor_sweep_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_roll24_daytrade_overlay_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_roll24_segment_governor_sweep_20260701.md"
MAX_ROLL_HOURS = 24.0
EPS = 1.0e-12


@dataclass(frozen=True)
class ExposureSpec:
    name: str
    long_factor: float
    short_factor: float
    cap_notional: float
    leverage_cap: float = 5.0
    max_margin_fraction: float = 1.0


@dataclass(frozen=True)
class SegmentGovernorSpec:
    name: str
    loss1_scale: float
    loss2_scale: float
    loss_window_hours: float


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def segment_mae_mfe(window: pd.DataFrame, side: int, entry_price: float) -> tuple[float, float]:
    if side > 0:
        mfe = float(window["high"].astype(float).max() / entry_price - 1.0)
        mae = float(window["low"].astype(float).min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / window["low"].astype(float).min() - 1.0)
        mae = float(entry_price / window["high"].astype(float).max() - 1.0)
    return mfe, mae


def split_roll24_base(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="raise")
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if float(row["notional"]) <= EPS:
            rows.append(row.to_dict())
            continue
        side = int(row["side"])
        parent_entry = pd.Timestamp(row["entry_timestamp"])
        parent_exit = pd.Timestamp(row["exit_timestamp"])
        cur_entry = parent_entry
        seg = 0
        roundtrip_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        while cur_entry < parent_exit:
            target_exit = min(cur_entry + pd.Timedelta(hours=MAX_ROLL_HOURS), parent_exit)
            window = market[(market["timestamp"] >= cur_entry) & (market["timestamp"] <= target_exit)].copy()
            if window.empty:
                break
            if cur_entry not in close_by_ts:
                cur_entry = pd.Timestamp(window["timestamp"].iloc[0])
            exit_ts = pd.Timestamp(window["timestamp"].iloc[-1])
            entry_price = float(close_by_ts[cur_entry])
            exit_price = float(close_by_ts[exit_ts])
            raw_move = float(side) * (exit_price / entry_price - 1.0)
            net_per_notional = raw_move - roundtrip_cost
            mfe, mae = segment_mae_mfe(window, side, entry_price)
            synthetic_entry_i = int(row["entry_i"]) + seg * 100_000
            hold_bars = int(round((exit_ts - cur_entry).total_seconds() / 300.0))
            out = row.to_dict()
            out.update(
                {
                    "entry_i": synthetic_entry_i,
                    "exit_i": synthetic_entry_i + hold_bars,
                    "entry_timestamp": cur_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_timestamp": exit_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_timestamp_dt": cur_entry,
                    "exit_timestamp_dt": exit_ts,
                    "reason": "roll24_final" if exit_ts >= parent_exit else "roll24_time_exit",
                    "raw_exit_price_move": raw_move,
                    "mfe_price_move": mfe,
                    "mae_price_move": mae,
                    "net_per_notional": net_per_notional,
                    "trade_return": net_per_notional * float(row["notional"]),
                    "win": int(net_per_notional > 0.0),
                    "hold_hours": float((exit_ts - cur_entry).total_seconds() / 3600.0),
                    "roll24_parent_entry_timestamp": parent_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll24_parent_exit_timestamp": parent_exit.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll24_segment": seg,
                    "roll24_roundtrip_cost": roundtrip_cost,
                    "pre_governor_notional": float(row["notional"]),
                    "roll24_segment_governor_spec": "",
                    "roll24_segment_multiplier": 1.0,
                }
            )
            rows.append(out)
            if exit_ts >= parent_exit:
                break
            cur_entry = exit_ts + pd.Timedelta(minutes=5)
            seg += 1
    return pd.DataFrame(rows)


def apply_segment_exposure_governor(
    df: pd.DataFrame,
    exposure: ExposureSpec,
    governor: SegmentGovernorSpec,
) -> pd.DataFrame:
    out = df.copy()
    out["paper_sizing_spec"] = exposure.name
    out["roll24_segment_governor_spec"] = governor.name
    out["roll24_segment_multiplier"] = 0.0
    active_idx = list(out[out["notional"].astype(float) > EPS].sort_values(["entry_i", "exit_i"]).index)
    loss_streak = 0
    last_loss_exit_ts: pd.Timestamp | None = None
    cap = min(float(exposure.cap_notional), float(exposure.leverage_cap) * float(exposure.max_margin_fraction))
    for idx in active_idx:
        row = out.loc[idx]
        side = int(row["side"])
        side_factor = float(exposure.long_factor if side > 0 else exposure.short_factor)
        base_notional = min(float(row["pre_governor_notional"]) * side_factor, cap)
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        effective_streak = loss_streak
        if last_loss_exit_ts is not None:
            hours_from_loss = (entry_ts - last_loss_exit_ts).total_seconds() / 3600.0
            if hours_from_loss > float(governor.loss_window_hours):
                effective_streak = 0
        if effective_streak >= 2:
            multiplier = float(governor.loss2_scale)
        elif effective_streak == 1:
            multiplier = float(governor.loss1_scale)
        else:
            multiplier = 1.0
        notional = base_notional * multiplier
        leverage = float(exposure.leverage_cap)
        margin = notional / leverage if leverage > EPS else 0.0
        trade_return = float(row["net_per_notional"]) * notional
        out.at[idx, "pre_governor_notional"] = base_notional
        out.at[idx, "roll24_segment_multiplier"] = multiplier
        out.at[idx, "loss_cluster_multiplier"] = multiplier
        out.at[idx, "notional"] = notional
        out.at[idx, "leverage"] = leverage
        out.at[idx, "margin_fraction"] = margin
        out.at[idx, "risk_notional"] = notional
        out.at[idx, "risk_leverage"] = leverage
        out.at[idx, "risk_margin_fraction"] = margin
        out.at[idx, "exit_input_notional"] = notional
        out.at[idx, "exit_input_leverage"] = leverage
        out.at[idx, "exit_input_exposure"] = notional
        out.at[idx, "trade_return"] = trade_return
        out.at[idx, "win"] = int(trade_return > 0.0)
        if float(row["net_per_notional"]) < 0.0:
            loss_streak += 1
            last_loss_exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        else:
            loss_streak = 0
            last_loss_exit_ts = None
    return out


def exposure_specs() -> list[ExposureSpec]:
    specs: list[ExposureSpec] = []
    for long_factor in [0.75, 1.00, 1.20]:
        for short_factor in [1.50, 1.65, 1.80, 1.95, 2.10, 2.25, 2.40]:
            cap = min(5.0, round(2.15 * short_factor, 2))
            specs.append(
                ExposureSpec(
                    name=(
                        f"long{int(round(long_factor * 100)):03d}_"
                        f"short{int(round(short_factor * 100)):03d}_cap{int(round(cap * 100)):03d}"
                    ),
                    long_factor=long_factor,
                    short_factor=short_factor,
                    cap_notional=cap,
                )
            )
    return specs


def governor_specs() -> list[SegmentGovernorSpec]:
    return [
        SegmentGovernorSpec("none", 1.0, 1.0, 0.0),
        SegmentGovernorSpec("loss1_85_win12", 0.85, 1.0, 12.0),
        SegmentGovernorSpec("loss1_75_win12", 0.75, 1.0, 12.0),
        SegmentGovernorSpec("loss1_65_win12", 0.65, 1.0, 12.0),
        SegmentGovernorSpec("streak85_65_win12", 0.85, 0.65, 12.0),
        SegmentGovernorSpec("streak75_55_win12", 0.75, 0.55, 12.0),
        SegmentGovernorSpec("loss1_85_win24", 0.85, 1.0, 24.0),
        SegmentGovernorSpec("loss1_75_win24", 0.75, 1.0, 24.0),
    ]


def gate_and_score(row: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, float]:
    pnl_gain = float(row["validation_pnl"]) - float(reference["validation_pnl"])
    avg_hold_ok = float(row["validation_avg_hold_hours"]) <= float(reference["validation_avg_hold_hours"]) + 1.0e-9
    max_hold_ok = float(row["validation_max_hold_hours"]) <= 24.0 + 1.0e-9
    mdd_abs = abs(float(row["validation_mdd"]))
    gate = bool(
        pnl_gain > 0.0
        and avg_hold_ok
        and max_hold_ok
        and mdd_abs <= 20.0
        and float(row["validation_pnl"]) >= 100.0
        and int(row["validation_overlap_count"]) == 0
        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
    )
    score = pnl_gain - max(0.0, mdd_abs - 19.80) * 15.0
    return gate, float(score)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 Roll24 Segment Governor Sweep - 2026-07-01

## Method

This sweep rebuilds the v4 90h exit as 24h-or-less roll segments, then retunes exposure and applies a path-causal segment-level loss-window governor.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Selection scope: `{report["selection_scope"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Exposure spec: `{selected["exposure_spec"]}`
- Segment governor: `{selected["segment_governor_spec"]}`
- Validation upgrade gate pass: `{selected["validation_upgrade_gate_pass"]}`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    v1 = load_module("omega462_loss_cluster_v1_for_roll24_sweep", V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll24_sweep", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    source_dir = stop_mod.resolve_path(runtime["source_report"]).parent
    val_path, oos_path = stop_mod.source_variant_ledgers(source_dir)
    val = stop_mod.ensure_time_columns(pd.read_csv(val_path))
    oos = stop_mod.ensure_time_columns(pd.read_csv(oos_path))
    reference = read_json(REFERENCE_REPORT)["selected_variant"]

    stop_spec = stop_mod.StopSpec(
        name="hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5",
        hard_stop_hours=90.0,
        loss_after_hours=48.0,
        loss_stop_move=-0.045,
        trail_after_hours=72.0,
        trail_arm_move=0.070,
        trail_giveback_move=0.025,
        trail_floor_move=0.025,
        stall_after_hours=72.0,
        stall_lookback_hours=24.0,
        stall_min_profit_move=0.055,
        stall_slope_max=0.0020,
    )
    val_base = split_roll24_base(stop_mod.apply_stop_spec(val, train_market, stop_spec), train_market)
    oos_base = split_roll24_base(stop_mod.apply_stop_spec(oos, eval_market, stop_spec), eval_market)
    rows: list[dict[str, Any]] = []
    exp_list = exposure_specs()
    gov_list = governor_specs()
    for exposure in exp_list:
        for governor in gov_list:
            val_work = apply_segment_exposure_governor(val_base, exposure, governor)
            oos_work = apply_segment_exposure_governor(oos_base, exposure, governor)
            row = {
                "exposure_spec": exposure.name,
                "segment_governor_spec": governor.name,
                **{f"exposure_{k}": v for k, v in asdict(exposure).items()},
                **{f"segment_governor_{k}": v for k, v in asdict(governor).items()},
                **flatten("validation", stop_mod.metrics(val_work)),
                **flatten("oos", stop_mod.metrics(oos_work)),
            }
            gate, score = gate_and_score(row, reference)
            row["validation_upgrade_gate_pass"] = gate
            row["selection_score"] = score
            rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(
        ["validation_upgrade_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_exp = next(spec for spec in exp_list if spec.name == selected["exposure_spec"])
    selected_gov = next(spec for spec in gov_list if spec.name == selected["segment_governor_spec"])
    selected_val = apply_segment_exposure_governor(val_base, selected_exp, selected_gov)
    selected_oos = apply_segment_exposure_governor(oos_base, selected_exp, selected_gov)
    status = (
        "VALIDATION_DAYTRADE_UPGRADE_IMPROVES_ROLL24_REFERENCE"
        if bool(selected["validation_upgrade_gate_pass"])
        else "NO_VALIDATION_DAYTRADE_UPGRADE_IMPROVED_ROLL24_REFERENCE"
    )
    safe = f"{selected['exposure_spec']}__{selected['segment_governor_spec']}".replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll24_segment_governor_ranking.csv"
    top20_path = OUT_DIR / "roll24_segment_governor_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only; OOS readout only",
        "reference_variant": reference,
        "variants_evaluated": int(len(ranking)),
        "selected_variant": selected,
        "top20": ranking.head(20).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "top20": str(top20_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_exposure": selected["exposure_spec"],
                "selected_segment_governor": selected["segment_governor_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "trades": selected["validation_trades"],
                    "gate": bool(selected["validation_upgrade_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "trades": selected["oos_trades"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
