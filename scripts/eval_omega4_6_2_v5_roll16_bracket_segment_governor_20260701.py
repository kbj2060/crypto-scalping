#!/usr/bin/env python3
"""16h bracket roll segment-governor sweep for Omega 4.6.2 v5."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_loss_cluster_governor_20260701.py"
SEGMENT_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_roll24_segment_governor_sweep_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll16_bracket_segment_governor_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll24_segment_governor_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
PARENT_REPORT = ROOT / "tmp/causal_regen_20260516" / PARENT_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll16_bracket_segment_governor_20260701.md"
MAX_ROLL_HOURS = 16.0
EPS = 1.0e-12


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


def split_roll_bracket(
    df: pd.DataFrame,
    market: pd.DataFrame,
    segment_mod: Any,
    max_hours: float,
    tp_move: float,
    sl_move: float,
) -> pd.DataFrame:
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
        seg_i = 0
        roundtrip_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        while cur_entry < parent_exit:
            target_exit = min(cur_entry + pd.Timedelta(hours=max_hours), parent_exit)
            window = market[(market["timestamp"] >= cur_entry) & (market["timestamp"] <= target_exit)].copy()
            if window.empty:
                break
            if cur_entry not in close_by_ts:
                cur_entry = pd.Timestamp(window["timestamp"].iloc[0])
            entry_price = float(close_by_ts[cur_entry])
            exit_ts = pd.Timestamp(window["timestamp"].iloc[-1])
            raw_move: float | None = None
            reason: str | None = None
            for _, bar in window.iloc[1:].iterrows():
                ts = pd.Timestamp(bar["timestamp"])
                high = float(bar["high"])
                low = float(bar["low"])
                if side > 0:
                    hit_sl = (low / entry_price - 1.0) <= -sl_move
                    hit_tp = (high / entry_price - 1.0) >= tp_move
                else:
                    hit_sl = (entry_price / high - 1.0) <= -sl_move
                    hit_tp = (entry_price / low - 1.0) >= tp_move
                if hit_sl or hit_tp:
                    exit_ts = ts
                    if hit_sl:
                        raw_move = -sl_move
                        reason = "roll16_bracket_sl"
                    else:
                        raw_move = tp_move
                        reason = "roll16_bracket_tp"
                    break
            if raw_move is None:
                exit_price = float(close_by_ts[exit_ts])
                raw_move = float(side) * (exit_price / entry_price - 1.0)
                reason = "roll16_final" if exit_ts >= parent_exit else "roll16_time_exit"
            used_window = market[(market["timestamp"] >= cur_entry) & (market["timestamp"] <= exit_ts)].copy()
            mfe, mae = segment_mod.segment_mae_mfe(used_window, side, entry_price)
            net_per_notional = raw_move - roundtrip_cost
            synthetic_entry_i = int(row["entry_i"]) + seg_i * 100_000
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
                    "reason": reason,
                    "raw_exit_price_move": raw_move,
                    "mfe_price_move": mfe,
                    "mae_price_move": mae,
                    "net_per_notional": net_per_notional,
                    "trade_return": net_per_notional * float(row["notional"]),
                    "win": int(net_per_notional > 0.0),
                    "hold_hours": float((exit_ts - cur_entry).total_seconds() / 3600.0),
                    "roll16_parent_entry_timestamp": parent_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll16_parent_exit_timestamp": parent_exit.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll16_segment": seg_i,
                    "roll16_roundtrip_cost": roundtrip_cost,
                    "roll16_tp_move": tp_move,
                    "roll16_sl_move": sl_move,
                    "pre_governor_notional": float(row["notional"]),
                    "roll24_segment_governor_spec": "",
                    "roll24_segment_multiplier": 1.0,
                }
            )
            rows.append(out)
            if exit_ts >= parent_exit:
                break
            cur_entry = exit_ts + pd.Timedelta(minutes=5)
            seg_i += 1
    return pd.DataFrame(rows)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll16 Bracket Segment Governor - 2026-07-01

## Method

This candidate starts from the v5 parent, splits positions into `<=16h` segments, and exits each segment early when a path-causal `4.5%` TP or `4.5%` SL is touched. Same-bar TP/SL ambiguity is handled conservatively by taking SL first. Segment exposure/governor selection is validation-primary with an OOS safety gate; fresh holdout is required before any live claim.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

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
- TP/SL: `{selected["roll16_tp_move"]:.4f}` / `{selected["roll16_sl_move"]:.4f}`
- Research gate pass: `{selected["research_upgrade_gate_pass"]}`

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
    v1 = load_module("omega462_loss_cluster_v1_for_roll16", V1_MODULE_PATH)
    segment_mod = load_module("omega462_segment_for_roll16", SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll16", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference = read_json(REFERENCE_REPORT)["selected_variant"]
    parent_report = read_json(PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])

    rows: list[dict[str, Any]] = []
    governors = [
        segment_mod.SegmentGovernorSpec("none", 1.00, 1.00, 0.0),
        segment_mod.SegmentGovernorSpec("loss1_90_win12", 0.90, 1.00, 12.0),
        segment_mod.SegmentGovernorSpec("streak85_60_win12", 0.85, 0.60, 12.0),
    ]
    exposure_specs = [
        segment_mod.ExposureSpec("long100_short100_cap430", 1.00, 1.00, 4.30),
        segment_mod.ExposureSpec("long085_short100_cap430", 0.85, 1.00, 4.30),
        segment_mod.ExposureSpec("long070_short100_cap410", 0.70, 1.00, 4.10),
        segment_mod.ExposureSpec("long100_short095_cap430", 1.00, 0.95, 4.30),
    ]
    bracket_specs = [(0.045, 0.045), (0.035, 0.045), (0.045, 0.060), (0.060, 0.060)]
    for tp_move, sl_move in bracket_specs:
        val_base = split_roll_bracket(
            val_parent, train_market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move
        )
        oos_base = split_roll_bracket(
            oos_parent, eval_market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move
        )
        for exposure in exposure_specs:
            for governor in governors:
                val_work = segment_mod.apply_segment_exposure_governor(val_base, exposure, governor)
                oos_work = segment_mod.apply_segment_exposure_governor(oos_base, exposure, governor)
                row = {
                    "exposure_spec": exposure.name,
                    "segment_governor_spec": governor.name,
                    "roll16_max_hours": MAX_ROLL_HOURS,
                    "roll16_tp_move": tp_move,
                    "roll16_sl_move": sl_move,
                    **{f"exposure_{k}": value for k, value in asdict(exposure).items()},
                    **{f"segment_governor_{k}": value for k, value in asdict(governor).items()},
                    **flatten("validation", stop_mod.metrics(val_work)),
                    **flatten("oos", stop_mod.metrics(oos_work)),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) > float(reference["validation_pnl"])
                    and float(row["validation_mdd"]) >= -20.0
                    and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                oos_gate = bool(
                    float(row["oos_pnl"]) > float(reference["oos_pnl"])
                    and float(row["oos_mdd"]) >= -20.0
                    and float(row["oos_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and int(row["oos_overlap_count"]) == 0
                    and float(row["oos_max_leverage"]) <= 5.0 + 1.0e-9
                    and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_upgrade_gate_pass"] = validation_gate
                row["oos_research_gate_pass"] = oos_gate
                row["research_upgrade_gate_pass"] = validation_gate and oos_gate
                row["selection_score"] = (
                    float(row["validation_pnl"])
                    - float(reference["validation_pnl"])
                    + 0.2 * (float(reference["validation_avg_hold_hours"]) - float(row["validation_avg_hold_hours"]))
                )
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["research_upgrade_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_exposure = segment_mod.ExposureSpec(
        selected["exposure_spec"],
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_governor = segment_mod.SegmentGovernorSpec(
        selected["segment_governor_spec"],
        float(selected["segment_governor_loss1_scale"]),
        float(selected["segment_governor_loss2_scale"]),
        float(selected["segment_governor_loss_window_hours"]),
    )
    selected_val_base = split_roll_bracket(
        val_parent,
        train_market,
        segment_mod,
        MAX_ROLL_HOURS,
        float(selected["roll16_tp_move"]),
        float(selected["roll16_sl_move"]),
    )
    selected_oos_base = split_roll_bracket(
        oos_parent,
        eval_market,
        segment_mod,
        MAX_ROLL_HOURS,
        float(selected["roll16_tp_move"]),
        float(selected["roll16_sl_move"]),
    )
    selected_val = segment_mod.apply_segment_exposure_governor(
        selected_val_base, selected_exposure, selected_governor
    )
    selected_oos = segment_mod.apply_segment_exposure_governor(
        selected_oos_base, selected_exposure, selected_governor
    )
    status = (
        "RESEARCH_ROLL16_BRACKET_UPGRADE_IMPROVES_PNL_AND_HOLD"
        if bool(selected["research_upgrade_gate_pass"])
        else "NO_RESEARCH_ROLL16_BRACKET_UPGRADE_IMPROVED_PNL_AND_HOLD"
    )
    safe = (
        f"{selected['exposure_spec']}__{selected['segment_governor_spec']}__"
        f"tp{float(selected['roll16_tp_move']):.3f}_sl{float(selected['roll16_sl_move']):.3f}"
    ).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll16_bracket_segment_governor_ranking.csv"
    top20_path = OUT_DIR / "roll16_bracket_segment_governor_top20.csv"
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
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_primary_with_oos_safety_gate; fresh_holdout_required",
        "reference_variant": reference,
        "parent_variant": parent_report["selected_variant"],
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
                "selected_tp": selected["roll16_tp_move"],
                "selected_sl": selected["roll16_sl_move"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gate": bool(selected["validation_upgrade_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "gate": bool(selected["oos_research_gate_pass"]),
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
