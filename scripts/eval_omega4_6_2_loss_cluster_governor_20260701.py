#!/usr/bin/env python3
"""Path-causal loss-cluster governor sweep for Omega 4.6.2 hold compression."""

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
STOP_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_paper_exit_stopping_20260701.py"
REFERENCE_MODEL_ID = "omega4_6_2_paper_optstop_exit_sizing_overlay_20260701"
MODEL_ID = "omega4_6_2_loss_cluster_governor_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
DEFAULT_RUNTIME = ROOT / "tmp/causal_regen_20260516" / BASE_MODEL_ID / "runtime_contract.json"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_loss_cluster_governor_20260701.md"
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
class GovernorSpec:
    name: str
    loss1_scale: float
    loss2_scale: float
    dd1_threshold: float
    dd1_scale: float
    dd2_threshold: float
    dd2_scale: float
    loss_window_hours: float = 1.0e9


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


def stop_specs(stop_mod: Any) -> list[Any]:
    return [
        stop_mod.StopSpec(
            name="hard90__loss48_4p5__trail72_7p0_gap2p5__stall84_lb24_min6p0",
            hard_stop_hours=90.0,
            loss_after_hours=48.0,
            loss_stop_move=-0.045,
            trail_after_hours=72.0,
            trail_arm_move=0.070,
            trail_giveback_move=0.025,
            trail_floor_move=0.025,
            stall_after_hours=84.0,
            stall_lookback_hours=24.0,
            stall_min_profit_move=0.060,
            stall_slope_max=0.0025,
        ),
        stop_mod.StopSpec(
            name="hard90__loss60_5p0__trail72_7p0_gap2p5__stall84_lb24_min6p0",
            hard_stop_hours=90.0,
            loss_after_hours=60.0,
            loss_stop_move=-0.050,
            trail_after_hours=72.0,
            trail_arm_move=0.070,
            trail_giveback_move=0.025,
            trail_floor_move=0.025,
            stall_after_hours=84.0,
            stall_lookback_hours=24.0,
            stall_min_profit_move=0.060,
            stall_slope_max=0.0025,
        ),
        stop_mod.StopSpec(
            name="hard96__loss48_4p5__trail72_7p0_gap2p5__stall96_lb24_min6p5",
            hard_stop_hours=96.0,
            loss_after_hours=48.0,
            loss_stop_move=-0.045,
            trail_after_hours=72.0,
            trail_arm_move=0.070,
            trail_giveback_move=0.025,
            trail_floor_move=0.025,
            stall_after_hours=96.0,
            stall_lookback_hours=24.0,
            stall_min_profit_move=0.065,
            stall_slope_max=0.0025,
        ),
    ]


def exposure_specs() -> list[ExposureSpec]:
    specs = []
    for factor in [1.48, 1.55, 1.62, 1.70, 1.78, 1.86, 1.94, 2.02]:
        cap = min(5.0, round(2.30 * factor, 2))
        specs.append(
            ExposureSpec(
                name=f"balanced{int(round(factor * 100)):03d}_cap{int(round(cap * 100)):03d}",
                long_factor=factor,
                short_factor=factor,
                cap_notional=cap,
            )
        )
    for short_factor in [1.62, 1.70, 1.78, 1.86, 1.94, 2.02]:
        cap = min(5.0, round(2.15 * short_factor, 2))
        specs.append(
            ExposureSpec(
                name=f"short{int(round(short_factor * 100)):03d}_long100_cap{int(round(cap * 100)):03d}",
                long_factor=1.0,
                short_factor=short_factor,
                cap_notional=cap,
            )
        )
    return specs


def governor_specs() -> list[GovernorSpec]:
    return [
        GovernorSpec("none", 1.00, 1.00, -1.0, 1.00, -1.0, 1.00),
        GovernorSpec("streak75_50", 0.75, 0.50, -1.0, 1.00, -1.0, 1.00),
        GovernorSpec("streak65_40", 0.65, 0.40, -1.0, 1.00, -1.0, 1.00),
        GovernorSpec("streak55_32", 0.55, 0.32, -1.0, 1.00, -1.0, 1.00),
        GovernorSpec("loss1_75_win12", 0.75, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        GovernorSpec("loss1_70_win12", 0.70, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        GovernorSpec("loss1_65_win12", 0.65, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        GovernorSpec("loss1_75_win24", 0.75, 1.00, -1.0, 1.00, -1.0, 1.00, 24.0),
        GovernorSpec("loss1_70_win24", 0.70, 1.00, -1.0, 1.00, -1.0, 1.00, 24.0),
        GovernorSpec("loss1_65_win24", 0.65, 1.00, -1.0, 1.00, -1.0, 1.00, 24.0),
        GovernorSpec("loss1_75_win48", 0.75, 1.00, -1.0, 1.00, -1.0, 1.00, 48.0),
        GovernorSpec("loss1_70_win48", 0.70, 1.00, -1.0, 1.00, -1.0, 1.00, 48.0),
        GovernorSpec("loss1_65_win48", 0.65, 1.00, -1.0, 1.00, -1.0, 1.00, 48.0),
        GovernorSpec("streak75_60_win48", 0.75, 0.60, -1.0, 1.00, -1.0, 1.00, 48.0),
        GovernorSpec("streak70_55_win48", 0.70, 0.55, -1.0, 1.00, -1.0, 1.00, 48.0),
        GovernorSpec("dd8_70_dd14_45", 1.00, 1.00, -0.08, 0.70, -0.14, 0.45),
        GovernorSpec("dd6_65_dd12_40", 1.00, 1.00, -0.06, 0.65, -0.12, 0.40),
        GovernorSpec("streak75_50_dd8_70_dd14_45", 0.75, 0.50, -0.08, 0.70, -0.14, 0.45),
        GovernorSpec("streak65_40_dd6_65_dd12_40", 0.65, 0.40, -0.06, 0.65, -0.12, 0.40),
    ]


def apply_exposure_governor(
    df: pd.DataFrame,
    exposure: ExposureSpec,
    governor: GovernorSpec,
) -> pd.DataFrame:
    out = df.copy()
    out["paper_sizing_spec"] = exposure.name
    out["loss_cluster_governor_spec"] = governor.name
    out["loss_cluster_multiplier"] = 0.0
    out["pre_governor_notional"] = 0.0

    active_mask = out["notional"].astype(float) > EPS
    out.loc[~active_mask, ["notional", "leverage", "margin_fraction", "trade_return"]] = 0.0
    active_idx = list(out[active_mask].sort_values(["entry_i", "exit_i"]).index)
    equity = 1.0
    peak = 1.0
    loss_streak = 0
    last_loss_exit_ts: pd.Timestamp | None = None
    max_notional = float(exposure.leverage_cap) * float(exposure.max_margin_fraction)
    cap = min(float(exposure.cap_notional), max_notional)

    for idx in active_idx:
        row = out.loc[idx]
        side = int(row["side"])
        side_factor = float(exposure.long_factor if side > 0 else exposure.short_factor)
        base_notional = min(float(row["notional"]) * side_factor, cap)
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        effective_loss_streak = loss_streak
        if last_loss_exit_ts is not None:
            hours_from_loss = (entry_ts - last_loss_exit_ts).total_seconds() / 3600.0
            if hours_from_loss > float(governor.loss_window_hours):
                effective_loss_streak = 0
        streak_multiplier = 1.0
        if effective_loss_streak >= 2:
            streak_multiplier = float(governor.loss2_scale)
        elif effective_loss_streak == 1:
            streak_multiplier = float(governor.loss1_scale)
        drawdown = equity / max(peak, EPS) - 1.0
        dd_multiplier = 1.0
        if drawdown <= float(governor.dd2_threshold):
            dd_multiplier = float(governor.dd2_scale)
        elif drawdown <= float(governor.dd1_threshold):
            dd_multiplier = float(governor.dd1_scale)
        multiplier = min(streak_multiplier, dd_multiplier)
        notional = base_notional * multiplier
        leverage = float(exposure.leverage_cap)
        margin = notional / leverage if leverage > EPS else 0.0
        trade_return = float(row["net_per_notional"]) * notional

        out.at[idx, "pre_governor_notional"] = base_notional
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

        equity *= 1.0 + trade_return
        peak = max(peak, equity)
        if float(row["net_per_notional"]) < 0.0:
            loss_streak += 1
            last_loss_exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        else:
            loss_streak = 0
            last_loss_exit_ts = None

    return out


def gate_and_score(row: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, float]:
    pnl_gain = float(row["validation_pnl"]) - float(reference["validation_pnl"])
    avg_hold_drop = float(reference["validation_avg_hold_hours"]) - float(row["validation_avg_hold_hours"])
    max_hold_drop = float(reference["validation_max_hold_hours"]) - float(row["validation_max_hold_hours"])
    mdd_abs = abs(float(row["validation_mdd"]))
    gate = bool(
        pnl_gain > 0.0
        and avg_hold_drop > 0.0
        and max_hold_drop > 0.0
        and mdd_abs <= 20.0
        and int(row["validation_overlap_count"]) == 0
        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
    )
    score = pnl_gain + 0.45 * avg_hold_drop + 0.20 * max_hold_drop
    if mdd_abs > 20.0:
        score -= 50.0 * (mdd_abs - 20.0)
    return gate, float(score)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 Loss-Cluster Governor - 2026-07-01

## Method

This sweep tests a path-causal risk governor: only prior closed trade losses and current realized drawdown can reduce the next trade's notional. It then tries higher base exposure to recover PnL while compressing max hold to 90h.

## Result

- Status: `{report["status"]}`
- Selection scope: `{report["selection_scope"]}`
- Reference model: `{report["reference_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Stop spec: `{selected["stop_spec"]}`
- Exposure spec: `{selected["exposure_spec"]}`
- Governor spec: `{selected["governor_spec"]}`
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
    stop_mod = load_module("omega462_stop_mod_for_governor", STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = stop_mod.read_json(stop_mod.resolve_path(DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    source_dir = stop_mod.resolve_path(runtime["source_report"]).parent
    val_path, oos_path = stop_mod.source_variant_ledgers(source_dir)
    val = stop_mod.ensure_time_columns(pd.read_csv(val_path))
    oos = stop_mod.ensure_time_columns(pd.read_csv(oos_path))
    reference = read_json(REFERENCE_REPORT)["selected_variant"]

    rows: list[dict[str, Any]] = []
    stop_list = stop_specs(stop_mod)
    exposure_list = exposure_specs()
    governor_list = governor_specs()
    for stop_spec in stop_list:
        val_stop = stop_mod.apply_stop_spec(val, train_market, stop_spec)
        oos_stop = stop_mod.apply_stop_spec(oos, eval_market, stop_spec)
        for exposure in exposure_list:
            for governor in governor_list:
                val_work = apply_exposure_governor(val_stop, exposure, governor)
                oos_work = apply_exposure_governor(oos_stop, exposure, governor)
                row = {
                    "stop_spec": stop_spec.name,
                    "exposure_spec": exposure.name,
                    "governor_spec": governor.name,
                    **{f"stop_{k}": v for k, v in asdict(stop_spec).items()},
                    **{f"exposure_{k}": v for k, v in asdict(exposure).items()},
                    **{f"governor_{k}": v for k, v in asdict(governor).items()},
                    **flatten("validation", stop_mod.metrics(val_work)),
                    **flatten("oos", stop_mod.metrics(oos_work)),
                }
                gate, score = gate_and_score(row, reference)
                row["validation_upgrade_gate_pass"] = gate
                row["selection_score"] = score
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_upgrade_gate_pass", "selection_score", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_stop = next(spec for spec in stop_list if spec.name == selected["stop_spec"])
    selected_exposure = next(spec for spec in exposure_list if spec.name == selected["exposure_spec"])
    selected_governor = next(spec for spec in governor_list if spec.name == selected["governor_spec"])
    selected_val = apply_exposure_governor(
        stop_mod.apply_stop_spec(val, train_market, selected_stop),
        selected_exposure,
        selected_governor,
    )
    selected_oos = apply_exposure_governor(
        stop_mod.apply_stop_spec(oos, eval_market, selected_stop),
        selected_exposure,
        selected_governor,
    )
    status = (
        "VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_AND_HOLD"
        if bool(selected["validation_upgrade_gate_pass"])
        else "NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_AND_HOLD"
    )
    safe = (
        f"{selected['stop_spec']}__{selected['exposure_spec']}__{selected['governor_spec']}"
        .replace(".", "p")
        .replace("/", "_")
    )
    ranking_path = OUT_DIR / "loss_cluster_governor_ranking.csv"
    top20_path = OUT_DIR / "loss_cluster_governor_top20.csv"
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
        "base_model_id": BASE_MODEL_ID,
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
                "selected_stop": selected["stop_spec"],
                "selected_exposure": selected["exposure_spec"],
                "selected_governor": selected["governor_spec"],
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
