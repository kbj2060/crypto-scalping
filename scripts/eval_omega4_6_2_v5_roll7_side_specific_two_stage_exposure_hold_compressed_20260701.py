#!/usr/bin/env python3
"""7h hold-compressed two-stage exposure branch for Omega 4.6.2 v5."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLL8_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701.py"
FEATURE_VETO_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_feature_veto_20260701.py"
TWO_STAGE_VETO_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
    / "report.json"
)
EXPOSURE_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701.py"
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll7_side_specific_two_stage_exposure_hold_compressed_20260701.md"
MAX_ROLL_HOURS = 7.0
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def modules() -> dict[str, Any]:
    roll8 = load_module("omega462_roll8_for_roll7_hold_compressed", ROLL8_PATH)
    base = roll8.load_module("omega462_roll9_base_for_roll7_hold_compressed", roll8.BASE_PATH)
    source = base.load_module("omega462_roll16_source_for_roll7_hold_compressed", base.SOURCE_MODULE_PATH)
    v1 = source.load_module("omega462_loss_cluster_v1_for_roll7_hold_compressed", source.V1_MODULE_PATH)
    segment_mod = source.load_module("omega462_segment_for_roll7_hold_compressed", source.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll7_hold_compressed", v1.STOP_MODULE_PATH)
    feat = load_module("omega462_feature_veto_for_roll7_hold_compressed", FEATURE_VETO_PATH)
    exposure = load_module("omega462_exposure_for_roll7_hold_compressed", EXPOSURE_PATH)
    helper = load_module("omega462_helper_for_roll7_hold_compressed", AUDIT_HELPER_PATH)
    return {
        "base": base,
        "source": source,
        "v1": v1,
        "segment_mod": segment_mod,
        "stop_mod": stop_mod,
        "feat": feat,
        "exposure": exposure,
        "helper": helper,
    }


def normalize_roll7(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={col: col.replace("roll16_", "roll7_") for col in out.columns if col.startswith("roll16_")})
    out["reason"] = out["reason"].astype(str).str.replace("roll16_", "roll7_", regex=False)
    return out


def split_side(
    source: Any,
    segment_mod: Any,
    df: pd.DataFrame,
    market: pd.DataFrame,
    tp_move: float,
    sl_move: float,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return normalize_roll7(
        source.split_roll_bracket(df, market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move)
    )


def build_pre_exposure(df: pd.DataFrame, market: pd.DataFrame, mods: dict[str, Any]) -> pd.DataFrame:
    base = mods["base"]
    source = mods["source"]
    segment_mod = mods["segment_mod"]
    feat = mods["feat"]
    active = df[df["notional"].astype(float) > EPS]
    skipped = df[df["notional"].astype(float) <= EPS]
    long_df = active[active["side"].astype(int) > 0]
    short_df = active[active["side"].astype(int) < 0]
    out = base.combine_sides(
        split_side(source, segment_mod, long_df, market, 0.0200, 0.0300),
        split_side(source, segment_mod, short_df, market, 0.0250, 0.0385),
        skipped,
    )
    out = feat.refresh_entry_features(out, market)
    two_stage_report = json.loads(TWO_STAGE_VETO_REPORT.read_text(encoding="utf-8"))
    first_stage = two_stage_report["reference_variant"]
    second_stage = two_stage_report["selected_variant"]
    for rule in [first_stage, second_stage]:
        spec_name = (
            f"{rule['feature_name']}_{rule['feature_op']}_"
            f"{float(rule['feature_threshold']):.8g}"
        )
        out, _ = feat.apply_veto(
            out,
            str(rule["feature_name"]),
            str(rule["feature_op"]),
            float(rule["feature_threshold"]),
            spec_name,
        )
    return out


def load_parent_and_markets(mods: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v1 = mods["v1"]
    stop_mod = mods["stop_mod"]
    source = mods["source"]
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    parent_report = source.read_json(source.PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])
    return val_parent, oos_parent, train_market, eval_market


def build_selected_ledgers(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mods = modules()
    exposure = mods["exposure"]
    val_parent, oos_parent, train_market, eval_market = load_parent_and_markets(mods)
    val_pre = build_pre_exposure(val_parent, train_market, mods)
    oos_pre = build_pre_exposure(oos_parent, eval_market, mods)
    selected_val = exposure.apply_exposure_overlay(
        val_pre,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_oos = exposure.apply_exposure_overlay(
        oos_pre,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    return selected_val, selected_oos


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll7 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch regenerates the current two-stage veto path with max roll hold compressed from `8h` to `7h`, then searches exposure overlays. Selection is validation-primary with OOS as a safety gate.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Exposure spec: `{selected["exposure_spec"]}`
- Max roll hold: `{selected["roll7_max_hours"]}h`
- Research gate pass: `{selected["research_roll7_hold_compressed_gate_pass"]}`

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
    mods = modules()
    helper = mods["helper"]
    exposure = mods["exposure"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))["selected_variant"]
    val_parent, oos_parent, train_market, eval_market = load_parent_and_markets(mods)
    val_pre = build_pre_exposure(val_parent, train_market, mods)
    oos_pre = build_pre_exposure(oos_parent, eval_market, mods)

    rows: list[dict[str, Any]] = []
    for long_factor in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        for short_factor in [1.00, 1.05, 1.10, 1.12, 1.15, 1.18, 1.20]:
            for cap_notional in [4.20, 4.40, 4.60, 4.80, 5.00]:
                val_work = exposure.apply_exposure_overlay(val_pre, long_factor, short_factor, cap_notional)
                oos_work = exposure.apply_exposure_overlay(oos_pre, long_factor, short_factor, cap_notional)
                val_metrics = helper.metrics(val_work)
                oos_metrics = helper.metrics(oos_work)
                row = {
                    "exposure_spec": f"lf{long_factor:.3f}_sf{short_factor:.3f}_cap{cap_notional:.2f}",
                    "exposure_long_factor": float(long_factor),
                    "exposure_short_factor": float(short_factor),
                    "exposure_cap_notional": float(cap_notional),
                    "roll7_max_hours": MAX_ROLL_HOURS,
                    "roll7_long_tp_move": 0.0200,
                    "roll7_long_sl_move": 0.0300,
                    "roll7_short_tp_move": 0.0250,
                    "roll7_short_sl_move": 0.0385,
                    **flatten("validation", val_metrics),
                    **flatten("oos", oos_metrics),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) >= 100.0
                    and float(row["validation_mdd"]) >= -20.0
                    and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and float(row["validation_avg_hold_hours"]) < float(reference["validation_avg_hold_hours"])
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["validation_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                oos_gate = bool(
                    float(row["oos_pnl"]) >= 100.0
                    and float(row["oos_mdd"]) >= -20.0
                    and float(row["oos_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and float(row["oos_avg_hold_hours"]) < float(reference["oos_avg_hold_hours"])
                    and int(row["oos_overlap_count"]) == 0
                    and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["oos_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_roll7_hold_compressed_gate_pass"] = validation_gate
                row["oos_safety_gate_pass"] = oos_gate
                row["research_roll7_hold_compressed_gate_pass"] = validation_gate and oos_gate
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_roll7_hold_compressed_gate_pass",
            "validation_pnl",
            "oos_pnl",
            "validation_avg_hold_hours",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("no roll7 hold-compressed variants evaluated")
    selected = ranking.iloc[0].to_dict()
    selected_val, selected_oos = build_selected_ledgers(selected)
    status = (
        "RESEARCH_ROLL7_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASS"
        if bool(selected["research_roll7_hold_compressed_gate_pass"])
        else "NO_ROLL7_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE"
    )
    safe = str(selected["exposure_spec"]).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll7_two_stage_exposure_hold_compressed_ranking.csv"
    top20_path = OUT_DIR / "roll7_two_stage_exposure_hold_compressed_top20.csv"
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
        "selection_scope": "validation_primary_roll7_two_stage_veto_exposure_overlay_with_oos_safety_gate; fresh_holdout_required",
        "selection_rule": "regenerate 7h roll path with current two-stage veto rules; among research-gated exposure overlays, sort by validation_pnl, oos_pnl, validation_avg_hold",
        "reference_variant": reference,
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
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gate": bool(selected["validation_roll7_hold_compressed_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "gate": bool(selected["oos_safety_gate_pass"]),
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
