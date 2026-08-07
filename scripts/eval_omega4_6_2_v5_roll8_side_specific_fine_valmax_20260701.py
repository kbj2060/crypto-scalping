#!/usr/bin/env python3
"""8h side-specific fine bracket compression branch for Omega 4.6.2 v5."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll9_side_specific_fine_valmax_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_fine_valmax_20260701.md"
MAX_ROLL_HOURS = 8.0
REFERENCE_CAP = 4.20
EPS = 1.0e-12


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_roll8(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={col: col.replace("roll16_", "roll8_") for col in out.columns if col.startswith("roll16_")})
    out["reason"] = out["reason"].astype(str).str.replace("roll16_", "roll8_", regex=False)
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
    return normalize_roll8(
        source.split_roll_bracket(df, market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move)
    )


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll8 Side-Specific Fine Valmax - 2026-07-01

## Method

This branch keeps the v5 parent and side-specific fine bracket family, then compresses the roll contract from the 9h reference to 8h. Selection is validation-primary with OOS as a safety gate only.

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

- Bracket spec: `{selected["side_bracket_spec"]}`
- Exposure spec: `{selected["exposure_spec"]}`
- Segment governor: `{selected["segment_governor_spec"]}`
- Long TP/SL: `{selected["roll8_long_tp_move"]:.4f}` / `{selected["roll8_long_sl_move"]:.4f}`
- Short TP/SL: `{selected["roll8_short_tp_move"]:.4f}` / `{selected["roll8_short_sl_move"]:.4f}`
- Research gate pass: `{selected["research_roll8_gate_pass"]}`

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
    base = load_module("omega462_roll9_base_for_roll8", BASE_PATH)
    source = base.load_module("omega462_roll16_source_for_roll8", base.SOURCE_MODULE_PATH)
    v1 = source.load_module("omega462_loss_cluster_v1_for_roll8", source.V1_MODULE_PATH)
    segment_mod = source.load_module("omega462_segment_for_roll8", source.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll8", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference = source.read_json(REFERENCE_REPORT)["selected_variant"]
    parent_report = source.read_json(source.PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])

    val_active = val_parent[val_parent["notional"].astype(float) > EPS]
    oos_active = oos_parent[oos_parent["notional"].astype(float) > EPS]
    val_skipped = val_parent[val_parent["notional"].astype(float) <= EPS]
    oos_skipped = oos_parent[oos_parent["notional"].astype(float) <= EPS]
    val_long = val_active[val_active["side"].astype(int) > 0]
    val_short = val_active[val_active["side"].astype(int) < 0]
    oos_long = oos_active[oos_active["side"].astype(int) > 0]
    oos_short = oos_active[oos_active["side"].astype(int) < 0]

    bracket_specs = [
        ("fine8_fast", 0.0200, 0.0300, 0.0250, 0.0400),
        ("fine8_faster", 0.0175, 0.0300, 0.0250, 0.0350),
        ("fine8_balanced", 0.0225, 0.0350, 0.0300, 0.0400),
        ("fine8_short_room", 0.0200, 0.0350, 0.0400, 0.0400),
        ("fine8_oos", 0.0250, 0.0350, 0.0400, 0.0400),
    ]
    bracket_bases: list[tuple[str, pd.DataFrame, pd.DataFrame, tuple[float, float, float, float]]] = []
    for name, long_tp, long_sl, short_tp, short_sl in bracket_specs:
        val_base = base.combine_sides(
            split_side(source, segment_mod, val_long, train_market, long_tp, long_sl),
            split_side(source, segment_mod, val_short, train_market, short_tp, short_sl),
            val_skipped,
        )
        oos_base = base.combine_sides(
            split_side(source, segment_mod, oos_long, eval_market, long_tp, long_sl),
            split_side(source, segment_mod, oos_short, eval_market, short_tp, short_sl),
            oos_skipped,
        )
        bracket_bases.append((name, val_base, oos_base, (long_tp, long_sl, short_tp, short_sl)))

    exposure_specs = [
        segment_mod.ExposureSpec(f"lf{lf:.2f}_sf{sf:.2f}_cap{cap:.2f}", lf, sf, cap)
        for lf in [0.55, 0.60, 0.65, 0.70, 0.75]
        for sf in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        for cap in [3.40, 3.60, 3.80, 4.00, 4.20]
    ]
    governors = [
        segment_mod.SegmentGovernorSpec("none", 1.00, 1.00, 0.0),
        segment_mod.SegmentGovernorSpec("loss1_90_win8", 0.90, 1.00, 8.0),
        segment_mod.SegmentGovernorSpec("streak85_60_win8", 0.85, 0.60, 8.0),
    ]

    rows: list[dict[str, Any]] = []
    for bracket_name, val_base, oos_base, bracket in bracket_bases:
        long_tp, long_sl, short_tp, short_sl = bracket
        for exposure in exposure_specs:
            for governor in governors:
                val_work = segment_mod.apply_segment_exposure_governor(val_base, exposure, governor)
                oos_work = segment_mod.apply_segment_exposure_governor(oos_base, exposure, governor)
                row = {
                    "side_bracket_spec": bracket_name,
                    "exposure_spec": exposure.name,
                    "segment_governor_spec": governor.name,
                    "roll8_max_hours": MAX_ROLL_HOURS,
                    "roll8_long_tp_move": long_tp,
                    "roll8_long_sl_move": long_sl,
                    "roll8_short_tp_move": short_tp,
                    "roll8_short_sl_move": short_sl,
                    **{f"exposure_{key}": value for key, value in asdict(exposure).items()},
                    **{f"segment_governor_{key}": value for key, value in asdict(governor).items()},
                    **base.flatten("validation", stop_mod.metrics(val_work)),
                    **base.flatten("oos", stop_mod.metrics(oos_work)),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) >= 100.0
                    and float(row["validation_mdd"]) >= -20.0
                    and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and float(row["validation_avg_hold_hours"]) < float(reference["validation_avg_hold_hours"])
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
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
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_roll8_gate_pass"] = validation_gate
                row["oos_safety_gate_pass"] = oos_gate
                row["research_roll8_gate_pass"] = validation_gate and oos_gate
                row["cap_reference_distance_abs"] = abs(float(row["exposure_cap_notional"]) - REFERENCE_CAP)
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_roll8_gate_pass",
            "validation_pnl",
            "validation_avg_hold_hours",
            "cap_reference_distance_abs",
        ],
        ascending=[False, False, True, True],
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
    selected_val_base = next(
        val_base for name, val_base, _, _ in bracket_bases if name == selected["side_bracket_spec"]
    )
    selected_oos_base = next(
        oos_base for name, _, oos_base, _ in bracket_bases if name == selected["side_bracket_spec"]
    )
    selected_val = segment_mod.apply_segment_exposure_governor(
        selected_val_base, selected_exposure, selected_governor
    )
    selected_oos = segment_mod.apply_segment_exposure_governor(
        selected_oos_base, selected_exposure, selected_governor
    )
    status = (
        "RESEARCH_ROLL8_SIDE_SPECIFIC_FINE_PASS"
        if bool(selected["research_roll8_gate_pass"])
        else "NO_ROLL8_SIDE_SPECIFIC_FINE_PASSING_CANDIDATE"
    )
    safe = (
        f"{selected['side_bracket_spec']}__{selected['exposure_spec']}__"
        f"{selected['segment_governor_spec']}"
    ).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll8_side_specific_fine_valmax_ranking.csv"
    top20_path = OUT_DIR / "roll8_side_specific_fine_valmax_top20.csv"
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
        "selection_scope": "validation_primary_roll8_hold_compression_with_oos_safety_gate; fresh_holdout_required",
        "selection_rule": f"among research-gated roll8 variants, sort by validation_pnl, validation_avg_hold, cap distance to {REFERENCE_CAP:.2f}; OOS is not an ordering key",
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
    source.write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_bracket": selected["side_bracket_spec"],
                "selected_exposure": selected["exposure_spec"],
                "selected_segment_governor": selected["segment_governor_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gate": bool(selected["validation_roll8_gate_pass"]),
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
            default=source.json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
