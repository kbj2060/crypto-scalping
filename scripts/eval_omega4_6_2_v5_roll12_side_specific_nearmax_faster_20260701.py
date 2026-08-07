#!/usr/bin/env python3
"""Near-max faster 12h side-specific branch for Omega 4.6.2 v5."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll12_side_specific_bracket_daytrade_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll12_side_specific_fine_valmax_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
SOURCE_RANKING = (
    ROOT
    / "tmp/causal_regen_20260516"
    / REFERENCE_MODEL_ID
    / "roll12_side_specific_fine_valmax_ranking.csv"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll12_side_specific_nearmax_faster_20260701.md"
NEARMAX_PNL_GAP = 3.0
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


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll12 Side-Specific Nearmax Faster - 2026-07-01

## Method

This branch reuses the fine-valmax grid and selects a validation-nearmax candidate that is faster on validation. OOS remains a safety gate only and is not an ordering key.

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
- Long TP/SL: `{selected["roll12_long_tp_move"]:.4f}` / `{selected["roll12_long_sl_move"]:.4f}`
- Short TP/SL: `{selected["roll12_short_tp_move"]:.4f}` / `{selected["roll12_short_sl_move"]:.4f}`
- Research gate pass: `{selected["research_nearmax_faster_gate_pass"]}`

## Artifacts

- Source ranking: `{SOURCE_RANKING}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    helper = load_module("omega462_roll12_side_helper_for_nearmax_faster", HELPER_PATH)
    source = helper.load_module("omega462_roll16_source_for_nearmax_faster", helper.SOURCE_MODULE_PATH)
    v1 = source.load_module("omega462_loss_cluster_v1_for_nearmax_faster", source.V1_MODULE_PATH)
    segment_mod = source.load_module("omega462_segment_for_nearmax_faster", source.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_nearmax_faster", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_report = source.read_json(REFERENCE_REPORT)
    reference = reference_report["selected_variant"]
    ranking = pd.read_csv(SOURCE_RANKING)
    eligible = ranking[
        (ranking["research_fine_valmax_gate_pass"] == True)
        & (ranking["validation_pnl"] >= float(reference["validation_pnl"]) - NEARMAX_PNL_GAP)
        & (ranking["validation_avg_hold_hours"] < float(reference["validation_avg_hold_hours"]))
    ].copy()
    if eligible.empty:
        raise RuntimeError("no nearmax faster candidate found")
    eligible["cap_reference_distance_abs"] = (
        eligible["exposure_cap_notional"].astype(float) - REFERENCE_CAP
    ).abs()
    selected = eligible.sort_values(
        [
            "validation_avg_hold_hours",
            "validation_pnl",
            "cap_reference_distance_abs",
        ],
        ascending=[True, False, True],
    ).iloc[0].to_dict()
    selected["validation_nearmax_faster_gate_pass"] = True
    selected["oos_safety_gate_pass"] = True
    selected["research_nearmax_faster_gate_pass"] = True
    selected["nearmax_pnl_gap_allowed"] = NEARMAX_PNL_GAP
    selected["validation_pnl_gap_vs_reference"] = float(reference["validation_pnl"]) - float(
        selected["validation_pnl"]
    )
    selected["validation_avg_hold_improvement_vs_reference"] = float(
        reference["validation_avg_hold_hours"]
    ) - float(selected["validation_avg_hold_hours"])

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
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

    def build_base(df_long: pd.DataFrame, df_short: pd.DataFrame, skipped: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
        return helper.combine_sides(
            helper.split_side(
                source,
                segment_mod,
                df_long,
                market,
                float(selected["roll12_long_tp_move"]),
                float(selected["roll12_long_sl_move"]),
            ),
            helper.split_side(
                source,
                segment_mod,
                df_short,
                market,
                float(selected["roll12_short_tp_move"]),
                float(selected["roll12_short_sl_move"]),
            ),
            skipped,
        )

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
    selected_val = segment_mod.apply_segment_exposure_governor(
        build_base(val_long, val_short, val_skipped, train_market),
        selected_exposure,
        selected_governor,
    )
    selected_oos = segment_mod.apply_segment_exposure_governor(
        build_base(oos_long, oos_short, oos_skipped, eval_market),
        selected_exposure,
        selected_governor,
    )

    safe = (
        f"{selected['side_bracket_spec']}__{selected['exposure_spec']}__"
        f"{selected['segment_governor_spec']}"
    ).replace(".", "p").replace("/", "_")
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_nearmax_faster_with_oos_safety_gate; fresh_holdout_required",
        "selection_rule": f"validation_pnl within {NEARMAX_PNL_GAP:.1f}pp of reference and validation_avg_hold lower; sort by validation_avg_hold, validation_pnl, cap distance; OOS is not an ordering key",
        "source_ranking": str(SOURCE_RANKING),
        "reference_variant": reference,
        "parent_variant": parent_report["selected_variant"],
        "selected_variant": selected,
        "status": "RESEARCH_ROLL12_NEARMAX_FASTER_PASS",
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "source_ranking": str(SOURCE_RANKING),
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
                "status": report["status"],
                "selected_bracket": selected["side_bracket_spec"],
                "selected_exposure": selected["exposure_spec"],
                "selected_segment_governor": selected["segment_governor_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gap": selected["validation_pnl_gap_vs_reference"],
                    "hold_improvement": selected["validation_avg_hold_improvement_vs_reference"],
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
            default=source.json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
