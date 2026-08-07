#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _json_default  # noqa: E402
from scripts import retrain_alpha7_1_01965_tp_sl_decontam_20260528 as base  # noqa: E402


MODEL_ID = "alpha7_01965_cleanfunding_teacher_parent_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_cleanfunding_v1_20260529"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
TEACHER_COLS = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


def _select_validation_baseline(report: dict[str, Any]) -> dict[str, Any]:
    rows = list(report.get("ranking", []))
    if not rows:
        raise RuntimeError("clean Alpha7 report has empty ranking")
    return max(rows, key=lambda r: float(r.get("val_cost3_pnl", -1e18)))


def _dedupe(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = base._read(TRAIN_CSV)
    eval_df = base._read(EVAL_CSV)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    base._assert_clean_frame(train_all, name="train")
    base._assert_clean_frame(eval_df, name="eval")

    missing_teacher = [c for c in TEACHER_COLS if c not in train_all.columns or c not in eval_df.columns]
    if missing_teacher:
        raise RuntimeError(f"missing teacher columns: {missing_teacher}")

    clean_report_path = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529/report.json"
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
    selected = _select_validation_baseline(clean_report)

    primary_key = str(selected["primary_model"])
    fallback_key = str(selected["fallback_model"])
    primary_path = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529" / primary_key / "parent.pkl"
    fallback_path = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529" / fallback_key / "parent.pkl"
    if not primary_path.exists() or not fallback_path.exists():
        raise FileNotFoundError(f"missing selected baseline artifact: {primary_path} / {fallback_path}")

    primary_base = joblib.load(primary_path)
    fallback_base = joblib.load(fallback_path)
    primary_cols = list(primary_base["feature_cols"])
    teacher_primary_cols = _dedupe(primary_cols + TEACHER_COLS)
    base._assert_feature_cols(train_all, teacher_primary_cols, name="teacher_primary")
    base._assert_feature_cols(eval_df, teacher_primary_cols, name="teacher_primary_eval")

    teacher_primary, teacher_rt, teacher_summary = base._load_or_train(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=teacher_primary_cols,
        seed=5292901,
        out_dir=OUT_DIR / f"{primary_key}_teacher",
    )

    fallback_rt = None
    fallback_summary_path = ROOT / "tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529" / fallback_key / "summary.json"
    fallback_summary = json.loads(fallback_summary_path.read_text(encoding="utf-8")) if fallback_summary_path.exists() else {}

    baseline_val = base._predict_metrics(
        primary_parent=primary_base,
        primary_rt=None,
        fallback_parent=fallback_base,
        fallback_rt=fallback_rt,
        df=val_df,
    )
    baseline_oos = base._predict_metrics(
        primary_parent=primary_base,
        primary_rt=None,
        fallback_parent=fallback_base,
        fallback_rt=fallback_rt,
        df=eval_df,
    )
    teacher_val = base._predict_metrics(
        primary_parent=teacher_primary,
        primary_rt=teacher_rt,
        fallback_parent=fallback_base,
        fallback_rt=fallback_rt,
        df=val_df,
    )
    teacher_oos = base._predict_metrics(
        primary_parent=teacher_primary,
        primary_rt=teacher_rt,
        fallback_parent=fallback_base,
        fallback_rt=fallback_rt,
        df=eval_df,
    )

    def row(name: str, val: dict[str, Any], oos: dict[str, Any]) -> dict[str, Any]:
        vc3 = val["combo_costs"]["cost3"]
        oc3 = oos["combo_costs"]["cost3"]
        return {
            "variant": name,
            "primary_model": primary_key if name == "baseline_selected" else f"{primary_key}_teacher",
            "fallback_model": fallback_key,
            "val_cost3_pnl": float(vc3["pnl"]),
            "val_cost3_mdd": float(vc3["mdd"]),
            "val_cost3_trades": int(vc3["trades"]),
            "val_cost3_wr": float(vc3["wr"]),
            "oos_cost3_pnl": float(oc3["pnl"]),
            "oos_cost3_mdd": float(oc3["mdd"]),
            "oos_cost3_trades": int(oc3["trades"]),
            "oos_cost3_wr": float(oc3["wr"]),
            "oos_combo_active": int(oos["combo_active"]),
            "oos_primary_active": int(oos["primary_active"]),
            "oos_fallback_active": int(oos["fallback_active"]),
        }

    rows = [
        row("baseline_selected", baseline_val, baseline_oos),
        row("teacher_primary_only", teacher_val, teacher_oos),
    ]
    ranking = pd.DataFrame(rows).sort_values(["val_cost3_pnl", "oos_cost3_pnl"], ascending=[False, False])
    ranking_path = OUT_DIR / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "scope": "Add teacher_* features to the validation-selected Clean Alpha7 01965 primary parent only; fallback remains fixed.",
        "selection_uses_2026": False,
        "inputs": {"train_csv": str(TRAIN_CSV), "eval_csv": str(EVAL_CSV), "baseline_report": str(clean_report_path)},
        "selected_baseline": selected,
        "teacher_cols_added": TEACHER_COLS,
        "baseline_primary_contract": base._contract_report(list(primary_base["feature_cols"])),
        "teacher_primary_contract": base._contract_report(list(teacher_primary["feature_cols"])),
        "fallback_contract": base._contract_report(list(fallback_base["feature_cols"])),
        "teacher_primary_summary": teacher_summary,
        "fallback_summary": fallback_summary,
        "ranking": rows,
        "validation_selected_variant": str(ranking.iloc[0]["variant"]),
        "artifacts": {
            "teacher_primary": str(OUT_DIR / f"{primary_key}_teacher" / "parent.pkl"),
            "teacher_primary_summary": str(OUT_DIR / f"{primary_key}_teacher" / "summary.json"),
            "ranking": str(ranking_path),
        },
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "ranking": str(ranking_path), "selected": ranking.iloc[0].to_dict()}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
