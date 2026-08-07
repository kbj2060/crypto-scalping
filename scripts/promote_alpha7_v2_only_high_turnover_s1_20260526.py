#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_TRAIN_CSV,
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402


CANDIDATE_NAME = "t0015_c015_h030_s6"
CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_high_turnover_rebuild_20260526" / CANDIDATE_NAME
BASELINE_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_live_20260526"
CURRENT_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha5_state24_sticky_fallback_alpha43_live_20260525"
LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_high_turnover_s1_live_20260526"

MODEL_ID = "alpha7_sniper_primary_state24_sticky_alpha43_fallback_v2only_highturnover_s1_20260526_live"
DISPLAY_NAME = "Alpha7 Sniper Primary v2-only high-turnover s1"
PRIMARY_MODEL_ID = "alpha7_primary_v2_only_high_turnover_s1_20260526"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    parent = joblib.load(CANDIDATE_DIR / "parent.pkl")
    candidate_summary = json.loads((CANDIDATE_DIR / "summary.json").read_text(encoding="utf-8"))
    baseline_combo_summary = json.loads((BASELINE_LIVE_DIR / "fallback_combo_summary.json").read_text(encoding="utf-8"))
    current_primary_summary = json.loads((BASELINE_LIVE_DIR / "primary_summary.json").read_text(encoding="utf-8"))
    current_fallback_summary = json.loads((FALLBACK_SUMMARY).read_text(encoding="utf-8"))

    best_experiment = next(
        exp for exp in candidate_summary["experiments"] if exp["name"] == candidate_summary["best_by_selection"]
    )
    parent_rt = None
    if best_experiment["name"] == "parent_direct_scaled_no_teacher":
        rt = best_experiment["selected_parent_scale_runtime"]
        parent_rt = alpha2.Alpha2Runtime(
            name=str(rt["name"]),
            confidence=float(rt["confidence"]),
            parent_notional_scale=float(rt["parent_notional_scale"]),
            max_notional=float(rt["max_notional"]),
        )

    train_all = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    eval_df = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    fallback = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    primary_val = _predict_scaled(parent, val_df, parent_rt)
    primary_eval = _predict_scaled(parent, eval_df, parent_rt)
    fallback_val = _predict_scaled(fallback, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback, eval_df, fallback_rt)
    combo_val = _combo_metrics(val_df, _combine_primary_fallback(primary_val, fallback_val))
    combo_eval = _combo_metrics(eval_df, _combine_primary_fallback(primary_eval, fallback_eval))

    shutil.copy2(CANDIDATE_DIR / "parent.pkl", LIVE_DIR / "primary_parent.pkl")
    shutil.copy2(FALLBACK_PARENT, LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl")
    shutil.copy2(FALLBACK_SUMMARY, LIVE_DIR / "fallback_alpha43_no_legacy_summary.json")
    shutil.copy2(CURRENT_LIVE_DIR / "tp_sl_path_edge_predictor.pkl", LIVE_DIR / "tp_sl_path_edge_predictor.pkl")

    primary_summary = dict(current_primary_summary)
    primary_summary["model_id"] = PRIMARY_MODEL_ID
    primary_summary["design"] = (
        "Alpha7 primary retrained on clean_regime4_state24_sticky090_v2_* only with mildly higher turnover. "
        "turnover_bonus is increased to 0.0015, cash_score is reduced to 0.015, and hold_penalty is reduced to 0.03."
    )
    primary_summary["selected_metrics"] = candidate_summary["selected_metrics"]
    primary_summary["selected_validation_metrics"] = candidate_summary["selected_validation_metrics"]
    primary_summary["artifacts"] = {
        "parent": str((LIVE_DIR / "primary_parent.pkl").relative_to(ROOT)),
        "candidate_summary": str((CANDIDATE_DIR / "summary.json").relative_to(ROOT)),
    }
    primary_summary["audit"] = {
        "status": "pass",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026-01-01..2026-02-28",
        "candidate_name": CANDIDATE_NAME,
        "turnover_bonus": 0.0015,
        "cash_score": 0.015,
        "hold_penalty": 0.03,
        "stride": 6,
        "clean_regime4_prefix": "clean_regime4_state24_sticky090_v2_",
        "legacy_clean_regime4_prefix_count": 0,
        "tp_sl_action_score_retained": True,
        "fallback_unchanged": True,
    }
    _write_json(LIVE_DIR / "primary_summary.json", primary_summary)

    manifest = {
        "model_id": MODEL_ID,
        "display_name": DISPLAY_NAME,
        "promoted_at": "2026-05-26",
        "role": "live_trading_bot_primary_model_candidate",
        "lineage": {
            "primary": PRIMARY_MODEL_ID,
            "fallback": "alpha4_3_no_legacy_parent",
            "tp_sl_action_score": "alpha4_2_tp_sl_action_score_20260517",
            "current_regime": "clean_regime4_state24_sticky090_v2_20260517",
            "future_regime": "regime4_pred_tft_h12_nomdjd_all74_20260517",
        },
        "selected_candidate": {
            "name": CANDIDATE_NAME,
            "turnover_bonus": 0.0015,
            "cash_score": 0.015,
            "hold_penalty": 0.03,
            "stride": 6,
            "val_cost3_pnl": float(combo_val["cost3"]["pnl"]),
            "val_cost3_mdd": float(combo_val["cost3"]["mdd"]),
            "val_cost3_trades": int(combo_val["cost3"]["trades"]),
            "oos_cost3_pnl": float(combo_eval["cost3"]["pnl"]),
            "oos_cost3_mdd": float(combo_eval["cost3"]["mdd"]),
            "oos_cost3_trades": int(combo_eval["cost3"]["trades"]),
            "oos_cost3_wr": float(combo_eval["cost3"]["wr"]),
            "delta_oos_trades": int(combo_eval["cost3"]["trades"] - baseline_combo_summary["selected_metrics"]["cost3"]["trades"]),
            "delta_oos_pnl": float(combo_eval["cost3"]["pnl"] - baseline_combo_summary["selected_metrics"]["cost3"]["pnl"]),
        },
        "baseline_reference": {
            "validation_cost3_trades": int(baseline_combo_summary["validation_metrics"]["cost3"]["trades"]),
            "oos_cost3_trades": int(baseline_combo_summary["selected_metrics"]["cost3"]["trades"]),
            "oos_cost3_pnl": float(baseline_combo_summary["selected_metrics"]["cost3"]["pnl"]),
        },
        "validation_2025_q4": {"cost3": combo_val["cost3"]},
        "runtime_native_oos_2026_01_02": {"cost3": combo_eval["cost3"]},
        "audit_report": str((CANDIDATE_DIR / "summary.json").relative_to(ROOT)),
    }
    _write_json(LIVE_DIR / "alpha7_live_manifest.json", manifest)

    combo_summary = {
        "model_id": MODEL_ID,
        "display_name": DISPLAY_NAME,
        "lineage_model_id": "alpha7_v2_only_high_turnover_s1_live_20260526",
        "cfg": {
            "mode": "fallback",
            "primary": "alpha7_primary_sniper_v2_only_high_turnover_s1",
            "secondary": "alpha43_no_legacy_cash_only_fallback",
            "primary_lineage": "alpha7_primary_v2_only_high_turnover_s1",
            "secondary_lineage": "alpha43_no_legacy",
        },
        "selection_score": float(
            combo_val["cost3"]["pnl"] / max(abs(float(combo_val["cost3"]["mdd"])), 1e-12)
            + 0.03 * (int(combo_val["cost3"]["trades"]) - int(baseline_combo_summary["validation_metrics"]["cost3"]["trades"]))
        ),
        "selected_metrics": {"cost3": combo_eval["cost3"]},
        "validation_metrics": {"cost3": combo_val["cost3"]},
        "audit": {
            "selection_uses_2026": False,
            "feature_sources": {
                "primary": "alpha7 v2-only current regime4 high-turnover s1",
                "secondary": "alpha43 no_legacy",
            },
        },
    }
    _write_json(LIVE_DIR / "fallback_combo_summary.json", combo_summary)

    print(
        json.dumps(
            {
                "live_dir": str(LIVE_DIR),
                "model_id": MODEL_ID,
                "combo_val_cost3": combo_val["cost3"],
                "combo_oos_cost3": combo_eval["cost3"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
