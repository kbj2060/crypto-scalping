#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402
from scripts import retrain_alpha7_1_01965_tp_sl_decontam_20260528 as base  # noqa: E402


MODEL_ID = "alpha7_submodel_01965_cleanfunding_v1_20260529"
RUN_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LIVE_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID
CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
TRAIN_CSV = CANDIDATE_DIR / "trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = CANDIDATE_DIR / "trade_candidates_2026_alpha6_current_tail111_exact.csv"
SOURCE_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"


def _copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_live_manifest(report: dict[str, Any]) -> None:
    ranking = list(report.get("ranking", []))
    selected = max(ranking, key=lambda r: float(r.get("val_cost3_pnl", -1e18))) if ranking else {}
    manifest = {
        "model_id": MODEL_ID,
        "display_name": "Alpha7.1-01965 clean funding v1",
        "promoted_at": "2026-05-29",
        "role": "research_baseline_for_alpha8_retest",
        "status": "research_only_not_live_wired",
        "source_candidate": "01965_random_alpha7_combo_primary_fallback",
        "lineage": {
            "base_live_model": "alpha7_submodel_01965_decontam_v2_tp_20260528",
            "clean_funding_remediation": "tmp/causal_regen_20260516/funding_clean_retrain_20260529",
            "candidate_csv": "alpha7_01965_cleanfunding_candidates_20260529",
            "decision_source": "alpha7_combo_primary_fallback",
            "primary": str(selected.get("primary_model", "")),
            "fallback": str(selected.get("fallback_model", "")),
            "current_regime": "clean_regime4_state24_sticky090_v2_20260517",
            "future_regime": "regime4_pred_tft_h12_nomdjd_all74_20260517_cleanfunding_rescore",
            "tp_sl_action_score": "alpha7_01965_cleanfunding_candidates_20260529",
        },
        "artifacts": {
            "primary_parent": str(LIVE_DIR / "primary_parent.pkl"),
            "primary_summary": str(LIVE_DIR / "primary_summary.json"),
            "fallback_parent": str(LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"),
            "fallback_summary": str(LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"),
            "tp_sl_action_score": str(LIVE_DIR / "tp_sl_path_edge_predictor.pkl"),
            "runtime_config": str(LIVE_DIR / "alpha7_01965_cleanfunding_runtime_config.json"),
        },
        "inputs": {"train_csv": str(TRAIN_CSV), "eval_csv": str(EVAL_CSV)},
        "validation_report": str(RUN_DIR / "report.json"),
        "selected_variant_by_validation": selected.get("variant"),
        "selected_validation_cost3": {
            "pnl": selected.get("val_cost3_pnl"),
            "mdd": selected.get("val_cost3_mdd"),
            "trades": selected.get("val_cost3_trades"),
            "wr": selected.get("val_cost3_wr"),
        },
        "reported_oos_cost3": {
            "pnl": selected.get("oos_cost3_pnl"),
            "mdd": selected.get("oos_cost3_mdd"),
            "trades": selected.get("oos_cost3_trades"),
            "wr": selected.get("oos_cost3_wr"),
        },
        "audit": {
            "legacy_regime_columns": int(report["frame_contract"]["train"]["forbidden_legacy_count"]),
            "current_regime_v2_columns": int(report["frame_contract"]["train"]["current_v2_count"]),
            "future_regime_pred_columns": int(report["frame_contract"]["train"]["future_regime_pred_count"]),
            "selection_uses_2026": False,
            "selection_note": "Selected by 2025Q4 validation only. This live-style directory is for Alpha8 retest input wiring; not live-wired.",
            "feature_contract_fail_fast": True,
            "legacy_contract_layer": False,
        },
    }
    (LIVE_DIR / "alpha7_live_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (LIVE_DIR / "alpha7_01965_cleanfunding_runtime_config.json").write_text(
        json.dumps({"model_id": MODEL_ID, "live_wired": False, "baseline_for_alpha8_retest": True}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (LIVE_DIR / "fallback_combo_summary.json").write_text(
        json.dumps({"model_id": MODEL_ID, "source_report": str(RUN_DIR / "report.json"), "ranking": report.get("ranking", [])}, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not TRAIN_CSV.exists() or not EVAL_CSV.exists():
        raise FileNotFoundError("clean funding candidate CSVs must be generated before retraining")
    base.TRAIN_CSV = TRAIN_CSV
    base.EVAL_CSV = EVAL_CSV
    base.OUT_DIR = RUN_DIR
    base.LIVE_DIR = SOURCE_LIVE_DIR
    base.PRIMARY_PARENT = SOURCE_LIVE_DIR / "primary_parent.pkl"
    base.FALLBACK_PARENT = SOURCE_LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
    rc = base.main()
    if int(rc) != 0:
        return int(rc)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN_DIR / "report.json").read_text(encoding="utf-8"))
    ranking = list(report.get("ranking", []))
    selected = max(ranking, key=lambda r: float(r.get("val_cost3_pnl", -1e18))) if ranking else {}
    primary_model = str(selected.get("primary_model", "primary_v2_tp"))
    fallback_model = str(selected.get("fallback_model", "fallback_v2_tp"))
    _copy_required(RUN_DIR / primary_model / "parent.pkl", LIVE_DIR / "primary_parent.pkl")
    _copy_required(RUN_DIR / primary_model / "summary.json", LIVE_DIR / "primary_summary.json")
    _copy_required(RUN_DIR / fallback_model / "parent.pkl", LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl")
    _copy_required(RUN_DIR / fallback_model / "summary.json", LIVE_DIR / "fallback_alpha43_no_legacy_summary.json")
    _copy_required(CANDIDATE_DIR / "tp_sl_path_edge_predictor.pkl", LIVE_DIR / "tp_sl_path_edge_predictor.pkl")

    for name in ("v31_state24_v2_plus_pred_runtime_report.json", "v31_state24_v2_plus_pred_runtime_audit.json"):
        src = SOURCE_LIVE_DIR / name
        if src.exists():
            shutil.copy2(src, LIVE_DIR / name)

    _write_live_manifest(report)
    print(json.dumps({"model_id": MODEL_ID, "run_dir": str(RUN_DIR), "live_dir": str(LIVE_DIR), "report": str(RUN_DIR / "report.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
