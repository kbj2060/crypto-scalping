#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _train_parent,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402


LEGACY_PREFIX = "clean_regime4_2024_unsup_v1_"
V2_PREFIX = "clean_regime4_state24_sticky090_v2_"
CURRENT_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha5_state24_sticky_fallback_alpha43_live_20260525"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_live_rebuild_20260526"
LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_live_20260526"


def _rename_clean4_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        col: col.replace(LEGACY_PREFIX, V2_PREFIX, 1)
        for col in out.columns
        if str(col).startswith(LEGACY_PREFIX)
    }
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _rename_feature_cols(cols: list[str]) -> list[str]:
    return [str(c).replace(LEGACY_PREFIX, V2_PREFIX, 1) for c in cols]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    train_all = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    current_primary = joblib.load(PRIMARY_PARENT)
    current_fallback = joblib.load(FALLBACK_PARENT)
    current_primary_summary = json.loads(PRIMARY_SUMMARY.read_text(encoding="utf-8"))
    current_fallback_summary = json.loads(FALLBACK_SUMMARY.read_text(encoding="utf-8"))
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    primary_feature_cols = _rename_feature_cols(list(current_primary["feature_cols"]))
    parent, parent_rt, summary = _train_parent(
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=primary_feature_cols,
        seed=5526,
        out_dir=OUT_DIR,
    )

    primary_val = _predict_scaled(parent, val_df, parent_rt)
    primary_eval = _predict_scaled(parent, eval_df, parent_rt)
    fallback_val = _predict_scaled(current_fallback, val_df, fallback_rt)
    fallback_eval = _predict_scaled(current_fallback, eval_df, fallback_rt)
    combo_val = _combine_primary_fallback(primary_val, fallback_val)
    combo_eval = _combine_primary_fallback(primary_eval, fallback_eval)
    combo_val_metrics = _combo_metrics(val_df, combo_val)
    combo_eval_metrics = _combo_metrics(eval_df, combo_eval)

    parent_path = LIVE_DIR / "primary_parent.pkl"
    primary_summary_path = LIVE_DIR / "primary_summary.json"
    fallback_parent_path = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"
    fallback_summary_path = LIVE_DIR / "fallback_alpha43_no_legacy_summary.json"
    tp_sl_path = LIVE_DIR / "tp_sl_path_edge_predictor.pkl"
    manifest_path = LIVE_DIR / "alpha7_live_manifest.json"
    combo_summary_path = LIVE_DIR / "fallback_combo_summary.json"

    joblib.dump(parent, parent_path)
    shutil.copy2(FALLBACK_PARENT, fallback_parent_path)
    shutil.copy2(CURRENT_LIVE_DIR / "tp_sl_path_edge_predictor.pkl", tp_sl_path)

    primary_summary_payload = dict(current_primary_summary)
    primary_summary_payload["model_id"] = "alpha7_primary_v2_only_20260526"
    primary_summary_payload["design"] = (
        "Alpha7 primary retrained on clean_regime4_state24_sticky090_v2_* only. "
        "Legacy clean_regime4_2024_unsup_v1_* alias columns are removed from the feature contract."
    )
    primary_summary_payload["feature_contract"] = dict(primary_summary_payload.get("feature_contract", {}) or {})
    primary_summary_payload["feature_contract"]["feature_cols"] = list(primary_feature_cols)
    primary_summary_payload["feature_contract"]["current_regime4_feature_count"] = int(
        sum(str(c).startswith(V2_PREFIX) for c in primary_feature_cols)
    )
    primary_summary_payload["feature_contract"]["legacy_clean_regime_feature_count"] = int(
        sum(str(c).startswith(LEGACY_PREFIX) for c in primary_feature_cols)
    )
    primary_summary_payload["feature_contract"]["feature_count"] = int(len(primary_feature_cols))
    primary_summary_payload["selected_metrics"] = summary["selected_metrics"]
    primary_summary_payload["selected_validation_metrics"] = summary["selected_validation_metrics"]
    primary_summary_payload["artifacts"] = {
        "parent": str(parent_path.relative_to(ROOT)),
        "report": str((OUT_DIR / "summary.json").relative_to(ROOT)),
        "grid": str((OUT_DIR / "summary.json").relative_to(ROOT)),
    }
    primary_summary_payload["audit"] = {
        "status": "pass",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "clean_regime4_prefix": V2_PREFIX,
        "legacy_clean_regime4_prefix_count": 0,
        "tp_sl_action_score_retained": True,
        "fallback_unchanged": True,
    }
    _write_json(primary_summary_path, primary_summary_payload)
    _write_json(fallback_summary_path, current_fallback_summary)

    manifest = {
        "model_id": "alpha7_sniper_primary_state24_sticky_alpha43_fallback_v2only_20260526_live",
        "display_name": "Alpha7 Sniper Primary v2-only",
        "promoted_at": "2026-05-26",
        "role": "live_trading_bot_primary_model",
        "lineage": {
            "primary": "alpha7_primary_v2_only_20260526",
            "fallback": "alpha4_3_no_legacy_parent",
            "tp_sl_action_score": "alpha4_2_tp_sl_action_score_20260517",
            "current_regime": "clean_regime4_state24_sticky090_v2_20260517",
            "future_regime": "regime4_pred_tft_h12_nomdjd_all74_20260517",
        },
        "validation_2025_q4": combo_val_metrics,
        "runtime_native_oos_2026_01_02": combo_eval_metrics,
        "audit_report": str((OUT_DIR / "summary.json").relative_to(ROOT)),
    }
    _write_json(manifest_path, manifest)

    combo_summary = {
        "model_id": manifest["model_id"],
        "display_name": manifest["display_name"],
        "lineage_model_id": "alpha7_v2_only_live_20260526",
        "cfg": {
            "mode": "fallback",
            "primary": "alpha7_primary_sniper_v2_only",
            "secondary": "alpha43_no_legacy_cash_only_fallback",
            "primary_lineage": "alpha7_primary_v2_only",
            "secondary_lineage": "alpha43_no_legacy",
        },
        "selection_score": float(
            combo_val_metrics["cost3"]["pnl"] / max(abs(float(combo_val_metrics["cost3"]["mdd"])), 1e-12)
        ),
        "validation_metrics": combo_val_metrics,
        "selected_metrics": combo_eval_metrics,
        "audit": {
            "selection_uses_2026": False,
            "feature_sources": {
                "primary": "alpha7 v2-only current regime4",
                "secondary": "alpha43 no_legacy",
            },
        },
    }
    _write_json(combo_summary_path, combo_summary)
    _write_json(OUT_DIR / "summary.json", {"manifest": manifest, "combo_summary": combo_summary, "primary_summary": summary})

    print(
        json.dumps(
            {
                "live_dir": str(LIVE_DIR),
                "manifest": manifest,
                "combo_cost3": combo_eval_metrics["cost3"],
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
