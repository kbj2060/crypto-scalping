#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    _active,
    _combine_primary_fallback,
    _combo_metrics,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_01543_alpha7_primary_cash_fallback_20260527 import (  # noqa: E402
    _backtest,
    _json_default,
    _load_01543_config,
    _load_train_val_eval_frames,
)


MODEL_ID = "alpha7_01543_parent_existing_alpha43_fallback_20260527"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01543_parent_existing_alpha43_fallback_20260527"


def _counts(primary: pd.DataFrame, fallback: pd.DataFrame, combined: pd.DataFrame) -> dict[str, Any]:
    p_active = _active(primary).to_numpy(dtype=bool)
    f_active = _active(fallback).to_numpy(dtype=bool)
    c_active = _active(combined).to_numpy(dtype=bool)
    p_side = pd.to_numeric(primary["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    f_side = pd.to_numeric(fallback["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return {
        "rows": int(len(primary)),
        "primary_active_rows": int(p_active.sum()),
        "fallback_active_rows": int(f_active.sum()),
        "combined_active_rows": int(c_active.sum()),
        "fallback_fill_rows": int((~p_active & f_active).sum()),
        "overlap_active_rows": int((p_active & f_active).sum()),
        "overlap_same_side_rows": int((p_active & f_active & (p_side == f_side)).sum()),
        "overlap_opposite_side_rows": int((p_active & f_active & (p_side != f_side)).sum()),
    }


def _loop_backtests(
    frame: pd.DataFrame,
    stack: dict[str, Any],
    q: np.ndarray,
    cfg_01543: dict[str, Any],
    primary: pd.DataFrame,
    combined: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "01543_primary_only": _backtest(frame, primary, stack=stack, q=q, cfg_01543=cfg_01543),
        "01543_plus_existing_fallback": _backtest(frame, combined, stack=stack, q=q, cfg_01543=cfg_01543),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_01543 = _load_01543_config()
    train_df, val_df, eval_df = _load_train_val_eval_frames()

    primary_parent = joblib.load(PRIMARY_PARENT)
    fallback_parent = joblib.load(FALLBACK_PARENT)
    loop._assert_parent_contract(primary_parent, val_df, name="alpha7_primary")
    loop._assert_parent_contract(fallback_parent, val_df, name="alpha7_existing_fallback")

    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)

    current_primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    current_primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    p01543_val = loop._apply_decision_mods(current_primary_val, cfg_01543)
    p01543_eval = loop._apply_decision_mods(current_primary_eval, cfg_01543)

    current_combo_val = _combine_primary_fallback(current_primary_val, fallback_val)
    current_combo_eval = _combine_primary_fallback(current_primary_eval, fallback_eval)
    combo01543_val = _combine_primary_fallback(p01543_val, fallback_val)
    combo01543_eval = _combine_primary_fallback(p01543_eval, fallback_eval)

    stack = loop._load_stack()
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    summary = {
        "model_id": MODEL_ID,
        "design": (
            "Replace Alpha7 parent decisions with 01543_random_alpha7_primary decision mods, "
            "while keeping the existing alpha43 no-legacy cash-only fallback unchanged. No fallback retraining."
        ),
        "audit": {
            "selection_uses_2026": False,
            "oos_window": "2026 fixed OOS",
            "fallback_retrained": False,
            "fallback_unchanged": True,
            "compat_alias_added": False,
            "live_artifacts_modified": False,
        },
        "paths": {
            "primary_parent": str(PRIMARY_PARENT),
            "primary_summary": str(PRIMARY_SUMMARY),
            "fallback_parent": str(FALLBACK_PARENT),
            "fallback_summary": str(FALLBACK_SUMMARY),
        },
        "01543_config": cfg_01543,
        "runtime_native_combo_metrics": {
            "current_alpha7_parent_existing_fallback": {
                "val": _combo_metrics(val_df, current_combo_val),
                "oos": _combo_metrics(eval_df, current_combo_eval),
            },
            "01543_parent_only": {
                "val": _combo_metrics(val_df, p01543_val),
                "oos": _combo_metrics(eval_df, p01543_eval),
            },
            "01543_parent_existing_fallback": {
                "val": _combo_metrics(val_df, combo01543_val),
                "oos": _combo_metrics(eval_df, combo01543_eval),
            },
        },
        "loop_style_exit_guard_metrics": {
            "val": _loop_backtests(val_df, stack, val_q, cfg_01543, p01543_val, combo01543_val),
            "oos": _loop_backtests(eval_df, stack, eval_q, cfg_01543, p01543_eval, combo01543_eval),
        },
        "decision_counts": {
            "val": _counts(p01543_val, fallback_val, combo01543_val),
            "oos": _counts(p01543_eval, fallback_eval, combo01543_eval),
        },
    }

    rt = summary["runtime_native_combo_metrics"]
    current_oos = rt["current_alpha7_parent_existing_fallback"]["oos"]["cost3"]
    primary_oos = rt["01543_parent_only"]["oos"]["cost3"]
    combo_oos = rt["01543_parent_existing_fallback"]["oos"]["cost3"]
    summary["deltas_cost3_oos"] = {
        "01543_combo_minus_current_combo_pnl": float(combo_oos["pnl"] - current_oos["pnl"]),
        "01543_combo_minus_01543_parent_only_pnl": float(combo_oos["pnl"] - primary_oos["pnl"]),
        "01543_combo_minus_current_combo_trades": int(combo_oos["trades"] - current_oos["trades"]),
        "01543_combo_minus_01543_parent_only_trades": int(combo_oos["trades"] - primary_oos["trades"]),
    }

    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(out),
                "current_combo_oos_cost3": current_oos,
                "01543_parent_only_oos_cost3": primary_oos,
                "01543_parent_existing_fallback_oos_cost3": combo_oos,
                "deltas": summary["deltas_cost3_oos"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
