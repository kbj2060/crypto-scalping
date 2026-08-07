#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FINAL_GOVERNOR_WINDOW_BARS", "25000")
os.environ.setdefault("FINAL_GOVERNOR_LIVE_MODEL_BARS", "25000")

import trading_bot  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _active, _copy_rows, _decision_audit, _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/live_alpha5_fallback_audit_20260525"
REGIME4_PREFIX = "clean_regime4_2024_unsup_v1_"
PRED4_PREFIX = "regime4_pred_"


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _compare_cols(reference: pd.DataFrame, generated: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in reference.columns or col not in generated.columns:
            rows.append({"column": col, "status": "missing"})
            continue
        a = _num(reference, col).to_numpy(dtype=np.float64)
        b = _num(generated, col).to_numpy(dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            rows.append({"column": col, "status": "no_finite_overlap"})
            continue
        diff = np.abs(a[mask] - b[mask])
        rows.append(
            {
                "column": col,
                "status": "ok",
                "max_abs_diff": float(np.max(diff)),
                "mean_abs_diff": float(np.mean(diff)),
                "p99_abs_diff": float(np.quantile(diff, 0.99)),
                "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]) if np.std(a[mask]) > 1e-12 and np.std(b[mask]) > 1e-12 else None,
            }
        )
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    return {
        "column_count": len(cols),
        "ok_count": len(ok_rows),
        "max_abs_diff": float(max((r["max_abs_diff"] for r in ok_rows), default=0.0)),
        "mean_abs_diff_max": float(max((r["mean_abs_diff"] for r in ok_rows), default=0.0)),
        "worst": sorted(ok_rows, key=lambda r: float(r["max_abs_diff"]), reverse=True)[:10],
        "missing": [r["column"] for r in rows if r.get("status") != "ok"],
    }


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    fb = fallback.copy().reset_index(drop=True)
    mask = ~_active(out) & _active(fb)
    return _copy_rows(out, fb, mask)


def _scale_with_runtime(runtime: trading_bot.FinalGovernorRuntime, decisions: pd.DataFrame, rt: dict[str, Any] | None) -> pd.DataFrame:
    return runtime._scale_fully_learned_decisions_with_runtime(decisions, rt)


def main() -> int:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_eval = _read(DEFAULT_EVAL)

    runtime = trading_bot.FinalGovernorRuntime()
    runtime.window_bars = max(runtime.window_bars, len(raw_eval))
    generated = runtime._prepare_frame(raw_eval, m7_last=None, trend_signal=None)

    primary_result = runtime._fully_learned_decision_frame(generated)
    if primary_result is None:
        raise RuntimeError("primary fully learned decision frame unavailable")
    primary_dec, _primary_features = primary_result

    if runtime.fully_learned_fallback_policy_bundle is None:
        fallback_dec = primary_dec.iloc[0:0].copy()
        final_dec = primary_dec.copy()
    else:
        fallback_result = runtime._fully_learned_decision_frame(
            generated,
            bundle=runtime.fully_learned_fallback_policy_bundle,
        )
        if fallback_result is None:
            raise RuntimeError("fallback decision frame unavailable")
        fallback_dec, _fallback_features = fallback_result
        fallback_dec = _scale_with_runtime(runtime, fallback_dec, runtime.fully_learned_fallback_scale_runtime)
        final_dec = _combine_primary_fallback(primary_dec, fallback_dec)

    ref_parent = joblib.load(v31.DEFAULT_PARENT)
    fee = float(ref_parent["config"]["fee"])
    slip = float(ref_parent["config"]["slip"])
    parent_for_features = _parent_for_features(list(ref_parent["feature_cols"]))
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    metrics = {
        "1.0": _compact_costs(
            _metrics(raw_eval, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=final_dec, fee=fee, slip=slip)
        ),
        "1.5": _compact_costs(
            _metrics(raw_eval, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=final_dec, fee=fee * 1.5, slip=slip * 1.5)
        ),
        "2.0": _compact_costs(
            _metrics(raw_eval, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=final_dec, fee=fee * 2.0, slip=slip * 2.0)
        ),
        "3.0": _compact_costs(
            _metrics(raw_eval, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=final_dec, fee=fee * 3.0, slip=slip * 3.0)
        ),
    }
    months: dict[str, Any] = {}
    for name, start, end in (("jan", "2026-01-01", "2026-02-01"), ("feb", "2026-02-01", "2026-03-01")):
        mask = (raw_eval["timestamp"] >= pd.Timestamp(start)) & (raw_eval["timestamp"] < pd.Timestamp(end))
        months[name] = _compact_costs(
            _metrics(
                raw_eval.loc[mask].reset_index(drop=True),
                parent_for_features=parent_for_features,
                runner=noop_runner,
                runner_cfg=noop_cfg,
                dec=final_dec.loc[mask].reset_index(drop=True),
                fee=fee,
                slip=slip,
            )
        )

    regime_cols = [c for c in raw_eval.columns if c.startswith(REGIME4_PREFIX)]
    pred_cols = [c for c in raw_eval.columns if c.startswith(PRED4_PREFIX)]
    tp_cols = ["tp_sl_action_score"] if "tp_sl_action_score" in raw_eval.columns else []
    report = {
        "model_id": "live_alpha5_state24_sticky_fallback_alpha43_audit_20260525",
        "eval_csv": str(DEFAULT_EVAL),
        "rows": int(len(raw_eval)),
        "range": [str(raw_eval["timestamp"].iloc[0]), str(raw_eval["timestamp"].iloc[-1])],
        "runtime_paths": {
            "primary": runtime.fully_learned_policy_path,
            "fallback": runtime.fully_learned_fallback_policy_path if runtime.fully_learned_fallback_policy_bundle is not None else "OFF",
            "clean4": runtime.clean_regime4_sticky_path if runtime.clean_regime4_sticky_bundle is not None else "OFF",
            "regime4_pred": runtime.regime4_pred_tft_path if runtime.regime4_pred_tft_model is not None else "OFF",
            "tp_sl": runtime.fully_learned_tp_sl_score_path if runtime.fully_learned_tp_sl_score_bundle is not None else "OFF",
        },
        "traces": {
            "clean_regime4_sticky": dict(generated.attrs.get("clean_regime4_sticky_trace", {}) or {}),
            "regime4_pred_tft": dict(generated.attrs.get("regime4_pred_tft_trace", {}) or {}),
            "tp_sl_action_score": dict(generated.attrs.get("tp_sl_action_score_trace", {}) or {}),
        },
        "feature_parity": {
            "clean4": _compare_cols(raw_eval, generated, regime_cols),
            "regime4_pred": _compare_cols(raw_eval, generated, pred_cols),
            "tp_sl_action_score": _compare_cols(raw_eval, generated, tp_cols),
        },
        "decision_audit": {
            "primary": _decision_audit(raw_eval, primary_dec),
            "fallback": _decision_audit(raw_eval, fallback_dec) if len(fallback_dec) else {"active_rows": 0},
            "final": _decision_audit(raw_eval, final_dec),
            "fallback_used_rows": int((~_active(primary_dec) & _active(fallback_dec)).sum()) if len(fallback_dec) else 0,
        },
        "metrics": metrics,
        "months": months,
        "pass_checks": {
            "feature_contract_not_blocked": not bool(runtime.fully_learned_contract_blocked),
            "clean4_enabled": bool(dict(generated.attrs.get("clean_regime4_sticky_trace", {}) or {}).get("enabled", False)),
            "regime4_pred_enabled": bool(dict(generated.attrs.get("regime4_pred_tft_trace", {}) or {}).get("enabled", False)),
            "tp_sl_enabled": bool(dict(generated.attrs.get("tp_sl_action_score_trace", {}) or {}).get("enabled", False)),
            "oos_cost3_gt_100": float(metrics["1.0"]["cost3"]["pnl"]) > 100.0,
            "jan_cost3_positive": float(months["jan"]["cost3"]["pnl"]) > 0.0,
            "feb_cost3_positive": float(months["feb"]["cost3"]["pnl"]) > 0.0,
            "stress2_cost3_positive": float(metrics["2.0"]["cost3"]["pnl"]) > 0.0,
        },
    }
    final_dec.assign(timestamp=raw_eval["timestamp"].to_numpy()).to_csv(out_dir / "live_runtime_decisions_2026.csv", index=False)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": metrics["1.0"], "pass_checks": report["pass_checks"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
