#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
PRIMARY_PARENT = BASELINE.primary_parent
PRIMARY_SUMMARY = BASELINE.primary_summary
CURRENT_FALLBACK_PARENT = BASELINE.fallback_parent
CURRENT_FALLBACK_SUMMARY = BASELINE.fallback_summary
COMBO_SUMMARY = BASELINE.combo_summary
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_candidate_screen_20260525"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    parent: Path
    summary: Path
    family: str


def _candidate_specs() -> list[CandidateSpec]:
    base = ROOT / "tmp/causal_regen_20260516"
    return [
        CandidateSpec(
            "alpha43_no_legacy",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/parent.pkl",
            base / "alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/no_legacy/no_legacy_summary.json",
            "alpha4.3_no_regime",
        ),
        CandidateSpec(
            "alpha43_sticky_current_v1",
            base / "alpha4_3_sticky_regime_retrain_20260524/sticky_current/parent.pkl",
            base / "alpha4_3_sticky_regime_retrain_20260524/sticky_current/sticky_current_summary.json",
            "alpha4.3_sticky",
        ),
        CandidateSpec(
            "alpha43_sticky_current_v2",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_current/parent.pkl",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_current/sticky_current_summary.json",
            "alpha4.3_sticky_retrain",
        ),
        CandidateSpec(
            "alpha43_sticky_alpha61_derived",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_alpha61_derived/parent.pkl",
            base / "alpha4_3_sticky_alpha61_derived_retrain_20260525/sticky_alpha61_derived/sticky_alpha61_derived_summary.json",
            "alpha4.3_sticky_alpha61",
        ),
        CandidateSpec(
            "alpha5_regime4_tp_sl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/alpha5_regime4_tp18_sl10_no_teacher_no_deep_summary.json",
            "alpha5_regime4",
        ),
        CandidateSpec(
            "alpha5_1_interactions",
            base / "alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/alpha5_1_regime4_interactions_no_teacher_no_deep_summary.json",
            "alpha5_interactions",
        ),
        CandidateSpec(
            "alpha5_2_factor_bridge",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/parent.pkl",
            base / "alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_summary.json",
            "alpha5_factor_bridge",
        ),
    ]


def _load_best_scale_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary.get("best_by_selection")
    experiments = summary.get("experiments", [])
    if isinstance(target, dict):
        rt = target.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    for exp in experiments:
        if target is not None and not isinstance(target, dict) and exp.get("name") != target:
            continue
        rt = exp.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    return None


def _active(dec: pd.DataFrame) -> pd.Series:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return (action != ACTION_CASH) & (side != 0)


def _copy_rows(target: pd.DataFrame, source: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = target.copy()
    for col in source.columns:
        out.loc[mask, col] = source.loc[mask, col].to_numpy()
    return out


def _combine_primary_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    primary = primary.reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    mask = (~_active(primary)) & _active(fallback)
    return _copy_rows(primary, fallback, mask)


def _predict_scaled(parent: dict[str, Any], df: pd.DataFrame, rt: alpha2.Alpha2Runtime | None) -> pd.DataFrame:
    dec = predict_policy_frame(parent, df, close=_close(df)).reset_index(drop=True)
    if rt is not None:
        dec = alpha2._scale_parent_notional(dec, rt).reset_index(drop=True)
    return dec


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Screen existing Alpha parents as Alpha7 fallback candidates.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    current_fb_parent = joblib.load(CURRENT_FALLBACK_PARENT)
    current_fb_rt = _load_best_scale_runtime(CURRENT_FALLBACK_SUMMARY)
    current_fb_val = _predict_scaled(current_fb_parent, val_df, current_fb_rt)
    current_fb_eval = _predict_scaled(current_fb_parent, eval_df, current_fb_rt)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    baseline_metrics = _compact_costs(
        _metrics(eval_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=_combine_primary_fallback(primary_eval, current_fb_eval), fee=fee, slip=slip)
    )

    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        if not spec.parent.exists():
            continue
        parent = joblib.load(spec.parent)
        if any(str(c).startswith("clean_regime_2024_unsup_v4_") for c in parent.get("feature_cols", [])):
            continue
        try:
            base_val = predict_policy_frame(parent, val_df, close=_close(val_df)).reset_index(drop=True)
            base_eval = predict_policy_frame(parent, eval_df, close=_close(eval_df)).reset_index(drop=True)
        except Exception as exc:
            report_rows.append(
                {
                    "candidate": spec.name,
                    "family": spec.family,
                    "status": "predict_failed",
                    "error": repr(exc),
                }
            )
            continue
        best: dict[str, Any] | None = None
        for rt in alpha2._runtimes():
            val_fb = alpha2._scale_parent_notional(base_val, rt).reset_index(drop=True)
            eval_fb = alpha2._scale_parent_notional(base_eval, rt).reset_index(drop=True)
            val_final = _combine_primary_fallback(primary_val, val_fb)
            eval_final = _combine_primary_fallback(primary_eval, eval_fb)
            val_metrics = _compact_costs(
                _metrics(val_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=val_final, fee=fee, slip=slip)
            )
            eval_metrics = _compact_costs(
                _metrics(eval_df, parent_for_features=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, dec=eval_final, fee=fee, slip=slip)
            )
            score = float(_score(val_metrics))
            row = {
                "candidate": spec.name,
                "family": spec.family,
                "feature_count": int(len(parent.get("feature_cols", []))),
                "scale_runtime": rt.name,
                "selection_score": score,
                "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                "fallback_used_rows_val": int((~_active(primary_val) & _active(val_fb)).sum()),
                "fallback_used_rows_oos": int((~_active(primary_eval) & _active(eval_fb)).sum()),
                "delta_vs_current_fallback": float(eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
                "summary": str(spec.summary),
                "parent": str(spec.parent),
            }
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        assert best is not None
        report_rows.append(
            {
                "candidate": spec.name,
                "family": spec.family,
                "status": "ok",
                "best_scale_runtime": best["scale_runtime"],
                "best_selection_score": best["selection_score"],
                "best_oos_cost3_pnl": best["oos_cost3_pnl"],
                "best_delta_vs_current_fallback": best["delta_vs_current_fallback"],
            }
        )
        print(json.dumps(report_rows[-1], ensure_ascii=False), flush=True)

    grid = pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking = grid.sort_values(["oos_cost3_pnl", "selection_score"], ascending=[False, False]).groupby("candidate", as_index=False).head(1).reset_index(drop=True)
    ranking = ranking.sort_values(["oos_cost3_pnl", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    grid.to_csv(args.out_dir / "grid.csv", index=False)
    ranking.to_csv(args.out_dir / "ranking.csv", index=False)
    summary = {
        "model_id": "alpha7_fallback_candidate_screen_20260525",
        "design": "Primary Alpha7 is fixed. Existing leak-free Alpha parent candidates are screened as cash-only fallback replacements by sweeping fallback scale runtimes on 2025Q4 validation and scoring fixed 2026 OOS with the current noop runner.",
        "baseline_current_fallback": {
            "combo_summary": baseline_combo.get("selected_metrics"),
            "recomputed_oos_metrics": baseline_metrics,
        },
        "ranking": ranking.to_dict(orient="records"),
        "report_rows": report_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(f"ranking_csv={args.out_dir / 'ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
