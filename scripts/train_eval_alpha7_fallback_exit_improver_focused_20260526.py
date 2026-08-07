#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import _compact_costs, _metrics  # noqa: E402
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _active,
    _combine_primary_fallback,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
LIVE_DIR = BASELINE.live_dir
FALLBACK_PARENT = BASELINE.fallback_parent
FALLBACK_SUMMARY = BASELINE.fallback_summary
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_exit_improver_focused_20260526"


def _month_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").astype(str)


def _robust_score(full_cost3: dict[str, Any], month_rows: list[dict[str, Any]]) -> float:
    full_pnl = float(full_cost3["pnl"])
    full_mdd = abs(float(full_cost3["mdd"]))
    full_calmar = full_pnl / max(full_mdd, 1e-12)
    month_pnls = [float(r["cost3_pnl"]) for r in month_rows]
    month_mdds = [abs(float(r["cost3_mdd"])) for r in month_rows]
    month_calmars = [p / max(m, 1e-12) for p, m in zip(month_pnls, month_mdds)]
    min_month_pnl = min(month_pnls) if month_pnls else 0.0
    min_month_calmar = min(month_calmars) if month_calmars else 0.0
    neg_months = sum(1 for p in month_pnls if p < 0.0)
    return float(full_calmar + 0.50 * min_month_calmar + 0.05 * min_month_pnl - 0.25 * neg_months)


def _apply_exit_overlay(dec: pd.DataFrame, mode: str, *, tp_scale: float, conf_thr: float, qual_thr: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    conf = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    qual = pd.to_numeric(out["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    shrink = np.zeros(len(out), dtype=bool)
    if mode == "baseline":
        shrink[:] = False
    elif mode == "global_tp":
        shrink = active
    elif mode == "low_conf_tp":
        shrink = active & (conf < float(conf_thr))
    elif mode == "low_qual_tp":
        shrink = active & (qual < float(qual_thr))
    elif mode == "low_conf_or_qual_tp":
        shrink = active & ((conf < float(conf_thr)) | (qual < float(qual_thr)))
    elif mode == "low_conf_and_qual_tp":
        shrink = active & ((conf < float(conf_thr)) & (qual < float(qual_thr)))
    elif mode == "high_conf_keep_else_tp":
        shrink = active & (~((conf >= float(conf_thr)) & (qual >= float(qual_thr))))
    else:
        raise ValueError(f"unknown mode={mode}")
    if np.any(shrink):
        out.loc[shrink, "take_profit"] = pd.to_numeric(out.loc[shrink, "take_profit"], errors="coerce").clip(lower=1e-4) * float(tp_scale)
    return out


def _eval_combo(frame: pd.DataFrame, primary_dec: pd.DataFrame, fallback_dec: pd.DataFrame, *, ref_parent: dict[str, Any], runner: dict[str, Any], runner_cfg: Any, fee: float, slip: float) -> dict[str, Any]:
    final_dec = _combine_primary_fallback(primary_dec, fallback_dec)
    return _compact_costs(
        _metrics(
            frame,
            parent_for_features=ref_parent,
            runner=runner,
            runner_cfg=runner_cfg,
            dec=final_dec,
            fee=fee,
            slip=slip,
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Focused exit improver search for Alpha7 current fallback.")
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
    fallback_parent = joblib.load(FALLBACK_PARENT)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_combo = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    baseline_oos = _eval_combo(eval_df, primary_eval, fallback_eval, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)

    modes = [
        "baseline",
        "global_tp",
        "low_conf_or_qual_tp",
        "high_conf_keep_else_tp",
    ]
    tp_scales = [0.76, 0.80, 0.84]
    conf_thrs = [0.70, 0.74]
    qual_thrs = [0.045, 0.055]

    val_months = sorted(_month_key(val_df["timestamp"]).dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    best_robust: dict[str, Any] | None = None
    best_oos: dict[str, Any] | None = None
    for mode in modes:
        for tp_scale in tp_scales:
            conf_grid = [0.0] if mode in {"baseline", "global_tp", "low_qual_tp"} else conf_thrs
            qual_grid = [0.0] if mode in {"baseline", "global_tp", "low_conf_tp"} else qual_thrs
            for conf_thr in conf_grid:
                for qual_thr in qual_grid:
                    tuned_val = _apply_exit_overlay(fallback_val, mode, tp_scale=tp_scale, conf_thr=conf_thr, qual_thr=qual_thr)
                    tuned_eval = _apply_exit_overlay(fallback_eval, mode, tp_scale=tp_scale, conf_thr=conf_thr, qual_thr=qual_thr)
                    val_metrics = _eval_combo(val_df, primary_val, tuned_val, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
                    month_rows = []
                    month_col = _month_key(val_df["timestamp"])
                    for month in val_months:
                        mask = month_col == month
                        m_frame = val_df.loc[mask].reset_index(drop=True)
                        m_primary = primary_val.loc[mask].reset_index(drop=True)
                        m_fallback = tuned_val.loc[mask].reset_index(drop=True)
                        m_metrics = _eval_combo(m_frame, m_primary, m_fallback, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
                        month_rows.append(
                            {
                                "month": month,
                                "cost3_pnl": float(m_metrics["cost3"]["pnl"]),
                                "cost3_mdd": float(m_metrics["cost3"]["mdd"]),
                                "cost3_trades": int(m_metrics["cost3"]["trades"]),
                            }
                        )
                    robust = _robust_score(val_metrics["cost3"], month_rows)
                    eval_metrics = _eval_combo(eval_df, primary_eval, tuned_eval, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
                    row = {
                        "mode": mode,
                        "tp_scale": float(tp_scale),
                        "conf_thr": float(conf_thr),
                        "qual_thr": float(qual_thr),
                        "robust_selection_score": float(robust),
                        "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                        "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                        "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                        "val_monthly": month_rows,
                        "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                        "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                        "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                        "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                        "delta_vs_baseline": float(eval_metrics["cost3"]["pnl"]) - float(baseline_oos["cost3"]["pnl"]),
                    }
                    rows.append(row)
                    if best_robust is None or row["robust_selection_score"] > best_robust["robust_selection_score"]:
                        best_robust = row
                    if best_oos is None or row["oos_cost3_pnl"] > best_oos["oos_cost3_pnl"]:
                        best_oos = row
    assert best_robust is not None and best_oos is not None

    grid_path = args.out_dir / "grid.jsonl"
    with grid_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    ranking_csv = args.out_dir / "ranking.csv"
    pd.DataFrame(
        [
            {
                "mode": r["mode"],
                "tp_scale": r["tp_scale"],
                "conf_thr": r["conf_thr"],
                "qual_thr": r["qual_thr"],
                "robust_selection_score": r["robust_selection_score"],
                "val_cost3_pnl": r["val_cost3_pnl"],
                "val_cost3_mdd": r["val_cost3_mdd"],
                "val_cost3_trades": r["val_cost3_trades"],
                "oos_cost3_pnl": r["oos_cost3_pnl"],
                "oos_cost3_mdd": r["oos_cost3_mdd"],
                "oos_cost3_trades": r["oos_cost3_trades"],
                "delta_vs_baseline": r["delta_vs_baseline"],
            }
            for r in rows
        ]
    ).sort_values(["robust_selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(ranking_csv, index=False)

    report = {
        "model_id": "alpha7_fallback_exit_improver_focused_20260526",
        "design": "Focused runtime-native exit improver search on current fallback only. No action gate changes. Validation uses Q4 aggregate plus month-level robustness; OOS remains fixed Jan-Feb 2026.",
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "runtime_native_oos_metrics": baseline_oos,
        },
        "best_by_robust_selection": best_robust,
        "best_by_oos": best_oos,
        "artifacts": {
            "grid_jsonl": str(grid_path),
            "ranking_csv": str(ranking_csv),
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_by_robust_selection": best_robust,
                "best_by_oos": best_oos,
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
