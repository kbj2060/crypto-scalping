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
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import _compact_costs, _metrics  # noqa: E402
from scripts.train_eval_alpha7_fallback_exit_improver_focused_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    LIVE_DIR,
    _eval_combo,
    _json_default,
    _load_best_scale_runtime,
    _month_key,
    _robust_score,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _active,
    _combine_primary_fallback,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_primary_exit_improver_20260526"


def _apply_primary_overlay(
    dec: pd.DataFrame,
    mode: str,
    *,
    tp_scale: float,
    hold_scale: float,
    conf_thr: float,
    qual_thr: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    conf = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    qual = pd.to_numeric(out["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if mode == "baseline":
        shrink = np.zeros(len(out), dtype=bool)
    elif mode == "global":
        shrink = active
    elif mode == "low_conf":
        shrink = active & (conf < float(conf_thr))
    elif mode == "low_qual":
        shrink = active & (qual < float(qual_thr))
    elif mode == "low_conf_or_qual":
        shrink = active & ((conf < float(conf_thr)) | (qual < float(qual_thr)))
    elif mode == "low_conf_and_qual":
        shrink = active & ((conf < float(conf_thr)) & (qual < float(qual_thr)))
    elif mode == "mid_band":
        shrink = active & (conf >= float(conf_thr)) & (qual < float(qual_thr))
    else:
        raise ValueError(f"unknown mode={mode}")
    if np.any(shrink):
        out.loc[shrink, "take_profit"] = (
            pd.to_numeric(out.loc[shrink, "take_profit"], errors="coerce").clip(lower=1e-4) * float(tp_scale)
        )
        hold = pd.to_numeric(out.loc[shrink, "max_hold_bars"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out.loc[shrink, "max_hold_bars"] = np.maximum(1, np.rint(hold * float(hold_scale)).astype(np.int64))
    return out


def _apply_fallback_tp(dec: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(float(scale) - 1.0) < 1e-12:
        return dec.copy().reset_index(drop=True)
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    if np.any(active):
        out.loc[active, "take_profit"] = (
            pd.to_numeric(out.loc[active, "take_profit"], errors="coerce").clip(lower=1e-4) * float(scale)
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Runtime-native-ish Alpha7 primary exit improver search.")
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
    baseline_oos = _eval_combo(
        eval_df,
        primary_eval,
        fallback_eval,
        ref_parent=ref_parent,
        runner=noop_runner,
        runner_cfg=noop_cfg,
        fee=fee,
        slip=slip,
    )

    modes = ["baseline", "low_conf", "low_qual", "low_conf_or_qual", "mid_band"]
    tp_scales = [1.0, 0.95, 0.90, 0.85]
    hold_scales = [1.0, 0.75, 0.50]
    conf_thrs = [0.65, 0.75]
    qual_thrs = [0.045, 0.055]
    fallback_tp_scales = [1.0, 0.80]
    val_months = sorted(_month_key(val_df["timestamp"]).dropna().unique().tolist())

    rows: list[dict[str, Any]] = []
    best_robust: dict[str, Any] | None = None
    best_oos: dict[str, Any] | None = None
    total = len(modes) * len(tp_scales) * len(hold_scales) * len(conf_thrs) * len(qual_thrs) * len(fallback_tp_scales)
    step = 0

    for mode in modes:
        for tp_scale in tp_scales:
            for hold_scale in hold_scales:
                for conf_thr in conf_thrs:
                    for qual_thr in qual_thrs:
                        for fallback_tp_scale in fallback_tp_scales:
                            if mode == "baseline" and (tp_scale, hold_scale, conf_thr, qual_thr) != (1.0, 1.0, conf_thrs[0], qual_thrs[0]):
                                continue
                            if mode != "baseline" and abs(tp_scale - 1.0) < 1e-12 and abs(hold_scale - 1.0) < 1e-12:
                                continue
                            step += 1
                            print(
                                f"[alpha7_primary_exit_improver] {step}/{total} mode={mode} tp={tp_scale:.2f} hold={hold_scale:.2f} conf={conf_thr:.2f} qual={qual_thr:.3f} fb_tp={fallback_tp_scale:.2f}",
                                flush=True,
                            )
                            tuned_primary_val = _apply_primary_overlay(
                                primary_val,
                                mode,
                                tp_scale=tp_scale,
                                hold_scale=hold_scale,
                                conf_thr=conf_thr,
                                qual_thr=qual_thr,
                            )
                            tuned_primary_eval = _apply_primary_overlay(
                                primary_eval,
                                mode,
                                tp_scale=tp_scale,
                                hold_scale=hold_scale,
                                conf_thr=conf_thr,
                                qual_thr=qual_thr,
                            )
                            tuned_fallback_val = _apply_fallback_tp(fallback_val, fallback_tp_scale)
                            tuned_fallback_eval = _apply_fallback_tp(fallback_eval, fallback_tp_scale)
                            combo_val = _combine_primary_fallback(tuned_primary_val, tuned_fallback_val)
                            combo_eval = _combine_primary_fallback(tuned_primary_eval, tuned_fallback_eval)
                            val_metrics = _compact_costs(
                                _metrics(
                                    val_df,
                                    parent_for_features=ref_parent,
                                    runner=noop_runner,
                                    runner_cfg=noop_cfg,
                                    dec=combo_val,
                                    fee=fee,
                                    slip=slip,
                                )
                            )
                            month_rows = []
                            month_col = _month_key(val_df["timestamp"])
                            for month in val_months:
                                mask = month_col == month
                                m_frame = val_df.loc[mask].reset_index(drop=True)
                                m_combo = combo_val.loc[mask].reset_index(drop=True)
                                m_metrics = _compact_costs(
                                    _metrics(
                                        m_frame,
                                        parent_for_features=ref_parent,
                                        runner=noop_runner,
                                        runner_cfg=noop_cfg,
                                        dec=m_combo,
                                        fee=fee,
                                        slip=slip,
                                    )
                                )
                                month_rows.append(
                                    {
                                        "month": month,
                                        "cost3_pnl": float(m_metrics["cost3"]["pnl"]),
                                        "cost3_mdd": float(m_metrics["cost3"]["mdd"]),
                                        "cost3_trades": int(m_metrics["cost3"]["trades"]),
                                    }
                                )
                            robust = _robust_score(val_metrics["cost3"], month_rows)
                            eval_metrics = _compact_costs(
                                _metrics(
                                    eval_df,
                                    parent_for_features=ref_parent,
                                    runner=noop_runner,
                                    runner_cfg=noop_cfg,
                                    dec=combo_eval,
                                    fee=fee,
                                    slip=slip,
                                )
                            )
                            row = {
                                "mode": mode,
                                "tp_scale": float(tp_scale),
                                "hold_scale": float(hold_scale),
                                "conf_thr": float(conf_thr),
                                "qual_thr": float(qual_thr),
                                "fallback_tp_scale": float(fallback_tp_scale),
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
                "hold_scale": r["hold_scale"],
                "conf_thr": r["conf_thr"],
                "qual_thr": r["qual_thr"],
                "fallback_tp_scale": r["fallback_tp_scale"],
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
    ).sort_values(["robust_selection_score", "oos_cost3_pnl"], ascending=False).to_csv(ranking_csv, index=False)

    summary = {
        "model_id": "alpha7_primary_exit_improver_20260526",
        "design": "Keep current Alpha7 primary/fallback parents. Search conditional primary TP/max-hold shrink plus optional fallback TP shrink using the existing Alpha7 noop-runner contract. Selection stays on 2025 Q4 with month-level robustness, OOS stays fixed 2026 Jan-Feb.",
        "baseline": {
            "combo_selected_metrics": baseline_combo.get("selected_metrics"),
            "runtime_native_oos_metrics": baseline_oos,
        },
        "best_by_robust_selection": best_robust,
        "best_by_oos": best_oos,
        "artifacts": {
            "grid": str(grid_path),
            "ranking_csv": str(ranking_csv),
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best_by_robust_selection": best_robust, "best_by_oos": best_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
