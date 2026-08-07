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
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    EVAL_CSV,
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
MODEL_ID = "alpha7_fallback_micro_trigger_exit_20260526"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_micro_trigger_exit_20260526"
REQUIRED_TRIGGER_COLS = (
    "smart_money_flow",
    "ofi_acceleration",
    "liquidity_vacuum",
    "funding_pressure",
    "ai_flow_exhaustion",
    "ai_flow_pressure",
    "net_taker_ratio",
    "volatility_z",
)


def _safe_col(frame: pd.DataFrame, col: str) -> np.ndarray:
    return (
        pd.to_numeric(frame[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _require_cols(frame: pd.DataFrame, cols: tuple[str, ...], *, name: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required trigger features: {missing}")


def _feat_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_safe_col(frame, c) for c in REQUIRED_TRIGGER_COLS]).astype(np.float64)


def _rule_trigger_mask(
    frame: pd.DataFrame,
    side: np.ndarray,
    *,
    q_smf: float,
    q_ofi: float,
    q_liq: float,
    q_flow_exhaust: float,
    q_abs_funding: float,
) -> np.ndarray:
    smf = _safe_col(frame, "smart_money_flow")
    ofi = _safe_col(frame, "ofi_acceleration")
    liq = _safe_col(frame, "liquidity_vacuum")
    flow_exh = _safe_col(frame, "ai_flow_exhaustion")
    fund = _safe_col(frame, "funding_pressure")

    abs_smf = np.abs(smf)
    abs_ofi = np.abs(ofi)
    abs_liq = np.abs(liq)
    abs_fund = np.abs(fund)

    q_smf_v = float(np.quantile(abs_smf, q_smf))
    q_ofi_v = float(np.quantile(abs_ofi, q_ofi))
    q_liq_v = float(np.quantile(abs_liq, q_liq))
    q_flow_exh_v = float(np.quantile(flow_exh, q_flow_exhaust))
    q_abs_fund_v = float(np.quantile(abs_fund, q_abs_funding))

    pulse = (abs_ofi >= q_ofi_v) & (abs_liq >= q_liq_v) & (abs_smf >= q_smf_v)
    exhaustion = (flow_exh >= q_flow_exh_v) | (abs_fund >= q_abs_fund_v)
    dir_ok = ((side > 0) & ((smf > 0.0) | (ofi > 0.0))) | ((side < 0) & ((smf < 0.0) | (ofi < 0.0)))
    return pulse & exhaustion & dir_ok


def _apply_trigger_exit_overlay(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    fallback_dec: pd.DataFrame,
    *,
    iforest: IsolationForest | None,
    q_smf: float,
    q_ofi: float,
    q_liq: float,
    q_flow_exhaust: float,
    q_abs_funding: float,
    conf_floor: float,
    qual_floor: float,
    tp_scale: float,
    sl_scale_low: float,
    sl_scale_high: float,
    hold_cap_low: int,
    hold_cap_high: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = fallback_dec.copy().reset_index(drop=True)
    p_active = _active(primary_dec.reset_index(drop=True))
    f_active = _active(out)
    cash_region = (~p_active) & f_active

    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    rule = _rule_trigger_mask(
        frame,
        side,
        q_smf=q_smf,
        q_ofi=q_ofi,
        q_liq=q_liq,
        q_flow_exhaust=q_flow_exhaust,
        q_abs_funding=q_abs_funding,
    )
    if iforest is None:
        anomaly = np.ones(len(frame), dtype=bool)
    else:
        anomaly = iforest.predict(_feat_matrix(frame)) == -1
    keep = cash_region & rule & anomaly
    block = cash_region & (~keep)

    out.loc[block, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score", "confidence"]] = 0
    out.loc[block, "leverage"] = 1.0

    if np.any(keep):
        conf = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        qual = pd.to_numeric(out["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        low_conv = keep & ((conf < float(conf_floor)) | (qual < float(qual_floor)))
        high_conv = keep & (~low_conv)

        tp = pd.to_numeric(out["take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        sl = pd.to_numeric(out["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        hold = pd.to_numeric(out["max_hold_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

        tp[keep] = np.maximum(1e-4, tp[keep]) * float(tp_scale)
        sl[low_conv] = np.maximum(1e-4, sl[low_conv]) * float(sl_scale_low)
        sl[high_conv] = np.maximum(1e-4, sl[high_conv]) * float(sl_scale_high)
        hold[low_conv] = np.maximum(1, np.minimum(hold[low_conv], int(hold_cap_low)))
        hold[high_conv] = np.maximum(1, np.minimum(hold[high_conv], int(hold_cap_high)))

        out["take_profit"] = tp
        out["stop_loss"] = sl
        out["max_hold_bars"] = hold

    stats = {
        "rows": int(len(out)),
        "primary_cash_and_fallback_active": int(cash_region.sum()),
        "trigger_rule_rows": int((cash_region & rule).sum()),
        "trigger_anomaly_rows": int((cash_region & anomaly).sum()),
        "trigger_keep_rows": int(keep.sum()),
        "trigger_block_rows": int(block.sum()),
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Alpha7 fallback micro trigger gate + micro exit overlay (runtime-native).")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    _require_cols(train_all, REQUIRED_TRIGGER_COLS, name="train")
    _require_cols(eval_df, REQUIRED_TRIGGER_COLS, name="eval")

    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    fit_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    if len(fit_df) < 1000:
        raise ValueError(f"fit partition too small: {len(fit_df)}")

    primary_parent = joblib.load(BASELINE.primary_parent)
    primary_rt = _load_best_scale_runtime(BASELINE.primary_summary)
    fallback_parent = joblib.load(BASELINE.fallback_parent)
    fallback_rt = _load_best_scale_runtime(BASELINE.fallback_summary)

    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    baseline_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, fallback_eval),
            fee=fee,
            slip=slip,
        )
    )

    # Keep the search bounded so one run finishes quickly while preserving the core idea.
    q_smf_grid = [0.75, 0.85]
    q_ofi_grid = [0.75, 0.85]
    q_liq_grid = [0.75, 0.85]
    q_flow_exh_grid = [0.60]
    q_abs_funding_grid = [0.60]
    anomaly_mode_grid = ["off", "on"]
    contamination_grid = [0.03]
    conf_floor_grid = [0.50]
    qual_floor_grid = [0.03]
    tp_scale_grid = [0.50, 0.65]
    sl_scale_low_grid = [0.70]
    sl_scale_high_grid = [0.90]
    hold_cap_low_grid = [6, 12]
    hold_cap_high_grid = [18]

    fit_x = _feat_matrix(fit_df)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for anomaly_mode in anomaly_mode_grid:
        if anomaly_mode == "on":
            iforest_specs: list[tuple[IsolationForest | None, float]] = []
            for contamination in contamination_grid:
                mdl = IsolationForest(
                    n_estimators=300,
                    contamination=float(contamination),
                    random_state=int(args.seed),
                    n_jobs=-1,
                )
                mdl.fit(fit_x)
                iforest_specs.append((mdl, float(contamination)))
        else:
            iforest_specs = [(None, 0.0)]
        for iforest, contamination in iforest_specs:
            for q_smf in q_smf_grid:
                for q_ofi in q_ofi_grid:
                    for q_liq in q_liq_grid:
                        for q_flow_exh in q_flow_exh_grid:
                            for q_abs_funding in q_abs_funding_grid:
                                for conf_floor in conf_floor_grid:
                                    for qual_floor in qual_floor_grid:
                                        for tp_scale in tp_scale_grid:
                                            for sl_scale_low in sl_scale_low_grid:
                                                for sl_scale_high in sl_scale_high_grid:
                                                    if sl_scale_high < sl_scale_low:
                                                        continue
                                                    for hold_cap_low in hold_cap_low_grid:
                                                        for hold_cap_high in hold_cap_high_grid:
                                                            if hold_cap_high < hold_cap_low:
                                                                continue
                                                            tuned_val, val_stats = _apply_trigger_exit_overlay(
                                                            val_df,
                                                            primary_val,
                                                            fallback_val,
                                                            iforest=iforest,
                                                            q_smf=q_smf,
                                                            q_ofi=q_ofi,
                                                            q_liq=q_liq,
                                                            q_flow_exhaust=q_flow_exh,
                                                            q_abs_funding=q_abs_funding,
                                                            conf_floor=conf_floor,
                                                            qual_floor=qual_floor,
                                                            tp_scale=tp_scale,
                                                            sl_scale_low=sl_scale_low,
                                                            sl_scale_high=sl_scale_high,
                                                            hold_cap_low=hold_cap_low,
                                                            hold_cap_high=hold_cap_high,
                                                        )
                                                        tuned_eval, eval_stats = _apply_trigger_exit_overlay(
                                                            eval_df,
                                                            primary_eval,
                                                            fallback_eval,
                                                            iforest=iforest,
                                                            q_smf=q_smf,
                                                            q_ofi=q_ofi,
                                                            q_liq=q_liq,
                                                            q_flow_exhaust=q_flow_exh,
                                                            q_abs_funding=q_abs_funding,
                                                            conf_floor=conf_floor,
                                                            qual_floor=qual_floor,
                                                            tp_scale=tp_scale,
                                                            sl_scale_low=sl_scale_low,
                                                            sl_scale_high=sl_scale_high,
                                                            hold_cap_low=hold_cap_low,
                                                            hold_cap_high=hold_cap_high,
                                                        )
                                                        val_final = _combine_primary_fallback(primary_val, tuned_val)
                                                        eval_final = _combine_primary_fallback(primary_eval, tuned_eval)
                                                        val_metrics = _compact_costs(
                                                            _metrics(
                                                                val_df,
                                                                parent_for_features=ref_parent,
                                                                runner=noop_runner,
                                                                runner_cfg=noop_cfg,
                                                                dec=val_final,
                                                                fee=fee,
                                                                slip=slip,
                                                            )
                                                        )
                                                        eval_metrics = _compact_costs(
                                                            _metrics(
                                                                eval_df,
                                                                parent_for_features=ref_parent,
                                                                runner=noop_runner,
                                                                runner_cfg=noop_cfg,
                                                                dec=eval_final,
                                                                fee=fee,
                                                                slip=slip,
                                                            )
                                                        )
                                                        row = {
                                                            "anomaly_mode": anomaly_mode,
                                                            "contamination": float(contamination),
                                                            "q_smf": float(q_smf),
                                                            "q_ofi": float(q_ofi),
                                                            "q_liq": float(q_liq),
                                                            "q_flow_exhaust": float(q_flow_exh),
                                                            "q_abs_funding": float(q_abs_funding),
                                                            "conf_floor": float(conf_floor),
                                                            "qual_floor": float(qual_floor),
                                                            "tp_scale": float(tp_scale),
                                                            "sl_scale_low": float(sl_scale_low),
                                                            "sl_scale_high": float(sl_scale_high),
                                                            "hold_cap_low": int(hold_cap_low),
                                                            "hold_cap_high": int(hold_cap_high),
                                                            "selection_score": float(_score(val_metrics)),
                                                            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                                                            "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                                                            "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                                                            "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                                                            "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                                                            "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                                                            "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                                                            "delta_vs_baseline": float(eval_metrics["cost3"]["pnl"]) - float(baseline_metrics["cost3"]["pnl"]),
                                                            "val_trigger_keep_rows": int(val_stats["trigger_keep_rows"]),
                                                            "eval_trigger_keep_rows": int(eval_stats["trigger_keep_rows"]),
                                                        }
                                                        rows.append(row)
                                                        if best is None or row["selection_score"] > best["selection_score"]:
                                                            best = row

    assert best is not None
    grid_df = pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).reset_index(drop=True)
    grid_path = args.out_dir / "grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_cfg = grid_df.iloc[0].to_dict()
    if str(best_cfg["anomaly_mode"]) == "on":
        best_iforest: IsolationForest | None = IsolationForest(
            n_estimators=300,
            contamination=float(best_cfg["contamination"]),
            random_state=int(args.seed),
            n_jobs=-1,
        )
        best_iforest.fit(fit_x)
    else:
        best_iforest = None
    best_tuned_eval, best_eval_stats = _apply_trigger_exit_overlay(
        eval_df,
        primary_eval,
        fallback_eval,
        iforest=best_iforest,
        q_smf=float(best_cfg["q_smf"]),
        q_ofi=float(best_cfg["q_ofi"]),
        q_liq=float(best_cfg["q_liq"]),
        q_flow_exhaust=float(best_cfg["q_flow_exhaust"]),
        q_abs_funding=float(best_cfg["q_abs_funding"]),
        conf_floor=float(best_cfg["conf_floor"]),
        qual_floor=float(best_cfg["qual_floor"]),
        tp_scale=float(best_cfg["tp_scale"]),
        sl_scale_low=float(best_cfg["sl_scale_low"]),
        sl_scale_high=float(best_cfg["sl_scale_high"]),
        hold_cap_low=int(best_cfg["hold_cap_low"]),
        hold_cap_high=int(best_cfg["hold_cap_high"]),
    )
    best_eval_metrics = _compact_costs(
        _metrics(
            eval_df,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=_combine_primary_fallback(primary_eval, best_tuned_eval),
            fee=fee,
            slip=slip,
        )
    )

    report = {
        "model_id": MODEL_ID,
        "design": "Primary/fallback models stay fixed. In primary-cash rows, fallback is executed only when microstructure pulse + anomaly trigger fires, then fallback exit is compacted with micro-TP and time-decay-style SL/hold caps.",
        "baseline_live_dir": str(BASELINE.live_dir),
        "baseline_model_id": BASELINE.model_id,
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "required_trigger_cols": list(REQUIRED_TRIGGER_COLS),
        "baseline_oos_cost3": baseline_metrics["cost3"],
        "best_by_selection": best_cfg,
        "best_oos_cost3": best_eval_metrics["cost3"],
        "best_eval_trigger_stats": best_eval_stats,
        "artifacts": {"grid": str(grid_path)},
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_selection_score": float(best_cfg["selection_score"]),
                "best_oos_cost3_pnl": float(best_eval_metrics["cost3"]["pnl"]),
                "best_oos_cost3_trades": int(best_eval_metrics["cost3"]["trades"]),
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
