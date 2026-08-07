"""Corrects a flawed comparison from earlier in this investigation: parent_only_metrics
(omega._metrics / _to_fixed_decisions) exits purely on FIXED take_profit/stop_loss/max_hold
constants from BASE_TEMPLATE -- it never calls the trained exit classifier at all. So swapping
the exit-head's training label (terminal_giveback heuristic vs DP suffix-max oracle) cannot
possibly move that metric; the VAL delta seen earlier (+5.26% -> +12.95%) was CatBoost seed noise
in the direction/quality experts (260713 vs 260715), not a real effect of the exit-label change.

This script builds the missing, correct comparison: a bar-by-bar backtest where the exit
decision is actually threshold-gated on the trained exit classifier's predicted probability
(mirroring train_eval_omega4_2_risk_sidecar_sol_20260707.py's _replay_with_risk /
_predict_exit_prob_one, ported from its TabM torch model to CatBoost's .predict_proba). Entries
(direction+quality decisions) are held FIXED -- taken from one saved bundle's prediction CSVs --
so only the exit-head's training label varies between the two arms of the comparison. TP/SL
still fire first (same as production), the exit-head threshold only decides trades that hit
neither.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

HEURISTIC_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_catboost_parent_20260713_zig075_q070_rerun"
DP_OPTIMAL_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_catboost_parent_path_optimal_20260715_zig075_path_optimal_q070"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_catboost_exit_threshold_ab_20260715"


def _load_bundle(path: Path) -> dict[str, Any]:
    with open(path / "catboost_3head_bundle.pkl", "rb") as f:
        return pickle.load(f)


def _prepare_exit_runtime(base_x: pd.DataFrame, exit_models: dict[str, Any]) -> tuple[np.ndarray, list[int], list[str]]:
    cols = list(base_x.columns)
    pos_idx = [cols.index(c) for c in parent.POS_COLS]
    return base_x.to_numpy(dtype=np.float32), pos_idx, cols


def _predict_exit_prob(
    base_np: np.ndarray, cols: list[str], pos_idx: list[int], exit_models: dict[str, Any],
    *, row_i: int, expert: str, pos_values: list[float],
) -> float:
    row = base_np[int(row_i)].copy()
    row[np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    x = pd.DataFrame([row], columns=cols)
    model = exit_models[expert]
    proba = model.predict_proba(x)
    idx1 = list(np.asarray(model.classes_, dtype=np.int64)).index(1)
    return float(proba[0, idx1])


def _replay_threshold(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    exit_models: dict[str, Any],
    *,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    take_profit_cfg = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss_cfg = float(omega.BASE_TEMPLATE["stop_loss"])
    notional_cfg = float(omega.BASE_TEMPLATE["notional"])
    leverage_cfg = float(omega.BASE_TEMPLATE["leverage"])
    max_hold_cfg = int(omega.BASE_TEMPLATE.get("max_hold", 0) or 0)
    route = hard._route_id(frame)
    base_np, pos_idx, cols = _prepare_exit_runtime(base_x, exit_models)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}

    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional_cfg
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            reason = ""
            hold = max(int(i) - int(entry_i), 0)
            if take_profit_cfg > 0.0 and unreal >= take_profit_cfg:
                reason = "take_profit"
            elif stop_loss_cfg > 0.0 and unreal <= -abs(stop_loss_cfg):
                reason = "stop_loss"
            elif max_hold_cfg > 0 and hold >= max_hold_cfg:
                reason = "max_hold"
            else:
                giveback = (float(mfe) - float(unreal)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = _predict_exit_prob(
                    base_np, cols, pos_idx, exit_models,
                    row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(unreal), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)),
                        float(take_profit_cfg - unreal), float(unreal + abs(stop_loss_cfg)),
                        float(notional_cfg), float(leverage_cfg), float(notional_cfg * leverage_cfg),
                        float(take_profit_cfg), float(stop_loss_cfg),
                    ],
                )
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional_cfg)
                cash -= before * exit_fee * notional_cfg
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                mfe = 0.0
                mae = 0.0
                continue
        if pos != 0:
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_i = int(i)
        mfe = 0.0
        mae = 0.0
        if side > 0:
            long_entries += 1
        else:
            short_entries += 1

    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heuristic_bundle = _load_bundle(HEURISTIC_DIR)
    dp_bundle = _load_bundle(DP_OPTIMAL_DIR)
    base_cols_h = heuristic_bundle["base_cols"]
    base_cols_d = dp_bundle["base_cols"]
    if base_cols_h != base_cols_d:
        raise RuntimeError("base_cols mismatch between the two bundles")
    heuristic_exit = {e: heuristic_bundle["models"][e]["exit"] for e in hard.EXPERT_NAMES}
    dp_exit = {e: dp_bundle["models"][e]["exit"] for e in hard.EXPERT_NAMES}

    fee, slip = omega._load_fee_slip()
    thresholds = [0.5, 0.7, 0.9, 0.95]
    results: dict[str, Any] = {}
    train_all, eval_df, _overlay_report = omega._load_omega_frames()
    for split, entry_bundle_dir in (("validation", HEURISTIC_DIR), ("oos", HEURISTIC_DIR)):
        dec_csv = entry_bundle_dir / f"{split}_predictions_q070.csv"
        src = pd.read_csv(dec_csv, parse_dates=["timestamp"])
        oof = split == "validation"
        year_frame = train_all if split == "validation" else eval_df
        year_frame, aligned_src = omega._align(year_frame, src, f"{split} exit-threshold-ab align")
        aligned_dec = parent._to_decisions(aligned_src, oof=oof)
        base_x = parent._base_input(year_frame, base_cols_h)
        for arm_name, exit_models in (("heuristic_terminal_giveback", heuristic_exit), ("dp_suffix_max_optimal", dp_exit)):
            for th in thresholds:
                key = f"{split}/{arm_name}/th={th}"
                res = _replay_threshold(
                    year_frame, base_x, aligned_dec, exit_models,
                    exit_threshold=th, fee=fee, slip=slip, cost_mult=3.0,
                )
                results[key] = res
                print(f"{key}: pnl={res['pnl']:.2f}% mdd={res['mdd']:.2f}% trades={res['trades']} wr={res['wr']:.3f} reasons={res['exit_reasons']}", flush=True)

    (OUT_DIR / "ab_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
