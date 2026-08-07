#!/usr/bin/env python3
"""RESEARCH ONLY -- continuous position-weight modulation for the LIVE ETH Omega4.6.1
h48qual/zig075 components (Jiang et al. 1706.10059-style continuous exposure fraction, NOT a
from-scratch RL policy -- see project memory project-omega4-7-rl-failed.md for why an
end-to-end policy-gradient agent was already tried and failed catastrophically on this same
low-trade-count data; this script deliberately avoids that failure mode by using a supervised
regressor trained on a DP-computed oracle target instead).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, or
.env. Setup/component/frame-loading conventions forked from research_eth_omega461_exit_sweep_20260721.py
("sweep") and research_eth_omega461_exit_ideas2_20260721.py ("ideas2"), matching
train_eth_omega461_reversal_risk_scaleout_20260721.py's TRAIN/internal-VAL split convention.

Idea: instead of a single one-time scale-out trigger (all 8 prior exit-logic experiments, see
project-eth-omega461-exit-logic-experiments-20260721.md), output a continuous position-weight
fraction w(t) in [0,1] at every bar while a trade is open, generalizing scale-out/scale-in into a
smooth trajectory. TP/SL price-move thresholds are unchanged (Futures Risk Sizing Contract:
they're price-move levels, independent of notional); only the *notional* multiplying the P&L at
each bar is modulated.

Oracle construction (label target, forward-looking ONLY for constructing the training label --
same convention as the project's existing "DP trajectory (oracle) label" approach, see
scripts/build_scalp_1m_dp_labels_20260716.py and scripts/build_omega1_2_1_dp_trajectory_labels_20260620.py
for the prior finite-state DP recursions this borrows the backward-induction idea from; this is a
NEW, smaller recursion over a discretized weight state (not a copy of either prior file) because
those recursions solve entry/hold/exit timing, not continuous exposure sizing):
  For each baseline TRAIN-window trade (replayed with the exact baseline lifecycle: TP/SL/
  exit_head@0.95, no trailing/partial mechanism -- identical to sweep.replay_exit_variant with no
  kwargs), discretize weight into 5 levels {0, 0.25, 0.5, 0.75, 1.0}. Backward DP over bars from
  the trade's exit bar to its entry bar: V(t, w_prev) = max_w [ w*r(t) - turnover_cost*|w-w_prev|
  + V(t+1, w) ], where r(t) is the bar's incremental signed price-move contribution and
  turnover_cost = fee_eff+slip_eff (the SAME realistic per-side cost rate used everywhere else in
  this harness, not a free rebalance). Forward-trace from w=1.0 at entry (baseline always enters
  at full notional; only the HOLD trajectory is modulated) gives the oracle w*(t) path.

Model: HistGradientBoostingRegressor (this project's convention for a single scalar in-position
task; matches the reversal-risk scaleout script's model choice) trained on
  features = pos_state (13 cols matching omega4_6_1_live.py::_Component.exit_probability) +
             full numeric base_cols (same base_x the exit head/reversal classifier consume) +
             regime3_current_sensitive_wide24_{chop,bull,bear}_prob + atr_pct
  target   = oracle w*(t) (continuous regression target, though only 5 discrete values appear in
             training data since the oracle itself is discretized)
TRAIN window: 2025-01-01 .. 2025-09-30 (SPLIT_TS, same as reversal-risk scaleout). Internal-VAL
(2025-10-01..12-31, same window as the eval script's VAL split) used only as an R^2/MAE training
diagnostic, not a backtest.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_ideas2_20260721 as ideas2  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head_ref  # noqa: E402 (SPLIT_TS reference)

OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
SPLIT_TS = exit_head_ref.SPLIT_TS  # pd.Timestamp("2025-10-01")
LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
POS_COLS = [
    "pos_side", "pos_hold_bars", "pos_unrealized", "pos_mfe", "pos_mae", "pos_giveback",
    "pos_dist_to_tp", "pos_dist_to_sl", "pos_notional", "pos_leverage", "pos_exposure", "pos_tp", "pos_sl",
]


def dp_oracle_weight_path(moves: np.ndarray, turnover_cost: float) -> np.ndarray:
    """moves: price-move-from-entry (side-adjusted) recorded at each decision bar PLUS the
    trade's terminal move, length n. Returns oracle weight levels w*(t) for the n-1 decision
    bars (weight applied moving from bar t to bar t+1), backward DP, forward-traced from w=1.0.
    """
    n = len(moves)
    if n < 2:
        return np.zeros(0)
    r = np.diff(moves)
    k = len(LEVELS)
    v_next = np.zeros(k)
    policy = np.zeros((n - 1, k), dtype=np.int64)
    for t in range(n - 2, -1, -1):
        rt = r[t]
        cand = LEVELS[None, :] * rt - turnover_cost * np.abs(LEVELS[:, None] - LEVELS[None, :]) + v_next[None, :]
        best_k = np.argmax(cand, axis=1)
        v_cur = cand[np.arange(k), best_k]
        policy[t] = best_k
        v_next = v_cur
    w_path = np.zeros(n - 1)
    kprev = k - 1  # start at weight=1.0 (baseline always enters full notional)
    for t in range(n - 1):
        kk = int(policy[t, kprev])
        w_path[t] = LEVELS[kk]
        kprev = kk
    return w_path


@torch.no_grad()
def build_dataset(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    regime: dict[str, np.ndarray],
    atr_pct: np.ndarray,
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Causal bar-by-bar replay, IDENTICAL lifecycle logic to sweep.replay_exit_variant(
    exit_threshold=0.95, no trailing) -- the exact baseline trade lifecycle the live model
    follows. Records per-bar (features, move) for every bar the position is held; on trade close
    runs the DP oracle over the trade's own recorded moves (forward-looking ONLY for label
    construction, matches the exit-head/reversal-classifier label convention already used in this
    research thread) to assign a w*(t) target to each recorded row.
    fresh_forward_bar_by_bar=true for the underlying simulation; no saved ledger used as input.
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    turnover_cost = fee_eff + slip_eff
    exit_threshold = sweep.BASELINE_EXIT_THRESHOLD
    pos = 0
    entry_price = 0.0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    pending_rows: list[dict[str, float]] = []
    pending_moves: list[float] = []
    all_rows: list[dict[str, float]] = []
    all_labels: list[float] = []
    trades = 0
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    base_cols = list(base_x.columns)

    def flush_trade(final_move: float) -> None:
        nonlocal pending_rows, pending_moves
        if not pending_rows:
            pending_rows, pending_moves = [], []
            return
        moves_arr = np.asarray(pending_moves + [final_move], dtype=np.float64)
        w_path = dp_oracle_weight_path(moves_arr, turnover_cost)
        for row, w in zip(pending_rows, w_path):
            all_rows.append(row)
            all_labels.append(float(w))
        pending_rows, pending_moves = [], []

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            hold = max(int(i) - int(entry_i), 0)
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = rs._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if not reason:
                giveback_now = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                base_row = {base_cols[j]: float(base_np[i, j]) for j in range(len(base_cols))}
                base_row.update({
                    "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                    "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback_now, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                    "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                    "regime_chop_prob": float(regime["chop"][i]), "regime_bull_prob": float(regime["bull"][i]),
                    "regime_bear_prob": float(regime["bear"][i]), "atr_pct": float(atr_pct[i]),
                })
                pending_rows.append(base_row)
                pending_moves.append(float(move))
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                trades += 1
                flush_trade(raw_exit)
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_i = min(int(i) + 1, len(frame) - 1)
        leverage = row_leverage
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        mfe = 0.0
        mae = 0.0

    if pos != 0 and pending_rows:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        trades += 1
        flush_trade(raw_exit)

    x = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(all_labels, dtype=np.float64)
    diag = {
        "trades": int(trades), "rows": int(len(y)),
        "label_mean": float(y.mean()) if len(y) else float("nan"),
        "label_frac_at_1": float((y >= 0.999).mean()) if len(y) else float("nan"),
        "label_frac_at_0": float((y <= 0.001).mean()) if len(y) else float("nan"),
    }
    return x, y, diag


def prep_split(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path) -> dict[str, Any]:
    p = sweep.prep_component(name, cfg, frame, pred_csv, oof=True)
    p["regime"] = ideas2.get_regime_arrays(p["frame"])
    p["atr"] = atr_eval._atr_pct(p["frame"], cfg["atr_window"])
    return p


def train_component(name: str, cfg: dict) -> dict[str, Any]:
    full_2025 = sweep.load_frame("2025-01-01", "2025-12-31", base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    train_pred = sweep.EXT_PRED_DIR / name / f"train_predictions_{cfg['q_tag']}.csv"
    val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"

    train_frame = full_2025[full_2025["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_frame = full_2025[full_2025["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    print(f"stage=prep component={name} split=TRAIN", flush=True)
    p_train = prep_split(name, cfg, train_frame, train_pred)
    print(f"stage=prep component={name} split=VAL(internal)", flush=True)
    p_val = prep_split(name, cfg, val_frame, val_pred)

    print(f"stage=build_dataset component={name} split=TRAIN rows_frame={len(p_train['frame'])}", flush=True)
    x_train, y_train, diag_train = build_dataset(
        p_train["frame"], p_train["x"], p_train["dec"], p_train["loaded"], p_train["regime"], p_train["atr"],
        risk_margin_fraction=p_train["margin"], risk_leverage=p_train["leverage"],
        fee=p_train["fee"], slip=p_train["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_train["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"stage=build_dataset component={name} split=VAL(internal) rows_frame={len(p_val['frame'])}", flush=True)
    x_val, y_val, diag_val = build_dataset(
        p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], p_val["regime"], p_val["atr"],
        risk_margin_fraction=p_val["margin"], risk_leverage=p_val["leverage"],
        fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"stage=train_diag component={name} train={diag_train} val={diag_val}", flush=True)

    feature_cols = list(x_train.columns)
    x_val = x_val.reindex(columns=feature_cols, fill_value=0.0)

    if len(y_train) < 20:
        raise RuntimeError(f"{name}: too few training rows ({len(y_train)}) to fit a regressor")

    reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=260721)
    reg.fit(x_train.to_numpy(dtype=np.float64), y_train)

    val_pred_y = reg.predict(x_val.to_numpy(dtype=np.float64)) if len(y_val) else np.zeros(0)
    val_r2 = float(r2_score(y_val, val_pred_y)) if len(y_val) > 1 else float("nan")
    val_mae = float(mean_absolute_error(y_val, val_pred_y)) if len(y_val) else float("nan")
    # naive baseline: always predict train-mean weight (sanity that the model beats a trivial constant)
    naive_mae = float(mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))) if len(y_val) else float("nan")

    out_dir = OUT_ROOT / f"eth_omega461_continuous_weight_20260721_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": reg, "feature_columns": feature_cols}, f)

    report = {
        "component": name, "model_kind": "hgb_regressor",
        "split_ts": str(SPLIT_TS), "train_window": ["2025-01-01", str(SPLIT_TS.date())],
        "internal_val_window": [str(SPLIT_TS.date()), "2025-12-31"],
        "train_diag": diag_train, "internal_val_diag": diag_val,
        "internal_val_r2": val_r2, "internal_val_mae": val_mae, "internal_val_naive_mae": naive_mae,
        "feature_count": len(feature_cols),
        "levels": LEVELS.tolist(),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"stage=done component={name} val_r2={val_r2:.4f} val_mae={val_mae:.4f} naive_mae={naive_mae:.4f} "
          f"train_rows={diag_train['rows']} train_label_mean={diag_train['label_mean']:.3f}", flush=True)
    return report


def main() -> int:
    reports = {}
    for name, cfg in sweep.COMPONENTS.items():
        reports[name] = train_component(name, cfg)
    print(json.dumps(reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
