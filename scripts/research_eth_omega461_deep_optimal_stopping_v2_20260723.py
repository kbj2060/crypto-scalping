#!/usr/bin/env python3
"""RESEARCH ONLY -- Deep Optimal Stopping v2, VALUE-COMPARISON variant (round 13 follow-up, same
day). Round 13 (research_eth_omega461_deep_optimal_stopping_20260723.py) trained a per-bucket
BINARY CLASSIFIER ("should have stopped") and applied it live via an arbitrary probability
threshold grid (0.5/0.6/0.7) -- a shortcut, not what Becker/Cheridito/Jentzen (arXiv 1804.05394)
actually specify. This script fixes that: at each backward-induction step tau, a REGRESSOR is
trained to predict the CONTINUATION VALUE (the expected forward payoff obtained by following the
ALREADY-TRAINED later-bucket policies from this bar onward), and the stopping decision falls out
of a direct value comparison: stop now iff immediate payoff (realized unrealized move) >=
predicted continuation value (+ a small optional robustness margin). No arbitrary probability
threshold is tuned -- the only free knob is the robustness margin, tested at a couple of small
values, not a wide grid.

Reused unmodified from round 13: trajectory construction (build_trajectories, causal bar-by-bar
replay identical to the baseline TP/SL/exit_head@0.95 lifecycle), bucket edges/definition
(BUCKET_EDGES, hold_bars ranges), feature set (pos_state + base_cols + proxy quality/direction),
and the diagnostic-first / sanity-check / VAL-select-then-single-OOS-touch discipline.

Changed: HistGradientBoostingClassifier -> HistGradientBoostingRegressor per bucket; label =
binary "should stop" -> target = continuation_value (a real number, the same continuation reward
computed by the backward chain, now used as the regression TARGET rather than only to construct a
binary label); the live decision rule is a value comparison, not a probability threshold; the
diagnostic metric is held-out R^2 (variance explained) per bucket instead of AUC.

Scope: zig075 ONLY. h48qual's near-terminal-bucket diagnostic in round 13 was already AUC~0.5
(no signal in the classifier's feature set for that component); switching the decision rule from
threshold-classifier to value-regression does not change what information the features carry, so
h48qual is not re-tried here.

CRITICAL data-scarcity caveat (same as round 13, restated): ~70-72 real TRAIN trade trajectories
for zig075. That is the entire path dataset for backward induction -- regression targets per
bucket are computed over however many BARS (not trades) fall in that bucket across those ~70
trajectories, which is a lot more than 70 rows but still a single real market history replayed
once, not i.i.d. simulated paths as the original paper assumes.

VAL window: 2025-10-01..2025-12-31. OOS: 2026-01-01..03-31 (single touch only if a VAL winner
exists). Fresh window: 2026-04-01..2026-07-12 (limited by the oos_predictions_q075.csv extended
prediction file's actual max timestamp, 2026-07-12 09:00 -- base feature CSVs extend to 07-20 but
predictions don't, so the fresh check stops there; NOT a selection window, reported only as an
additional non-selection robustness data point per task instructions).

fresh_forward_bar_by_bar=true; trade_ledgers_used_as_input=false; saved_parent_exit_timestamps_used=false;
future_rows_used_for_entry=false. Does NOT touch trading_bot_modules/omega4_6_1_live.py,
trading_bot.py, runtime_config.py, config/pipeline.yaml, dashboard files, ensemble/*, or any
data/ensemble/ckpt/ artifact.

Baselines (reused unmodified from round 13, itself bit-for-bit reproduced from the live baseline):
  VAL  zig075: pnl +40.31% mdd -13.07% (29 trades)
  OOS  zig075: pnl +17.89% mdd -11.01% (25 trades)
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
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head_ref  # noqa: E402 (SPLIT_TS reference)

# reuse round 13's trajectory builder verbatim (identical lifecycle / feature construction)
import research_eth_omega461_deep_optimal_stopping_20260723 as dos1  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260723/deep_optimal_stopping_v2_valuecompare_20260723"
SPLIT_TS = exit_head_ref.SPLIT_TS  # pd.Timestamp("2025-10-01")
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"
FRESH_START, FRESH_END = "2026-04-01", "2026-07-12"  # capped by oos_predictions_q075.csv max ts

PROXY_QUALITY_COL = dos1.PROXY_QUALITY_COL
PROXY_DIR_LONG_COL = dos1.PROXY_DIR_LONG_COL
PROXY_DIR_SHORT_COL = dos1.PROXY_DIR_SHORT_COL

BUCKET_EDGES = dos1.BUCKET_EDGES
N_BUCKETS = dos1.N_BUCKETS
TERMINAL_BUCKET = dos1.TERMINAL_BUCKET
TRADE_HOLDOUT_FRAC = dos1.TRADE_HOLDOUT_FRAC
MIN_BUCKET_TRAIN_ROWS = 20  # below this, skip fitting a regressor for that bucket (treat as always-continue)
MARGINS = (0.0, 0.005)  # robustness margin only, not a threshold-tuning grid

bucket_of = dos1.bucket_of
_load_proxy_columns = dos1._load_proxy_columns
prep_split = dos1.prep_split
build_trajectories = dos1.build_trajectories
_continuation_rewards = dos1._continuation_rewards

BASELINES = {
    ("zig075", "VAL"): {"pnl": 40.31, "mdd": -13.07},
    ("zig075", "OOS"): {"pnl": 17.89, "mdd": -11.01},
}


def beats_baseline(name: str, split: str, pnl: float, mdd: float) -> bool:
    b = BASELINES[(name, split)]
    return pnl > b["pnl"] and mdd > b["mdd"]


def _stop_decision_array_reg(
    traj: dict[str, Any], buckets: np.ndarray, models: dict[int, Any], feature_cols: list[str], margin: float,
) -> np.ndarray:
    """Vectorized: for every bar j in this trajectory, True iff the ALREADY-TRAINED value-regressor
    policy for j's bucket would stop there. stop iff immediate payoff (move) >= predicted
    continuation value + margin. Terminal bucket = always True. Bucket with no fitted regressor
    (too few training rows) = never stops there (treated as always-continue, matching round 13's
    single-class-skip convention)."""
    n = len(buckets)
    stop = np.zeros(n, dtype=bool)
    for b in np.unique(buckets):
        b = int(b)
        idx = np.where(buckets == b)[0]
        if len(idx) == 0:
            continue
        if b == TERMINAL_BUCKET:
            stop[idx] = True
            continue
        reg = models.get(b)
        if reg is None:
            continue
        rows = traj["x"].iloc[idx][feature_cols].to_numpy(dtype=np.float64)
        pred_cont = reg.predict(rows)
        move_vals = traj["move"][idx]
        stop[idx] = move_vals >= (pred_cont + margin)
    return stop


def _bucket_training_rows_reg(
    trajectories: list[dict[str, Any]], tau: int, models: dict[int, Any], feature_cols: list[str], margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Regression target y = continuation_value(i) = payoff obtained by CONTINUING past bar i and
    following the already-trained later-bucket value-comparison policies onward (identical
    computation to round 13's cont_reward, now used directly as the regression target rather than
    only to build a binary label)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for traj in trajectories:
        buckets = bucket_of(traj["hold"])
        idx_tau = np.where(buckets == tau)[0]
        if len(idx_tau) == 0:
            continue
        move = traj["move"]
        stop_decision = _stop_decision_array_reg(traj, buckets, models, feature_cols, margin)
        cont_reward = _continuation_rewards(move, stop_decision)
        xs.append(traj["x"].iloc[idx_tau][feature_cols].to_numpy(dtype=np.float64))
        ys.append(cont_reward[idx_tau])
    if not xs:
        return np.empty((0, len(feature_cols))), np.empty((0,), dtype=np.float64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_dos_sequence_reg(
    trajectories: list[dict[str, Any]], feature_cols: list[str], margin: float,
) -> dict[int, Any]:
    """Backward induction over buckets. Returns {bucket_id: fitted regressor or None (terminal /
    too-few-rows skip)}."""
    models: dict[int, Any] = {TERMINAL_BUCKET: None}
    for tau in range(TERMINAL_BUCKET - 1, -1, -1):
        x, y = _bucket_training_rows_reg(trajectories, tau, models, feature_cols, margin)
        if len(y) < MIN_BUCKET_TRAIN_ROWS:
            print(f"stage=train_bucket tau={tau} margin={margin} TOO_FEW_ROWS n={len(y)} -- skipping (treat as always-continue)", flush=True)
            models[tau] = None
            continue
        reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=260723)
        reg.fit(x, y)
        models[tau] = reg
        print(f"stage=train_bucket tau={tau} margin={margin} n={len(y)} y_mean={y.mean():.5f} y_std={y.std():.5f}", flush=True)
    return models


def diagnostic_r2(trajectories: list[dict[str, Any]], feature_cols: list[str], margin: float) -> dict[str, Any]:
    """Trade-level chronological 80/20 split. Trains the backward-induction regression chain on the
    fit portion only, then reports held-out R^2 per bucket (predicted continuation value vs the
    continuation value realized by the SAME fit-side-trained chain applied to held-out trades' own
    real paths -- no leakage of held-out trades into training)."""
    n = len(trajectories)
    n_fit = max(int(round(n * (1.0 - TRADE_HOLDOUT_FRAC))), 1)
    fit_traj = trajectories[:n_fit]
    hold_traj = trajectories[n_fit:]
    print(f"stage=diagnostic n_trades_total={n} n_fit={len(fit_traj)} n_holdout={len(hold_traj)} margin={margin}", flush=True)
    models = train_dos_sequence_reg(fit_traj, feature_cols, margin)

    results = {}
    for tau in range(TERMINAL_BUCKET - 1, -1, -1):
        reg = models.get(tau)
        if reg is None:
            results[tau] = {"status": "no_model_too_few_rows"}
            continue
        x, y = _bucket_training_rows_reg(hold_traj, tau, models, feature_cols, margin)
        if len(x) < 5:
            results[tau] = {"status": "no_holdout_data", "n": int(len(y))}
            continue
        pred = reg.predict(x)
        r2 = float(r2_score(y, pred))
        results[tau] = {"status": "ok", "n": int(len(y)), "y_mean": float(y.mean()), "y_std": float(y.std()), "r2": r2}
        print(f"stage=diagnostic_r2 tau={tau} n={len(y)} y_mean={y.mean():.5f} y_std={y.std():.5f} r2={r2:.4f}", flush=True)
    return {"n_trades_total": n, "n_fit": len(fit_traj), "n_holdout": len(hold_traj), "margin": margin, "per_bucket": results}


@torch.no_grad()
def replay_with_dos_reg(
    frame: pd.DataFrame, base_x: pd.DataFrame, dec: pd.DataFrame, loaded_models: dict[str, tuple],
    *, risk_margin_fraction: np.ndarray, risk_leverage: np.ndarray, fee: float, slip: float, cost_mult: float,
    notional_scaled_sltp: bool, device: torch.device, dos_models: dict[int, Any] | None, feature_cols: list[str],
    margin: float = 0.0, use_dos: bool = True, use_exit_head: bool = True,
    exit_threshold: float = sweep.BASELINE_EXIT_THRESHOLD,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay, structurally identical to round 13's replay_with_dos except the
    DOS decision is a value comparison (stop iff move >= predicted continuation value + margin)
    instead of a probability threshold. use_dos=False, use_exit_head=True reduces to the exact
    baseline (sanity-check config). fresh_forward_bar_by_bar=true; no saved ledger used as input."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)

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
            if not reason and use_dos and dos_models is not None:
                tau = int(bucket_of(np.asarray([hold]))[0])
                if tau == TERMINAL_BUCKET:
                    reason = "dos_terminal"
                else:
                    reg = dos_models.get(tau)
                    if reg is not None:
                        giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                        base_row = {base_x.columns[j]: float(base_np[i, j]) for j in range(base_x.shape[1])}
                        base_row.update({
                            "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                            "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                            "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                            "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                            "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                            "proxy_quality_for_action": float(dos1.dos_proxy_quality[i]),
                            "proxy_dir_p_side": float(dos1.dos_proxy_dir_long[i] if pos > 0 else dos1.dos_proxy_dir_short[i]),
                        })
                        xrow = np.asarray([base_row[c] for c in feature_cols], dtype=np.float64).reshape(1, -1)
                        pred_cont = float(reg.predict(xrow)[0])
                        if move >= (pred_cont + margin):
                            reason = "dos_stop"
            if not reason and use_exit_head:
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
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
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
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage),
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()

    train_frame = sweep.load_frame(TRAIN_START, TRAIN_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    fresh_frame = sweep.load_frame(FRESH_START, FRESH_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"TRAIN frame rows={len(train_frame)} VAL frame rows={len(val_frame)} OOS frame rows={len(oos_frame)} "
          f"FRESH frame rows={len(fresh_frame)}", flush=True)

    name = "zig075"
    cfg = sweep.COMPONENTS[name]
    print(f"===== component={name} (value-comparison DOS, no threshold tuning) =====", flush=True)
    train_pred = sweep.EXT_PRED_DIR / name / f"train_predictions_{cfg['q_tag']}.csv"
    val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
    oos_pred = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"

    p_train, proxy_train = prep_split(name, cfg, train_frame, train_pred, oof=True)
    p_val, proxy_val = prep_split(name, cfg, val_frame, val_pred, oof=True)

    print(f"stage=build_trajectories component={name} split=TRAIN", flush=True)
    traj_train = build_trajectories(
        p_train["frame"], p_train["x"], p_train["dec"], p_train["loaded"], proxy_train,
        risk_margin_fraction=p_train["margin"], risk_leverage=p_train["leverage"],
        fee=p_train["fee"], slip=p_train["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_train["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    print(f"stage=trajectories_built component={name} n_trades={len(traj_train)} "
          f"total_bars={sum(len(t['hold']) for t in traj_train)}", flush=True)

    feature_cols = list(p_train["x"].columns) + [
        "pos_side", "pos_hold_bars", "pos_unrealized", "pos_mfe", "pos_mae", "pos_giveback",
        "pos_dist_to_tp", "pos_dist_to_sl", "pos_notional", "pos_leverage", "pos_exposure",
        "pos_tp", "pos_sl", "proxy_quality_for_action", "proxy_dir_p_side",
    ]

    # ---------------- Diagnostic gate (R^2 per bucket, margin=0 reference) ----------------
    diag = diagnostic_r2(traj_train, feature_cols, margin=0.0)
    with open(OUT_DIR / f"diagnostic_report_{name}.json", "w") as f:
        json.dump(diag, f, indent=2, default=str)
    near_terminal = diag["per_bucket"].get(TERMINAL_BUCKET - 1, {})
    degenerate = near_terminal.get("status") == "ok" and near_terminal.get("r2", 0.0) < 0.05
    print(f"stage=diagnostic_gate component={name} near_terminal_bucket_result={near_terminal} degenerate={degenerate}", flush=True)

    # ---------------- Sanity check on VAL: use_dos=False must reproduce baseline exactly ----------------
    dos1.dos_proxy_quality = proxy_val[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
    dos1.dos_proxy_dir_long = proxy_val[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
    dos1.dos_proxy_dir_short = proxy_val[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
    m_noop, _ = replay_with_dos_reg(
        p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], risk_margin_fraction=p_val["margin"],
        risk_leverage=p_val["leverage"], fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
        notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=None,
        feature_cols=feature_cols, use_dos=False, use_exit_head=True,
    )
    b = BASELINES[(name, "VAL")]
    sane = abs(m_noop["pnl"] - b["pnl"]) < 0.01 and abs(m_noop["mdd"] - b["mdd"]) < 0.01
    print(f"stage=sanity component={name} noop_pnl={m_noop['pnl']:.4f} noop_mdd={m_noop['mdd']:.4f} "
          f"baseline_pnl={b['pnl']} baseline_mdd={b['mdd']} sane={sane}", flush=True)
    if not sane:
        print(f"stage=STOP component={name} sanity check FAILED -- aborting", flush=True)
        return 1

    if degenerate:
        print(f"stage=NOTE component={name} near-terminal bucket R^2 is degenerate -- still running "
              f"VAL configs for completeness but not expecting a win", flush=True)

    # ---------------- Train one DOS value-regressor sequence per margin ----------------
    val_rows = []
    winners = []
    trained_by_margin: dict[float, dict[int, Any]] = {}
    for margin in MARGINS:
        print(f"stage=train_final component={name} margin={margin}", flush=True)
        dos_models = train_dos_sequence_reg(traj_train, feature_cols, margin)
        trained_by_margin[margin] = dos_models
        with open(OUT_DIR / f"dos_reg_models_{name}_margin{margin}.pkl", "wb") as f:
            pickle.dump({"models": dos_models, "feature_cols": feature_cols, "bucket_edges": BUCKET_EDGES, "margin": margin}, f)

        for use_exit_head, tag in ((False, "dos_replace_exit_head"), (True, "dos_before_exit_head")):
            m, _ = replay_with_dos_reg(
                p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], risk_margin_fraction=p_val["margin"],
                risk_leverage=p_val["leverage"], fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
                notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=dos_models,
                feature_cols=feature_cols, margin=margin, use_dos=True, use_exit_head=use_exit_head,
            )
            row = {"component": name, "variant": tag, "margin": margin, **m}
            val_rows.append(row)
            cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
            print(f"  VAL component={name} variant={tag} margin={margin} pnl={m['pnl']:.2f} mdd={m['mdd']:.2f} "
                  f"trades={m['trades']} wr={m['wr']:.2f} cleared_vs_baseline={cleared}", flush=True)
            if cleared:
                winners.append({"component": name, "variant": tag, "margin": margin})

    val_df = pd.DataFrame(val_rows)
    val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
    val_df.to_csv(OUT_DIR / f"dos_reg_VAL_{name}.csv", index=False)

    any_val_winner = len(winners) > 0
    print(f"\nstage=val_summary any_val_winner={any_val_winner} winners={winners}", flush=True)

    if not any_val_winner:
        print("stage=done no_val_winners -- no OOS confirmation run (single-touch discipline); "
              "fresh-window check also skipped (non-selection extra data point, not needed once VAL rejects)", flush=True)
        return 0

    # ---------------- Single OOS touch, then fresh-window extra data point ----------------
    print("stage=oos_confirm", flush=True)
    p_oos, proxy_oos = prep_split(name, cfg, oos_frame, oos_pred, oof=False)
    fresh_pred = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
    p_fresh, proxy_fresh = prep_split(name, cfg, fresh_frame, fresh_pred, oof=False)

    oos_rows = []
    fresh_rows = []
    for w in winners:
        margin = w["margin"]
        with open(OUT_DIR / f"dos_reg_models_{name}_margin{margin}.pkl", "rb") as f:
            saved = pickle.load(f)
        use_exit_head = w["variant"] == "dos_before_exit_head"

        dos1.dos_proxy_quality = proxy_oos[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
        dos1.dos_proxy_dir_long = proxy_oos[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
        dos1.dos_proxy_dir_short = proxy_oos[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
        m_oos, _ = replay_with_dos_reg(
            p_oos["frame"], p_oos["x"], p_oos["dec"], p_oos["loaded"], risk_margin_fraction=p_oos["margin"],
            risk_leverage=p_oos["leverage"], fee=p_oos["fee"], slip=p_oos["slip"], cost_mult=sweep.COST_MULT,
            notional_scaled_sltp=p_oos["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=saved["models"],
            feature_cols=saved["feature_cols"], margin=margin, use_dos=True, use_exit_head=use_exit_head,
        )
        b_oos = BASELINES[(name, "OOS")]
        cleared_oos = beats_baseline(name, "OOS", m_oos["pnl"], m_oos["mdd"])
        oos_rows.append({**w, "oos_pnl": m_oos["pnl"], "oos_mdd": m_oos["mdd"], "oos_trades": m_oos["trades"],
                          "oos_wr": m_oos["wr"], "oos_baseline_pnl": b_oos["pnl"], "oos_baseline_mdd": b_oos["mdd"],
                          "cleared_oos": cleared_oos})
        print(f"  {w} -> OOS pnl={m_oos['pnl']:.2f}% mdd={m_oos['mdd']:.2f}% trades={m_oos['trades']} "
              f"(baseline pnl={b_oos['pnl']:.2f}% mdd={b_oos['mdd']:.2f}%) cleared={cleared_oos}", flush=True)

        dos1.dos_proxy_quality = proxy_fresh[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
        dos1.dos_proxy_dir_long = proxy_fresh[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
        dos1.dos_proxy_dir_short = proxy_fresh[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
        m_fresh, _ = replay_with_dos_reg(
            p_fresh["frame"], p_fresh["x"], p_fresh["dec"], p_fresh["loaded"], risk_margin_fraction=p_fresh["margin"],
            risk_leverage=p_fresh["leverage"], fee=p_fresh["fee"], slip=p_fresh["slip"], cost_mult=sweep.COST_MULT,
            notional_scaled_sltp=p_fresh["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=saved["models"],
            feature_cols=saved["feature_cols"], margin=margin, use_dos=True, use_exit_head=use_exit_head,
        )
        fresh_rows.append({**w, "fresh_pnl": m_fresh["pnl"], "fresh_mdd": m_fresh["mdd"], "fresh_trades": m_fresh["trades"],
                            "fresh_wr": m_fresh["wr"]})
        print(f"  {w} -> FRESH({FRESH_START}..{FRESH_END}, non-selection) pnl={m_fresh['pnl']:.2f}% "
              f"mdd={m_fresh['mdd']:.2f}% trades={m_fresh['trades']}", flush=True)

    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "dos_reg_OOS_confirm.csv", index=False)
    pd.DataFrame(fresh_rows).to_csv(OUT_DIR / "dos_reg_FRESH_check.csv", index=False)
    any_oos_winner = any(r["cleared_oos"] for r in oos_rows)
    print(f"stage=ALL_DONE any_oos_winner={any_oos_winner}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
