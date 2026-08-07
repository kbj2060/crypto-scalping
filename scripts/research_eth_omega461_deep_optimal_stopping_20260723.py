#!/usr/bin/env python3
"""RESEARCH ONLY -- Deep Optimal Stopping (Becker/Cheridito/Jentzen, arXiv 1804.05394) adapted
exit-logic experiment for the LIVE ETH Omega4.6.1 h48qual/zig075 components (round 13, likely
final round, of the exit-logic research thread; see docs/... memory
project-eth-omega461-exit-logic-experiments-20260721 for rounds 1-12).

Motivation / how this differs from round 10 (train_eth_omega461_dp_reversal_scaleout_20260721.py,
which got AUC=0.5 at all tested horizons trying to predict a single DP-computed "optimal action"
label with bar-local features): instead of ONE classifier predicting a whole-window
backward-induction label, Deep Optimal Stopping trains a SEQUENCE of stopping networks f_tau, one
per discretized hold-time bucket, via backward induction over REALIZED trajectories -- the label
at each bucket is derived by comparing "stop now" reward against the "continuation value" implied
by replaying the ALREADY-TRAINED later-bucket networks forward along that SAME real trade's path.
This is the classic Longstaff-Schwartz-style regression/classification backward induction the
Becker et al. paper formalizes with neural nets; here HistGradientBoostingClassifier is used per
bucket (same model family as round 10) since this project doesn't have GPU-scale path counts.

CRITICAL, STATED-UP-FRONT data-scarcity caveat: the original paper assumes effectively unlimited
simulated paths. This project only has the project's own REAL trade trajectories -- roughly
70-72 trades in the TRAIN window (2025-01-01..2025-09-30), 29 in VAL (2025-10-01..12-31), 25 in
OOS (2026-01-01..03-31, zig075) / 14 (h48qual). That is the entire path dataset available for a
multi-step backward induction. This is flagged as a likely-fatal constraint before building the
full pipeline (diagnostic gate below); if the diagnostic already looks degenerate this script
stops and reports rather than grinding through a full VAL/OOS funnel that can't possibly work.

Trajectory construction: causal bar-by-bar replay IDENTICAL in lifecycle logic to the baseline
(TP/SL/exit_head@0.95, no trailing/DOS) -- i.e. the real live trade lifecycle -- recording every
bar of every open position (hold, per-bar unrealized "move", and the full pos_state+base_cols+
proxy feature row) from entry to the trade's REAL baseline exit bar. Using the baseline's own
TP/SL/exit-head lifecycle to define each trajectory's real extent (rather than e.g. forcing every
trade to run to TP/SL/window-end with exit_head disabled) keeps the "path" data grounded in what
actually happened live -- the same choice round 10's build_dataset made for its labeling window.

Bucket backward induction: buckets are hold_bars ranges (see BUCKET_EDGES, chosen from the actual
TRAIN+VAL+OOS hold_bars distribution -- min 5, median ~600-650, p90 ~1200-2200, max up to ~5000
bars -- computed via tmp/research_20260723/diag_holdtimes.py). The LAST bucket is TRIVIAL: any
bar reaching it is forced to stop (matches the project's existing "boundary" convention elsewhere,
e.g. the DP recursion's boundary_start cutoff) -- no network is trained for it. Working backward
from the second-to-last bucket, each f_tau is trained on: stop_reward = move at that bar;
continuation_value = the reward obtained by simulating the ALREADY-TRAINED later buckets' decision
functions forward along the SAME real trajectory (a bar-by-bar walk: at each later bar, evaluate
that bar's bucket's trained f; if it says "stop", continuation_value = move there; if the
trajectory's real (baseline) exit happens first, continuation_value = the trade's realized final
move). label = 1 (stop) iff stop_reward >= continuation_value.

Diagnostic-first discipline (per task): before building the full backward chain, a trade-level
held-out AUC check is run for the near-terminal buckets (the ones with the most data and the
"easiest" decision) using a chronological 80/20 split of TRAIN trades. If that is already
degenerate (AUC ~0.5, matching round 10's failure mode), this is reported as the final result.

Fresh-forward discipline: causal bar-by-bar single forward pass for BOTH trajectory construction
and the VAL/OOS DOS-driven replay; no saved ledger used as an input to the replay logic (past
trajectories are training data only, exactly like every other trained-classifier round in this
thread). Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py,
config/pipeline.yaml, dashboard files, ensemble/*, or any data/ensemble/ckpt/ artifact.

VAL window: 2025-10-01..2025-12-31 (this model's established VAL start, one month short of
CLAUDE.md's canonical 09-01 -- OOF prediction CSVs only exist from 10-01, same note as every
other round in this thread). OOS: 2026-01-01..03-31, single touch only if a VAL winner exists.

Baselines (reused unmodified, NOT recomputed -- reproduced bit-for-bit by
tmp/research_20260723/diag_holdtimes.py before this script was written):
  VAL  h48qual: pnl +5.45%  mdd -11.62% (29 trades)
  VAL  zig075:  pnl +40.31% mdd -13.07% (29 trades)
  OOS  h48qual: pnl +9.49%  mdd -6.54%  (14 trades)
  OOS  zig075:  pnl +17.89% mdd -11.01% (25 trades)
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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

OUT_DIR = ROOT / "tmp/research_20260723/deep_optimal_stopping_20260723"
SPLIT_TS = exit_head_ref.SPLIT_TS  # pd.Timestamp("2025-10-01")
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"

PROXY_QUALITY_COL = "omega1_regime3_expertdq_oof_quality_for_action"
PROXY_DIR_LONG_COL = "omega1_regime3_expertdq_oof_dir_p_long"
PROXY_DIR_SHORT_COL = "omega1_regime3_expertdq_oof_dir_p_short"

# Chosen from the actual hold_bars distribution (see module docstring / diag_holdtimes.py):
# TRAIN median ~600-650, p75 ~1200-1275, p90 ~1900-2200, max up to ~5000. 4 buckets -> 3 trained
# networks + 1 trivial terminal bucket.
BUCKET_EDGES = [0, 150, 400, 900, 10**9]
N_BUCKETS = len(BUCKET_EDGES) - 1
TERMINAL_BUCKET = N_BUCKETS - 1  # trivial: forced stop
TRADE_HOLDOUT_FRAC = 0.2  # chronological trade-level split for the diagnostic AUC only


def bucket_of(hold: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(BUCKET_EDGES, hold, side="right") - 1, 0, N_BUCKETS - 1)


def _load_proxy_columns(pred_csv: Path, keep_ts: set) -> pd.DataFrame:
    # OOF (train/validation) prediction CSVs use the "..._oof_..." column prefix; OOS (held-out)
    # prediction CSVs drop "_oof_" since they aren't out-of-fold (same naming difference round 3's
    # eval script -- research_eth_omega461_reversal_risk_scaleout_eval_20260721.py -- handles).
    header = pd.read_csv(pred_csv, nrows=0).columns
    if PROXY_QUALITY_COL in header:
        qual_col, long_col, short_col = PROXY_QUALITY_COL, PROXY_DIR_LONG_COL, PROXY_DIR_SHORT_COL
    else:
        qual_col = PROXY_QUALITY_COL.replace("_oof_", "_")
        long_col = PROXY_DIR_LONG_COL.replace("_oof_", "_")
        short_col = PROXY_DIR_SHORT_COL.replace("_oof_", "_")
        if qual_col not in header:
            raise RuntimeError(f"proxy columns not found in {pred_csv} (tried oof and non-oof naming)")
    src = pd.read_csv(pred_csv, usecols=["timestamp", qual_col, long_col, short_col])
    src = src.rename(columns={qual_col: PROXY_QUALITY_COL, long_col: PROXY_DIR_LONG_COL, short_col: PROXY_DIR_SHORT_COL})
    src["timestamp"] = pd.to_datetime(src["timestamp"])
    src = src[src["timestamp"].isin(keep_ts)].reset_index(drop=True)
    return src


def prep_split(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path, *, oof: bool) -> tuple[dict, pd.DataFrame]:
    p = sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)
    proxy = _load_proxy_columns(pred_csv, set(p["frame"]["timestamp"]))
    if len(proxy) != len(p["frame"]) or not proxy["timestamp"].equals(p["frame"]["timestamp"]):
        raise RuntimeError(f"{name}: proxy/frame timestamp mismatch ({len(proxy)} vs {len(p['frame'])})")
    return p, proxy


@torch.no_grad()
def build_trajectories(
    frame: pd.DataFrame, base_x: pd.DataFrame, dec: pd.DataFrame, loaded_models: dict[str, tuple],
    proxy: pd.DataFrame, *, risk_margin_fraction: np.ndarray, risk_leverage: np.ndarray,
    fee: float, slip: float, cost_mult: float, notional_scaled_sltp: bool, device: torch.device,
) -> list[dict[str, Any]]:
    """Causal bar-by-bar replay, IDENTICAL lifecycle to sweep.replay_exit_variant's baseline
    (TP/SL/exit_head@0.95). Records the full per-bar path (features, hold, move) of every trade
    from entry to its real baseline exit bar. fresh_forward_bar_by_bar=true; used ONLY as training
    path data, never consulted by the eval replay's own decisions."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
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
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)
    proxy_quality = proxy[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
    proxy_dir_long = proxy[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
    proxy_dir_short = proxy[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
    base_cols = list(base_x.columns)

    trajectories: list[dict[str, Any]] = []
    cur_rows: list[dict[str, float]] = []
    cur_holds: list[int] = []
    cur_moves: list[float] = []

    def flush_trade() -> None:
        if cur_rows:
            trajectories.append({
                "x": pd.DataFrame(cur_rows), "hold": np.asarray(cur_holds, dtype=np.int64),
                "move": np.asarray(cur_moves, dtype=np.float64),
            })
        cur_rows.clear()
        cur_holds.clear()
        cur_moves.clear()

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
            giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
            base_row = {base_cols[j]: float(base_np[i, j]) for j in range(len(base_cols))}
            base_row.update({
                "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                "proxy_quality_for_action": float(proxy_quality[i]),
                "proxy_dir_p_side": float(proxy_dir_long[i] if pos > 0 else proxy_dir_short[i]),
            })
            cur_rows.append(base_row)
            cur_holds.append(hold)
            cur_moves.append(float(move))
            if not reason:
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
                pos = 0
                flush_trade()
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

    if pos != 0:
        flush_trade()

    return trajectories


def _stop_decision_array(traj: dict[str, Any], buckets: np.ndarray, models: dict[int, Any], feature_cols: list[str]) -> np.ndarray:
    """Vectorized: for every bar j in this trajectory, True iff the ALREADY-TRAINED policy for
    j's bucket would stop there (terminal bucket = always True; a bucket with no fitted model
    (single-class/skipped) = never stops there; otherwise batched predict_proba >= 0.5)."""
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
        clf = models.get(b)
        if clf is None:
            continue
        rows = traj["x"].iloc[idx][feature_cols].to_numpy(dtype=np.float64)
        proba = clf.predict_proba(rows)[:, 1]
        stop[idx] = proba >= 0.5
    return stop


def _continuation_rewards(move: np.ndarray, stop_decision: np.ndarray) -> np.ndarray:
    """cont_reward[i] = move[j*] where j* = min{j>i : stop_decision[j]} else move[-1]. O(n)."""
    n = len(move)
    cont = np.empty(n, dtype=np.float64)
    running = float(move[-1])
    cont[n - 1] = running
    for j in range(n - 2, -1, -1):
        cont[j] = running
        if stop_decision[j + 1]:
            running = float(move[j + 1])
    return cont


def _bucket_training_rows(
    trajectories: list[dict[str, Any]], tau: int, models: dict[int, Any], feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for traj in trajectories:
        buckets = bucket_of(traj["hold"])
        idx_tau = np.where(buckets == tau)[0]
        if len(idx_tau) == 0:
            continue
        move = traj["move"]
        stop_decision = _stop_decision_array(traj, buckets, models, feature_cols)
        cont_reward = _continuation_rewards(move, stop_decision)
        stop_reward = move[idx_tau]
        labels = (stop_reward >= cont_reward[idx_tau]).astype(np.int64)
        xs.append(traj["x"].iloc[idx_tau][feature_cols].to_numpy(dtype=np.float64))
        ys.append(labels)
    if not xs:
        return np.empty((0, len(feature_cols))), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_dos_sequence(trajectories: list[dict[str, Any]], feature_cols: list[str]) -> dict[int, Any]:
    """Backward induction over buckets. Returns {bucket_id: fitted classifier or None (terminal /
    single-class skip)}."""
    models: dict[int, Any] = {TERMINAL_BUCKET: None}
    for tau in range(TERMINAL_BUCKET - 1, -1, -1):
        x, y = _bucket_training_rows(trajectories, tau, models, feature_cols)
        if len(np.unique(y)) < 2:
            print(f"stage=train_bucket tau={tau} SINGLE_CLASS n={len(y)} pos_rate={y.mean() if len(y) else float('nan')} -- skipping (treat as always-continue)", flush=True)
            models[tau] = None
            continue
        clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=260723)
        clf.fit(x, y)
        models[tau] = clf
        print(f"stage=train_bucket tau={tau} n={len(y)} pos_rate={y.mean():.4f}", flush=True)
    return models


def diagnostic_auc(trajectories: list[dict[str, Any]], feature_cols: list[str]) -> dict[str, Any]:
    """Trade-level chronological 80/20 split. Trains the backward-induction chain on the fit
    portion only, then reports held-out AUC per bucket using the SAME chain (continuation values
    also only ever computed from fit-side-trained models, applied to held-out trades' own realized
    paths -- no leakage of held-out trades into training)."""
    n = len(trajectories)
    n_fit = max(int(round(n * (1.0 - TRADE_HOLDOUT_FRAC))), 1)
    fit_traj = trajectories[:n_fit]
    hold_traj = trajectories[n_fit:]
    print(f"stage=diagnostic n_trades_total={n} n_fit={len(fit_traj)} n_holdout={len(hold_traj)}", flush=True)
    models = train_dos_sequence(fit_traj, feature_cols)

    results = {}
    for tau in range(TERMINAL_BUCKET - 1, -1, -1):
        clf = models.get(tau)
        if clf is None:
            results[tau] = {"status": "no_model_or_single_class"}
            continue
        x, y = _bucket_training_rows(hold_traj, tau, models, feature_cols)
        if len(x) == 0 or len(np.unique(y)) < 2:
            results[tau] = {"status": "no_holdout_data_or_single_class", "n": int(len(y))}
            continue
        proba = clf.predict_proba(x)[:, 1]
        auc = float(roc_auc_score(y, proba))
        results[tau] = {"status": "ok", "n": int(len(y)), "pos_rate": float(y.mean()), "auc": auc}
        print(f"stage=diagnostic_auc tau={tau} n={len(y)} pos_rate={y.mean():.4f} auc={auc:.4f}", flush=True)
    return {"n_trades_total": n, "n_fit": len(fit_traj), "n_holdout": len(hold_traj), "per_bucket": results}


@torch.no_grad()
def replay_with_dos(
    frame: pd.DataFrame, base_x: pd.DataFrame, dec: pd.DataFrame, loaded_models: dict[str, tuple],
    *, risk_margin_fraction: np.ndarray, risk_leverage: np.ndarray, fee: float, slip: float, cost_mult: float,
    notional_scaled_sltp: bool, device: torch.device, dos_models: dict[int, Any] | None, feature_cols: list[str],
    dos_stop_thr: float = 0.5, use_dos: bool = True, use_exit_head: bool = True,
    exit_threshold: float = sweep.BASELINE_EXIT_THRESHOLD,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Same TP/SL-first structure and fill/cost model as
    sweep.replay_exit_variant. When use_dos=True: after TP/SL, the trained DOS bucket sequence is
    checked (forced stop in the terminal bucket, else classifier prob>=dos_stop_thr) BEFORE (in
    place of, when use_exit_head=False) the exit-head@0.95 check. use_dos=False, use_exit_head=True
    reduces to the exact baseline (sanity-check config). fresh_forward_bar_by_bar=true; no saved
    ledger used as input."""
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
    proxy = None  # filled lazily below via closures on frame-level arrays passed by caller

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
                    clf = dos_models.get(tau)
                    if clf is not None:
                        giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                        base_row = {base_x.columns[j]: float(base_np[i, j]) for j in range(base_x.shape[1])}
                        base_row.update({
                            "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                            "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                            "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                            "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                            "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                            "proxy_quality_for_action": float(dos_proxy_quality[i]),
                            "proxy_dir_p_side": float(dos_proxy_dir_long[i] if pos > 0 else dos_proxy_dir_short[i]),
                        })
                        xrow = np.asarray([base_row[c] for c in feature_cols], dtype=np.float64).reshape(1, -1)
                        p_stop = float(clf.predict_proba(xrow)[0, 1])
                        if p_stop >= float(dos_stop_thr):
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


# module-level scratch arrays set by run_replay() before calling replay_with_dos() with use_dos=True
dos_proxy_quality = None
dos_proxy_dir_long = None
dos_proxy_dir_short = None

BASELINES = {
    ("h48qual", "VAL"): {"pnl": 5.45, "mdd": -11.62},
    ("zig075", "VAL"): {"pnl": 40.31, "mdd": -13.07},
    ("h48qual", "OOS"): {"pnl": 9.49, "mdd": -6.54},
    ("zig075", "OOS"): {"pnl": 17.89, "mdd": -11.01},
}


def beats_baseline(name: str, split: str, pnl: float, mdd: float) -> bool:
    b = BASELINES[(name, split)]
    return pnl > b["pnl"] and mdd > b["mdd"]  # mdd is negative; "beats" = smaller drawdown magnitude


def main() -> int:
    global dos_proxy_quality, dos_proxy_dir_long, dos_proxy_dir_short
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()

    train_frame = sweep.load_frame(TRAIN_START, TRAIN_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"TRAIN frame rows={len(train_frame)} VAL frame rows={len(val_frame)} OOS frame rows={len(oos_frame)}", flush=True)

    all_diag = {}
    all_val_results = {}
    any_val_winner = False
    winners = []

    for name, cfg in sweep.COMPONENTS.items():
        print(f"===== component={name} =====", flush=True)
        train_pred = sweep.EXT_PRED_DIR / name / f"train_predictions_{cfg['q_tag']}.csv"
        val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        oos_pred = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"

        p_train, proxy_train = prep_split(name, cfg, train_frame, train_pred, oof=True)
        p_val, proxy_val = prep_split(name, cfg, val_frame, val_pred, oof=True)
        p_oos, proxy_oos = prep_split(name, cfg, oos_frame, oos_pred, oof=False)

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

        # ---------------- Diagnostic gate ----------------
        diag = diagnostic_auc(traj_train, feature_cols)
        all_diag[name] = diag
        near_terminal = diag["per_bucket"].get(TERMINAL_BUCKET - 1, {})
        degenerate = near_terminal.get("status") == "ok" and abs(near_terminal.get("auc", 0.5) - 0.5) < 0.05
        print(f"stage=diagnostic_gate component={name} near_terminal_bucket_result={near_terminal} degenerate={degenerate}", flush=True)

        # ---------------- Train final DOS sequence on full TRAIN trajectories ----------------
        print(f"stage=train_final component={name}", flush=True)
        dos_models = train_dos_sequence(traj_train, feature_cols)
        with open(OUT_DIR / f"dos_models_{name}.pkl", "wb") as f:
            pickle.dump({"models": dos_models, "feature_cols": feature_cols, "bucket_edges": BUCKET_EDGES}, f)

        # ---------------- Sanity check on VAL: use_dos=False must reproduce baseline exactly ----------------
        dos_proxy_quality = proxy_val[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
        dos_proxy_dir_long = proxy_val[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
        dos_proxy_dir_short = proxy_val[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
        m_noop, _ = replay_with_dos(
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
            print(f"stage=STOP component={name} sanity check FAILED -- not proceeding to DOS VAL run", flush=True)
            continue

        if degenerate:
            print(f"stage=SKIP_VAL_GRID component={name} near-terminal bucket AUC is degenerate "
                  f"(matches round 10's AUC~0.5 failure mode) -- still running ONE DOS VAL config for completeness "
                  f"but not expecting a win", flush=True)

        # ---------------- VAL: DOS as full replacement of exit_head, plus DOS-as-addition variant ----------------
        val_rows = []
        for use_exit_head, tag in ((False, "dos_replace_exit_head"), (True, "dos_before_exit_head")):
            for thr in (0.5, 0.6, 0.7):
                m, _ = replay_with_dos(
                    p_val["frame"], p_val["x"], p_val["dec"], p_val["loaded"], risk_margin_fraction=p_val["margin"],
                    risk_leverage=p_val["leverage"], fee=p_val["fee"], slip=p_val["slip"], cost_mult=sweep.COST_MULT,
                    notional_scaled_sltp=p_val["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=dos_models,
                    feature_cols=feature_cols, dos_stop_thr=thr, use_dos=True, use_exit_head=use_exit_head,
                )
                row = {"component": name, "variant": tag, "dos_stop_thr": thr, **m}
                val_rows.append(row)
                cleared = beats_baseline(name, "VAL", m["pnl"], m["mdd"])
                print(f"  VAL component={name} variant={tag} thr={thr} pnl={m['pnl']:.2f} mdd={m['mdd']:.2f} "
                      f"trades={m['trades']} wr={m['wr']:.2f} cleared_vs_baseline={cleared}", flush=True)
                if cleared:
                    winners.append({"component": name, "variant": tag, "dos_stop_thr": thr})
                    any_val_winner = True
        val_df = pd.DataFrame(val_rows)
        val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
        val_df.to_csv(OUT_DIR / f"dos_VAL_{name}.csv", index=False)
        all_val_results[name] = val_rows

    with open(OUT_DIR / "diagnostic_report.json", "w") as f:
        json.dump(all_diag, f, indent=2, default=str)

    print(f"\nstage=val_summary any_val_winner={any_val_winner} winners={winners}", flush=True)

    if not any_val_winner:
        print("stage=done no_val_winners -- no OOS confirmation run (single-touch discipline)", flush=True)
        return 0

    # ---------------- Single OOS touch ----------------
    print("stage=oos_confirm", flush=True)
    oos_rows = []
    for w in winners:
        name = w["component"]
        cfg = sweep.COMPONENTS[name]
        oos_pred = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        p_oos, proxy_oos = prep_split(name, cfg, oos_frame, oos_pred, oof=False)
        with open(OUT_DIR / f"dos_models_{name}.pkl", "rb") as f:
            saved = pickle.load(f)
        dos_proxy_quality = proxy_oos[PROXY_QUALITY_COL].to_numpy(dtype=np.float64)
        dos_proxy_dir_long = proxy_oos[PROXY_DIR_LONG_COL].to_numpy(dtype=np.float64)
        dos_proxy_dir_short = proxy_oos[PROXY_DIR_SHORT_COL].to_numpy(dtype=np.float64)
        use_exit_head = w["variant"] == "dos_before_exit_head"
        m_cand, _ = replay_with_dos(
            p_oos["frame"], p_oos["x"], p_oos["dec"], p_oos["loaded"], risk_margin_fraction=p_oos["margin"],
            risk_leverage=p_oos["leverage"], fee=p_oos["fee"], slip=p_oos["slip"], cost_mult=sweep.COST_MULT,
            notional_scaled_sltp=p_oos["notional_scaled_sltp"], device=sweep.DEVICE, dos_models=saved["models"],
            feature_cols=saved["feature_cols"], dos_stop_thr=w["dos_stop_thr"], use_dos=True, use_exit_head=use_exit_head,
        )
        b = BASELINES[(name, "OOS")]
        cleared = beats_baseline(name, "OOS", m_cand["pnl"], m_cand["mdd"])
        row = {**w, "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"],
               "oos_wr": m_cand["wr"], "oos_baseline_pnl": b["pnl"], "oos_baseline_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  {w} -> OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "dos_OOS_confirm.csv", index=False)
    any_oos_winner = any(r["cleared_oos"] for r in oos_rows)
    print(f"stage=ALL_DONE any_oos_winner={any_oos_winner}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
