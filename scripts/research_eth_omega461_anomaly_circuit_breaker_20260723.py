#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH live Omega4.6.1 exit logic, round 14: MARKET-LEVEL emergency circuit
breaker (a new mechanism CATEGORY, not a 14th trade-level variant).

Rounds 1-13 (see project memory `project-eth-omega461-exit-logic-experiments-20260721` plus
today's rounds 11-13 in tmp/research_20260722/ and tmp/research_20260723/) all tried to predict
something about an INDIVIDUAL TRADE's own path (giveback, reversal, DP-optimal continuation
value, regime flip-COUNT) using only the ~30-70 realized trade trajectories per component as
training/tuning signal -- fundamentally data-starved.

This round instead builds a detector trained/computed on EVERY 5m bar in the window (tens of
thousands of rows, not tens of trades) that flags anomalous/dangerous MARKET conditions,
independent of any specific trade's own P&L state. When the detector fires while a position
happens to be open, it force-flats the position immediately, overriding TP/SL/exit_head for
that bar (checked FIRST, before TP/SL/exit_head, per the task spec).

Three candidate signals, all computed once per bar over the full frame (cheap, no per-trade
fitting):
  1. realized-vol z-score spike -- pure rule-based/causal, rolling short-window realized vol of
     log-returns vs its own longer trailing-window mean/std. No training.
  2. regime3 transition MAGNITUDE -- bar-over-bar L1 change in
     (bull_prob, bear_prob, chop_prob) from the current-HMM sensitive wide24 overlay
     (regime3_current_sensitive_wide24_{bull,bear,chop}_prob), rolling-summed over a trailing
     window. Different from round 11 (2026-07-22)'s flip-RATE-of-argmax-label signal -- this is
     the continuous magnitude of probability movement, not a discrete label-flip count.
  3. GMM volatility-regime rank -- REUSED archived unsupervised detector
     (data/ensemble/unsupervised/gmm_volatility.pkl, a GaussianMixture over
     [bb_width_z, garch_vol_z, realized_vol_ratio, rogers_satchell_vol, garman_klass_vol,
     parkinson_vol] with a cluster_rank_map into 0=calm..5=extreme). All 6 feature columns exist
     in the base 2025/2026 feature CSVs already used by the rest of the ETH pipeline, so this is
     effectively free to score per-bar (single .predict() call, no retraining). The archived
     isolation_forest.pkl was checked but is NOT usable as-is: 2 of its 6 feature_cols
     (sig_whale, sig_oi_divergence) are absent from the base feature CSVs.

Force-flat mechanism (identical causal bar-by-bar replay to
research_eth_omega461_exit_sweep_20260721.replay_exit_variant / round 11's
replay_regime_rate_variant -- same TP/SL/exit-head order otherwise, same fill/cost model), with
one new FIRST check per bar with an open position:
    if emergency_signal[i] >= threshold: reason = "circuit_breaker"  (checked BEFORE TP/SL/exit_head)

Mandatory sanity check: threshold set unreachably high (signal can never cross it) must
reproduce the static exit_threshold=0.95 baseline bit-for-bit (PnL/MDD/trade-count within
tolerance).

Windows: VAL = 2025-10-01..2025-12-31 (same VAL-window note as rounds 1-13: this model's OOF
predictions don't exist before 2025-10-01, the canonical 2025-09-01 start would leak into the
parent's own TRAIN split). OOS = 2026-01-01..2026-03-31, single touch, only if a VAL config beats
baseline on BOTH PnL and MDD for a component.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. All three signals are
causal (trailing rolling windows / instantaneous per-bar scoring of already-causal feature
columns) with one EXCEPTION scoped strictly to a diagnostic printed before the grid (not used in
any replay decision): a bar-level "would a hypothetical position have taken an adverse move in
the next K bars" label, used only to sanity-check the signals' correlation with genuinely bad
market conditions before spending grid time on them. That label uses future bars by
construction (it's a look-ahead diagnostic label, exactly as specified in the task's design
guidance point 2) and is NEVER read by the replay loop or by any threshold/decision logic.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

PFX = "regime3_current_sensitive_wide24_"
OUT_DIR = ROOT / "tmp/research_20260723/anomaly_circuit_breaker_20260723"
BASELINE_EXIT_THRESHOLD = sweep.BASELINE_EXIT_THRESHOLD  # 0.95
DEVICE = sweep.DEVICE
COST_MULT = sweep.COST_MULT
GMM_PKL = ROOT / "data/ensemble/unsupervised/gmm_volatility.pkl"


# ---------------------------------------------------------------------------
# Candidate emergency signals (all causal, computed once per bar over the full frame)
# ---------------------------------------------------------------------------

def compute_vol_zscore(frame: pd.DataFrame, short_window: int, long_window: int) -> np.ndarray:
    """Causal: realized_vol_short[i] = std of log-returns over the trailing short_window bars
    ending at i. z[i] = (realized_vol_short[i] - trailing long_window mean) / trailing
    long_window std, both computed over realized_vol_short itself (also causal, ending at i)."""
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    logret = np.zeros(len(close), dtype=np.float64)
    logret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
    rv = pd.Series(logret).rolling(window=short_window, min_periods=short_window).std().to_numpy(dtype=np.float64)
    rv_s = pd.Series(rv)
    mu = rv_s.rolling(window=long_window, min_periods=max(long_window // 4, 8)).mean().to_numpy(dtype=np.float64)
    sd = rv_s.rolling(window=long_window, min_periods=max(long_window // 4, 8)).std().to_numpy(dtype=np.float64)
    sd = np.where((sd < 1e-10) | np.isnan(sd), 1.0, sd)
    z = (rv - mu) / sd
    return np.nan_to_num(z, nan=0.0)


def compute_regime_magnitude(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Causal: bar-over-bar L1 change in (bull_prob, bear_prob, chop_prob), rolling-summed over
    a trailing window ending at i. Continuous magnitude, not a discrete argmax label flip."""
    bull = pd.to_numeric(frame[f"{PFX}bull_prob"], errors="raise").to_numpy(dtype=np.float64)
    bear = pd.to_numeric(frame[f"{PFX}bear_prob"], errors="raise").to_numpy(dtype=np.float64)
    chop = pd.to_numeric(frame[f"{PFX}chop_prob"], errors="raise").to_numpy(dtype=np.float64)
    delta = np.zeros(len(bull), dtype=np.float64)
    delta[1:] = np.abs(np.diff(bull)) + np.abs(np.diff(bear)) + np.abs(np.diff(chop))
    mag = pd.Series(delta).rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64)
    return mag


def compute_gmm_rank(frame: pd.DataFrame) -> np.ndarray:
    """Reused archived unsupervised detector (no retraining). Instantaneous per-bar cluster rank
    (0=calm..5=extreme) from the frozen GaussianMixture, scored on already-causal feature
    columns (no future bars used in any single row's own feature computation)."""
    with open(GMM_PKL, "rb") as f:
        pkl = pickle.load(f)
    cols = pkl["feature_cols"]
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    mean = np.asarray(pkl["mean"], dtype=np.float64)
    std = np.asarray(pkl["std"], dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x - mean) / std
    cluster = pkl["model"].predict(x_norm)
    rank_map = pkl["cluster_rank_map"]
    rank = np.array([rank_map[int(c)] for c in cluster], dtype=np.float64)
    return rank


# ---------------------------------------------------------------------------
# Force-flat causal replay: identical to sweep.replay_exit_variant except a new FIRST check
# (before TP/SL/exit_head) when a position is open.
# ---------------------------------------------------------------------------

@torch.no_grad()
def replay_circuit_breaker_variant(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
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
            exit_prob = 0.0
            if float(signal[int(i)]) >= float(threshold):
                reason = "circuit_breaker"
            elif take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
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
                exit_prob = float(prob)
                if prob >= float(BASELINE_EXIT_THRESHOLD):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
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
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
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
        filled, px, fee_paid, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
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
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
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
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
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
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


def run_one(name: str, p: dict[str, Any], *, sig_name: str, signal: np.ndarray, threshold: float, extra: dict[str, Any]) -> dict[str, Any]:
    m, _ledger = replay_circuit_breaker_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        signal=signal, threshold=threshold, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )
    return {"component": name, "signal": sig_name, "threshold": threshold, **extra, **m,
            "fire_rate_pct": float(np.mean(signal >= threshold) * 100.0),
            "exit_reasons": json.dumps(m["exit_reasons"])}


def diagnostic_label_correlation(frame: pd.DataFrame, signals: dict[str, np.ndarray], K: int = 24, adverse_thresh: float = 0.03) -> pd.DataFrame:
    """LOOK-AHEAD DIAGNOSTIC ONLY, never fed into the replay loop or any decision. For every bar
    i, label bad[i]=1 if a hypothetical position opened at i (either direction) would see an
    adverse move >= adverse_thresh within the next K bars. Reports base rate and simple
    point-biserial-style mean-signal-given-label split, to sanity check the signals before
    spending grid time."""
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    n = len(close)
    bad = np.zeros(n, dtype=np.int64)
    for i in range(n - K):
        c0 = close[i]
        fwd_low = low[i + 1: i + 1 + K].min()
        fwd_high = high[i + 1: i + 1 + K].max()
        long_adverse = (fwd_low - c0) / c0
        short_adverse = (c0 - fwd_high) / c0
        if long_adverse <= -adverse_thresh or short_adverse <= -adverse_thresh:
            bad[i] = 1
    rows = []
    base_rate = float(bad[: n - K].mean()) if n > K else 0.0
    for sig_name, sig in signals.items():
        s = sig[: n - K]
        b = bad[: n - K]
        mean_bad = float(s[b == 1].mean()) if (b == 1).any() else float("nan")
        mean_good = float(s[b == 0].mean()) if (b == 0).any() else float("nan")
        rows.append({"signal": sig_name, "n_bars": int(n - K), "base_rate_pct": base_rate * 100.0,
                     "mean_signal_given_bad": mean_bad, "mean_signal_given_good": mean_good})
    return pd.DataFrame(rows)


def main() -> int:
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)

    val_prepped: dict[str, dict[str, Any]] = {}
    oos_prepped: dict[str, dict[str, Any]] = {}
    for name, cfg in sweep.COMPONENTS.items():
        val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        oos_pred_full = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        print(f"stage=prep component={name} split=VAL", flush=True)
        val_prepped[name] = sweep.prep_component(name, cfg, val_frame, val_pred, oof=True)
        print(f"stage=prep component={name} split=OOS", flush=True)
        oos_prepped[name] = sweep.prep_component(name, cfg, oos_frame, oos_pred_full, oof=False)

    # prepped frames may be filtered/re-indexed inside prep_component (timestamp intersection
    # with the frozen prediction CSVs); recompute signals off each component's OWN post-prep
    # frame so lengths/indices line up exactly with what the replay consumes.
    val_signals: dict[str, dict[str, np.ndarray]] = {}
    for name, p in val_prepped.items():
        f = p["frame"]
        val_signals[name] = {
            "vol_zscore_w12_240": compute_vol_zscore(f, short_window=12, long_window=240),
            "vol_zscore_w24_480": compute_vol_zscore(f, short_window=24, long_window=480),
            "regime_mag_w12": compute_regime_magnitude(f, window=12),
            "regime_mag_w24": compute_regime_magnitude(f, window=24),
            "gmm_rank": compute_gmm_rank(f),
        }

    # --- Diagnostic (look-ahead label, NOT used by any replay decision): base rate + simple
    # bad-vs-good signal-mean split, to sanity-check signals before grid search. ---
    print("stage=diagnostic_label_correlation", flush=True)
    diag_rows = []
    for name, p in val_prepped.items():
        d = diagnostic_label_correlation(p["frame"], val_signals[name], K=24, adverse_thresh=0.03)
        d.insert(0, "component", name)
        diag_rows.append(d)
    diag_df = pd.concat(diag_rows, ignore_index=True)
    diag_df.to_csv(OUT_DIR / "diagnostic_label_correlation_VAL.csv", index=False)
    print(diag_df.to_string(index=False), flush=True)

    # --- Sanity check: threshold set unreachably high must reproduce the static
    # exit_threshold=0.95 baseline bit-for-bit. ---
    print("stage=sanity_noop", flush=True)
    sanity_rows = []
    for name, p in val_prepped.items():
        baseline_m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        for sig_name, sig in val_signals[name].items():
            unreachable = float(np.nanmax(sig)) + 1.0e6
            noop = run_one(name, p, sig_name=sig_name, signal=sig, threshold=unreachable, extra={})
            sanity_rows.append({"component": name, "signal": sig_name, "baseline_pnl": baseline_m["pnl"],
                                 "variant_pnl": noop["pnl"], "baseline_mdd": baseline_m["mdd"], "variant_mdd": noop["mdd"],
                                 "baseline_trades": baseline_m["trades"], "variant_trades": noop["trades"]})
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUT_DIR / "sanity_checks_VAL.csv", index=False)
    print(sanity_df.to_string(index=False), flush=True)
    for row in sanity_rows:
        if abs(row["baseline_pnl"] - row["variant_pnl"]) > 0.01 or abs(row["baseline_mdd"] - row["variant_mdd"]) > 0.01 or row["baseline_trades"] != row["variant_trades"]:
            print(f"SANITY CHECK FAILED: {row}", flush=True)
            return 1
    print("sanity checks PASSED (unreachable threshold reproduces baseline within tolerance)", flush=True)

    # --- VAL-only grid: for each signal, sweep a handful of quantile-based thresholds so the
    # fire-rate is comparable/interpretable across signals with different scales. ---
    print("stage=val_grid", flush=True)
    quantiles = [0.999, 0.995, 0.99, 0.98, 0.95]
    grid_rows = []
    for name, p in val_prepped.items():
        for sig_name, sig in val_signals[name].items():
            for q in quantiles:
                thr = float(np.nanquantile(sig, q))
                grid_rows.append(run_one(name, p, sig_name=sig_name, signal=sig, threshold=thr, extra={"quantile": q}))
    val_grid = pd.DataFrame(grid_rows)
    val_grid.to_csv(OUT_DIR / "circuit_breaker_grid_VAL.csv", index=False)
    print(val_grid[["component", "signal", "quantile", "threshold", "fire_rate_pct", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

    baseline_val = {}
    for name, p in val_prepped.items():
        m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        baseline_val[name] = m
    print("baseline VAL:", {k: {"pnl": v["pnl"], "mdd": v["mdd"], "trades": v["trades"]} for k, v in baseline_val.items()}, flush=True)

    winners = []
    for _, r in val_grid.iterrows():
        b = baseline_val[r["component"]]
        if r["pnl"] > b["pnl"] and r["mdd"] > b["mdd"]:  # mdd is negative; "beats" means less negative
            winners.append(r.to_dict())
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(OUT_DIR / "val_winners.csv", index=False)
    print(f"VAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    if len(winners):
        print(winners_df[["component", "signal", "quantile", "threshold", "pnl", "mdd", "trades"]].to_string(index=False), flush=True)

    if not len(winners):
        print("stage=done no_val_winners -- skipping OOS run per established discipline (round 4/8/11 precedent)", flush=True)
        return 0

    # --- Single OOS touch, only for VAL-winning configs (best pnl per component). ---
    print("stage=oos_confirm", flush=True)
    oos_signals: dict[str, dict[str, np.ndarray]] = {}
    for name, p in oos_prepped.items():
        f = p["frame"]
        oos_signals[name] = {
            "vol_zscore_w12_240": compute_vol_zscore(f, short_window=12, long_window=240),
            "vol_zscore_w24_480": compute_vol_zscore(f, short_window=24, long_window=480),
            "regime_mag_w12": compute_regime_magnitude(f, window=12),
            "regime_mag_w24": compute_regime_magnitude(f, window=24),
            "gmm_rank": compute_gmm_rank(f),
        }

    oos_rows = []
    best_by_component: dict[str, dict[str, Any]] = {}
    for w in winners:
        comp = w["component"]
        if comp not in best_by_component or w["pnl"] > best_by_component[comp]["pnl"]:
            best_by_component[comp] = w
    for comp, w in best_by_component.items():
        p = oos_prepped[comp]
        sig_name = w["signal"]
        # re-derive the VAL threshold value at the SAME quantile on OOS's own signal
        # distribution (thresholds are quantile-calibrated per split, consistent with how the
        # VAL grid picked them; the winning VAL threshold value itself is also reported for
        # comparison).
        oos_sig = oos_signals[comp][sig_name]
        oos_thr_val_value = float(w["threshold"])
        oos_m = run_one(comp, p, sig_name=sig_name, signal=oos_sig, threshold=oos_thr_val_value, extra={"quantile": w["quantile"]})
        baseline_oos_m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        oos_rows.append({**oos_m, "baseline_pnl": baseline_oos_m["pnl"], "baseline_mdd": baseline_oos_m["mdd"], "baseline_trades": baseline_oos_m["trades"]})
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print(oos_df.to_string(index=False), flush=True)

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
