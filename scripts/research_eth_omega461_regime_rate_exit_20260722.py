#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH live Omega4.6.1 exit logic, round 11: regime CHANGE-RATE adaptive
EXIT_THRESHOLD (not a static chop/bull/bear state veto -- round 3 on 2026-07-21 already tried
that and found only marginal/noise-level improvement).

Signal: rolling flip-rate of the argmax(bull_prob, bear_prob, chop_prob) regime label over a
trailing causal window, from the SAME regime3 "current-HMM sensitive wide24" feature file round
3 used (regime3_current_sensitive_wide24_{bull,bear,chop}_prob in
data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/). High flip-rate =
regime is churning fast (uncertain, transitioning) -> lower the exit-head probability gate (more
willing to fire, protect gains sooner). Low/stable flip-rate = regime is settled -> stays at the
baseline EXIT_THRESHOLD=0.95 (let winners run, matching current live behavior).

Modulation mirrors the existing exit_trend_threshold_scale mechanism in
train_eval_omega4_2_risk_sidecar_20260622.py:389-398 (effective_exit_threshold =
clip(exit_threshold + scale * signal, floor, cap)), but the signal here is regime_change_rate
instead of a trend-context difference, and the direction is inverted (higher rate -> lower
threshold, so we SUBTRACT k * rate and cap at the baseline 0.95 rather than allowing overshoot
above it):

    effective_threshold[i] = clip(0.95 - k * regime_change_rate[i], floor, 0.95)

Reuses the Omega461LiveAdapter-based fresh-forward causal replay loop built for rounds 1-2
(research_eth_omega461_exit_sweep_20260721.py) verbatim except for the threshold-gate line --
same TP/SL/exit-head order, same fill/cost model, same COMPONENTS bundle/sidecar/ATR config,
same VAL/OOS windows and frozen prediction CSVs.

Windows: VAL = 2025-10-01..2025-12-31 (this model's OOF predictions don't exist before
2025-10-01; 2025-09 is inside its own TRAIN split -- same VAL-window note as rounds 1-2, not
silently "fixed" back to the canonical 2025-09-01 start). OOS = 2026-01-01..2026-03-31, single
touch, only run if a VAL config beats baseline on BOTH PnL and MDD for a component.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false: regime_change_rate[i]
is a rolling function of bars <= i only (causal), matching the frozen replay loop's per-bar
forward order.
"""

from __future__ import annotations

import json
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
OUT_DIR = ROOT / "tmp/research_20260722/regime_rate_adaptive_exit_20260722"
BASELINE_EXIT_THRESHOLD = sweep.BASELINE_EXIT_THRESHOLD  # 0.95
DEVICE = sweep.DEVICE
COST_MULT = sweep.COST_MULT


def compute_regime_change_rate(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Causal rolling flip-rate of argmax(bull,bear,chop) over a trailing window ending at i
    (inclusive). rate[i] in [0, 1] = fraction of the last `window` bar-to-bar transitions (up to
    and including bar i) where the regime label changed. Only uses frame rows <= i."""
    bull = pd.to_numeric(frame[f"{PFX}bull_prob"], errors="raise").to_numpy(dtype=np.float64)
    bear = pd.to_numeric(frame[f"{PFX}bear_prob"], errors="raise").to_numpy(dtype=np.float64)
    chop = pd.to_numeric(frame[f"{PFX}chop_prob"], errors="raise").to_numpy(dtype=np.float64)
    labels = np.argmax(np.stack([bull, bear, chop], axis=1), axis=1)
    flips = np.zeros(len(labels), dtype=np.float64)
    flips[1:] = (labels[1:] != labels[:-1]).astype(np.float64)
    rate = pd.Series(flips).rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64) / float(window)
    return rate


@torch.no_grad()
def replay_regime_rate_variant(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    regime_rate: np.ndarray,
    k: float,
    floor: float,
    ceiling: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Identical causal bar-by-bar replay to sweep.replay_exit_variant (same TP/SL/exit-head
    order, same fill/cost model), with the static exit_threshold replaced by
    effective_threshold[i] = clip(ceiling - k * regime_rate[i], floor, ceiling)."""
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
            if take_profit > 0.0 and move >= take_profit:
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
                effective_threshold = float(np.clip(ceiling - float(k) * float(regime_rate[int(i)]), floor, ceiling))
                if prob >= effective_threshold:
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


def run_one(name: str, p: dict[str, Any], *, window: int, k: float, floor: float, ceiling: float) -> dict[str, Any]:
    rate = compute_regime_change_rate(p["frame"], window=window)
    m, _ledger = replay_regime_rate_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        regime_rate=rate, k=k, floor=floor, ceiling=ceiling, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )
    return {"component": name, "window": window, "k": k, "floor": floor, "ceiling": ceiling, **m,
            "exit_reasons": json.dumps(m["exit_reasons"])}


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

    # --- Sanity check (a): no-op, k=0.0 must reproduce the static exit_threshold=0.95 baseline
    # bit-for-bit (within 0.01 tolerance). ---
    print("stage=sanity_noop", flush=True)
    sanity_rows = []
    for name, p in val_prepped.items():
        baseline_m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        noop = run_one(name, p, window=24, k=0.0, floor=BASELINE_EXIT_THRESHOLD, ceiling=BASELINE_EXIT_THRESHOLD)
        impossible = run_one(name, p, window=24, k=0.30, floor=BASELINE_EXIT_THRESHOLD, ceiling=BASELINE_EXIT_THRESHOLD)
        sanity_rows.append({"component": name, "check": "a_k_zero", "baseline_pnl": baseline_m["pnl"],
                             "variant_pnl": noop["pnl"], "baseline_mdd": baseline_m["mdd"], "variant_mdd": noop["mdd"],
                             "baseline_trades": baseline_m["trades"], "variant_trades": noop["trades"]})
        sanity_rows.append({"component": name, "check": "b_floor_eq_ceiling", "baseline_pnl": baseline_m["pnl"],
                             "variant_pnl": impossible["pnl"], "baseline_mdd": baseline_m["mdd"], "variant_mdd": impossible["mdd"],
                             "baseline_trades": baseline_m["trades"], "variant_trades": impossible["trades"]})
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUT_DIR / "sanity_checks_VAL.csv", index=False)
    print(sanity_df.to_string(index=False), flush=True)
    for row in sanity_rows:
        if abs(row["baseline_pnl"] - row["variant_pnl"]) > 0.01 or abs(row["baseline_mdd"] - row["variant_mdd"]) > 0.01 or row["baseline_trades"] != row["variant_trades"]:
            print(f"SANITY CHECK FAILED: {row}", flush=True)
            return 1
    print("sanity checks PASSED (both a and b reproduce baseline within tolerance)", flush=True)

    # --- VAL-only grid: window in {12, 24}, k in {0.05, 0.15, 0.30}, floor=0.80, ceiling=0.95
    # (6 configs per component, 12 total). ---
    print("stage=val_grid", flush=True)
    grid_rows = []
    for name, p in val_prepped.items():
        for window in (12, 24):
            for k in (0.05, 0.15, 0.30):
                grid_rows.append(run_one(name, p, window=window, k=k, floor=0.80, ceiling=BASELINE_EXIT_THRESHOLD))
    val_grid = pd.DataFrame(grid_rows)
    val_grid.to_csv(OUT_DIR / "regime_rate_grid_VAL.csv", index=False)
    print(val_grid[["component", "window", "k", "floor", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

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
        print(winners_df[["component", "window", "k", "floor", "pnl", "mdd", "trades"]].to_string(index=False), flush=True)

    if not len(winners):
        print("stage=done no_val_winners -- skipping OOS run per established discipline (round 4/8 precedent)", flush=True)
        return 0

    # --- Single OOS touch, only for VAL-winning configs (best pnl per component). ---
    print("stage=oos_confirm", flush=True)
    oos_rows = []
    best_by_component: dict[str, dict[str, Any]] = {}
    for w in winners:
        comp = w["component"]
        if comp not in best_by_component or w["pnl"] > best_by_component[comp]["pnl"]:
            best_by_component[comp] = w
    for comp, w in best_by_component.items():
        p = oos_prepped[comp]
        oos_m = run_one(comp, p, window=int(w["window"]), k=float(w["k"]), floor=float(w["floor"]), ceiling=BASELINE_EXIT_THRESHOLD)
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
