#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH live Omega4.6.1 exit logic, round 14 v2: add an AND-gate (adverse own-P&L
condition) to the round-14 market-anomaly circuit breaker
(scripts/research_eth_omega461_anomaly_circuit_breaker_20260723.py,
tmp/research_20260723/anomaly_circuit_breaker_20260723/).

Round 14 finding: the best VAL config (h48qual, regime_mag_w24 @ q0.98) beat baseline hugely on
VAL (PnL +26.47%/MDD -6.10% vs baseline +5.45%/-11.62%) but COLLAPSED on the single OOS touch
(PnL +1.17%/MDD -6.39% vs baseline +9.49%/-6.54%). Post-hoc trade-ledger inspection found the
breaker was force-flatting positions that were still net-positive or about to hit take-profit --
i.e. it fired on market-level anomaly alone, with no regard for whether the OPEN POSITION itself
was actually in trouble. zig075 had zero VAL winners at all in round 14.

This round adds exactly the diagnosed missing condition: force-exit now requires BOTH (a) the
market anomaly signal fires (identical signal computation, reused verbatim from round 14 --
regime_mag_w{12,24} and vol_zscore_w24_480, the three signals that showed real VAL promise) AND
(b) the position's OWN current unrealized P&L (`move`, exactly the price-move fraction already
computed every bar in the causal replay loop) is adverse, in one of 3 formulations:
  - move_lt_zero:  move < 0                              (loosest -- any unrealized loss)
  - atr_0.5:       move < -0.5 * atr_pct[i]               (half an ATR underwater)
  - atr_1.0:       move < -1.0 * atr_pct[i]               (a full ATR underwater, strictest)
atr_pct[i] is the SAME causal ATR-percentage series (atr_eval._atr_pct, trailing atr_window bars
ending at i, no future data) already used by the parent's own ATR-scaled TP/SL, reused verbatim
-- not recomputed or redefined.

Force-flat mechanism, replay harness, TP/SL/exit_head ordering, fill/cost model: byte-identical
to round 14's replay_circuit_breaker_variant except the trigger condition is now
`signal[i] >= threshold AND adverse(move, atr_pct[i])` instead of `signal[i] >= threshold` alone.

Mandatory sanity checks (both must reproduce the static exit_threshold=0.95 baseline bit-for-bit):
  1. signal threshold set unreachably high (signal component of the AND never true) -- identical
     check to round 14.
  2. adverse_mult set to an absurdly large value (adverse component of the AND never true, even
     though the signal alone would fire constantly) -- NEW, specific to this round's AND-gate.

Windows: VAL = 2025-10-01..2025-12-31 (same OOF-availability constraint as rounds 1-14). OOS =
2026-01-01..2026-03-31, single touch, only if a VAL config beats baseline on BOTH pnl and mdd.
Fresh window = 2026-04-01..(latest date covered by the extended prediction CSVs, currently
2026-07-12 -- the nominal 2026-07-21 upper bound is NOT reachable because
tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/*/oos_predictions_q0*.csv only extends
to 2026-07-12 09:00; this is reported explicitly, not silently truncated), diagnostic-only, run
AFTER OOS and never used for selection.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_anomaly_circuit_breaker_20260723 as cb1  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260723/anomaly_circuit_breaker_v2_and_condition_20260723"
BASELINE_EXIT_THRESHOLD = sweep.BASELINE_EXIT_THRESHOLD  # 0.95
DEVICE = sweep.DEVICE
COST_MULT = sweep.COST_MULT

FRESH_START, FRESH_END = "2026-04-01", "2026-07-12"  # bounded by extended prediction CSV coverage

SIGNAL_NAMES = ["regime_mag_w24", "regime_mag_w12", "vol_zscore_w24_480"]  # the 3 round-14 signals with real VAL promise
QUANTILES = [0.95, 0.98]  # reuse the round-14 quantiles that produced the winner (0.98) plus a looser one
ADVERSE_MODES: dict[str, float | None] = {"move_lt_zero": None, "atr_0.5": 0.5, "atr_1.0": 1.0}


def adverse_ok(mode: str, mult: float | None, move: float, atr_at_i: float) -> bool:
    if mode == "move_lt_zero":
        return move < 0.0
    return move < -float(mult) * float(atr_at_i)


@torch.no_grad()
def replay_circuit_breaker_and_variant(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    adverse_mode: str,
    adverse_mult: float | None,
    atr_pct: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Identical to round 14's replay_circuit_breaker_variant except the force-flat trigger is
    now `signal[i] >= threshold AND adverse_ok(mode, mult, move, atr_pct[i])`."""
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
            if float(signal[int(i)]) >= float(threshold) and adverse_ok(adverse_mode, adverse_mult, float(move), float(atr_pct[int(i)])):
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


def run_one(name: str, p: dict[str, Any], *, sig_name: str, signal: np.ndarray, threshold: float,
            adverse_mode: str, adverse_mult: float | None, atr_pct: np.ndarray, extra: dict[str, Any]) -> dict[str, Any]:
    m, _ledger = replay_circuit_breaker_and_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        signal=signal, threshold=threshold, adverse_mode=adverse_mode, adverse_mult=adverse_mult, atr_pct=atr_pct,
        fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )
    return {"component": name, "signal": sig_name, "threshold": threshold, "adverse_mode": adverse_mode,
            "adverse_mult": adverse_mult, **extra, **m,
            "fire_rate_pct": float(np.mean(signal >= threshold) * 100.0),
            "exit_reasons": json.dumps(m["exit_reasons"])}


def compute_signals(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "regime_mag_w24": cb1.compute_regime_magnitude(frame, window=24),
        "regime_mag_w12": cb1.compute_regime_magnitude(frame, window=12),
        "vol_zscore_w24_480": cb1.compute_vol_zscore(frame, short_window=24, long_window=480),
    }


def prep_all(frame: pd.DataFrame, pred_dir_key: str, *, oof: bool) -> dict[str, dict[str, Any]]:
    prepped: dict[str, dict[str, Any]] = {}
    for name, cfg in sweep.COMPONENTS.items():
        pred_csv = sweep.EXT_PRED_DIR / name / f"{pred_dir_key}_{cfg['q_tag']}.csv"
        prepped[name] = sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)
    return prepped


def baseline_metrics(prepped: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, p in prepped.items():
        m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        out[name] = m
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = prep_all(val_frame, "validation_predictions", oof=True)

    val_signals: dict[str, dict[str, np.ndarray]] = {}
    val_atr: dict[str, np.ndarray] = {}
    for name, p in val_prepped.items():
        f = p["frame"]
        val_signals[name] = compute_signals(f)
        val_atr[name] = atr_eval._atr_pct(f, sweep.COMPONENTS[name]["atr_window"])

    baseline_val = baseline_metrics(val_prepped)
    print("baseline VAL:", {k: {"pnl": v["pnl"], "mdd": v["mdd"], "trades": v["trades"]} for k, v in baseline_val.items()}, flush=True)

    # --- Sanity checks (both must reproduce baseline bit-for-bit) ---
    print("stage=sanity", flush=True)
    sanity_rows = []
    for name, p in val_prepped.items():
        sig = val_signals[name]["regime_mag_w24"]
        atr_pct = val_atr[name]
        b = baseline_val[name]
        # (1) signal component never true (unreachable threshold)
        unreachable_thr = float(np.nanmax(sig)) + 1.0e6
        r1 = run_one(name, p, sig_name="regime_mag_w24", signal=sig, threshold=unreachable_thr,
                     adverse_mode="move_lt_zero", adverse_mult=None, atr_pct=atr_pct, extra={"sanity": "signal_unreachable"})
        sanity_rows.append({"component": name, "check": "signal_unreachable", "baseline_pnl": b["pnl"], "variant_pnl": r1["pnl"],
                             "baseline_mdd": b["mdd"], "variant_mdd": r1["mdd"], "baseline_trades": b["trades"], "variant_trades": r1["trades"]})
        # (2) adverse component never true (signal threshold reachable, but adverse_mult absurd)
        reachable_thr = float(np.nanquantile(sig, 0.95))
        r2 = run_one(name, p, sig_name="regime_mag_w24", signal=sig, threshold=reachable_thr,
                     adverse_mode="atr_0.5", adverse_mult=1.0e9, atr_pct=atr_pct, extra={"sanity": "adverse_unreachable"})
        sanity_rows.append({"component": name, "check": "adverse_unreachable", "baseline_pnl": b["pnl"], "variant_pnl": r2["pnl"],
                             "baseline_mdd": b["mdd"], "variant_mdd": r2["mdd"], "baseline_trades": b["trades"], "variant_trades": r2["trades"]})
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUT_DIR / "sanity_checks_VAL.csv", index=False)
    print(sanity_df.to_string(index=False), flush=True)
    for row in sanity_rows:
        if abs(row["baseline_pnl"] - row["variant_pnl"]) > 0.01 or abs(row["baseline_mdd"] - row["variant_mdd"]) > 0.01 or row["baseline_trades"] != row["variant_trades"]:
            print(f"SANITY CHECK FAILED: {row}", flush=True)
            return 1
    print("sanity checks PASSED (both unreachable-AND-component configs reproduce baseline)", flush=True)

    # --- VAL grid: signal x quantile-threshold x adverse formulation/strictness x component ---
    print("stage=val_grid", flush=True)
    grid_rows = []
    for name, p in val_prepped.items():
        atr_pct = val_atr[name]
        for sig_name in SIGNAL_NAMES:
            sig = val_signals[name][sig_name]
            for q in QUANTILES:
                thr = float(np.nanquantile(sig, q))
                for adverse_mode, adverse_mult in ADVERSE_MODES.items():
                    grid_rows.append(run_one(name, p, sig_name=sig_name, signal=sig, threshold=thr,
                                              adverse_mode=adverse_mode, adverse_mult=adverse_mult, atr_pct=atr_pct,
                                              extra={"quantile": q}))
    val_grid = pd.DataFrame(grid_rows)
    val_grid.to_csv(OUT_DIR / "circuit_breaker_and_grid_VAL.csv", index=False)
    print(val_grid[["component", "signal", "quantile", "adverse_mode", "fire_rate_pct", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

    winners = []
    for _, r in val_grid.iterrows():
        b = baseline_val[r["component"]]
        if r["pnl"] > b["pnl"] and r["mdd"] > b["mdd"]:  # mdd is negative; "beats" means less negative
            winners.append(r.to_dict())
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(OUT_DIR / "val_winners.csv", index=False)
    print(f"VAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    if len(winners):
        print(winners_df[["component", "signal", "quantile", "adverse_mode", "pnl", "mdd", "trades"]].to_string(index=False), flush=True)

    if not len(winners):
        print("stage=done no_val_winners -- skipping OOS run per established discipline (round 4/8/11/14 precedent)", flush=True)
        return 0

    # --- Single OOS touch, only for VAL-winning configs (best pnl per component). ---
    print("stage=oos_confirm", flush=True)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = prep_all(oos_frame, "oos_predictions", oof=False)
    baseline_oos = baseline_metrics(oos_prepped)

    oos_signals: dict[str, dict[str, np.ndarray]] = {}
    oos_atr: dict[str, np.ndarray] = {}
    for name, p in oos_prepped.items():
        f = p["frame"]
        oos_signals[name] = compute_signals(f)
        oos_atr[name] = atr_eval._atr_pct(f, sweep.COMPONENTS[name]["atr_window"])

    best_by_component: dict[str, dict[str, Any]] = {}
    for w in winners:
        comp = w["component"]
        if comp not in best_by_component or w["pnl"] > best_by_component[comp]["pnl"]:
            best_by_component[comp] = w

    oos_rows = []
    fresh_rows = []
    fresh_prepped_cache: dict[str, dict[str, Any]] = {}
    fresh_signals_cache: dict[str, dict[str, np.ndarray]] = {}
    fresh_atr_cache: dict[str, np.ndarray] = {}
    for comp, w in best_by_component.items():
        p = oos_prepped[comp]
        sig_name = w["signal"]
        adverse_mode = w["adverse_mode"]
        adverse_mult = w["adverse_mult"]
        oos_sig = oos_signals[comp][sig_name]
        oos_thr_val_value = float(w["threshold"])
        oos_m = run_one(comp, p, sig_name=sig_name, signal=oos_sig, threshold=oos_thr_val_value,
                         adverse_mode=adverse_mode, adverse_mult=adverse_mult, atr_pct=oos_atr[comp],
                         extra={"quantile": w["quantile"]})
        b = baseline_oos[comp]
        oos_rows.append({**oos_m, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"], "baseline_trades": b["trades"],
                          "beats_baseline_both": bool(oos_m["pnl"] > b["pnl"] and oos_m["mdd"] > b["mdd"])})

        # --- Fresh window (2026-04-01..07-12), diagnostic-only, NOT selection-influencing ---
        if comp not in fresh_prepped_cache:
            fresh_frame = sweep.load_frame(FRESH_START, FRESH_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
            fresh_prepped = prep_all(fresh_frame, "oos_predictions", oof=False)
            for c2, p2 in fresh_prepped.items():
                fresh_prepped_cache[c2] = p2
                f2 = p2["frame"]
                fresh_signals_cache[c2] = compute_signals(f2)
                fresh_atr_cache[c2] = atr_eval._atr_pct(f2, sweep.COMPONENTS[c2]["atr_window"])
        pf = fresh_prepped_cache[comp]
        print(f"fresh frame component={comp} rows={len(pf['frame'])} range=[{pf['frame']['timestamp'].min()}, {pf['frame']['timestamp'].max()}]", flush=True)
        fresh_sig = fresh_signals_cache[comp][sig_name]
        fresh_m = run_one(comp, pf, sig_name=sig_name, signal=fresh_sig, threshold=oos_thr_val_value,
                           adverse_mode=adverse_mode, adverse_mult=adverse_mult, atr_pct=fresh_atr_cache[comp],
                           extra={"quantile": w["quantile"]})
        baseline_fresh_m, _ = sweep.replay_exit_variant(
            pf["frame"], pf["x"], pf["dec"], pf["loaded"], risk_margin_fraction=pf["margin"], risk_leverage=pf["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=pf["fee"], slip=pf["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=pf["notional_scaled_sltp"], device=DEVICE,
        )
        fresh_rows.append({**fresh_m, "baseline_pnl": baseline_fresh_m["pnl"], "baseline_mdd": baseline_fresh_m["mdd"],
                            "baseline_trades": baseline_fresh_m["trades"]})

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print(oos_df.to_string(index=False), flush=True)

    fresh_df = pd.DataFrame(fresh_rows)
    fresh_df.to_csv(OUT_DIR / "fresh_window_diagnostic.csv", index=False)
    print(f"fresh window diagnostic (2026-{FRESH_START}..{FRESH_END}, NOT selection-influencing):", flush=True)
    print(fresh_df.to_string(index=False), flush=True)

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
