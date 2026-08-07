#!/usr/bin/env python3
"""RESEARCH ONLY -- adapt the V31/V49 pyramid gate design (scripts/eval_hf_v13_v49_profit_state_pyramid_v56.py,
an UNRELATED, older, non-live model lineage that used this gate to trigger scale-IN/pyramiding
adds) as a SCALE-OUT trigger for the LIVE ETH Omega4.6.1 h48qual/zig075 components.

That model's gate (V56Config / backtest_v56 in eval_hf_v13_v49_profit_state_pyramid_v56.py) added
to a winning position only when ALL of: unrealized profit above a threshold, MFE above a trigger,
the model's own learned "close probability" still LOW (still confident in continuing), drawdown
within bounds, AND opposite-direction utility/Q-value blocked (low). Grid search on that model
picked "no pyramiding" as best -- a real negative precedent, but for a different, unrelated model.

Adapted here with INVERTED logic -- instead of adding to the position when the model is "still
confident", this fires a partial REDUCE when the model's confidence is flipping (exit-head
probability rising but not yet at the live EXIT_THRESHOLD=0.95 gate) while price action already
looks favorable and the opposite side isn't gaining conviction:
  unrealized   >= profit_min   * take_profit   (profit_min in {0.3, 0.5})
  mfe          >= mfe_trigger  * take_profit   (mfe_trigger in {0.6, 0.8}, reuses "activate_frac")
  exit_head_prob >= confidence_fade_thr         (confidence_fade_thr in {0.3, 0.5, 0.7} -- the
      SAME exit-head probability the baseline replay already computes every bar for the
      EXIT_THRESHOLD=0.95 check; this catches it starting to rise, not waiting for it to fully
      fire, so it is NOT redundant with the final exit_head check below)
  opposite_dir_p <= opposite_block_thr          (fixed 0.35; proxy_dir_short for a long position,
      proxy_dir_long for a short -- same proxy columns already wired in
      research_eth_omega461_reversal_risk_scaleout_eval_20260721.py)
All four ANDed (matching V31/V49's AND-gate design). On fire: partial reduce close_frac=0.5
(fixed), reusing the exact partial-scaleout two-tranche accounting block verbatim from
research_eth_omega461_exit_ideas2_20260721.py / research_eth_omega461_reversal_risk_scaleout_eval_20260721.py.
Fires at most once per trade (partial_done flag).

Forked from research_eth_omega461_reversal_risk_scaleout_eval_20260721.py's structure (proxy
column loading, prep_with_proxy, sanity-check funnel, VAL-then-OOS confirmation). Does NOT use
the reversal-risk classifier from that script -- this experiment's signal is the baseline exit
head's own probability plus the opposite-direction proxy column, no new model is trained. Does
NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, or .env.

Baselines (reused unmodified from ideas2 / exit_sweep, NOT recomputed):
  VAL  h48qual: pnl +5.45%  mdd -11.62%
  VAL  zig075:  pnl +40.31% mdd -13.07%
  OOS  h48qual: pnl +9.49%  mdd -6.54%
  OOS  zig075:  pnl +17.89% mdd -11.01%

Fresh-forward discipline: causal bar-by-bar single forward pass; no saved ledger used as input.
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
import research_eth_omega461_exit_ideas2_20260721 as ideas2  # noqa: E402
import research_eth_omega461_reversal_risk_scaleout_eval_20260721 as reversal_eval  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260721"
BASELINES = ideas2.BASELINES
beats_baseline = ideas2.beats_baseline
OPPOSITE_BLOCK_THR = 0.35
CLOSE_FRAC = 0.5


@torch.no_grad()
def replay_gate_scaleout(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    exit_threshold: float = sweep.BASELINE_EXIT_THRESHOLD,
    proxy_dir_long: np.ndarray | None = None,
    proxy_dir_short: np.ndarray | None = None,
    profit_min: float | None = None,
    mfe_trigger: float | None = None,
    confidence_fade_thr: float | None = None,
    opposite_block_thr: float | None = None,
    close_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Identical structure to reversal_eval.replay_reversal (TP/SL
    first), then the new V31/V49-adapted gate (positioned before the final exit-head/
    EXIT_THRESHOLD check, reusing the SAME exit-head probability computed for that check -- no
    duplicate model call), then the same partial-scaleout two-tranche accounting block reused
    verbatim. Reduces exactly to the baseline when gate kwargs are None. fresh_forward_bar_by_bar
    =true; no saved ledger used as input.
    """
    gate_enabled = (
        profit_min is not None
        and mfe_trigger is not None
        and confidence_fade_thr is not None
        and opposite_block_thr is not None
        and close_frac is not None
    )
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
    notional_initial = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    partial_done = False
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
            exit_prob = 0.0
            hold = max(int(i) - int(entry_i), 0)
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                # Exit-head probability computed once, unconditionally -- consumed by BOTH the
                # new gate below (fade-detection) and the final EXIT_THRESHOLD check, matching
                # the baseline replay's own single computation (no duplicate model call).
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
                if (
                    gate_enabled
                    and (not partial_done)
                    and take_profit > 0.0
                    and move >= float(profit_min) * take_profit
                    and mfe >= float(mfe_trigger) * take_profit
                    and exit_prob >= float(confidence_fade_thr)
                ):
                    opposite_dir_p = float(proxy_dir_short[i] if pos > 0 else proxy_dir_long[i])
                    if opposite_dir_p <= float(opposite_block_thr):
                        reason = "v31v49_gate_scaleout"
                if not reason and exit_prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason == "v31v49_gate_scaleout":
                # Reused verbatim from ideas2.replay_idea / reversal_eval.replay_reversal's
                # partial-scaleout two-tranche block.
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                part_notional = notional * float(close_frac)
                before = cash
                cash = cash * (1.0 + raw_exit * part_notional)
                cash -= before * exit_fee * part_notional
                reasons["v31v49_gate_scaleout"] = reasons.get("v31v49_gate_scaleout", 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": "v31v49_gate_scaleout",
                    "win": int(raw_exit > 0), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float("nan"),
                    "net_per_notional": float("nan"), "notional": float(part_notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                notional = notional - part_notional
                partial_done = True
                continue
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
                    "net_per_notional": float(trade_return / max(notional_initial, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                partial_done = False
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
        notional_initial = row_notional
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
        partial_done = False

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
            "net_per_notional": float(trade_return / max(notional_initial, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    closed = ledger[ledger["reason"] != "v31v49_gate_scaleout"] if len(ledger) else ledger
    hold_bars = (closed["exit_i"] - closed["entry_i"]).clip(lower=0) if len(closed) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(closed["trade_return"].max() * 100.0) if len(closed) else 0.0,
            "p95_trade_pnl": float(closed["trade_return"].quantile(0.95) * 100.0) if len(closed) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


def run_one(p: dict, **kwargs) -> tuple[dict[str, Any], pd.DataFrame]:
    return replay_gate_scaleout(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
        device=sweep.DEVICE, **kwargs,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = {name: reversal_eval.prep_with_proxy(name, cfg, val_frame, sweep.EXT_PRED_DIR, oof=True) for name, cfg in sweep.COMPONENTS.items()}

    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = {name: reversal_eval.prep_with_proxy(name, cfg, oos_frame, sweep.EXT_PRED_DIR, oof=False) for name, cfg in sweep.COMPONENTS.items()}

    # ---------------- Mandatory sanity checks ----------------
    print("stage=sanity_checks", flush=True)
    sanity_rows = []
    all_sane = True
    for split_name, prepped, base_key in (("VAL", val_prepped, "VAL"), ("OOS", oos_prepped, "OOS")):
        for name, p in prepped.items():
            b = BASELINES[(name, base_key)]
            m_noop, _ = run_one(p)  # all gate kwargs None -> fully disabled no-op path
            m_impossible, _ = run_one(
                p, proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                profit_min=0.3, mfe_trigger=0.6, confidence_fade_thr=1.01,
                opposite_block_thr=OPPOSITE_BLOCK_THR, close_frac=CLOSE_FRAC,
            )
            # BASELINES constants are rounded to 2 decimal places -- tolerance matches that
            # rounding precision, same as the reversal-scaleout round's sanity check.
            tol = 0.01
            ok_noop = abs(m_noop["pnl"] - b["pnl"]) < tol and abs(m_noop["mdd"] - b["mdd"]) < tol
            ok_impossible_vs_baseline = abs(m_impossible["pnl"] - b["pnl"]) < tol and abs(m_impossible["mdd"] - b["mdd"]) < tol
            ok_impossible_vs_noop = abs(m_impossible["pnl"] - m_noop["pnl"]) < 1e-9 and abs(m_impossible["mdd"] - m_noop["mdd"]) < 1e-9
            ok_impossible = ok_impossible_vs_baseline and ok_impossible_vs_noop
            all_sane = all_sane and ok_noop and ok_impossible
            sanity_rows.append({
                "split": split_name, "component": name, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"],
                "noop_pnl": m_noop["pnl"], "noop_mdd": m_noop["mdd"], "noop_ok": ok_noop,
                "impossible_pnl": m_impossible["pnl"], "impossible_mdd": m_impossible["mdd"], "impossible_ok": ok_impossible,
                "impossible_vs_noop_exact": ok_impossible_vs_noop,
            })
            print(f"  sanity split={split_name} component={name} noop_ok={ok_noop} impossible_ok={ok_impossible} "
                  f"impossible_vs_noop_exact={ok_impossible_vs_noop} "
                  f"(noop pnl={m_noop['pnl']:.4f}/mdd={m_noop['mdd']:.4f} impossible pnl={m_impossible['pnl']:.4f}/mdd={m_impossible['mdd']:.4f} "
                  f"vs baseline pnl={b['pnl']:.4f}/mdd={b['mdd']:.4f})", flush=True)
    pd.DataFrame(sanity_rows).to_csv(OUT_DIR / "v31v49_gate_scaleout_sanity_checks.csv", index=False)
    if not all_sane:
        print("stage=STOP sanity checks FAILED -- see v31v49_gate_scaleout_sanity_checks.csv, not proceeding to grid", flush=True)
        return 1
    print("stage=sanity_checks PASSED", flush=True)

    # ---------------- VAL grid ----------------
    print("stage=val_grid", flush=True)
    val_rows = []
    winners = []
    for name, p in val_prepped.items():
        for profit_min in (0.3, 0.5):
            for mfe_trigger in (0.6, 0.8):
                for confidence_fade_thr in (0.3, 0.5, 0.7):
                    m, _ = run_one(
                        p, proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
                        profit_min=profit_min, mfe_trigger=mfe_trigger, confidence_fade_thr=confidence_fade_thr,
                        opposite_block_thr=OPPOSITE_BLOCK_THR, close_frac=CLOSE_FRAC,
                    )
                    row = {
                        "component": name, "profit_min": profit_min, "mfe_trigger": mfe_trigger,
                        "confidence_fade_thr": confidence_fade_thr, "opposite_block_thr": OPPOSITE_BLOCK_THR,
                        "close_frac": CLOSE_FRAC, **m,
                    }
                    val_rows.append(row)
                    if beats_baseline(name, "VAL", m["pnl"], m["mdd"]):
                        winners.append({
                            "component": name, "profit_min": profit_min, "mfe_trigger": mfe_trigger,
                            "confidence_fade_thr": confidence_fade_thr, "opposite_block_thr": OPPOSITE_BLOCK_THR,
                            "close_frac": CLOSE_FRAC,
                        })
    val_df = pd.DataFrame(val_rows)
    val_df["exit_reasons"] = val_df["exit_reasons"].apply(json.dumps)
    val_df.to_csv(OUT_DIR / "v31v49_gate_scaleout_VAL.csv", index=False)
    cols = ["component", "profit_min", "mfe_trigger", "confidence_fade_thr", "opposite_block_thr",
            "close_frac", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]
    print(val_df[cols].to_string(index=False), flush=True)
    print(f"\nVAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    for w in winners:
        print(f"  {w}", flush=True)

    if not winners:
        print("stage=done no_val_winners -> no OOS confirmation run", flush=True)
        return 0

    # ---------------- OOS confirmation ----------------
    print("stage=oos_confirm", flush=True)
    oos_rows = []
    for w in winners:
        p = oos_prepped[w["component"]]
        m_cand, _ = run_one(
            p, proxy_dir_long=p["proxy_dir_long"], proxy_dir_short=p["proxy_dir_short"],
            profit_min=w["profit_min"], mfe_trigger=w["mfe_trigger"], confidence_fade_thr=w["confidence_fade_thr"],
            opposite_block_thr=w["opposite_block_thr"], close_frac=w["close_frac"],
        )
        b = BASELINES[(w["component"], "OOS")]
        cleared = beats_baseline(w["component"], "OOS", m_cand["pnl"], m_cand["mdd"])
        row = {**w, "oos_pnl": m_cand["pnl"], "oos_mdd": m_cand["mdd"], "oos_trades": m_cand["trades"],
               "oos_wr": m_cand["wr"], "oos_baseline_pnl": b["pnl"], "oos_baseline_mdd": b["mdd"], "cleared_oos": cleared}
        oos_rows.append(row)
        print(f"  {w} -> OOS pnl={m_cand['pnl']:.2f}% mdd={m_cand['mdd']:.2f}% trades={m_cand['trades']} "
              f"(baseline pnl={b['pnl']:.2f}% mdd={b['mdd']:.2f}%) cleared={cleared}", flush=True)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "v31v49_gate_scaleout_OOS_confirm.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
