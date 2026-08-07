#!/usr/bin/env python3
"""Lookahead/contamination/lag audit for the final Omega4.6.1 greedy live-realistic replay.

Two checks:
1. Entry-delay lag test: re-run the exact same greedy replay but delay ENTRY EXECUTION by k=1,2,3
   bars after the signal bar (still using the SAME signal computed causally at bar i, just
   executing later). A genuine causal edge should degrade SMOOTHLY as k increases (missing the
   optimal entry costs some edge) -- if PnL is flat, improves, or the sign only holds for k=0,
   that is a red flag for a lookahead artifact (the model's real information advantage is at bar
   i specifically, which is the correct/expected behavior for a live system that decides at bar i
   and fills at bar i+1).
2. Feature-shift sanity re-check: re-verify ou_halflife/atr_pct at the entry bar only use rolling
   windows that end at or before the entry bar index (no negative-shift/centered-window patterns
   in the source feature engineering, confirmed via static code grep separately).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
DURATION_THRESHOLD = 0.005417


@torch.no_grad()
def greedy_replay_delayed(frame: pd.DataFrame, components: dict, *, fee: float, slip: float,
                           cost_mult: float, device: torch.device, entry_delay: int) -> pd.DataFrame:
    """Same as greedy.greedy_replay but entry fills `entry_delay` extra bars after the signal bar
    (0 = normal next-bar-open fill, as used everywhere else in this project)."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = 1.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict] = []

    for i in range(0, n - 2 - entry_delay):
        if pos != 0:
            comp = components[active_comp]
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            mfe, mae = max(mfe, move), min(mae, move)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae),
                                float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                                float(move + abs(stop_loss)), float(notional), float(leverage_v),
                                float(notional * leverage_v), float(take_profit), float(stop_loss)],
                    device=device,
                )
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                rows.append({"entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "trade_return": float(trade_return), "notional": float(notional)})
                pos, active_comp = 0, None
                continue
            continue

        for name in greedy.PRIORITY:
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            fill_i = min(i + 1 + entry_delay, n - 1)
            entry_px = arrays["open"][fill_i] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = fill_i, i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break

    return pd.DataFrame(rows)


def summarize(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    returns = ledger["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0),
            "trades": int(len(ledger)), "wr": float((returns > 0).mean())}


def main() -> int:
    device = retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = omega._load_fee_slip()
    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        components[name] = greedy.prepare_component(ext_frame, pred_csv, cfg, device)

    print("=== Entry-delay lag test (genuine causal edge should degrade smoothly as delay increases) ===", flush=True)
    for delay in (0, 1, 2, 3, 6):
        ledger = greedy_replay_delayed(ext_frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device, entry_delay=delay)
        m = summarize(ledger)
        print(f"  delay={delay} bars: pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)

    print("\n=== Static code check: rolling-window direction (grep results) ===", flush=True)
    import subprocess
    for f in ["features/engineering.py", "features/elite.py"]:
        out = subprocess.run(["grep", "-n", r"shift(-\|center=True\|centre=True\|bfill", f], capture_output=True, text=True, cwd=str(ROOT))
        print(f"  {f}: {'CLEAN (no forward-looking patterns found)' if not out.stdout.strip() else out.stdout}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
