#!/usr/bin/env python3
"""Stage-2b prototype: non-directional "straddle breakout" strategy.

Rationale: stage-1 gate (research_btc_event_gate_prototype_20260804.py) shows a
stable ~3.3x lift on "a big move is coming" (VAL 3.40x, OOS 3.33x), but the stage-2
direction head (research_btc_direction_head_prototype_20260804.py) failed the
Fresh-Forward test -- VAL 60.9% -> OOS 46.3% accuracy, "confident" subset OOS 33.3%
(worse than random), the same VAL-positive/OOS-negative signature that has closed
every other BTC directional line in this project's history.

Since the gate does NOT need to know direction to have edge, this strategy removes
direction prediction entirely: when the gate fires, place symmetric stop-entry
"breakout" orders on both sides of price (long above, short below). Whichever side
is touched first becomes the position -- the market picks the direction, not a model.
If the move is genuinely large (which the gate says is more likely than base rate),
one side fills and rides toward TP; if the bar chops without a real breakout, neither
side fills and there is no trade (no whipsaw cost beyond the two cancelled orders).

Fresh-Forward compliance: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. No model is fit
here at all (no train/fit step, no leakage risk) -- this is a rule-based execution
overlay on top of the already-causal stage-1 gate, walked bar-by-bar forward through
VAL/OOS using only information available at or before each bar.

Futures Risk Sizing Contract (per CLAUDE.md): all TP/SL/entry levels below are price-move
fractions; fee_cost reuses the exact (FEE_RATE+SLIP_RATE)*2*3 convention from
build_omega1_2_triple_barrier_labels_btc_20260708.py for consistency with the rest of
the BTC pipeline, not a new number invented for this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_btc_event_gate_prototype_20260804 as gate_mod  # noqa: E402
import build_omega1_2_triple_barrier_labels_btc_20260708 as tb_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260804/btc_nondirectional_straddle_prototype"

BREAKOUT_ATR_MULT = 1.0    # stop-entry offset from signal-bar close, in ATRs
TP_MULT, SL_MULT = 1.6, 1.0     # same as h48_balanced (h48qual family) for comparability
MIN_TP, MIN_SL = 0.006, 0.004
FEE_COST = float(tb_mod.FEE_RATE + tb_mod.SLIP_RATE) * 2.0 * 3.0  # reuse repo convention


def _cluster_dedupe(ts: pd.Series, gap: str = "4h") -> np.ndarray:
    new_cluster = ts.diff() > pd.Timedelta(gap)
    if len(new_cluster):
        new_cluster.iloc[0] = True
    return new_cluster.to_numpy()


def _simulate_straddle(df: pd.DataFrame, signal_idx: np.ndarray, atr: np.ndarray) -> list[dict]:
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    ts = df["timestamp"]
    n = len(df)
    trades = []
    for i in signal_idx:
        if i + gate_mod.EVENT_HORIZON + 1 >= n:
            continue
        vol = float(atr[i])
        if not np.isfinite(vol) or vol <= 0:
            continue
        ref = close[i]
        up_trigger = ref * (1.0 + BREAKOUT_ATR_MULT * vol)
        down_trigger = ref * (1.0 - BREAKOUT_ATR_MULT * vol)

        side = 0
        entry = 0.0
        trigger_bar = None
        for j in range(i + 1, i + 1 + gate_mod.EVENT_HORIZON):
            hit_up = high[j] >= up_trigger
            hit_down = low[j] <= down_trigger
            if hit_up and hit_down:
                trigger_bar = j
                break  # ambiguous same-bar double-touch: untradeable, skip
            if hit_up:
                side, entry, trigger_bar = 1, up_trigger, j
                break
            if hit_down:
                side, entry, trigger_bar = -1, down_trigger, j
                break
        if side == 0:
            trades.append(
                {"timestamp": ts.iloc[i], "outcome": "no_breakout" if trigger_bar is None else "ambiguous_same_bar", "ret": 0.0}
            )
            continue

        remaining = (i + gate_mod.EVENT_HORIZON) - trigger_bar
        if remaining <= 0:
            trades.append({"timestamp": ts.iloc[i], "outcome": "no_room_left", "ret": 0.0})
            continue
        tp_move = max(MIN_TP, TP_MULT * vol)
        sl_move = max(MIN_SL, SL_MULT * vol)
        fut_high = high[trigger_bar + 1 : trigger_bar + 1 + remaining]
        fut_low = low[trigger_bar + 1 : trigger_bar + 1 + remaining]
        fut_close = close[trigger_bar + 1 : trigger_bar + 1 + remaining]
        ret, reason, mae, mfe, bars = tb_mod._reason_and_return(
            side=side, entry=entry, future_high=fut_high, future_low=fut_low,
            future_close=fut_close, tp_move=tp_move, sl_move=sl_move,
        )
        net_ret = ret - FEE_COST
        trades.append(
            {
                "timestamp": ts.iloc[i], "outcome": reason, "side": side, "gross_ret": float(ret),
                "ret": float(net_ret), "bars_to_exit": int(bars),
            }
        )
    return trades


def _summarize(trades: list[dict]) -> dict:
    df = pd.DataFrame(trades)
    n_signals = len(df)
    filled = df[~df["outcome"].isin(["no_breakout", "ambiguous_same_bar", "no_room_left"])]
    n_filled = len(filled)
    if n_filled == 0:
        return {"n_signals": n_signals, "n_filled": 0, "fill_rate": 0.0}
    wins = filled[filled["ret"] > 0]
    equity = filled["ret"].cumsum()
    running_max = equity.cummax()
    mdd = float((equity - running_max).min()) if len(equity) else 0.0
    return {
        "n_signals": n_signals,
        "n_filled": n_filled,
        "fill_rate": float(n_filled / n_signals),
        "win_rate": float(len(wins) / n_filled),
        "avg_net_ret_per_trade": float(filled["ret"].mean()),
        "sum_net_ret": float(filled["ret"].sum()),
        "mdd_price_move_units": mdd,
        "outcome_breakdown": filled["outcome"].value_counts().to_dict(),
        "long_short_split": filled["side"].value_counts().to_dict() if "side" in filled else {},
    }


def main() -> int:
    df = gate_mod._load_ohlc()
    gmm = pd.read_csv(gate_mod.GMM_SCORES, usecols=["timestamp", "gmm_cluster_rank", "gmm_confidence"])
    ifs = pd.read_csv(gate_mod.IF_SCORES, usecols=["timestamp", "if_score"])
    gmm["timestamp"] = pd.to_datetime(gmm["timestamp"])
    ifs["timestamp"] = pd.to_datetime(ifs["timestamp"])
    df = df.merge(gmm, on="timestamp", how="inner").merge(ifs, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    atr = gate_mod._causal_atr(df)
    gate = gate_mod._multi_timescale_gate(df["gmm_cluster_rank"], df["gmm_confidence"], df["if_score"])
    df = pd.concat([df, gate], axis=1)
    df["threshold"] = gate_mod._online_conformal_threshold(df["raw_score"])
    df["gate_fired"] = (df["raw_score"] >= df["threshold"]) & (df["agreement"] >= gate_mod.AGREEMENT_MIN)

    fired = df[df["gate_fired"]].copy()
    fired["is_cluster_start"] = _cluster_dedupe(fired["timestamp"])
    signal_idx = fired.index[fired["is_cluster_start"]].to_numpy()

    atr_arr = atr.to_numpy()
    trades = _simulate_straddle(df, signal_idx, atr_arr)
    trades_df = pd.DataFrame(trades)

    def _window(start, end):
        m = (trades_df["timestamp"] >= start) & (trades_df["timestamp"] <= end)
        return _summarize(trades_df[m].to_dict("records"))

    result = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "config": {
            "breakout_atr_mult": BREAKOUT_ATR_MULT,
            "tp_mult": TP_MULT, "sl_mult": SL_MULT, "min_tp": MIN_TP, "min_sl": MIN_SL,
            "fee_cost_per_trade": FEE_COST,
            "event_horizon_bars": gate_mod.EVENT_HORIZON,
        },
        "validation_2025_09_to_12": _window(gate_mod.VAL_START, gate_mod.VAL_END),
        "oos_2026_01_to_03": _window(gate_mod.OOS_START, gate_mod.OOS_END),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "straddle_eval_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    trades_df.to_csv(OUT_DIR / "straddle_trades.csv", index=False)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
