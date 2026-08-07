#!/usr/bin/env python3
"""Priority-5 test: does wiring in the (currently unused) L3 TCN sequence gate add value to the
frozen v2 winner? The gate only fires when both primary AND fallback are CASH (per
trading_bot_modules/omega6_live.py::decide_latest), producing a synthetic short-only candidate.

Reconstructs the gate's exact input features from the cached decision tape (all needed columns
are already there), runs the retrained-for-Omega6 TCN artifact
(tmp/causal_regen_20260516/omega6_sequence_gate_20260703/tcn_seq_gate_L24_omega6.pt) with its
24-bar lookback, and adds L3-sourced short candidates into the frozen winner's effective side
array before applying persistence/ATR-barrier/cooldown -- same risk controls as normal entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

GATE_PATH = ROOT / "tmp/causal_regen_20260516/omega6_sequence_gate_20260703/tcn_seq_gate_L24_omega6.pt"


class SequenceEntryTCN(nn.Module):
    def __init__(self, seq_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.net(seq.transpose(1, 2))
        pooled = x.mean(dim=-1)
        last = x[:, :, -1]
        return self.head(torch.cat([pooled, last], dim=1)).squeeze(-1)


def build_l3_features(tape: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(tape["timestamp"])
    dow = ts.dt.dayofweek + ts.dt.hour / 24.0
    out = pd.DataFrame(index=tape.index)
    out["primary_dir_p_cash"] = tape["primary_dir_p_cash"]
    out["primary_dir_p_long"] = tape["primary_dir_p_long"]
    out["primary_dir_p_short"] = tape["primary_dir_p_short"]
    out["primary_dir_confidence"] = tape[["primary_dir_p_cash", "primary_dir_p_long", "primary_dir_p_short"]].max(axis=1)
    out["primary_quality_p_cash"] = tape["primary_quality_p_cash"]
    out["primary_expert_bull"] = (tape["primary_expert"] == "bull").astype(float)
    out["primary_expert_bear"] = (tape["primary_expert"] == "bear").astype(float)
    out["primary_expert_chop"] = (tape["primary_expert"] == "chop").astype(float)
    out["primary_route_confidence"] = tape["primary_route_confidence"]
    out["fallback_dir_p_cash"] = tape["fallback_dir_p_cash"]
    out["fallback_dir_p_long"] = tape["fallback_dir_p_long"]
    out["fallback_dir_p_short"] = tape["fallback_dir_p_short"]
    out["fallback_dir_confidence"] = tape[["fallback_dir_p_cash", "fallback_dir_p_long", "fallback_dir_p_short"]].max(axis=1)
    out["fallback_quality_p_cash"] = tape["fallback_quality_p_cash"]
    out["fallback_expert_bull"] = (tape["fallback_expert"] == "bull").astype(float)
    out["fallback_expert_bear"] = (tape["fallback_expert"] == "bear").astype(float)
    out["fallback_expert_chop"] = (tape["fallback_expert"] == "chop").astype(float)
    out["fallback_route_confidence"] = tape["fallback_route_confidence"]
    out["atr_pct"] = tape["atr_pct"]
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out


def score_tcn_gate(tape: pd.DataFrame) -> np.ndarray:
    payload = torch.load(GATE_PATH, map_location="cpu", weights_only=False)
    feature_cols = list(payload["feature_cols"])
    lookback = int(payload["lookback"])
    model = SequenceEntryTCN(seq_dim=len(feature_cols))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)

    feats = build_l3_features(tape)[feature_cols].to_numpy(dtype=np.float32)
    n = len(feats)
    z = (feats - mean[None, :]) / std[None, :]
    scores = np.full(n, np.nan, dtype=np.float64)
    batch_size = 4096
    with torch.no_grad():
        for start in range(lookback - 1, n, batch_size):
            end = min(start + batch_size, n)
            seqs = np.stack([z[i - lookback + 1 : i + 1] for i in range(start, end)], axis=0)
            out = model(torch.from_numpy(seqs.astype(np.float32)))
            scores[start:end] = out.numpy()
    return scores


def run_with_l3(tape: pd.DataFrame, scores: np.ndarray, threshold: float, *, start, end, fee_mult: float, use_l3: bool) -> dict:
    sub_mask = (tape["timestamp"] >= start) & (tape["timestamp"] <= end)
    sub = tape[sub_mask].reset_index(drop=True)
    sub_scores = scores[sub_mask.to_numpy()]
    n = len(sub)
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)

    primary_side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    fallback_side_arr = sub["fallback_side"].to_numpy(dtype=np.int64)
    eff_side = np.where(primary_side_arr != 0, primary_side_arr, fallback_side_arr)
    if use_l3:
        both_cash = (primary_side_arr == 0) & (fallback_side_arr == 0)
        l3_fire = both_cash & np.isfinite(sub_scores) & (sub_scores >= threshold)
        eff_side = np.where(l3_fire, -1, eff_side)

    persistence_ok = eff_side != 0
    for k in range(1, 3):
        shifted = np.roll(eff_side, k)
        shifted[:k] = 0
        persistence_ok &= shifted == eff_side

    atr_pct_arr = sub["atr_pct"].to_numpy(dtype=np.float64)
    FEE = 0.00020 * fee_mult
    SLIP = 0.00050 * fee_mult
    notional = 0.30 * 2.0
    tp_atr_mult, sl_atr_mult, cooldown_bars = 15.0, 5.0, 12

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    hold_start = 0
    take_profit = stop_loss = 0.0
    max_hold = 288
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or not persistence_ok[i] or eff_side[i] == 0:
                i += 1
                continue
            side = int(eff_side[i])
            atr = max(atr_pct_arr[i], 1e-6)
            tp, sl = tp_atr_mult * atr, sl_atr_mult * atr
            entry_price = float(open_[min(i + 1, n - 1)]) * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
            pos = side
            take_profit, stop_loss = tp, sl
            hold_start = i
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1.0 - SLIP) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + SLIP)) / max(entry_price, 1e-12)
        unreal = raw * notional
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and unreal >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= max_hold:
            reason = "time_stop"
        if reason:
            exit_price = close[i] * (1.0 - SLIP if pos > 0 else 1.0 + SLIP)
            raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
            before = cash
            cash = cash * (1.0 + raw_exit * notional)
            cash -= before * FEE * notional
            trades.append({"win": bool(cash > entry_equity)})
            pos = 0
            cooldown_until = i + cooldown_bars
        i += 1
    wins = sum(1 for t in trades if t["win"])
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": len(trades), "wr": float(wins / len(trades)) if trades else 0.0}


def main() -> int:
    tape_raw = v2.load_tape()
    tape = v2.apply_quality_threshold(tape_raw, 0.58)
    print("scoring TCN gate over full tape...", flush=True)
    scores = score_tcn_gate(tape)
    threshold = float(torch.load(GATE_PATH, map_location="cpu", weights_only=False)["threshold"])
    print(f"threshold={threshold}", flush=True)

    both_cash = (tape["primary_side"].to_numpy() == 0) & (tape["fallback_side"].to_numpy() == 0)
    fires = both_cash & np.isfinite(scores) & (scores >= threshold)
    print(f"both-cash bars: {int(both_cash.sum())}, L3 would fire on: {int(fires.sum())} bars", flush=True)

    for name, use_l3 in (("baseline_no_l3", False), ("with_l3_gate", True)):
        out = {}
        for tag, mult in (("cost1", 1.0), ("cost3", 3.0)):
            out[tag] = run_with_l3(tape, scores, threshold, start=v2.VAL_START, end=v2.VAL_END, fee_mult=mult, use_l3=use_l3)
        print(
            f"{name}: cost1 pnl={out['cost1']['pnl']:.2f}% mdd={out['cost1']['mdd']:.2f}% trades={out['cost1']['trades']} wr={out['cost1']['wr']:.3f} | "
            f"cost3 pnl={out['cost3']['pnl']:.2f}% mdd={out['cost3']['mdd']:.2f}% trades={out['cost3']['trades']} wr={out['cost3']['wr']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
