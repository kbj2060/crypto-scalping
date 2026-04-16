#!/usr/bin/env python3
"""Simulate DSAC pure vs balanced overlay for chosen test slices."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor

@dataclass
class Metrics:
    pnl_pct: float
    trades: int
    win_rate: float
    avg_hold: float
    mdd_pct: float
    trend_reversal_exits: int
    risk_reduces: int
    sharpe: float
    overlay_mode: str
    overlay_reduce_counts: dict[str, int] = field(default_factory=dict)
    overlay_exit_counts: dict[str, int] = field(default_factory=dict)


def load_actor(path: str, device: str) -> GaussianActor:
    ckpt = torch.load(path, map_location=device)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM)))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def simulate(df: pd.DataFrame, router: DSACRouter, overlay: bool) -> Metrics:
    n = len(df)
    actions = np.zeros(n, dtype=int)
    fee = 0.0005
    slip = 0.0002
    balance = 1.0
    pos = None
    entry = 0.0
    lev = 0.0
    hold = 0
    lev_cur = 0.0
    peak = 1.0
    liqudebug = []
    trades = 0
    wins = 0
    eq_curve = [1.0]
    trend_hist: list[int] = []
    risk_reduces = 0
    trend_exits = 0

    close = df["close"].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    features = df.to_dict(orient="records")
    numeric_cols = [c for c in df.columns if c != "timestamp"]

    reduce_counter: Counter[str] = Counter()
    exit_counter: Counter[str] = Counter()

    overlay_name = "balanced" if overlay else "off"

    for i in range(n - 1):
        row = features[i]
        stats = {k: float(row.get(k, 0.0) or 0.0) for k in numeric_cols}
        action_float, lev_raw, info = router.decide(stats, {
            "type": pos,
            "entry_price": entry,
            "unrealized": 0.0,
            "mdd": 0.0,
            "hold_norm": float(min(hold / 96.0, 1.0)),
        })
        action = int(action_float)
        lev = float(np.clip(lev_raw, 0.0, 1.0))
        actions[i] = action

        next_open = float(open_np[i + 1])
        next_close = float(close[i + 1])
        hib = max(
            1.0 if float(row.get("m7_iso_anom", 0.0)) >= 0.5 else 0.0,
            min(float(row.get("m7_vae_error", 0.0)) / max(float(row.get("m7_vae_threshold", 1e-6)), 1e-6) / 1.35, 1.5),
            min(abs(float(row.get("jump_z", 0.0))) / 3.0, 1.5),
            min(abs(float(row.get("evt_excess_z", 0.0))) / 3.0, 1.5),
        ) / 1.5
        amihud = float(row.get("amihud_illiquidity_z", 0.0))
        trend_dir = 1
        if float(row.get("m7_prob_up", 0.0)) - float(row.get("m7_prob_dn", 0.0)) > 0.25:
            trend_dir = 2
        elif float(row.get("m7_prob_dn", 0.0)) - float(row.get("m7_prob_up", 0.0)) > 0.25:
            trend_dir = 0
        trend_hist.append(trend_dir)
        if len(trend_hist) > 2:
            trend_hist.pop(0)
        net_score = float(info.get("raw_action", 0.0))

        special_exit = False
        special_reduce = False
        overlay_reduce_reason = None
        overlay_exit_reason = None
        if overlay and pos is not None:
            high_risk = hib >= 0.85 or abs(amihud) >= 6.0
            pos_support = float(info.get("own_support", 0.0))
            net_edge = float(info.get("net_edge", 0.0))
            low_support = pos_support < 1.10
            if net_edge <= -0.10:
                special_exit = True
                overlay_exit_reason = "net_edge"
            elif high_risk and len(trend_hist) >= 2 and trend_hist[-1] != trend_hist[-2]:
                if (pos == "LONG" and trend_hist[-1] == 0) or (pos == "SHORT" and trend_hist[-1] == 2):
                    special_exit = True
                    overlay_exit_reason = "trend_flip"
            elif low_support:
                special_reduce = True
                overlay_reduce_reason = "low_support"
            elif high_risk:
                special_reduce = True
                overlay_reduce_reason = "hib" if hib >= 0.85 else "amihud"

        if action == 0 and pos is not None:
            special_exit = False
        if special_exit:
            trend_exits += 1
            action = 0
            lev = 0.0
        elif special_reduce:
            risk_reduces += 1
            lev = float(np.clip(lev * 0.5, 0.0, 1.0))
            if overlay_reduce_reason:
                reduce_counter[overlay_reduce_reason] += 1
        if special_exit and overlay_exit_reason:
            exit_counter[overlay_exit_reason] += 1

        if pos is None:
            if action == 1 and lev > 0.0:
                pos = "LONG"
                entry = next_open * (1.0 + slip)
                lev_cur = lev
                balance -= balance * fee * lev
                hold = 0
            elif action == 2 and lev > 0.0:
                pos = "SHORT"
                entry = next_open * (1.0 - slip)
                lev_cur = lev
                balance -= balance * fee * lev
                hold = 0
        else:
            if action == 0 or (pos == "LONG" and action == 2) or (pos == "SHORT" and action == 1):
                exit_price = next_open
                pnl = ((exit_price * (1.0 - slip) - entry) / entry) if pos == "LONG" else ((entry - exit_price * (1.0 + slip)) / entry)
                pnl *= lev_cur
                balance *= 1 + pnl
                balance -= balance * fee * lev_cur
                trades += 1
                if pnl > 0:
                    wins += 1
                hold = 0
                pos = None
                entry = 0.0
                lev_cur = 0.0
            else:
                hold += 1
        if pos is not None:
            cur = balance * (1 + (next_close - entry) / entry) if pos == "LONG" else balance * (1 + (entry - next_close) / entry)
            eq_curve.append(cur)
        else:
            eq_curve.append(balance)

    eq = np.asarray(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    pnl = (eq[-1] - 1.0) * 100.0
    mdd = float(np.min(dd)) * 100.0
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)) if len(rets) >= 3 and np.std(rets) > 0 else 0.0
    wr = float(wins) / trades if trades else 0.0
    avg_hold = 0.0
    if trades:
        avg_hold = float(len(eq_curve)) / trades
    return Metrics(
        pnl_pct=pnl,
        trades=trades,
        win_rate=wr * 100.0,
        avg_hold=avg_hold,
        mdd_pct=mdd,
        trend_reversal_exits=trend_exits,
        risk_reduces=risk_reduces,
        sharpe=sharpe,
        overlay_mode=overlay_name,
        overlay_reduce_counts=dict(reduce_counter),
        overlay_exit_counts=dict(exit_counter),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-csv", default="data/splits/year_oos/rl_base_2024.csv")
    parser.add_argument("--feature-csv", default="data/splits/year_oos/rl_base_2024.csv")
    parser.add_argument("--ckpt", default="data/ensemble/ckpt/best_dsac_agents.pth")
    parser.add_argument("--test-start", default="2024-10-01")
    parser.add_argument("--test-end", default="2025-06-30")
    parser.add_argument("--split", choices=["2024Q4","2025H1"], default="2024Q4")
    parser.add_argument("--overlay", choices=["off","balanced"], default="off")
    parser.add_argument("--out-json", default="data/ensemble/metrics/ab_dsac_balanced.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cpu"
    actor = load_actor(args.ckpt, device)
    df = pd.read_csv(args.rl_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    mask = {
        "2024Q4": (df["timestamp"] >= "2024-10-01") & (df["timestamp"] <= "2024-12-31"),
        "2025H1": (df["timestamp"] >= "2025-01-01") & (df["timestamp"] <= "2025-06-30"),
    }[args.split]
    sliced = df[mask].copy()
    router = DSACRouter(actor, device=device)
    metrics = simulate(sliced, router, overlay=args.overlay == "balanced")
    out = {
        "split": args.split,
        "overlay": args.overlay,
        "metrics": asdict(metrics),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
