#!/usr/bin/env python3
"""DSAC A/B split comparison (fast path vs slow path).

A: fast path  -> HMM off
B: slow path  -> HMM on

Both runs use the same checkpoint and the same test split.
Outputs include avg hold time in bars.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_agent import MultiTimeframeFeatures, OnlineHMMDetector
from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, DSACCompactTradingEnv, GaussianActor


@dataclass
class EvalResult:
    mode: str
    pnl_pct: float
    trades: int
    wr_pct: float
    avg_hold_bars: float
    median_hold_bars: float
    elapsed_sec: float


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_actor(ckpt_path: str, device: str, hidden_dim: int = 256) -> GaussianActor:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor" not in ckpt:
        raise KeyError(f"'actor' key missing in checkpoint: {ckpt_path}")
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM))
    actor = GaussianActor(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _split_by_time(df: pd.DataFrame, test_start: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required in csv")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().all():
        raise ValueError("timestamp parsing failed: all rows are NaT")
    work = df.copy()
    work["_ts"] = ts
    work = work.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    if end < start:
        raise ValueError(f"invalid range: end({end}) < start({start})")

    train_df = work[work["_ts"] < start].drop(columns=["_ts"]).reset_index(drop=True)
    test_df = work[(work["_ts"] >= start) & (work["_ts"] <= end)].drop(columns=["_ts"]).reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError("train split is empty; adjust test_start")
    if len(test_df) < 2:
        raise ValueError("test split too short; need at least 2 rows")
    return train_df, test_df


def _evaluate_one(
    actor: GaussianActor,
    device: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    fee: float,
    slip: float,
    hmm_fit_iter: int,
) -> EvalResult:
    t0 = time.time()

    hmm = None
    if mode == "slow":
        hmm_master = OnlineHMMDetector()
        hmm_master.fit(train_df, n_iter=int(hmm_fit_iter))
        hmm = copy.deepcopy(hmm_master)

    mtf = MultiTimeframeFeatures(test_df["close"].values.astype(np.float32))
    env = DSACCompactTradingEnv(
        test_df,
        phase="val",
        fee=float(fee),
        slip=float(slip),
        hmm_detector=hmm,
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    hold_bars_current = 0
    closed_hold_bars: list[int] = []

    while not done:
        prev_pos = env.pos
        if prev_pos is not None:
            hold_bars_current += 1

        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = float(actor.deterministic(st).cpu().item())
        state, _, done, _ = env.step(action)

        if prev_pos is None and env.pos is not None:
            hold_bars_current = 0
        elif prev_pos is not None and env.pos is None:
            closed_hold_bars.append(int(hold_bars_current))
            hold_bars_current = 0

    pnl_pct = float((env.balance / env.initial_balance - 1.0) * 100.0)
    wr_pct = float(env.win_rate * 100.0)
    avg_hold = float(np.mean(closed_hold_bars)) if closed_hold_bars else 0.0
    med_hold = float(np.median(closed_hold_bars)) if closed_hold_bars else 0.0
    elapsed = float(time.time() - t0)

    return EvalResult(
        mode=mode,
        pnl_pct=pnl_pct,
        trades=int(env.total_trades),
        wr_pct=wr_pct,
        avg_hold_bars=avg_hold,
        median_hold_bars=med_hold,
        elapsed_sec=elapsed,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare DSAC fast/slow path on one time split")
    p.add_argument("--csv-path", default="data/rl_training_data_full.csv")
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
    p.add_argument("--test-start", default="2026-01-01")
    p.add_argument("--test-end", default="2026-02-28 23:59:59")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--hmm-fit-iter", type=int, default=30)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"csv not found: {args.csv_path}")

    device = _resolve_device(args.device)
    df = pd.read_csv(args.csv_path)
    train_df, test_df = _split_by_time(df, args.test_start, args.test_end)
    actor = _load_actor(args.ckpt_path, device=device)

    fast = _evaluate_one(
        actor=actor,
        device=device,
        train_df=train_df,
        test_df=test_df,
        mode="fast",
        fee=args.fee,
        slip=args.slip,
        hmm_fit_iter=args.hmm_fit_iter,
    )
    slow = _evaluate_one(
        actor=actor,
        device=device,
        train_df=train_df,
        test_df=test_df,
        mode="slow",
        fee=args.fee,
        slip=args.slip,
        hmm_fit_iter=args.hmm_fit_iter,
    )

    out = {
        "config": {
            "csv_path": args.csv_path,
            "ckpt_path": args.ckpt_path,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "fee": float(args.fee),
            "slip": float(args.slip),
            "hmm_fit_iter": int(args.hmm_fit_iter),
            "device": device,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "fast": asdict(fast),
        "slow": asdict(slow),
        "delta": {
            "pnl_pct": float(slow.pnl_pct - fast.pnl_pct),
            "trades": int(slow.trades - fast.trades),
            "wr_pct": float(slow.wr_pct - fast.wr_pct),
            "avg_hold_bars": float(slow.avg_hold_bars - fast.avg_hold_bars),
            "median_hold_bars": float(slow.median_hold_bars - fast.median_hold_bars),
            "elapsed_sec": float(slow.elapsed_sec - fast.elapsed_sec),
            "runtime_ratio_slow_over_fast": float(slow.elapsed_sec / max(fast.elapsed_sec, 1e-9)),
        },
    }

    out_json = args.out_json.strip()
    if not out_json:
        os.makedirs("data/ensemble/metrics", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = f"data/ensemble/metrics/dsac_ab_compare_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("== DSAC A/B (one split) ==")
    print(f"test_range: {args.test_start} ~ {args.test_end}")
    print(f"rows: train={len(train_df)} test={len(test_df)}")
    print(
        "FAST | pnl={:.2f}% tr={} wr={:.2f}% avg_hold={:.2f}bars med_hold={:.2f}bars t={:.2f}s".format(
            fast.pnl_pct, fast.trades, fast.wr_pct, fast.avg_hold_bars, fast.median_hold_bars, fast.elapsed_sec
        )
    )
    print(
        "SLOW | pnl={:.2f}% tr={} wr={:.2f}% avg_hold={:.2f}bars med_hold={:.2f}bars t={:.2f}s".format(
            slow.pnl_pct, slow.trades, slow.wr_pct, slow.avg_hold_bars, slow.median_hold_bars, slow.elapsed_sec
        )
    )
    print(
        "DELTA(slow-fast) | pnl={:+.2f}% tr={:+d} wr={:+.2f}% avg_hold={:+.2f}bars med_hold={:+.2f}bars runtime×={:.2f}".format(
            out["delta"]["pnl_pct"],
            out["delta"]["trades"],
            out["delta"]["wr_pct"],
            out["delta"]["avg_hold_bars"],
            out["delta"]["median_hold_bars"],
            out["delta"]["runtime_ratio_slow_over_fast"],
        )
    )
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
