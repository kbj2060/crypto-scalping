#!/usr/bin/env python3
"""RESEARCH ONLY -- builds a DIRECTION+QUALITY label pair grid for ETH Omega4.6.1, all
triple-barrier based (replacing both zigzag_action direction AND the mismatched same_as_direction
quality that broke zig075 in the 2026-07-28 5-seed run).

Two label types per barrier config:
  - DIRECTION: first-touch triple barrier (whichever side's own tp is hit first wins), sequential/
    non-overlapping (position-gating), same mechanism as
    scripts/chart_eth_triple_barrier_label_ground_truth_20260728.py's label_event -- reused
    unmodified via import.
  - QUALITY: independent-per-bar (NOT sequential -- every bar tested regardless of a nearby open
    "position"), net-of-cost profitability filter, same methodology as the EXISTING production
    h48qual quality label (scripts/build_omega1_2_triple_barrier_labels_20260619.py's
    _reason_and_return + quality scoring: ret - fee_cost - 0.20*max(-mae,0) - 0.003*(reason==sl)),
    ported here to run on the 2024+2025 tape instead of that script's 2025-only source, and
    parameterized on the SAME barrier width as its paired direction config (not the original
    script's fixed h24/h48/h96 grid).

3 barrier widths tested, spanning what this session already found:
  dense  : tp=0.006  sl=0.0032 vertical=12  bars (1h)  -- 2026-07-28's max-density config
  medium : tp=0.020  sl=0.011  vertical=72  bars (6h)
  sparse : tp=0.045  sl=0.025  vertical=96  bars (8h)  -- close to h48qual's OWN existing
           h48_conservative quality label's ballpark (tp=0.006*1.2mult..., different formula but
           similar realized width), included so the grid brackets the known-working region.

Each component gets ITS OWN quality label at the SAME width as its own direction label (fixes
zig075's same_as_direction failure -- it gets a real, independently-built quality label instead
of literally reusing the direction label).

Output layout (drop-in --direction-label-dir / --quality-label-dir per config):
  tmp/causal_regen_20260516/eth_triple_barrier_grid_20260728/label_contracts/
    direction_{dense,medium,sparse}/zigzag_action_labels_{2024,2025,2026}.csv
    quality_{dense,medium,sparse}/zigzag_action_labels_{2024,2025,2026}.csv

Does NOT retrain anything, does NOT touch trading_bot_modules/, trading_bot.py, .env.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chart_eth_triple_barrier_label_ground_truth_20260728 as lbl  # noqa: E402
import train_eval_omega4_3head_parent72_pinned102_2024tape_20260727 as tape2024  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

ACTION_CASH, ACTION_LONG, ACTION_SHORT = 0, 1, 2
OUT_ROOT = ROOT / "tmp/causal_regen_20260516/eth_triple_barrier_grid_20260728/label_contracts"

CONFIGS = {
    "dense": dict(tp=0.006, sl=0.0032, vertical=12),
    "medium": dict(tp=0.020, sl=0.011, vertical=72),
    "sparse": dict(tp=0.045, sl=0.025, vertical=96),
}


# ---------------------------------------------------------------------------
# DIRECTION: sequential, first-touch (reuses lbl.label_event unmodified)
# ---------------------------------------------------------------------------
def build_direction_events(frame: pd.DataFrame, *, tp: float, sl: float, vertical: int) -> list[dict]:
    events = []
    t = 0
    last_end = len(frame) - vertical - 1
    t0 = time.time()
    while t < last_end:
        ev = lbl.label_event(frame, t, min_tp=tp, min_sl=sl, vertical_bars=vertical)
        events.append({"t": t, "exit_i": ev["exit_i"], "label": ev["label"]})
        t = ev["exit_i"] + 1 if ev["label"] != "CASH" else t + 1
        if len(events) % 40000 == 0:
            print(f"    ...direction {t}/{last_end} bars, {len(events)} events, {time.time()-t0:.0f}s", flush=True)
    return events


def expand_direction_to_per_bar(frame: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    action = np.full(len(frame), ACTION_CASH, dtype=np.int64)
    for ev in events:
        if ev["label"] == "CASH":
            continue
        code = ACTION_LONG if ev["label"] == "LONG" else ACTION_SHORT
        action[ev["t"]: ev["exit_i"] + 1] = code
    return pd.DataFrame({"timestamp": frame["timestamp"], "zigzag_action": action})


# ---------------------------------------------------------------------------
# QUALITY: independent per-bar, net-of-cost profitability filter (ports
# build_omega1_2_triple_barrier_labels_20260619.py's _reason_and_return/quality scoring)
# ---------------------------------------------------------------------------
def _reason_and_return(*, side: int, entry: float, future_high: np.ndarray, future_low: np.ndarray,
                        future_close: np.ndarray, tp_move: float, sl_move: float) -> tuple[float, str, float]:
    if entry <= 0.0:
        return 0.0, "invalid_entry", 0.0
    if side > 0:
        tp_level = entry * (1.0 + tp_move)
        sl_level = entry * (1.0 - sl_move)
        rel_low = future_low / entry - 1.0
        mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
        for hi, lo in zip(future_high, future_low):
            if lo <= sl_level:
                return -float(sl_move), "sl", mae
            if hi >= tp_level:
                return float(tp_move), "tp", mae
        return float(future_close[-1] / entry - 1.0) if len(future_close) else 0.0, "timeout", mae
    tp_level = entry * (1.0 - tp_move)
    sl_level = entry * (1.0 + sl_move)
    rel_low = 1.0 - future_high / entry
    mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
    for hi, lo in zip(future_high, future_low):
        if hi >= sl_level:
            return -float(sl_move), "sl", mae
        if lo <= tp_level:
            return float(tp_move), "tp", mae
    return float(1.0 - future_close[-1] / entry) if len(future_close) else 0.0, "timeout", mae


def build_quality_per_bar(frame: pd.DataFrame, *, tp: float, sl: float, vertical: int, fee_cost: float) -> pd.DataFrame:
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    ts = frame["timestamp"]
    action = np.full(n, ACTION_CASH, dtype=np.int64)
    last_i = n - vertical - 2
    t0 = time.time()
    for i in range(max(last_i, 0)):
        entry_i = i + 1
        end_i = entry_i + vertical
        entry = float(open_px[entry_i])
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, long_mae = _reason_and_return(side=1, entry=entry, future_high=fh, future_low=fl, future_close=fc, tp_move=tp, sl_move=sl)
        short_ret, short_reason, short_mae = _reason_and_return(side=-1, entry=entry, future_high=fh, future_low=fl, future_close=fc, tp_move=tp, sl_move=sl)
        long_q = long_ret - fee_cost - 0.20 * max(-long_mae, 0.0) - 0.003 * int(long_reason == "sl")
        short_q = short_ret - fee_cost - 0.20 * max(-short_mae, 0.0) - 0.003 * int(short_reason == "sl")
        if long_q > 0.0 and long_q >= short_q:
            action[i] = ACTION_LONG
        elif short_q > 0.0:
            action[i] = ACTION_SHORT
        if i % 60000 == 0 and i > 0:
            print(f"    ...quality {i}/{last_i} bars, {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame({"timestamp": ts, "zigzag_action": action})


def write_label(out: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out["year"] = pd.to_datetime(out["timestamp"]).dt.year
    for year, sub in out.groupby("year"):
        path = out_dir / f"zigzag_action_labels_{int(year)}.csv"
        sub[["timestamp", "zigzag_action"]].to_csv(path, index=False)
        counts = sub["zigzag_action"].value_counts().to_dict()
        print(f"    wrote {len(sub)} rows -> {path} (CASH={counts.get(0,0)} LONG={counts.get(1,0)} SHORT={counts.get(2,0)})", flush=True)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("stage=load_frame (2024+2025 tape, cmamba-fix applied)", flush=True)
    train_all, eval_df, _overlay = tape2024._load_omega_frames_2024tape()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"])
    print(f"train: {len(train_all)} rows, eval(OOS): {len(eval_df)} rows", flush=True)
    fee, slip = omega._load_fee_slip()
    fee_cost = float(fee + slip) * 2.0 * 3.0  # matches build_omega1_2_triple_barrier_labels_20260619.py's cost model

    for name, cfg in CONFIGS.items():
        if which not in ("all", name):
            continue
        print(f"=== config={name} tp={cfg['tp']} sl={cfg['sl']} vertical={cfg['vertical']} ===", flush=True)

        print(f"  [direction] TRAIN", flush=True)
        d_train = expand_direction_to_per_bar(train_all, build_direction_events(train_all, tp=cfg["tp"], sl=cfg["sl"], vertical=cfg["vertical"]))
        print(f"  [direction] EVAL(OOS)", flush=True)
        d_eval = expand_direction_to_per_bar(eval_df, build_direction_events(eval_df, tp=cfg["tp"], sl=cfg["sl"], vertical=cfg["vertical"]))
        write_label(pd.concat([d_train, d_eval], ignore_index=True), OUT_ROOT / f"direction_{name}")

        print(f"  [quality] TRAIN", flush=True)
        q_train = build_quality_per_bar(train_all, tp=cfg["tp"], sl=cfg["sl"], vertical=cfg["vertical"], fee_cost=fee_cost)
        print(f"  [quality] EVAL(OOS)", flush=True)
        q_eval = build_quality_per_bar(eval_df, tp=cfg["tp"], sl=cfg["sl"], vertical=cfg["vertical"], fee_cost=fee_cost)
        write_label(pd.concat([q_train, q_eval], ignore_index=True), OUT_ROOT / f"quality_{name}")

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
