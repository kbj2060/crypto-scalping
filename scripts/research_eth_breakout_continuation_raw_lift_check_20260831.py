#!/usr/bin/env python3
"""Raw-lift precheck for a NEW "특화 감지기" candidate: 돌파 지속 (breakout follow-through) --
the natural complement to V자반등 (which asks "does a sweep/extreme REVERSE cleanly"), this asks
"does a genuine close-confirmed breakout of the causal 48-bar level CONTINUE without reverting,
more than a random bar's forward move would". Same infrastructure as V자반등 (sweep_level/atr from
add_causal_columns), same "trigger vs random-bar baseline" precheck methodology as V_REBOUND's own
precheck (research_eth_sweep_v_rebound_random_bar_baseline_20260829.py) -- reused in spirit, not
copy-pasted (that script compared against V_REBOUND's specific label; this defines a fresh,
symmetric "continuation" outcome since none existed to reuse).

Trigger (deliberately the near-mirror of liquidity_sweep, not identical): sweep = wick pierces the
level AND close reclaims back inside (rejected). Breakout here = CLOSE itself is beyond the level
(confirmed, not just a wick) -- no reclaim requirement at the trigger bar itself; whether it holds
is exactly the question being tested.

Outcome ("genuine continuation", checked at 3 horizons matching this project's K_HORIZONS
convention: 1h/4h/8h = 12/48/96 bars): CLOSE stays beyond the level (using the SAME level fixed at
the trigger bar, not a rolling one) for EVERY bar in the window -- i.e. never closes back inside.
This is the direct mirror of what "sweep" penalizes (a reclaim) -- a breakout is doing its job
exactly when reclaim never happens. ATR-normalized forward move magnitude reported alongside as a
secondary, richer stat (not the hit/lift metric itself).

NOT a full label design -- this is the cheap first look only, matching this project's Homer
candidate-pool discipline (raw lift precheck before any phase1 diagnostic / TabPFN commitment).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_breakout_continuation_raw_lift_check_20260831"

START = pd.Timestamp("2024-01-01", tz="UTC")
K_HORIZONS = {"K12_1h": 12, "K48_4h": 48, "K96_8h": 96}


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_breakout_20260831", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df[df["timestamp"] >= START].reset_index(drop=True)


def continuation_stats(idx: np.ndarray, is_up: bool, close: np.ndarray, level: np.ndarray,
                        atr: np.ndarray, K: int, n: int) -> dict:
    """hit = close never crosses back inside `level` (fixed at the trigger bar) for all K bars
    forward. Also reports mean ATR-normalized forward move at K (secondary/descriptive only)."""
    valid = idx[(idx + K < n) & np.isfinite(atr[idx]) & (atr[idx] > 0)]
    if len(valid) == 0:
        return {"n": 0, "hit_rate": float("nan"), "mean_move_atr": float("nan")}
    lvl = level[valid]
    hits = np.zeros(len(valid), dtype=bool)
    move_atr = np.zeros(len(valid))
    for j, i in enumerate(valid):
        fut_close = close[i + 1: i + K + 1]
        if is_up:
            hits[j] = bool((fut_close > lvl[j]).all())
        else:
            hits[j] = bool((fut_close < lvl[j]).all())
        end_move = (close[i + K] - close[i]) if is_up else (close[i] - close[i + K])
        move_atr[j] = end_move / atr[i]
    return {"n": int(len(valid)), "hit_rate": float(hits.mean()),
            "mean_move_atr": float(move_atr.mean())}


def random_baseline(all_pos: np.ndarray, is_up: bool, close: np.ndarray, level_up: np.ndarray,
                     level_down: np.ndarray, atr: np.ndarray, K: int, n: int) -> dict:
    """Same 'stays beyond a level' check, but the level is each random bar's OWN causal 48-bar
    level (not conditioned on having broken it) -- answers 'how often does price just happen to
    stay beyond ITS current level for K bars anyway, breakout or not'."""
    level = level_up if is_up else level_down
    return continuation_stats(all_pos, is_up, close, level, atr, K, n)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    impl = load_impl()
    kl = load_klines()
    causal = impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())

    close = kl["close"].to_numpy()
    level_high = causal["sweep_level_high"].to_numpy()
    level_low = causal["sweep_level_low"].to_numpy()
    atr = causal["atr"].to_numpy()
    n = len(kl)

    is_breakout_up = np.isfinite(level_high) & (close > level_high)
    is_breakout_down = np.isfinite(level_low) & (close < level_low)
    up_idx = np.flatnonzero(is_breakout_up)
    down_idx = np.flatnonzero(is_breakout_down)
    all_idx = np.arange(n)

    print(f"bars={n} ({kl['timestamp'].iloc[0]} ~ {kl['timestamp'].iloc[-1]})")
    print(f"raw breakout triggers (undeduped): up={len(up_idx)} ({len(up_idx)/n*100:.2f}% of bars), "
          f"down={len(down_idx)} ({len(down_idx)/n*100:.2f}% of bars)\n")

    rows = []
    for side, idx, is_up in (("up", up_idx, True), ("down", down_idx, False)):
        for k_name, K in K_HORIZONS.items():
            trig = continuation_stats(idx, is_up, close, level_high if is_up else level_low, atr, K, n)
            base = random_baseline(all_idx, is_up, close, level_high, level_low, atr, K, n)
            lift = (trig["hit_rate"] / base["hit_rate"]) if base["hit_rate"] else float("nan")
            row = {"side": side, "horizon": k_name, "K_bars": K,
                   "n_triggers": trig["n"], "trigger_hit_rate": round(trig["hit_rate"], 4),
                   "baseline_hit_rate": round(base["hit_rate"], 4), "lift_x": round(lift, 3),
                   "trigger_mean_move_atr": round(trig["mean_move_atr"], 3),
                   "baseline_mean_move_atr": round(base["mean_move_atr"], 3)}
            rows.append(row)
            print(f"  {side:5s} {k_name:8s}: n={row['n_triggers']:6d}  hit={row['trigger_hit_rate']:.1%} "
                  f"vs baseline={row['baseline_hit_rate']:.1%}  lift={row['lift_x']:.2f}x  "
                  f"move(ATR): trig={row['trigger_mean_move_atr']:+.2f} base={row['baseline_mean_move_atr']:+.2f}")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
