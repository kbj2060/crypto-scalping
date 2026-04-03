#!/usr/bin/env python3
"""Directional bias diagnostics for RL training CSV.

Checks whether the dataset itself structurally favors LONG or SHORT after costs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIME_COLS = ["regime_chop", "regime_whipsaw", "regime_bull", "regime_bear", "regime_normal"]


@dataclass
class Stats:
    n: int
    long_pos_rate: float
    short_pos_rate: float
    long_mean: float
    short_mean: float
    long_p50: float
    short_p50: float
    long_p90: float
    short_p90: float


def _safe_q(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.nanquantile(x, q))


def _calc_stats(long_edge: np.ndarray, short_edge: np.ndarray) -> Stats:
    n = int(long_edge.shape[0])
    return Stats(
        n=n,
        long_pos_rate=float(np.mean(long_edge > 0.0)) if n > 0 else 0.0,
        short_pos_rate=float(np.mean(short_edge > 0.0)) if n > 0 else 0.0,
        long_mean=float(np.mean(long_edge)) if n > 0 else 0.0,
        short_mean=float(np.mean(short_edge)) if n > 0 else 0.0,
        long_p50=_safe_q(long_edge, 0.50),
        short_p50=_safe_q(short_edge, 0.50),
        long_p90=_safe_q(long_edge, 0.90),
        short_p90=_safe_q(short_edge, 0.90),
    )


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:+.3f}%"


def diagnose(
    csv_path: str,
    horizon: int,
    fee: float,
    slip: float,
    use_log_return: bool,
) -> int:
    df = pd.read_csv(csv_path)
    if "close" not in df.columns:
        raise ValueError("`close` column is required")

    close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    valid = np.isfinite(close) & (close > 0.0)
    if valid.sum() < horizon + 10:
        raise ValueError("not enough valid close rows")

    close_s = pd.Series(close)
    fwd = close_s.shift(-horizon).to_numpy(dtype=np.float64)
    base = close_s.to_numpy(dtype=np.float64)
    mask = np.isfinite(base) & np.isfinite(fwd) & (base > 0.0) & (fwd > 0.0)
    base = base[mask]
    fwd = fwd[mask]

    round_cost = 2.0 * (float(fee) + float(slip))
    if use_log_return:
        fwd_ret = np.log(fwd / base)
        long_edge = fwd_ret - round_cost
        short_edge = -fwd_ret - round_cost
    else:
        fwd_ret = (fwd / base) - 1.0
        long_edge = fwd_ret - round_cost
        short_edge = -fwd_ret - round_cost

    all_stats = _calc_stats(long_edge, short_edge)
    long_win = long_edge[long_edge > 0.0]
    short_win = short_edge[short_edge > 0.0]

    print("=== Side Bias Diagnosis ===")
    print(f"csv={csv_path}")
    print(f"rows_raw={len(df)} | rows_used={all_stats.n} | horizon={horizon}")
    print(f"cost_roundtrip={round_cost:.6f} | return_mode={'log' if use_log_return else 'simple'}")
    print("")
    print("[Overall Opportunity Frequency]")
    print(
        f"long_edge>0: {all_stats.long_pos_rate:.4f} | short_edge>0: {all_stats.short_pos_rate:.4f} | "
        f"gap(short-long): {all_stats.short_pos_rate - all_stats.long_pos_rate:+.4f}"
    )
    print("[Overall Edge Distribution]")
    print(
        f"long mean={_fmt_pct(all_stats.long_mean)} p50={_fmt_pct(all_stats.long_p50)} p90={_fmt_pct(all_stats.long_p90)}"
    )
    print(
        f"short mean={_fmt_pct(all_stats.short_mean)} p50={_fmt_pct(all_stats.short_p50)} p90={_fmt_pct(all_stats.short_p90)}"
    )
    print("[Conditional Positive Edge]")
    print(
        f"long count={long_win.size} mean={_fmt_pct(float(np.mean(long_win)) if long_win.size else 0.0)} | "
        f"short count={short_win.size} mean={_fmt_pct(float(np.mean(short_win)) if short_win.size else 0.0)}"
    )
    print("")

    # Regime decomposition
    reg_cols = [c for c in REGIME_COLS if c in df.columns]
    if reg_cols:
        reg_raw = df[reg_cols].to_numpy(dtype=np.float64)
        reg_idx = np.argmax(reg_raw, axis=1)
        reg_name = [reg_cols[i].replace("regime_", "") for i in reg_idx]
        reg_name = np.asarray(reg_name)[mask]
        print("[Regime Breakdown]")
        for rn in sorted(set(reg_name.tolist())):
            m = reg_name == rn
            s = _calc_stats(long_edge[m], short_edge[m])
            print(
                f"{rn:8s} n={s.n:6d} | long>0={s.long_pos_rate:.4f} short>0={s.short_pos_rate:.4f} "
                f"| long_mean={_fmt_pct(s.long_mean)} short_mean={_fmt_pct(s.short_mean)}"
            )
        print("")

    # M7 model directional preference (if available)
    has_m7 = {"m7_trend_xgb_up", "m7_trend_xgb_dn"}.issubset(df.columns)
    if has_m7:
        up = pd.to_numeric(df["m7_trend_xgb_up"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)[mask]
        dn = pd.to_numeric(df["m7_trend_xgb_dn"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)[mask]
        pref_long = float(np.mean(up > dn)) if up.size else 0.0
        pref_short = float(np.mean(dn > up)) if up.size else 0.0
        tie = 1.0 - pref_long - pref_short
        print("[M7 Direction Preference]")
        print(f"pref_long={pref_long:.4f} | pref_short={pref_short:.4f} | tie={tie:.4f}")
        print("")

    bias_msg = "SHORT_BIAS" if all_stats.short_pos_rate > all_stats.long_pos_rate else "LONG_BIAS"
    print("[Conclusion]")
    print(f"data_bias={bias_msg} | opportunity_gap={all_stats.short_pos_rate - all_stats.long_pos_rate:+.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose directional bias in RL CSV")
    p.add_argument("--csv-path", default="data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--horizon", type=int, default=3, help="Forward steps (5m bars => 3 means 15m)")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--simple-return", action="store_true", help="Use simple return instead of log return")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        diagnose(
            csv_path=args.csv_path,
            horizon=int(args.horizon),
            fee=float(args.fee),
            slip=float(args.slip),
            use_log_return=not bool(args.simple_return),
        )
    )

