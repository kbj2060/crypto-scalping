#!/usr/bin/env python3
"""Oracle-ceiling verification for the BTC h48_conservative triple-barrier label -- the label that
feeds h48qual's Quality head via --quality-mode quality_label_action
(tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708, itself padded
from tb_action_h48_conservative in
tmp/causal_regen_20260516/btc_omega1_2_triple_barrier_labels_20260708/*_triple_barrier_labels.csv,
built by scripts/build_omega1_2_triple_barrier_labels_btc_20260708.py).

Three checks, all against the OOS split (2026-01-01..03-31):

1. RE-DERIVATION SPOT CHECK. Independently recompute the TP/SL/timeout outcome for a random sample
   of rows directly from data/splits/year_oos/btc_features_2026.csv (raw OHLC), using the exact same
   rule documented in the builder (entry = next bar open, ATR = past-only rolling(96, min24).shift(1),
   same-bar tie -> stop-loss-first) and diff against the saved tb_action/tb_long_reason/
   tb_short_reason/tb_long_ret/tb_short_ret. This tests the LABEL ITSELF, not the code that built it
   -- a bug in build_omega1_2_triple_barrier_labels_btc_20260708.py would reproduce identically here
   only if this script makes the same mistake, so an independent re-implementation is the point.

2. ORACLE CEILING. The per-bar label is defined conditional on entering (not a position path -- spans
   overlap), so it is converted to one the same way scripts/stage0d_btc_oracle_dp_vs_tripbarrier_
   20260808.py converts triple-barrier labels: walk forward, take the first directional action on a
   free bar, hold for exactly that side's realized bar-count (tb_long_bars/tb_short_bars), become free
   again. This is the h48_conservative-specific analogue of that script's tripbarrier_seq (which used
   a DIFFERENT triple-barrier build, btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808 --
   not directly comparable to this label).

3. WARM-START ARTIFACT. tb_atr_price_move is computed from a rolling(96, min_periods=24) window on
   the OOS frame ALONE (no 2025 tail carried over), so the first ~24 OOS bars fall back to the floor
   tp/sl (min_tp/min_sl) instead of the true ATR-scaled barrier. Quantified, not fixed here.

Outputs a JSON summary + two PNGs: full-OOS oracle equity curve, and a 1-week zoomed price+trade
chart (first calendar week of OOS, 2026-01-01..01-08, chosen for neutrality -- not cherry-picked).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))

TB_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega1_2_triple_barrier_labels_20260708"
RAW_OOS_CSV = ROOT / "data/splits/year_oos/btc_features_2026.csv"
OUT_DIR = ROOT / "tmp/research_20260810_h48qual_oracle"
CFG_NAME = "h48_conservative"
HORIZON = 48
TP_MULT, SL_MULT = 1.2, 0.8
MIN_TP, MIN_SL = 0.006, 0.004
FEE_RATE, SLIP_RATE = 0.0005, 0.0002
FEE_COST = (FEE_RATE + SLIP_RATE) * 2.0 * 3.0  # matches builder's cost_mult=3.0 convention

WEEK_START = pd.Timestamp("2026-01-01")
WEEK_END = pd.Timestamp("2026-01-08")

COLOR_TP = "#1B8A5A"
COLOR_SL = "#C0392B"
COLOR_TIMEOUT = "#7F8C8D"
COLOR_PRICE = "#7A828C"
COLOR_LONG = "#2C6FBB"
COLOR_SHORT = "#B5651D"


def _atr_price_move(frame: pd.DataFrame) -> np.ndarray:
    high, low, close = (pd.to_numeric(frame[c], errors="raise").astype(float) for c in ("high", "low", "close"))
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = (tr / close.replace(0.0, np.nan)).rolling(96, min_periods=24).mean().shift(1)
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _reason_and_return(side: int, entry: float, fut_high: np.ndarray, fut_low: np.ndarray, fut_close: np.ndarray,
                        tp_move: float, sl_move: float) -> tuple[float, str, int]:
    if side > 0:
        tp_level, sl_level = entry * (1 + tp_move), entry * (1 - sl_move)
        for bars, (hi, lo) in enumerate(zip(fut_high, fut_low), start=1):
            if lo <= sl_level:
                return -sl_move, "sl", bars
            if hi >= tp_level:
                return tp_move, "tp", bars
        return float(fut_close[-1] / entry - 1.0), "timeout", len(fut_close)
    tp_level, sl_level = entry * (1 - tp_move), entry * (1 + sl_move)
    for bars, (hi, lo) in enumerate(zip(fut_high, fut_low), start=1):
        if hi >= sl_level:
            return -sl_move, "sl", bars
        if lo <= tp_level:
            return tp_move, "tp", bars
    return float(1.0 - fut_close[-1] / entry), "timeout", len(fut_close)


def redo_spot_check(raw: pd.DataFrame, labels: pd.DataFrame, *, n_sample: int, seed: int) -> dict:
    """Independently re-derive tb_action/reason/return for a random sample and diff vs saved labels."""
    rng = np.random.default_rng(seed)
    open_px = pd.to_numeric(raw["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(raw["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(raw["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(raw["close"], errors="raise").to_numpy(dtype=np.float64)
    atr = _atr_price_move(raw)
    n = len(raw)
    last_i = n - HORIZON - 2
    candidates = rng.choice(np.arange(24, max(last_i, 25)), size=min(n_sample, max(last_i - 24, 0)), replace=False)

    lab = labels.set_index("timestamp")
    mismatches = []
    for i in sorted(int(x) for x in candidates):
        ts = raw["timestamp"].iloc[i]
        if ts not in lab.index:
            continue
        row = lab.loc[ts]
        entry_i = i + 1
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(MIN_TP, TP_MULT * vol)
        sl_move = max(MIN_SL, SL_MULT * vol)
        end_i = entry_i + HORIZON
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, _long_bars = _reason_and_return(1, entry, fh, fl, fc, tp_move, sl_move)
        short_ret, short_reason, _short_bars = _reason_and_return(-1, entry, fh, fl, fc, tp_move, sl_move)
        exp = {
            "long_ret": long_ret, "short_ret": short_ret,
            "long_reason": long_reason, "short_reason": short_reason,
        }
        got = {
            "long_ret": float(row[f"tb_long_ret_{CFG_NAME}"]), "short_ret": float(row[f"tb_short_ret_{CFG_NAME}"]),
            "long_reason": str(row[f"tb_long_reason_{CFG_NAME}"]), "short_reason": str(row[f"tb_short_reason_{CFG_NAME}"]),
        }
        diffs = {}
        for k in ("long_reason", "short_reason"):
            if exp[k] != got[k]:
                diffs[k] = {"expected": exp[k], "got": got[k]}
        for k in ("long_ret", "short_ret"):
            if abs(exp[k] - got[k]) > 1e-9:
                diffs[k] = {"expected": exp[k], "got": got[k]}
        if diffs:
            mismatches.append({"timestamp": str(ts), "diffs": diffs})
    return {"n_checked": len(candidates), "n_mismatch": len(mismatches), "mismatches": mismatches[:10]}


def oracle_sequential_path(labels: pd.DataFrame) -> pd.DataFrame:
    """Sequential non-overlapping oracle trader: first directional action on a free bar, hold for
    that side's realized bar count. Same conversion convention as stage0d_btc_oracle_dp_vs_
    tripbarrier_20260808.py's tb_position_path, applied to h48_conservative specifically."""
    action = labels[f"tb_action_{CFG_NAME}"].to_numpy()
    long_ret = labels[f"tb_long_ret_{CFG_NAME}"].to_numpy(dtype=np.float64)
    short_ret = labels[f"tb_short_ret_{CFG_NAME}"].to_numpy(dtype=np.float64)
    long_bars = labels[f"tb_long_bars_{CFG_NAME}"].to_numpy(dtype=np.int64)
    short_bars = labels[f"tb_short_bars_{CFG_NAME}"].to_numpy(dtype=np.int64)
    long_reason = labels[f"tb_long_reason_{CFG_NAME}"].to_numpy()
    short_reason = labels[f"tb_short_reason_{CFG_NAME}"].to_numpy()
    ts = labels["timestamp"].to_numpy()
    n = len(labels)

    trades = []
    i = 0
    while i < n:
        a = int(action[i])
        if a == 0:
            i += 1
            continue
        if a == 1:
            ret, bars, reason = float(long_ret[i]) - FEE_COST, int(long_bars[i]), str(long_reason[i])
        else:
            ret, bars, reason = float(short_ret[i]) - FEE_COST, int(short_bars[i]), str(short_reason[i])
        j = min(i + max(bars, 1), n)
        trades.append({"entry_i": i, "exit_i": j - 1, "entry_ts": ts[i], "exit_ts": ts[j - 1] if j - 1 < n else ts[-1],
                        "side": "LONG" if a == 1 else "SHORT", "reason": reason, "net_ret": ret, "bars": bars})
        i = j
    return pd.DataFrame(trades)


def compound_curve(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cash, peak, mdd = 1.0, 1.0, 0.0
    curve = []
    wins = 0
    for _, t in trades.iterrows():
        cash *= (1.0 + float(t["net_ret"]))
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        wins += int(t["net_ret"] > 0.0)
        curve.append({"exit_ts": t["exit_ts"], "equity": cash})
    curve_df = pd.DataFrame(curve)
    metrics = {
        "trades": int(len(trades)),
        "total_return_pct": round((cash - 1.0) * 100.0, 2),
        "mdd_pct": round(mdd * 100.0, 2),
        "win_rate": round(wins / max(len(trades), 1), 4),
        "avg_hold_bars": round(float(trades["bars"].mean()), 1) if len(trades) else 0.0,
        "reason_counts": trades["reason"].value_counts().to_dict() if len(trades) else {},
        "side_counts": trades["side"].value_counts().to_dict() if len(trades) else {},
    }
    return curve_df, metrics


def chart_full_equity(curve: pd.DataFrame, metrics: dict) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(pd.to_datetime(curve["exit_ts"]), (curve["equity"] - 1.0) * 100.0, color=COLOR_LONG, linewidth=1.2)
    ax.axhline(0, color="#B0B7BF", linewidth=0.8, linestyle=":")
    ax.set_title(f"BTC h48_conservative oracle ceiling, sequential non-overlapping, OOS 2026-01-01..03-31\n"
                 f"{metrics['trades']} trades, total {metrics['total_return_pct']:+.1f}%, "
                 f"MDD {metrics['mdd_pct']:.1f}%, win rate {metrics['win_rate']:.1%}, "
                 f"avg hold {metrics['avg_hold_bars']:.0f} bars", fontsize=11)
    ax.set_ylabel("Cumulative return (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "h48qual_oracle_oos_full_equity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_one_week(raw: pd.DataFrame, trades: pd.DataFrame) -> Path:
    week_raw = raw[(raw["timestamp"] >= WEEK_START) & (raw["timestamp"] < WEEK_END)].reset_index(drop=True)
    week_trades = trades[(pd.to_datetime(trades["entry_ts"]) >= WEEK_START) & (pd.to_datetime(trades["entry_ts"]) < WEEK_END)]

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(week_raw["timestamp"], week_raw["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1,
            label="BTC close (OOS)")

    reason_color = {"tp": COLOR_TP, "sl": COLOR_SL, "timeout": COLOR_TIMEOUT}
    reason_label = {"tp": "TP touched", "sl": "SL touched", "timeout": f"{HORIZON}-bar timeout"}
    seen_reason: set[str] = set()
    seen_side: set[str] = set()
    px_by_ts = week_raw.set_index("timestamp")["close"].astype(float)

    for _, t in week_trades.iterrows():
        entry_ts, exit_ts = pd.Timestamp(t["entry_ts"]), pd.Timestamp(t["exit_ts"])
        entry_px = float(px_by_ts.get(entry_ts, np.nan))
        if not np.isfinite(entry_px):
            continue
        side = t["side"]
        tp_line = entry_px * (1 + MIN_TP) if side == "LONG" else entry_px * (1 - MIN_TP)
        sl_line = entry_px * (1 - MIN_SL) if side == "LONG" else entry_px * (1 + MIN_SL)
        ax.plot([entry_ts, exit_ts], [entry_px, entry_px], color="#B0B7BF", linewidth=0.8, linestyle=":", zorder=2)
        marker = "^" if side == "LONG" else "v"
        color = COLOR_LONG if side == "LONG" else COLOR_SHORT
        ax.scatter([entry_ts], [entry_px], marker=marker, s=70, color=color, zorder=4, edgecolor="white",
                   linewidth=0.6, label=f"{side} entry" if side not in seen_side else None)
        seen_side.add(side)
        exit_px = float(px_by_ts.get(exit_ts, entry_px))
        exit_marker = {"tp": "o", "sl": "X", "timeout": "s"}[t["reason"]]
        ax.scatter([exit_ts], [exit_px], marker=exit_marker, s=60, color=reason_color[t["reason"]], zorder=4,
                   edgecolor="white", linewidth=0.6, label=reason_label[t["reason"]] if t["reason"] not in seen_reason else None)
        seen_reason.add(t["reason"])

    ax.set_title(f"BTC h48_conservative oracle trades, OOS week {WEEK_START.date()}..{(WEEK_END - pd.Timedelta(days=1)).date()}\n"
                 f"{len(week_trades)} sequential non-overlapping trades this week (min_tp={MIN_TP:.3f}, min_sl={MIN_SL:.3f} floor, ATR-scaled above)",
                 fontsize=11)
    ax.set_ylabel("BTC price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncols=2)
    fig.tight_layout()
    out = OUT_DIR / "h48qual_oracle_oos_week1_20260101_0107.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(TB_DIR / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"], low_memory=False)
    raw = pd.read_csv(RAW_OOS_CSV, parse_dates=["timestamp"], low_memory=False).sort_values("timestamp").reset_index(drop=True)

    report: dict = {"config": {"name": CFG_NAME, "horizon": HORIZON, "tp_mult": TP_MULT, "sl_mult": SL_MULT,
                                "min_tp": MIN_TP, "min_sl": MIN_SL, "fee_cost_per_trade": FEE_COST}}

    print("=== 1. re-derivation spot check (n=300, seed=20260810) ===", flush=True)
    spot = redo_spot_check(raw, labels, n_sample=300, seed=20260810)
    report["spot_check"] = spot
    print(json.dumps(spot, indent=2, default=str)[:2000], flush=True)

    print("\n=== 2. ATR warm-start window ===", flush=True)
    atr_col = f"tb_atr_price_move_{CFG_NAME}"
    zero_atr = labels[labels[atr_col] == 0.0]
    warmup = {"n_zero_atr_rows": int(len(zero_atr)), "n_total_rows": int(len(labels)),
              "last_zero_atr_ts": str(zero_atr["timestamp"].max()) if len(zero_atr) else None}
    report["atr_warmup_artifact"] = warmup
    print(json.dumps(warmup, indent=2), flush=True)

    print("\n=== 3. oracle sequential non-overlapping trade path (OOS) ===", flush=True)
    trades = oracle_sequential_path(labels)
    curve, metrics = compound_curve(trades)
    report["oracle_ceiling"] = metrics
    print(json.dumps(metrics, indent=2, default=str), flush=True)

    trades_path = OUT_DIR / "h48qual_oracle_oos_trades.csv"
    trades.to_csv(trades_path, index=False)
    report["artifacts"] = {"trades_csv": str(trades_path)}

    print("\n=== charts ===", flush=True)
    eq_png = chart_full_equity(curve, metrics)
    week_png = chart_one_week(raw, trades)
    report["artifacts"]["full_equity_png"] = str(eq_png)
    report["artifacts"]["week1_png"] = str(week_png)
    print(f"  {eq_png}\n  {week_png}", flush=True)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
