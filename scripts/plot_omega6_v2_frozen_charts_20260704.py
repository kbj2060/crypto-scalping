#!/usr/bin/env python3
"""Plot price+trade-marker and equity-curve charts for the frozen Omega6 v2 winner
(scripts/replay_omega6_v2_oos_freeze_20260704.py::FROZEN), separately for the validation
window (2025-10-01..12-31) and the one-shot OOS window (2026-01-01.., tape caps 2026-02-28).

Uses cost1 (1x fee/slip) for the visualization; this is the same trade set already reported
in the contract doc, just rendered visually instead of only as summary stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import replay_omega6_v2_oos_freeze_20260704 as f  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega6_v2_oos_freeze_20260704"
FEE = 0.00020
SLIP = 0.00050


def _reconstruct(tape: pd.DataFrame, cfg, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict], np.ndarray]:
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    result = v2.run_variant(tape, cfg, start=start, end=end)
    trades = result["_trade_list"]
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)
    n = len(sub)

    equity = np.ones(n, dtype=np.float64)
    cash = 1.0
    enriched = []
    for t in trades:
        entry_i, exit_i, side = t["entry_i"], t["exit_i"], t["side"]
        entry_price = open_[min(entry_i + 1, n - 1)] * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
        exit_price = close[exit_i] * (1.0 - SLIP if side > 0 else 1.0 + SLIP)
        raw = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        notional = cfg.fixed_margin * cfg.fixed_leverage
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * FEE * notional
        equity[exit_i:] = cash
        enriched.append({**t, "entry_price": entry_price, "exit_price": exit_price, "pnl_pct": (cash / before - 1.0) * 100.0})
    return sub, enriched, equity


def _plot(sub: pd.DataFrame, trades: list[dict], equity: np.ndarray, title: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[2.2, 1])
    ts = sub["timestamp"]
    ax1.plot(ts, sub["close"], color="#888888", linewidth=0.8, zorder=1)

    for t in trades:
        entry_ts = sub.iloc[t["entry_i"] + 1]["timestamp"] if t["entry_i"] + 1 < len(sub) else sub.iloc[t["entry_i"]]["timestamp"]
        exit_ts = sub.iloc[t["exit_i"]]["timestamp"]
        win = t["win"]
        entry_color = "#1a7f37" if t["side"] > 0 else "#b91c1c"
        exit_color = "#1a7f37" if win else "#b91c1c"
        marker_in = "^" if t["side"] > 0 else "v"
        ax1.scatter([entry_ts], [t["entry_price"]], marker=marker_in, color=entry_color, s=45, zorder=3, edgecolors="black", linewidths=0.4)
        ax1.scatter([exit_ts], [t["exit_price"]], marker="x", color=exit_color, s=45, zorder=3, linewidths=1.6)
        ax1.plot([entry_ts, exit_ts], [t["entry_price"], t["exit_price"]], color=entry_color if win else "#999999", alpha=0.35, linewidth=1.0, zorder=2)

    ax1.set_ylabel("Close price")
    ax1.set_title(title)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1a7f37", markeredgecolor="black", markersize=8, label="Long entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#b91c1c", markeredgecolor="black", markersize=8, label="Short entry"),
        Line2D([0], [0], marker="x", color="#1a7f37", markersize=8, label="Win exit"),
        Line2D([0], [0], marker="x", color="#b91c1c", markersize=8, label="Loss exit"),
    ]
    ax1.legend(handles=legend, loc="upper left", fontsize=8)

    eq_pct = (equity - 1.0) * 100.0
    ax2.plot(ts, eq_pct, color="#1f6feb", linewidth=1.3)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.fill_between(ts, eq_pct, 0, where=eq_pct >= 0, color="#1f6feb", alpha=0.12)
    ax2.fill_between(ts, eq_pct, 0, where=eq_pct < 0, color="#b91c1c", alpha=0.12)
    ax2.set_ylabel("Cumulative PnL (%)")
    ax2.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> int:
    tape = v2.load_tape()
    tape_qt = v2.apply_quality_threshold(tape, f.FROZEN.quality_threshold)

    val_sub, val_trades, val_eq = _reconstruct(tape_qt, f.FROZEN, v2.VAL_START, v2.VAL_END)
    oos_sub, oos_trades, oos_eq = _reconstruct(tape_qt, f.FROZEN, f.OOS_START, f.OOS_END)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot(
        val_sub,
        val_trades,
        val_eq,
        f"Omega6 v2 frozen -- Validation 2025-10-01..12-31 ({len(val_trades)} trades, cost1)",
        OUT_DIR / "val_chart.png",
    )
    _plot(
        oos_sub,
        oos_trades,
        oos_eq,
        f"Omega6 v2 frozen -- OOS 2026-01-01..02-28 one-shot ({len(oos_trades)} trades, cost1)",
        OUT_DIR / "oos_chart.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
