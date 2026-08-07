#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/causal_regen_20260516/coin_trade_charts_20260709"

PRICE_FILES = {
    ("eth", "val"): ROOT / "data/splits/year_oos/training_features_2025.csv",
    ("eth", "oos"): ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
    ("sol", "val"): ROOT / "data/splits/year_oos/sol_features_2025.csv",
    ("sol", "oos"): ROOT / "data/splits/year_oos/sol_features_2026.csv",
    ("btc", "val"): ROOT / "data/splits/year_oos/btc_features_2025.csv",
    ("btc", "oos"): ROOT / "data/splits/year_oos/btc_features_2026.csv",
}

LEDGERS = {
    ("eth", "val"): ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv",
    ("eth", "oos"): ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv",
    ("sol", "val"): ROOT / "tmp/causal_regen_20260516/sol_val_stability_exact_20260708/validation_ledger.csv",
    ("sol", "oos"): ROOT / "tmp/causal_regen_20260516/sol_val_stability_exact_20260708/oos_ledger.csv",
    ("btc", "val"): ROOT / "tmp/causal_regen_20260516/btc_omega4_6_1_two_component_router_20260708/validation_router_ledger.csv",
    ("btc", "oos"): ROOT / "tmp/causal_regen_20260516/btc_omega4_6_1_fast_param_search_20260708_min15/selected_oos_gated_ledger.csv",
}

DATE_WINDOWS = {
    ("eth", "val"): ("2025-10-01", "2025-12-31 23:59:59"),
    ("eth", "oos"): ("2026-01-01", "2026-06-30 23:59:59"),
    ("sol", "val"): ("2025-10-01", "2025-12-31 23:59:59"),
    ("sol", "oos"): ("2026-01-01", "2026-06-30 23:59:59"),
    ("btc", "val"): ("2025-10-01", "2025-12-31 23:59:59"),
    ("btc", "oos"): ("2026-01-01", "2026-06-30 23:59:59"),
}


def load_price(asset: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(
        PRICE_FILES[(asset, split_name)],
        usecols=lambda c: c in {"timestamp", "close", "open", "high", "low"},
        low_memory=False,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start, end = map(pd.Timestamp, DATE_WINDOWS[(asset, split_name)])
    return (
        df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def load_ledger(asset: str, split_name: str) -> pd.DataFrame:
    led = pd.read_csv(LEDGERS[(asset, split_name)], low_memory=False)
    if led.empty:
        return led
    led["entry_timestamp"] = pd.to_datetime(led["entry_timestamp"])
    led["exit_timestamp"] = pd.to_datetime(led["exit_timestamp"])
    if asset == "eth" and split_name == "val" and "ou_halflife" in led.columns:
        led = led[pd.to_numeric(led["ou_halflife"], errors="coerce") > 0.005417].copy()
    led["trade_return"] = pd.to_numeric(led["trade_return"], errors="coerce").fillna(0.0)
    led = led[led["trade_return"].abs() > 1e-12].copy()
    return led.sort_values("entry_timestamp").reset_index(drop=True)


def metrics(led: pd.DataFrame) -> tuple[float, float, int, float]:
    if led.empty:
        return 0.0, 0.0, 0, 0.0
    rets = led["trade_return"].to_numpy(dtype=np.float64)
    eq = np.cumprod(1.0 + rets)
    curve = np.r_[1.0, eq]
    peak = np.maximum.accumulate(curve)
    mdd = (curve / np.maximum(peak, 1e-12) - 1.0).min()
    return float((eq[-1] - 1.0) * 100.0), float(mdd * 100.0), int(len(led)), float((rets > 0.0).mean() * 100.0)


def px_at(price: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = int(price["timestamp"].searchsorted(ts))
    if idx >= len(price):
        idx = len(price) - 1
    if idx > 0 and abs(price["timestamp"].iloc[idx] - ts) > abs(price["timestamp"].iloc[idx - 1] - ts):
        idx -= 1
    return float(price["close"].iloc[idx])


def tp_sl_prices(entry_px: float, side: int, take_profit: float, stop_loss: float) -> tuple[float | None, float | None]:
    tp = float(take_profit) if np.isfinite(float(take_profit)) else 0.0
    sl = abs(float(stop_loss)) if np.isfinite(float(stop_loss)) else 0.0
    if side > 0:
        tp_px = entry_px * (1.0 + tp) if tp > 0.0 else None
        sl_px = entry_px * (1.0 - sl) if sl > 0.0 else None
    else:
        tp_px = entry_px * (1.0 - tp) if tp > 0.0 else None
        sl_px = entry_px * (1.0 + sl) if sl > 0.0 else None
    return tp_px, sl_px


def plot_asset(asset: str) -> list[dict[str, object]]:
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.5), gridspec_kw={"height_ratios": [2.1, 1.0]}, sharex="col")
    summary: list[dict[str, object]] = []
    for col, split_name in enumerate(["val", "oos"]):
        price = load_price(asset, split_name)
        led = load_ledger(asset, split_name)
        pnl, mdd, n, wr = metrics(led)
        summary.append(
            {
                "asset": asset,
                "split": split_name,
                "pnl_pct": pnl,
                "mdd_pct": mdd,
                "trades": n,
                "wr_pct": wr,
                "ledger": str(LEDGERS[(asset, split_name)].relative_to(ROOT)),
            }
        )
        ax = axes[0, col]
        ax.plot(price["timestamp"], price["close"], color="#222222", lw=0.8, alpha=0.85, label="close")
        if not led.empty:
            if {"take_profit", "stop_loss"}.issubset(led.columns):
                for _, row in led.iterrows():
                    entry_px = px_at(price, row["entry_timestamp"])
                    tp_px, sl_px = tp_sl_prices(
                        entry_px,
                        int(row["side"]),
                        float(row["take_profit"]),
                        float(row["stop_loss"]),
                    )
                    x0, x1 = row["entry_timestamp"], row["exit_timestamp"]
                    if tp_px is not None:
                        ax.plot([x0, x1], [tp_px, tp_px], color="#2ca02c", lw=0.7, ls="--", alpha=0.45)
                    if sl_px is not None:
                        ax.plot([x0, x1], [sl_px, sl_px], color="#d62728", lw=0.7, ls="--", alpha=0.45)
            longs = led[led["side"].astype(int) > 0]
            shorts = led[led["side"].astype(int) < 0]
            for sub, marker, color, label in [
                (longs, "^", "#0a8f4d", "long entry"),
                (shorts, "v", "#c0392b", "short entry"),
            ]:
                if not sub.empty:
                    y = [px_at(price, ts) for ts in sub["entry_timestamp"]]
                    ax.scatter(
                        sub["entry_timestamp"],
                        y,
                        marker=marker,
                        s=42,
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                        label=label,
                        zorder=5,
                    )
            y_exit = [px_at(price, ts) for ts in led["exit_timestamp"]]
            colors = np.where(led["trade_return"].to_numpy(dtype=np.float64) > 0.0, "#1f77b4", "#ff7f0e")
            ax.scatter(led["exit_timestamp"], y_exit, marker="x", s=36, color=colors, label="exit", zorder=6)
            if {"take_profit", "stop_loss"}.issubset(led.columns):
                ax.plot([], [], color="#2ca02c", lw=0.9, ls="--", alpha=0.65, label="take profit")
                ax.plot([], [], color="#d62728", lw=0.9, ls="--", alpha=0.65, label="stop loss")
        title_split = "VAL 2025-10..12" if split_name == "val" else "OOS 2026-01..06"
        ax.set_title(f"{asset.upper()} {title_split} | PnL {pnl:.1f}% MDD {mdd:.1f}% N {n} WR {wr:.0f}%")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper left", fontsize=8, ncols=2)

        ax_eq = axes[1, col]
        if not led.empty:
            eq = np.cumprod(1.0 + led["trade_return"].to_numpy(dtype=np.float64))
            ax_eq.step(led["exit_timestamp"], eq, where="post", color="#2454a6", lw=1.6)
        ax_eq.axhline(1.0, color="#999999", lw=0.8)
        ax_eq.set_ylabel("equity")
        ax_eq.grid(True, alpha=0.22)
        ax_eq.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax_eq.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_eq.xaxis.get_major_locator()))

    fig.suptitle(f"{asset.upper()} Individual Model Trade Chart", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"{asset}_val_oos_trade_chart.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for asset in ["eth", "sol", "btc"]:
        rows.extend(plot_asset(asset))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(str(OUT))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
