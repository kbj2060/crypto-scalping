"""One-week OOS BTC price chart with regime background shading, HMM vs JM, as two separate PNGs.
Ad-hoc visualization for the HMM->JM regime3 swap line (see
project-btc-regime3-jm-lam4-swap-retrain-20260809 memory); not a project artifact, just a chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

WEEK_START = pd.Timestamp("2026-01-01 00:00:00")
WEEK_END = pd.Timestamp("2026-01-08 00:00:00")

PRICE_CSV = ROOT / "data/splits/year_oos/btc_features_2026.csv"
HMM_CSV = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/btc_features_2026_regime3_current_sensitive_hmm_wide24.csv"
JM_CSV = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"
OUT_DIR = Path("/mnt/c/Users/kbj20/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-llewyn-crypto-scalping/ade1f4c3-6c01-43de-b3c0-978e2dc35016/scratchpad")

COLORS = {"bull": "#2ca02c", "bear": "#d62728", "chop": "#7f7f7f"}
PREFIX = "regime3_current_sensitive_wide24_"


def load_regime(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", f"{PREFIX}bull_prob", f"{PREFIX}bear_prob", f"{PREFIX}chop_prob"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= WEEK_START) & (df["timestamp"] < WEEK_END)].reset_index(drop=True)
    probs = df[[f"{PREFIX}bull_prob", f"{PREFIX}bear_prob", f"{PREFIX}chop_prob"]].to_numpy()
    classes = np.array(["bull", "bear", "chop"])
    df["regime"] = classes[np.argmax(probs, axis=1)]
    df["source"] = label
    return df


def plot_one(price: pd.DataFrame, regime: pd.DataFrame, title: str, out_path: Path) -> None:
    merged = pd.merge_asof(price.sort_values("timestamp"), regime[["timestamp", "regime"]].sort_values("timestamp"), on="timestamp")
    fig, ax = plt.subplots(figsize=(14, 5.5))

    # shade contiguous regime runs
    ts = merged["timestamp"].to_numpy()
    reg = merged["regime"].to_numpy()
    start = 0
    for i in range(1, len(reg) + 1):
        if i == len(reg) or reg[i] != reg[start]:
            t0 = merged["timestamp"].iloc[start]
            t1 = merged["timestamp"].iloc[i] if i < len(reg) else merged["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
            ax.axvspan(t0, t1, color=COLORS[reg[start]], alpha=0.22, linewidth=0)
            start = i

    ax.plot(merged["timestamp"], merged["close"], color="black", linewidth=1.1)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("BTC close")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.35, label=k) for k in ["bull", "bear", "chop"]]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    n_switch = int((reg[1:] != reg[:-1]).sum())
    ax.text(0.99, 0.02, f"transitions={n_switch}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}  (bars={len(merged)}, transitions={n_switch})")


def main() -> None:
    price = pd.read_csv(PRICE_CSV, usecols=["timestamp", "close"])
    price["timestamp"] = pd.to_datetime(price["timestamp"])
    price = price[(price["timestamp"] >= WEEK_START) & (price["timestamp"] < WEEK_END)].reset_index(drop=True)

    hmm = load_regime(HMM_CSV, "hmm")
    jm = load_regime(JM_CSV, "jm")

    plot_one(price, hmm, f"BTC OOS week {WEEK_START.date()}..{WEEK_END.date()} — live HMM regime3", OUT_DIR / "btc_regime_hmm_oos_week1.png")
    plot_one(price, jm, f"BTC OOS week {WEEK_START.date()}..{WEEK_END.date()} — JM(k=3,lam=4) regime3", OUT_DIR / "btc_regime_jm_oos_week1.png")


if __name__ == "__main__":
    main()
