"""One-week OOS ETH price chart with regime shading: redesigned ranked-m6 JM vs the incumbent.

Ad-hoc visualization for the JM-only regime3 redesign line; not a project artifact, just a chart.
The redesigned detector is refit here from its own definition (ANOVA-F top-6 features, standard
scaler, K=3, lambda_per_dim=2, temperature ratio 0.5) rather than read from a saved CSV, because
no artifact has been built for it yet -- the fit is on 2024 only and the decode is causal, so the
week shown is genuinely out of sample.

The incumbent panel is the live-shadow eth JM lambda=4 build read from its emitted per-bar CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, FIT_YEAR, SOURCES, _class_proba, _read, _state_class_matrix, causal_decode_V,
    fit_jm, softmax_states,
)
from scripts.ranked_jm_feature_selection_20260810 import load_pool, rankings_for  # noqa: E402

ASSET = "eth"
M, RANKING, SCALER, K, LPD, TEMP_RATIO, SEED = 6, "f_rank", "standard", 3, 2.0, 0.5, 7529
WEEK_START = pd.Timestamp("2026-03-25 00:00:00")
WEEK_END = pd.Timestamp("2026-04-01 00:00:00")

INCUMBENT_CSV = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"
PREFIX = "regime3_current_sensitive_wide24_"
COLORS = {"bull": "#2ca02c", "bear": "#d62728", "chop": "#7f7f7f"}
OUT_DIR = Path("/mnt/c/Users/kbj20/AppData/Local/Temp/claude/"
               "--wsl-localhost-ubuntu-home-llewyn-crypto-scalping/"
               "543afe45-c3db-4c32-a239-e5ba56172716/scratchpad")


def redesigned_regime(lpd: float = LPD) -> tuple[pd.DataFrame, list[str]]:
    pool = load_pool(ASSET, SCALER)
    idx = [int(i) for i in rankings_for(ASSET, SCALER)[RANKING][:M]]
    cols = [pool["cols"][i] for i in idx]
    lam = lpd * M
    mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=K, lam=lam, seed=SEED, n_init=5, n_iter=15)
    v_fit = causal_decode_V(pool[f"x_{FIT_YEAR}"][:, idx], mu, lam)
    spread = max(float(np.median(v_fit.max(axis=1) - v_fit.min(axis=1))), 1e-9)
    sp_fit = softmax_states(v_fit, TEMP_RATIO * spread)
    state_class = _state_class_matrix(sp_fit, pool[f"y_frozen_{FIT_YEAR}"])

    v26 = causal_decode_V(pool["x_2026"][:, idx], mu, lam)
    proba = _class_proba(softmax_states(v26, TEMP_RATIO * spread), state_class)
    ts = _read(SOURCES[ASSET]["2026"])["timestamp"]
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    out["regime"] = np.array(CLASSES3)[np.argmax(proba, axis=1)]
    return out, cols


def incumbent_regime() -> pd.DataFrame:
    cols = [f"{PREFIX}{c}_prob" for c in CLASSES3]
    df = pd.read_csv(INCUMBENT_CSV, usecols=["timestamp"] + cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["regime"] = np.array(CLASSES3)[np.argmax(df[cols].to_numpy(), axis=1)]
    return df[["timestamp", "regime"]]


def shade(ax, frame: pd.DataFrame) -> None:
    reg = frame["regime"].to_numpy()
    start = 0
    for i in range(1, len(reg) + 1):
        if i == len(reg) or reg[i] != reg[start]:
            t0 = frame["timestamp"].iloc[start]
            t1 = (frame["timestamp"].iloc[i] if i < len(reg)
                  else frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=5))
            ax.axvspan(t0, t1, color=COLORS[reg[start]], alpha=0.22, linewidth=0)
            start = i


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new, cols = redesigned_regime(LPD)
    # Same features, stickier jump penalty. Selection on balanced accuracy alone picks lpd=2, the
    # flippiest setting; lpd=8 is MORE persistent than the incumbent (52 vs 66 runs this week) and
    # still beats it on agreement by ~8pp, which is the trade the accuracy-only criterion hid.
    sticky, _ = redesigned_regime(8.0)
    old = incumbent_regime()

    price = _read(SOURCES[ASSET]["2026"])[["timestamp", "close"]]
    week = price[(price["timestamp"] >= WEEK_START) & (price["timestamp"] < WEEK_END)].reset_index(drop=True)

    panels = [
        ("REDESIGNED  m=6, lambda_per_dim=2  (accuracy-selected)   "
         "OOS bal_acc 0.931 | OOS sep_t -2.10", new),
        ("REDESIGNED  m=6, lambda_per_dim=8  (persistence-favoured)   "
         "OOS bal_acc 0.843 | OOS sep_t -2.09", sticky),
        ("INCUMBENT  JM lambda=4 on wide24 (live ETH shadow)   "
         "OOS bal_acc 0.765 | OOS sep_t +0.36", old),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    for ax, (title, reg) in zip(axes, panels):
        merged = pd.merge_asof(week.sort_values("timestamp"),
                               reg.sort_values("timestamp"), on="timestamp")
        shade(ax, merged)
        ax.plot(merged["timestamp"], merged["close"], color="black", linewidth=1.1)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("ETH close")
        ax.grid(alpha=0.15)
        runs = (merged["regime"] != merged["regime"].shift()).cumsum()
        ax.text(0.995, 0.04, f"{runs.nunique()} regime runs this week",
                transform=ax.transAxes, ha="right", fontsize=9, color="#444")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    axes[0].legend(handles=[Patch(facecolor=COLORS[c], alpha=0.4, label=c) for c in CLASSES3],
                   loc="upper left", ncol=3, fontsize=9, framealpha=0.9)
    fig.suptitle(f"ETH 5m regime, OOS week {WEEK_START:%Y-%m-%d} to {WEEK_END:%Y-%m-%d}  "
                 f"(fit on 2024 only, causal forward decode)", fontsize=13)
    fig.text(0.01, 0.005, "m=6 features: " + ", ".join(cols), fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = OUT_DIR / "eth_jm_ranked_m6_vs_incumbent_oos_week.png"
    fig.savefig(out, dpi=140)
    print(f"-> {out}")
    print("features:", cols)


if __name__ == "__main__":
    main()
