"""Oracle zigzag long/short chart (2026-08-08, user request).

Strategy illustrated: LONG during up-waves, SHORT during down-waves.
  - ORACLE version: waves as retrospectively segmented (lookahead BY DESIGN -- a ceiling, not a
    strategy). Position flips exactly at pivots.
  - CAUSAL version: same rule on the causal zigzag state (flip only after the 4% confirmation)
    -- what the rule actually earns live. The gap between the two curves IS the confirmation tax,
    and the causal strip's long bear stretches are that tax made visible.
Costs: 10bps roundtrip on notional per position flip, margin 0.30 x leverage 3 sizing.
Windows: full period and last 6 months. Stats: wave count/amplitude, both equities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle, causal_zigzag_regime, contiguous_runs  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
OUT_DIR = ROOT / "tmp/jump_model_regimes_20260808"
THRESH = 0.04
NOTIONAL = 0.30 * 3.0
FLIP_COST = 0.0010 * NOTIONAL  # one exit + one entry per flip
C_BULL, C_BEAR = "#2563EB", "#D9542B"
INK = "#1F2430"


def equity_of(direction: np.ndarray, close: np.ndarray) -> np.ndarray:
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    pos = np.where(direction > 0, 1.0, np.where(direction < 0, -1.0, 0.0))
    pnl = np.zeros(len(close))
    pnl[1:] = pos[:-1] * ret[1:] * NOTIONAL
    flips = np.zeros(len(close))
    flips[1:] = (np.abs(np.diff(pos)) > 0).astype(float)
    return np.cumprod(1.0 + pnl - flips * FLIP_COST)


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    odir, pivots = zigzag_oracle(close, THRESH)
    cdir = causal_zigzag_regime(close, THRESH)
    eq_oracle = equity_of(odir, close)
    eq_causal = equity_of(cdir, close)

    amps = np.abs(np.diff(np.log(close[pivots])))
    stats = {
        "n_waves": len(pivots) - 1,
        "median_wave_amplitude_pct": round(float(np.median(amps)) * 100, 2),
        "median_wave_bars": int(np.median(np.diff(pivots))),
        "oracle_ls_final_equity": round(float(eq_oracle[-1]), 2),
        "causal_ls_final_equity": round(float(eq_causal[-1]), 4),
        "confirmation_tax_note": "each wave loses ~2x threshold (miss first 4% + give back 4% at end)",
    }
    (OUT_DIR / "zigzag_longshort_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    windows = {"full": np.arange(0, len(close), 12),
               "6mo": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=180)).to_numpy())[::3]}
    for tag, idx in windows.items():
        h_ts = ts.to_numpy()[idx]
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [8, 6, 0.7, 0.7], "hspace": 0.08})
        ax = axes[0]
        o3 = np.where(odir > 0, 2, 0)[idx]
        for s, e, stt in contiguous_runs(o3):
            ax.axvspan(h_ts[s], h_ts[e], color=C_BULL if stt == 2 else C_BEAR, alpha=0.15, linewidth=0)
        ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"BTC — ZIGZAG ORACLE long/short (4% waves: up=LONG, down=SHORT) — {tag}",
                     loc="left", fontsize=13, color=INK)
        ax.set_ylabel("price (oracle shading)", fontsize=9, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=C_BULL, alpha=0.5, label="up-wave (long)"),
                           Patch(facecolor=C_BEAR, alpha=0.5, label="down-wave (short)")],
                  loc="upper left", frameon=False, fontsize=9, ncol=2)
        ax2 = axes[1]
        base_o = eq_oracle[idx][0]
        base_c = eq_causal[idx][0]
        ax2.plot(h_ts, eq_oracle[idx] / base_o, color=INK, linewidth=1.4, label="ORACLE L/S (lookahead ceiling)")
        ax2.plot(h_ts, eq_causal[idx] / base_c, color=C_BEAR, linewidth=1.4, label="CAUSAL zigzag L/S (live-feasible)")
        ax2.set_yscale("log")
        ax2.set_ylabel("equity (window-rebased)", fontsize=9, color=INK)
        ax2.legend(loc="upper left", frameon=False, fontsize=9)
        ax2.grid(axis="y", color="#000000", alpha=0.08)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        for strip_ax, dirs, label in ((axes[2], odir[idx], "ORACLE  "), (axes[3], cdir[idx], "causal  ")):
            s3 = np.where(dirs > 0, 2, np.where(dirs < 0, 0, 1))
            for s, e, stt in contiguous_runs(s3):
                strip_ax.axvspan(h_ts[s], h_ts[e], color={0: C_BEAR, 1: "#9AA0A6", 2: C_BULL}[stt], alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        fig.savefig(OUT_DIR / f"zigzag_oracle_ls_{tag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT_DIR / f'zigzag_oracle_ls_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
