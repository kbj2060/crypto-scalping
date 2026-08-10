"""Causal-zigzag regime detector sweep (2026-08-08).

Sweeps the confirmation threshold {4%, 6%, 8%} and an optional low-volatility chop
overlay (trailing-24h high/low range < range_floor -> chop), scoring each variant
against the retrospective 4% zigzag oracle per window (full / VAL / OOS / last30d):
bull/bear agreement on covered bars, coverage, median run length, and flip count.
Charts the leading variants for visual acceptance.  Adds the chosen states to
data/research/btc_jm_regime_states_20260808.parquet as extra columns.
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
from chart_btc_jm_regime_verification_20260808 import (  # noqa: E402
    causal_zigzag, agreement, REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK,
)
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    VAL_START, VAL_END, OOS_START, OOS_END,
)

STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"
OUT_DIR = ROOT / "tmp/jm_regime_verification_20260808"
THRESHOLDS = [0.04, 0.06, 0.08]
RANGE_FLOORS = [0.0, 0.03]  # 0 = no chop overlay


def runs_stats(named, idx):
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named[idx])]
    return float(np.median(runs)), len(runs)


def main() -> int:
    st = pd.read_parquet(STATES_PATH)
    ts = pd.to_datetime(st["timestamp"])
    close = st["close"].to_numpy(dtype=np.float64)
    oracle_dir = st["oracle_dir"].to_numpy()
    oracle_named = np.where(oracle_dir == 1, 2, np.where(oracle_dir == -1, 0, 1))

    hi = pd.Series(close).rolling(288, min_periods=288).max().to_numpy()
    lo = pd.Series(close).rolling(288, min_periods=288).min().to_numpy()
    range24 = (hi - lo) / np.where(lo > 0, lo, np.nan)

    windows = {
        "full": np.arange(len(close)),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "last30d": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=30)).to_numpy()),
    }

    variants = {}
    report = {}
    for thr in THRESHOLDS:
        czz = causal_zigzag(close, threshold=thr)
        for floor in RANGE_FLOORS:
            named = np.where(czz == 1, 2, np.where(czz == -1, 0, 1)).astype(np.int8)
            if floor > 0:
                named[np.nan_to_num(range24, nan=1.0) < floor] = 1
            key = f"czz{int(thr * 100)}" + (f"_chop{int(floor * 100)}" if floor > 0 else "")
            variants[key] = named
            rep = {}
            for wtag, idx in windows.items():
                ag, cov = agreement(named, oracle_dir, idx)
                med_run, n_runs = runs_stats(named, idx)
                rep[wtag] = {"agree": ag, "cov": cov, "med_run": med_run, "n_runs": n_runs}
            report[key] = rep
            print(json.dumps({key: rep}), flush=True)

    (OUT_DIR / "czz_sweep.json").write_text(json.dumps(report, indent=2))

    # chart the two leading variants against the oracle for the windows that matter visually
    chart_keys = ["czz4", "czz6", "czz6_chop3", "czz8"]
    for wtag, ds in (("full", 12), ("oos_2026Q1", 3), ("val_2025Q4", 3)):
        idx = windows[wtag][::ds]
        h_ts = ts.to_numpy()[idx]
        fig, axes = plt.subplots(len(chart_keys) + 2, 1, figsize=(16, 10),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [10] + [0.7] * (len(chart_keys) + 1),
                                              "hspace": 0.08})
        ax = axes[0]
        main_named = variants["czz6_chop3"]
        for s, e, stt in contiguous_runs(main_named[idx]):
            seg = slice(s, min(e + 2, len(idx)))
            ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.1)
        if wtag == "full":
            ax.set_yscale("log")
        heads = "  ".join(f"{k} {report[k][wtag]['agree']}%/{report[k][wtag]['n_runs']}fl" for k in chart_keys)
        ax.set_title(f"BTC — causal zigzag sweep (line = czz6_chop3) — {wtag}   [{heads}]",
                     loc="left", fontsize=11, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, alpha=0.8, label=l) for l, c in
                           (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
                  loc="upper left", frameon=False, fontsize=9, ncol=3)
        strip_specs = [(variants[k], k + "  ") for k in chart_keys] + [(oracle_named, "zigzag oracle*  ")]
        for strip_ax, (arr, label) in zip(axes[1:], strip_specs):
            for s, e, stt in contiguous_runs(arr[idx]):
                strip_ax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt],
                                 alpha=0.9, linewidth=0)
            strip_ax.set_yticks([])
            strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
            for side in ("top", "right", "left", "bottom"):
                strip_ax.spines[side].set_visible(False)
        fig.savefig(OUT_DIR / f"czz_sweep_{wtag}.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT_DIR / f'czz_sweep_{wtag}.png'}", flush=True)

    # persist the sweep variants for expert-model gating
    for key, named in variants.items():
        st[key] = named
    st.to_parquet(STATES_PATH, index=False)
    print("states parquet updated with", list(variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
