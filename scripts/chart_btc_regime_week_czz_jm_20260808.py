"""BTC regime chart, most recent week, with the NEW default detectors (2026-08-08).

Replaces chart_btc_hmm_regime_week_20260808.py as the daily eyeball chart: price line
colored by the causal 4% zigzag wave direction (the human bull/bear criterion made
causal), with alignment strips for causal-zigzag, Jump Model k3 lam32, and the retired
HMM.  Reads the decoded states from data/research/btc_jm_regime_states_20260808.parquet
(written by scripts/chart_btc_jm_regime_verification_20260808.py) -- no refit here.
"""
from __future__ import annotations

import argparse
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
from chart_btc_jm_regime_verification_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402

STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="tmp/jm_regime_verification_20260808/btc_regime_week_new.png")
    args = ap.parse_args()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    st = pd.read_parquet(STATES_PATH)
    ts = pd.to_datetime(st["timestamp"])
    close = st["close"].to_numpy(dtype=np.float64)
    czz = st["czz4"].to_numpy()
    jm32 = st["jm_lam32"].to_numpy()
    hmm = st["hmm"].to_numpy()

    start = ts.iloc[-1] - pd.Timedelta(days=args.days)
    idx = np.flatnonzero((ts >= start).to_numpy())
    h_ts = ts.to_numpy()[idx]
    occ = {n: round(float((czz[idx] == k).mean() * 100), 1) for k, n in ((0, "bear"), (1, "chop"), (2, "bull"))}
    print(json.dumps({"window": [str(start), str(ts.iloc[-1])], "czz4_occupancy_pct": occ}))

    fig, axes = plt.subplots(4, 1, figsize=(16, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7], "hspace": 0.06})
    ax = axes[0]
    for s, e, stt in contiguous_runs(czz[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.1)
    ax.set_title(f"BTC 5m — causal-zigzag 4% regimes (line) — last {args.days} days "
                 f"({pd.Timestamp(start).date()} .. {ts.iloc[-1].date()})",
                 loc="left", fontsize=13, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.8, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    for strip_ax, arr, label in ((axes[1], czz[idx], "causal zigzag  "),
                                 (axes[2], jm32[idx], "JM lam32  "),
                                 (axes[3], hmm[idx], "old HMM  ")):
        for s, e, stt in contiguous_runs(arr):
            strip_ax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt],
                             alpha=0.9, linewidth=0)
        strip_ax.set_yticks([])
        strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            strip_ax.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
