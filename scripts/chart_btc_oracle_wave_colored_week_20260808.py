"""Week chart with the PRICE LINE ITSELF colored by the zigzag oracle at a chosen threshold.

The oracle is RETROSPECTIVE (it uses future bars to place pivots) -- this is the ground-truth
picture of "what a human would call up-wave / down-wave at this scale", not a live signal.  The
causal zigzag at the same threshold is drawn underneath so the confirmation lag (the price move
you must pay before the wave is knowable live) is visible on the same window.

Usage:  --theta 0.005   (default; the scale the user asked to see)
        --theta 0.015   (the eye-matching scale for a 7-day 5m window, per the scale diagnosis)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402

ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_classifier_zoo_20260808"
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}
INK = "#1F2430"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def to_named(direction: np.ndarray) -> np.ndarray:
    return np.where(direction == 1, 2, np.where(direction == -1, 0, 1)).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.005)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--ref-thetas", type=float, nargs="*", default=[0.01, 0.015])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zoo = pd.read_parquet(ZOO_PATH)
    ts = pd.to_datetime(zoo["timestamp"])
    close = zoo["close"].to_numpy(dtype=np.float64)
    idx = np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy())
    h_ts = ts.to_numpy()[idx]

    odir, pivots = zigzag_oracle(close, threshold=args.theta)
    onamed = to_named(odir)
    cnamed = to_named(causal_zigzag(close, threshold=args.theta))
    week_pivots = [p for p in pivots if idx[0] <= p <= idx[-1]]
    agree_causal = float(np.mean(cnamed[idx] == onamed[idx])) * 100
    stats = {"theta": args.theta, "pivots_in_window": len(week_pivots),
             "oracle_runs_in_window": len(contiguous_runs(onamed[idx])),
             "causal_matches_oracle_pct": round(agree_causal, 1)}
    print(json.dumps(stats), flush=True)

    refs = [(t, to_named(zigzag_oracle(close, threshold=t)[0])) for t in args.ref_thetas]
    n_strips = 1 + len(refs)
    fig, axes = plt.subplots(1 + n_strips, 1, figsize=(15, 7.4 + 0.4 * n_strips), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.75] * n_strips, "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(onamed[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    for p in week_pivots:
        j = int(np.searchsorted(idx, p))
        if 0 <= j < len(idx):
            ax.plot(h_ts[j], close[idx][j], marker="o", markersize=3.4,
                    markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
    ax.set_title(f"BTC 최근 {args.days}일 — 라인 색 = 지그재그 오라클 {args.theta * 100:g}% 파동 방향 "
                 f"(사후 기준, 전환점 {len(week_pivots)}개)   "
                 f"[같은 임계값 causal 재현율 {stats['causal_matches_oracle_pct']}%]",
                 loc="left", fontsize=12.5, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=C_BULL, label="상승 파동"), Patch(facecolor=C_BEAR, label="하락 파동"),
                       plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                                  markeredgecolor=INK, label="전환점(pivot)")],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    strips = [(cnamed, f"causal {args.theta * 100:g}% (실전)  ")] + \
             [(nm, f"오라클 {t * 100:g}% (비교)  ") for t, nm in refs]
    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[idx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    out = OUT_DIR / f"week_oracle_colored_theta{int(round(args.theta * 1000)):03d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
