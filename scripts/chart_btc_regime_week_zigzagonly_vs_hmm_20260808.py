"""One-week eyeball chart: frozen zigzag-only theta=0.5% detector vs the retired HMM (2026-08-08).

SCALE WARNING, stated on the figure itself.  These two detectors answer DIFFERENT questions and a
side-by-side without that caveat is exactly the oracle-scale mistake diagnosed earlier today:
  zigzagonly_S2fine5_lam05  frozen theta=0.5% nowcaster, ~8-bar (40 min) median runs
  hmm_old                   3-state sticky HMM, multi-day scale, ~231-bar (~19 h) median runs
So the HMM is not "wrong" where it disagrees inside a wave -- it is not trying to resolve one.
Two oracle strips are drawn for that reason: theta=0.5% (the frozen detector's own reference) and
theta=1.5% (the scale diagnosis found to be eye-matching for a 7-day 5m chart; theta=4% has ZERO
turning points inside a week and would be a flat strip).

Reads decoded states only -- nothing is refit here.
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
from chart_btc_jm_regime_verification_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402

ZZ_PATH = ROOT / "data/research/btc_regime_theta005_zigzagonly_20260808.parquet"
ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def named(dir_arr: np.ndarray) -> np.ndarray:
    return np.where(dir_arr == 1, 2, np.where(dir_arr == -1, 0, 1)).astype(np.int8)


def stats(state: np.ndarray, idx: np.ndarray, oracle: np.ndarray) -> dict:
    runs = [e - s + 1 for s, e, _ in contiguous_runs(state[idx])]
    d = np.where(state == 2, 1, np.where(state == 0, -1, 0))
    m = idx[(d[idx] != 0) & (oracle[idx] != 0)]
    return {"median_run_min": round(float(np.median(runs)) * 5, 1) if runs else None,
            "flips": max(len(runs) - 1, 0),
            "agree_pct": round(float(np.mean(d[m] == oracle[m])) * 100, 1) if len(m) >= 20 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="tmp/regime_theta005_20260808/week_zigzagonly_vs_hmm.png")
    args = ap.parse_args()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zz = pd.read_parquet(ZZ_PATH)[["timestamp", "close", "zigzagonly_final"]]
    zoo = pd.read_parquet(ZOO_PATH)[["timestamp", "hmm_old"]]
    st = zz.merge(zoo, on="timestamp", how="inner")
    ts = pd.to_datetime(st["timestamp"])
    close = st["close"].to_numpy(dtype=np.float64)

    o005 = zigzag_oracle(close, threshold=0.005)[0]
    o015 = zigzag_oracle(close, threshold=0.015)[0]
    zzs = st["zigzagonly_final"].to_numpy().astype(np.int8)
    hmm = st["hmm_old"].to_numpy().astype(np.int8)

    start = ts.iloc[-1] - pd.Timedelta(days=args.days)
    idx = np.flatnonzero((ts >= start).to_numpy())
    h_ts = ts.to_numpy()[idx]

    report = {"window": [str(start), str(ts.iloc[-1])], "bars": int(len(idx))}
    for name, arr in (("zigzagonly_S2fine5", zzs), ("hmm_old", hmm)):
        report[name] = {"vs_oracle_0.5pct": stats(arr, idx, o005),
                        "vs_oracle_1.5pct": stats(arr, idx, o015),
                        "occupancy_pct": {n: round(float((arr[idx] == k).mean() * 100), 1)
                                          for k, n in ((0, "bear"), (1, "chop"), (2, "bull"))}}
    report["disagreement_pct"] = round(float((zzs[idx] != hmm[idx]).mean()) * 100, 1)
    print(json.dumps(report, indent=2), flush=True)

    strips = [(named(o005), "오라클 0.5% (사후)"), (zzs, "zigzagonly S2fine5  (θ=0.5%)"),
              (hmm, "HMM (은퇴, 멀티데이 스케일)"), (named(o015), "오라클 1.5% (사후)")]
    fig, axes = plt.subplots(1 + len(strips), 1, figsize=(16, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.78] * len(strips), "hspace": 0.07})

    ax = axes[0]
    for s, e, stt in contiguous_runs(zzs[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.2)
    z, h = report["zigzagonly_S2fine5"], report["hmm_old"]
    ax.set_title(
        f"BTC 5m — 라인 색 = zigzagonly S2fine5 (θ=0.5% frozen) — 최근 {args.days}일\n"
        f"zigzagonly: 중앙런 {z['vs_oracle_0.5pct']['median_run_min']}분 · 전환 {z['vs_oracle_0.5pct']['flips']}회 · "
        f"θ0.5 일치 {z['vs_oracle_0.5pct']['agree_pct']}% / θ1.5 일치 {z['vs_oracle_1.5pct']['agree_pct']}%   |   "
        f"HMM: 중앙런 {h['vs_oracle_0.5pct']['median_run_min']}분 · 전환 {h['vs_oracle_0.5pct']['flips']}회 · "
        f"θ0.5 {h['vs_oracle_0.5pct']['agree_pct']}% / θ1.5 {h['vs_oracle_1.5pct']['agree_pct']}%   |   "
        f"두 탐지기 불일치 {report['disagreement_pct']}%\n"
        f"※ 스케일이 다릅니다 — zigzagonly는 θ=0.5% 나우캐스터, HMM은 멀티데이 상태. 파동 내부 불일치는 HMM의 오류가 아닙니다.",
        loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)

    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[idx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb + "  ", rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
