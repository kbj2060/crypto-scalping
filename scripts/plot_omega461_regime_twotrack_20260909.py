"""투트랙 확정본 차트 — balgbm+K3 vs balnobb+K6.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 결정(2026-09-09) "balnobb+K6, balgbm+K3 의 차트를 다시 보여줘."

두 트랙
------
  트랙 A  balgbm  + K=3 : balancedish 원본(BB 덮어쓰기 有) + 15분 연속 확인
  트랙 B  balnobb + K=6 : balancedish − BB 덮어쓰기       + 30분 연속 확인

K 를 다르게 준 이유: 두 변형의 원본 전환수가 다르다(845 vs 1674). 같은 K 를 주면 안정성이
맞지 않는다. 이 조합은 debounce 스윕에서 **전환수를 비슷하게 맞춘 지점**이다
(588회 vs 656회) — 즉 **안정성을 통제하고 분류 품질만 다르게 한 대조**가 된다.

debounce 는 라벨이 아니라 **모델 예측(argmax) 위에 후처리**로 건다(스윕과 동일 방식).
`_debounce` 는 저장소 원본을 import 한다.

구간 선택은 앞선 모든 차트와 동일 규칙(OOS 내 |5일 순수익| 최대=추세 / 최소=횡보, 라벨 안 봄).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plot_korean_font_20260909 import use_korean_font  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import _debounce  # noqa: E402

TRACKS = [
    ("트랙 A · balgbm + K=3", ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909",
     "regime3_balgbm_cut2509_", 3, "BB 덮어쓰기 有 · 15분 연속 확인"),
    ("트랙 B · balnobb + K=6", ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909",
     "regime3_balnobb_cut2509_", 6, "BB 덮어쓰기 제거 · 30분 연속 확인"),
]
CLASSES = ("bull", "bear", "chop")
COLORS = {"bull": "#2e9e4f", "bear": "#d1495b", "chop": "#b8bcc2"}
OOS = ("2026-01-01", "2026-02-28 23:55:00")
WIN_BARS = 5 * 288
OUT = ROOT / "docs/charts/omega461_regimegbm_20260909"


def load() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                    usecols=["timestamp", "close"], parse_dates=["timestamp"])
    for _, d, pref, _, _ in TRACKS:
        s = pd.read_csv(d / f"training_features_2026_rebuilt_{pref}sidecar.csv",
                        parse_dates=["timestamp"],
                        usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
        b = b.merge(s, on="timestamp", how="inner")
    b = b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return b[(b.timestamp >= OOS[0]) & (b.timestamp <= OOS[1])].reset_index(drop=True)


def runs_of(pred):
    r, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            r.append(c); c = 1
    r.append(c)
    return np.array(r)


def bands(ax, ts, pred, close, title):
    ax.plot(ts, close, color="#101010", lw=2.5, zorder=3)
    s = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[s]:
            ax.axvspan(ts[s], ts[i - 1], color=COLORS[CLASSES[pred[s]]], alpha=0.55, lw=0)
            s = i
    sh = {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)}
    ax.set_title(f"{title}   —   전환 {int((pred[1:] != pred[:-1]).sum())}회 / {len(pred)}봉   ·   "
                 f"bull {sh['bull']*100:.0f}%  bear {sh['bear']*100:.0f}%  chop {sh['chop']*100:.0f}%",
                 pad=10)
    ax.set_ylabel("ETH 종가")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0)


def main() -> int:
    use_korean_font()
    plt.rcParams.update({"font.size": 21, "axes.titlesize": 26, "axes.labelsize": 23,
                         "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 20})
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    ts = df["timestamp"].to_numpy()
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)

    preds = {}
    for label, _, pref, k, _ in TRACKS:
        p = df[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64).argmax(1)
        preds[label] = _debounce(p, k) if k > 1 else p

    net = np.full(len(close), np.nan)
    net[:-WIN_BARS] = np.abs(close[WIN_BARS:] - close[:-WIN_BARS]) / close[:-WIN_BARS]
    ok = np.isfinite(net)
    wins = [("추세 구간 (5일, |순수익| 최대)", int(np.nanargmax(np.where(ok, net, -np.inf)))),
            ("횡보 구간 (5일, |순수익| 최소)", int(np.nanargmin(np.where(ok, net, np.inf))))]

    rows = len(wins) * len(TRACKS) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(32, 6.4 * rows), dpi=135)
    r = 0
    for wname, i0 in wins:
        sl = slice(i0, i0 + WIN_BARS)
        print(f"{wname}: {ts[i0]} ~ {ts[i0+WIN_BARS-1]}  "
              f"순수익 {(close[i0+WIN_BARS-1]/close[i0]-1)*100:+.2f}%", flush=True)
        for (label, _, _, _, note) in TRACKS:
            bands(axes[r], ts[sl], preds[label][sl], close[sl], f"{wname}  ·  {label}  ({note})")
            r += 1

    ax = axes[r]
    ax.plot(ts, close, color="#101010", lw=1.7, zorder=3)
    lo, hi = close.min(), close.max()
    span = hi - lo
    for j, label in enumerate(preds):
        base = lo - span * (0.11 + 0.10 * j)
        h = span * 0.075
        pred = preds[label]
        s = 0
        for i in range(1, len(pred) + 1):
            if i == len(pred) or pred[i] != pred[s]:
                ax.add_patch(mpatches.Rectangle(
                    (matplotlib.dates.date2num(ts[s]), base),
                    matplotlib.dates.date2num(ts[i - 1]) - matplotlib.dates.date2num(ts[s]), h,
                    color=COLORS[CLASSES[pred[s]]], lw=0))
                s = i
        ax.text(ts[8], base + h * 0.3, label, fontsize=21, weight="bold")
    ax.set_ylim(lo - span * 0.36, hi + span * 0.05)
    ax.set_title("OOS 전체 2026-01-01 ~ 02-28  ·  두 트랙 리본 비교", pad=10)
    ax.set_ylabel("ETH 종가")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0)

    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.999), frameon=False)
    fig.suptitle("투트랙 확정본 — balgbm+K3 vs balnobb+K6 (안정성 통제, 분류 품질만 다름)",
                 fontsize=32, y=1.001)
    fig.tight_layout(rect=[0, 0, 1, 0.986])
    p = OUT / "regime_twotrack_balgbmK3_balnobbK6.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"\n저장: {p}", flush=True)

    print(f"\n{'트랙':26s}{'전환':>8s}{'중앙':>7s}{'평균':>8s}{'≤5봉':>8s}   bull/bear/chop", flush=True)
    for label in preds:
        pr = preds[label]; rr = runs_of(pr)
        sh = [float((pr == i).mean()) * 100 for i in range(3)]
        print(f"{label:26s}{int((pr[1:]!=pr[:-1]).sum()):8d}{np.median(rr):7.0f}{rr.mean():8.1f}"
              f"{float((rr<=5).mean())*100:7.1f}%   {sh[0]:.1f}/{sh[1]:.1f}/{sh[2]:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
