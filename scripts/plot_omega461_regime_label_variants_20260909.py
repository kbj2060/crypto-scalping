"""레짐 라벨 3변형 시각 비교 — balgbm / balnobb(BB덮어쓰기 제거) / s12k3.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지시(2026-09-09) "BB 덮어쓰기 제거한 변형 만들어서 같은 잣대로 차트를 만들어서
비교해줘. 진입과 청산은 어차피 딥러닝 모델이 측정하고 우린 차트에 맞게 레짐만 잘 분류하면 돼."

→ 판단 기준이 **전방수익 경제성이 아니라 "차트를 구조적으로 잘 나누는가"** 로 명시됐다.
   따라서 이 차트가 1차 판정 도구이고, 숫자(전환수/지속)는 보조다.

세 변형은 **모델급이 완전히 통제**돼 있다 — 동일 HGB HP/136피쳐/컷오프(≤2025-09-30)/SEED/OOF.
차이는 학습 타깃 라벨 하나뿐.

  balgbm  : balancedish 원본            (덮어쓰기 있음 → 추세 후보의 42% 가 chop 으로)
  balnobb : balancedish − BB 덮어쓰기    (이번 변형)
  s12k3   : S12_K3 효율비 라벨           (참고, 이미 closed-negative)

구간 선택 규칙(체리피킹 방지): OOS 안 모든 5일 창 중 |순수익| 최대 = 추세 구간,
최소 = 횡보 구간. **라벨을 보지 않고** 가격만으로 고른다. 앞선 2변형 차트와 동일 규칙이라
같은 구간이 선택되어 직접 비교된다.
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
ARMS = [
    ("balgbm   balancedish 원본 (BB 덮어쓰기 有)",
     ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909", "regime3_balgbm_cut2509_"),
    ("balnobb  balancedish − BB 덮어쓰기 (신규)",
     ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909", "regime3_balnobb_cut2509_"),
    ("s12k3    S12_K3 효율비 라벨 (참고)",
     ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909", "regime3_s12k3_cut2509_"),
]
CLASSES = ("bull", "bear", "chop")
COLORS = {"bull": "#2e9e4f", "bear": "#d1495b", "chop": "#b8bcc2"}
OOS = ("2026-01-01", "2026-02-28 23:55:00")
WIN_BARS = 5 * 288
OUT = ROOT / "docs/charts/omega461_regimegbm_20260909"

use_korean_font()
plt.rcParams.update({"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 22,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 19})


def load() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                    usecols=["timestamp", "close"], parse_dates=["timestamp"])
    for _, d, pref in ARMS:
        s = pd.read_csv(d / f"training_features_2026_rebuilt_{pref}sidecar.csv",
                        parse_dates=["timestamp"],
                        usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
        b = b.merge(s, on="timestamp", how="inner")
    b = b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return b[(b.timestamp >= OOS[0]) & (b.timestamp <= OOS[1])].reset_index(drop=True)


def bands(ax, ts, pred, close, title):
    ax.plot(ts, close, color="#101010", lw=2.4, zorder=3)
    start = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[start]:
            ax.axvspan(ts[start], ts[i - 1], color=COLORS[CLASSES[pred[start]]], alpha=0.55, lw=0)
            start = i
    flips = int((pred[1:] != pred[:-1]).sum())
    sh = {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)}
    ax.set_title(f"{title}   —   전환 {flips}회 / {len(pred)}봉   ·   "
                 f"bull {sh['bull']*100:.0f}% bear {sh['bear']*100:.0f}% chop {sh['chop']*100:.0f}%", pad=10)
    ax.set_ylabel("ETH 종가")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0)


def run_lengths(pred):
    runs, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    return np.array(runs)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    ts = df["timestamp"].to_numpy()
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)
    preds = {}
    for name, _, pref in ARMS:
        p = df[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
        preds[name] = p.argmax(1)

    net = np.full(len(close), np.nan)
    net[:-WIN_BARS] = np.abs(close[WIN_BARS:] - close[:-WIN_BARS]) / close[:-WIN_BARS]
    ok = np.isfinite(net)
    wins = [("추세 구간 (|5일 순수익| 최대)", int(np.nanargmax(np.where(ok, net, -np.inf)))),
            ("횡보 구간 (|5일 순수익| 최소)", int(np.nanargmin(np.where(ok, net, np.inf))))]

    n_rows = len(wins) * len(ARMS) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(32, 6.2 * n_rows), dpi=140)
    k = 0
    for lbl, i0 in wins:
        sl = slice(i0, i0 + WIN_BARS)
        print(f"{lbl}: {ts[i0]} ~ {ts[i0+WIN_BARS-1]}  "
              f"순수익 {(close[i0+WIN_BARS-1]/close[i0]-1)*100:+.2f}%", flush=True)
        for name in preds:
            bands(axes[k], ts[sl], preds[name][sl], close[sl], f"{lbl}  ·  {name}")
            k += 1

    ax = axes[k]
    bins = np.arange(1, 82, 3)
    for name in preds:
        r = run_lengths(preds[name])
        ax.hist(np.clip(r, 1, 80), bins=bins, alpha=0.5, label=
                f"{name.split()[0]}  전환 {int((preds[name][1:]!=preds[name][:-1]).sum())}회 · "
                f"중앙 {np.median(r):.0f}봉 · 평균 {r.mean():.1f}봉 · ≤5봉 {float((r<=5).mean())*100:.0f}%")
    ax.set_title("상태 지속시간 분포 (OOS 전체 2026-01-01~02-28, 80봉 초과는 80으로 클립) — 오른쪽일수록 안정적", pad=10)
    ax.set_xlabel("연속 유지 봉 수 (1봉 = 5분)")
    ax.set_ylabel("구간 수")
    ax.legend()
    ax.grid(alpha=0.25)

    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.998), frameon=False)
    fig.suptitle("레짐 라벨 3변형 비교 — BB 덮어쓰기 제거 효과 (모델급 통제: 동일 HGB·136피쳐·컷오프·시드)",
                 fontsize=31, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = OUT / "regime_label_variants_bb_override.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"\n저장: {p}", flush=True)

    print(f"\n{'변형':10s}{'전환':>8s}{'중앙':>7s}{'평균':>8s}{'≤5봉':>8s}   클래스비중(bull/bear/chop)", flush=True)
    for name in preds:
        pr = preds[name]; r = run_lengths(pr)
        sh = [float((pr == i).mean()) for i in range(3)]
        print(f"{name.split()[0]:10s}{int((pr[1:]!=pr[:-1]).sum()):8d}{np.median(r):7.0f}"
              f"{r.mean():8.1f}{float((r<=5).mean())*100:7.1f}%   "
              f"{sh[0]*100:.1f}% / {sh[1]*100:.1f}% / {sh[2]*100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
