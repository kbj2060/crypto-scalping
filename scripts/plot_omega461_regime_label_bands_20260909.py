"""레짐 라벨 시각 비교 — balancedish(balgbm) vs S12_K3(s12k3) 밴드 차트.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지적(2026-09-09) "일관성이 없이 계속 플립되면 정확도는 높아서 의미가 없어."

무엇을 그리는가
--------------
두 후보는 **모델급이 통제**돼 있다(동일 HGB HP/136피쳐/컷오프/시드/OOF). 차이는 학습 타깃
라벨 하나뿐이다. 이 차트는 그 차이가 **시간축 위에서 어떻게 보이는지**를 보여준다 —
flip율 0.0497 vs 0.0978 같은 숫자 하나로는 "이 정도면 쓸 만한가"가 전달되지 않기 때문이다.

패널 구성
--------
  1-2. 추세 구간(5일): balgbm / s12k3 밴드를 같은 가격 위에 각각
  3-4. 횡보 구간(5일): 동상
  5.   OOS 전체(2개월) 리본 비교 — 거시적 전환 패턴
  6.   상태 지속시간 분포 — 플립 문제의 정량 대응물

구간 선택 규칙(사후 체리피킹 방지): OOS 안의 모든 5일 창을 훑어 **|순수익|이 최대인 창**을
추세 구간으로, **최소인 창**을 횡보 구간으로 자동 선택한다. 라벨을 보고 고르지 않는다.

색: bull=초록, bear=빨강, chop=회색. 두 패널 동일.
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
BAL_DIR = ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909"
S12_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"
ARMS = [("balgbm  (balancedish 라벨)", BAL_DIR, "regime3_balgbm_cut2509_"),
        ("s12k3  (S12_K3 라벨)", S12_DIR, "regime3_s12k3_cut2509_")]
CLASSES = ("bull", "bear", "chop")
COLORS = {"bull": "#2e9e4f", "bear": "#d1495b", "chop": "#b8bcc2"}
OOS = ("2026-01-01", "2026-02-28 23:55:00")
WIN_BARS = 5 * 288          # 5일
OUT = ROOT / "docs/charts/omega461_regimegbm_20260909"

plt.rcParams.update({"font.size": 20, "axes.titlesize": 26, "axes.labelsize": 22,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 20})


def load() -> pd.DataFrame:
    parts = []
    for tag in ("2026_rebuilt",):
        b = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{tag}.csv",
                        usecols=["timestamp", "close"], parse_dates=["timestamp"])
        for _, d, pref in ARMS:
            s = pd.read_csv(d / f"training_features_{tag}_{pref}sidecar.csv",
                            parse_dates=["timestamp"],
                            usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
            b = b.merge(s, on="timestamp", how="inner")
        parts.append(b)
    df = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return df[(df.timestamp >= OOS[0]) & (df.timestamp <= OOS[1])].reset_index(drop=True)


def bands(ax, ts, pred, close, title):
    ax.plot(ts, close, color="#1b1b1b", lw=2.2, zorder=3)
    start = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[start]:
            ax.axvspan(ts[start], ts[i - 1], color=COLORS[CLASSES[pred[start]]], alpha=0.55, lw=0)
            start = i
    flips = int((pred[1:] != pred[:-1]).sum())
    ax.set_title(f"{title}   —   전환 {flips}회 / {len(pred)}봉", pad=12)
    ax.set_ylabel("ETH 종가 (USDT)")
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

    # --- 구간 자동 선택: |순수익| 최대 = 추세, 최소 = 횡보 (라벨을 보지 않는다) ---
    net = np.full(len(close), np.nan)
    net[:-WIN_BARS] = np.abs(close[WIN_BARS:] - close[:-WIN_BARS]) / close[:-WIN_BARS]
    ok = np.isfinite(net)
    i_tr = int(np.nanargmax(np.where(ok, net, -np.inf)))
    i_ch = int(np.nanargmin(np.where(ok, net, np.inf)))
    wins = [("추세 구간 (|5일 순수익| 최대)", i_tr), ("횡보 구간 (|5일 순수익| 최소)", i_ch)]
    for lbl, i0 in wins:
        print(f"{lbl}: {ts[i0]} ~ {ts[i0+WIN_BARS-1]}  "
              f"순수익 {(close[i0+WIN_BARS-1]/close[i0]-1)*100:+.2f}%", flush=True)

    fig, axes = plt.subplots(6, 1, figsize=(32, 40), dpi=145,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.1, 0.9]})
    k = 0
    for lbl, i0 in wins:
        sl = slice(i0, i0 + WIN_BARS)
        for name in preds:
            bands(axes[k], ts[sl], preds[name][sl], close[sl], f"{lbl}  ·  {name}")
            k += 1

    # --- 패널 5: OOS 전체 리본 ---
    ax = axes[4]
    ax.plot(ts, close, color="#1b1b1b", lw=1.6, zorder=3)
    lo, hi = close.min(), close.max()
    span = hi - lo
    for j, name in enumerate(preds):
        base = lo - span * (0.10 + 0.09 * j)
        h = span * 0.07
        pred = preds[name]
        start = 0
        for i in range(1, len(pred) + 1):
            if i == len(pred) or pred[i] != pred[start]:
                ax.add_patch(mpatches.Rectangle(
                    (matplotlib.dates.date2num(ts[start]), base),
                    matplotlib.dates.date2num(ts[i - 1]) - matplotlib.dates.date2num(ts[start]), h,
                    color=COLORS[CLASSES[pred[start]]], lw=0))
                start = i
        ax.text(ts[5], base + h * 0.35, name.split("(")[0].strip(), fontsize=20, weight="bold")
    ax.set_ylim(lo - span * 0.33, hi + span * 0.05)
    ax.set_title("OOS 전체 2026-01-01 ~ 02-28  ·  리본 비교 (아래 두 띠)", pad=12)
    ax.set_ylabel("ETH 종가 (USDT)")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0)

    # --- 패널 6: 상태 지속시간 분포 ---
    ax = axes[5]
    bins = np.arange(1, 62, 2)
    for name in preds:
        r = run_lengths(preds[name])
        ax.hist(np.clip(r, 1, 60), bins=bins, alpha=0.55, label=
                f"{name.split('(')[0].strip()}  중앙 {np.median(r):.0f}봉 · 평균 {r.mean():.1f}봉 · n={len(r)}")
    ax.set_title("상태 지속시간 분포 (OOS 전체, 60봉 초과는 60으로 클립)  —  오른쪽일수록 안정적", pad=12)
    ax.set_xlabel("연속 유지 봉 수 (1봉 = 5분)")
    ax.set_ylabel("구간 수")
    ax.legend()
    ax.grid(alpha=0.25)

    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.997), frameon=False)
    fig.suptitle("레짐 라벨 시각 비교 — balancedish vs S12_K3 (모델급 통제: 동일 HGB·136피쳐·컷오프·시드)",
                 fontsize=32, y=0.9995)
    fig.tight_layout(rect=[0, 0, 1, 0.982])
    p = OUT / "regime_label_bands_balgbm_vs_s12k3.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"\n저장: {p}", flush=True)

    for name in preds:
        r = run_lengths(preds[name])
        print(f"  {name:28s} 전환 {int((preds[name][1:]!=preds[name][:-1]).sum()):4d}회  "
              f"중앙지속 {np.median(r):.0f}봉  평균 {r.mean():.1f}봉  "
              f"5봉이하 비율 {float((r<=5).mean())*100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
