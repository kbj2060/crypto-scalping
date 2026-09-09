"""레짐 debounce(K봉 연속 확인) 스윕 차트 — balgbm / balnobb × K∈{0,3,6,12,24}.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지시(2026-09-09) "K를 3/6/12/24로 걸어서 차트 만들어줘. 그리고 최선의 K가 적용된
balgbm / balnobb 를 투트랙으로 진행하자."

왜 debounce 인가
---------------
직전 비교에서 두 변형이 같은 트레이드오프 곡선의 양 끝에 놓였다(OOS 전체):
  balgbm  전환  845회 · 평균 20.1봉 — 안정적이나 강추세의 42% 를 chop 으로 오분류
  balnobb 전환 1674회 · 평균 10.1봉 — 분류는 맞으나 하루 28회 전환
debounce 는 **분류 기준을 건드리지 않고 지속성만 올리는** 유일한 손잡이다.

적용 위치 — 라벨이 아니라 예측 위
--------------------------------
S12_K3 는 K=3 을 **라벨에** 걸고 학습한다. 여기서는 **모델 예측(argmax) 위에 후처리**로 건다:
  · 재학습이 필요 없어 K 격자를 즉시 비교할 수 있다
  · 실제 라우팅에 적용될 형태와 동일하다(라우팅 신호에 거는 필터)
  · 라벨 정의 문제와 지속성 문제를 분리해 각각 판단할 수 있다
채택 K 가 정해지면, 그 K 를 라벨에 넣어 재학습한 판(S12_K3 관례)과 대조해볼 수 있다 — 이 차트는
그 전 단계의 선택 도구다.

debounce 정의는 저장소 원본(`research_eth_regime_scalping_label_geometry_20260902._debounce`)을
그대로 import 한다 — 새 상태가 **K봉 연속** 나와야 전환을 확정하고, 아니면 직전 상태를 유지한다.

읽는 법
------
· 위 5패널: 횡보 구간(플립이 문제되는 곳) — K 를 올릴수록 밴드가 뭉쳐야 한다
· 아래 5패널: 추세 구간 — K 를 올릴수록 **진입이 늦어진다**(반응속도 대가)
· 마지막 패널: K 대비 전환수/평균지속/진입지연 곡선 — 두 변형 동시

구간 선택은 앞선 차트들과 동일 규칙(|5일 순수익| 최대/최소, 라벨 안 봄)이라 직접 비교된다.
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

ARMS = [
    ("balgbm", ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909",
     "regime3_balgbm_cut2509_", "BB 덮어쓰기 有"),
    ("balnobb", ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909",
     "regime3_balnobb_cut2509_", "BB 덮어쓰기 제거"),
]
KS = (0, 3, 6, 12, 24)
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
    for _, d, pref, _ in ARMS:
        s = pd.read_csv(d / f"training_features_2026_rebuilt_{pref}sidecar.csv",
                        parse_dates=["timestamp"],
                        usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
        b = b.merge(s, on="timestamp", how="inner")
    b = b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return b[(b.timestamp >= OOS[0]) & (b.timestamp <= OOS[1])].reset_index(drop=True)


def apply_k(pred: np.ndarray, k: int) -> np.ndarray:
    return pred.copy() if k <= 1 else _debounce(pred, k)


def runs_of(pred):
    r, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            r.append(c); c = 1
    r.append(c)
    return np.array(r)


def entry_delay(base: np.ndarray, deb: np.ndarray) -> float:
    """K=0 이 새 상태로 진입한 시점 대비, debounce 판이 같은 상태를 처음 보고할 때까지의 지연(봉)."""
    d = []
    for i in range(1, len(base)):
        if base[i] != base[i - 1]:
            j = i
            while j < len(deb) and deb[j] != base[i]:
                j += 1
                if j - i > 200:
                    break
            if j < len(deb) and deb[j] == base[i]:
                d.append(j - i)
    return float(np.mean(d)) if d else float("nan")


def bands(ax, ts, pred, close, title):
    ax.plot(ts, close, color="#101010", lw=2.4, zorder=3)
    s = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[s]:
            ax.axvspan(ts[s], ts[i - 1], color=COLORS[CLASSES[pred[s]]], alpha=0.55, lw=0)
            s = i
    ax.set_title(f"{title}   —   전환 {int((pred[1:] != pred[:-1]).sum())}회 / {len(pred)}봉", pad=10)
    ax.set_ylabel("ETH 종가")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    ts = df["timestamp"].to_numpy()
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)
    base = {}
    for name, _, pref, _ in ARMS:
        p = df[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
        base[name] = p.argmax(1)
    deb = {(n, k): apply_k(base[n], k) for n, _, _, _ in ARMS for k in KS}

    net = np.full(len(close), np.nan)
    net[:-WIN_BARS] = np.abs(close[WIN_BARS:] - close[:-WIN_BARS]) / close[:-WIN_BARS]
    ok = np.isfinite(net)
    wins = {"횡보": int(np.nanargmin(np.where(ok, net, np.inf))),
            "추세": int(np.nanargmax(np.where(ok, net, -np.inf)))}

    # ---------- 차트 1: 변형별 K 스윕 (횡보 위 / 추세 아래) ----------
    for name, _, _, note in ARMS:
        rows = len(KS) * 2 + 1
        fig, axes = plt.subplots(rows, 1, figsize=(32, 5.9 * rows), dpi=140)
        r = 0
        for wname in ("횡보", "추세"):
            i0 = wins[wname]
            sl = slice(i0, i0 + WIN_BARS)
            for k in KS:
                bands(axes[r], ts[sl], deb[(name, k)][sl], close[sl],
                      f"{wname} 구간  ·  {name} ({note})  ·  K={k}"
                      + ("  (원본)" if k == 0 else f"  = {k*5}분 연속 확인"))
                r += 1
        ax = axes[r]
        for nm, _, _, nt in ARMS:
            tr = [int((deb[(nm, k)][1:] != deb[(nm, k)][:-1]).sum()) for k in KS]
            ax.plot(KS, tr, marker="o", ms=14, lw=3.5, label=f"{nm} ({nt}) 전환수")
        ax.set_xticks(KS)
        ax.set_xlabel("K (연속 확인 봉 수)")
        ax.set_ylabel("OOS 전체 전환 횟수")
        ax.set_title("K 대비 전환 횟수 — 두 변형 동시 (OOS 2026-01-01~02-28, 16,992봉)", pad=10)
        ax.legend(); ax.grid(alpha=0.3)
        h = [mpatches.Patch(color=COLORS[c], label=c) for c in CLASSES]
        fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.999), frameon=False)
        fig.suptitle(f"debounce 스윕 — {name} ({note})", fontsize=31, y=1.001)
        fig.tight_layout(rect=[0, 0, 1, 0.988])
        p = OUT / f"regime_debounce_sweep_{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"저장: {p}", flush=True)

    # ---------- 수치 표 ----------
    print(f"\n{'변형':10s}{'K':>4s}{'전환':>8s}{'중앙':>7s}{'평균':>8s}{'≤5봉':>8s}"
          f"{'K0일치':>9s}{'진입지연':>9s}   bull/bear/chop", flush=True)
    for name, _, _, _ in ARMS:
        for k in KS:
            p = deb[(name, k)]
            r_ = runs_of(p)
            agree = float((p == base[name]).mean())
            lag = entry_delay(base[name], p) if k else 0.0
            sh = [float((p == i).mean()) * 100 for i in range(3)]
            print(f"{name:10s}{k:4d}{int((p[1:]!=p[:-1]).sum()):8d}{np.median(r_):7.0f}"
                  f"{r_.mean():8.1f}{float((r_<=5).mean())*100:7.1f}%{agree*100:8.1f}%"
                  f"{lag:8.1f}봉   {sh[0]:.1f}/{sh[1]:.1f}/{sh[2]:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
