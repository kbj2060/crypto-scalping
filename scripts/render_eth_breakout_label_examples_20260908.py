#!/usr/bin/env python3
"""돌파/되돌림 라벨 예시 차트 -- **H=1시간(12봉) · 배리어 ±0.3%** (2026-09-08).

사용자: *"h 를 1시간으로 하고 배리어를 0.3%로 만든 라벨 차트를 각각 10개를 한 이미지로"*

각 패널: 1분봉 종가 경로 · 앵커 진입 기준가 · 트리거 레벨(진입) · ±0.3% 배리어 ·
트리거 분(수직선) · 배리어 첫터치 지점. 표본외 창(VAL/OOS/HOLDOUT)에서 돌파 5 · 되돌림 5.
⚠️피쳐 창은 트리거 분 직전까지, 라벨은 트리거 분부터 -- 차트에도 그 경계를 표시한다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

for cand in (Path("/mnt/c/Windows/Fonts/malgun.ttf"),
             Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")):
    if cand.exists():
        fm.fontManager.addfont(str(cand))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(cand)).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = MY / "label_examples_H1h_P030.png"
H, P, SEED = 12, 0.003, 20260908
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")


def main() -> int:
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(MY / "dataset_v4.parquet").sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)

    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    ref = O5[np.minimum(bi + 1, len(O5) - 1)]
    entry = ref * (1 + sgn * T)
    s0 = np.searchsorted(ts1, ts5[np.minimum(bi + 1, len(ts5) - 1)])
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    NM = H * 5
    ok = (s1 > 0) & (s1 + NM + 5 < len(ts1)) & (bt + H < len(C5)) & np.isin(d["split"].to_numpy(), WINS)

    up = entry * (1 + P); dn = entry * (1 - P)
    idx = np.flatnonzero(ok)
    win = np.full(len(d), -1)          # 1 돌파 · 0 되돌림 · -1 시간청산
    tmin = np.full(len(d), np.nan)
    for i in idx:
        seg = slice(s1[i], s1[i] + NM)
        hu = np.flatnonzero(hi1[seg] >= up[i]); hd = np.flatnonzero(lo1[seg] <= dn[i])
        a = hu[0] if len(hu) else 10**9; b = hd[0] if len(hd) else 10**9
        if a == b == 10**9: continue
        upfirst = a < b
        win[i] = int(upfirst if sgn[i] > 0 else (not upfirst))
        tmin[i] = min(a, b)
    res = idx[win[idx] >= 0]
    print(f"평가 가능 {len(idx):,} · 배리어 해소 {len(res):,} ({len(res)/len(idx):.1%}) · "
          f"돌파율 {win[res].mean():.3f}")

    pick = []
    for lab in (1, 0):
        c = res[win[res] == lab]
        pick += list(rng.choice(c, 5, replace=False))
    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    plt.rcParams.update({"font.size": 11})
    for ax, i in zip(axes.ravel(), pick):
        a0 = max(s0[i] - 10, 0); a1 = s1[i] + NM + 2
        x = np.arange(a0 - s1[i], a1 - s1[i])
        ax.plot(x, cl1[a0:a1], lw=1.3, color="#222", zorder=3)
        ax.fill_between(x, lo1[a0:a1], hi1[a0:a1], color="#999", alpha=.25, lw=0, zorder=1)
        ax.axhline(ref[i], color="#888", ls=":", lw=1.2)
        ax.axhline(entry[i], color="#0057d9", lw=1.8)
        ax.axhline(up[i], color="#0a6b2c" if sgn[i] > 0 else "#a01c1c", ls="--", lw=1.5)
        ax.axhline(dn[i], color="#a01c1c" if sgn[i] > 0 else "#0a6b2c", ls="--", lw=1.5)
        ax.axvline(0, color="#0057d9", lw=1.6)
        ax.axvspan(x[0], 0, color="#4a90d9", alpha=.07, lw=0)      # 피쳐 구간
        ax.axvspan(0, x[-1], color="#d99a4a", alpha=.07, lw=0)     # 라벨 구간
        ax.plot(tmin[i], up[i] if ((sgn[i] > 0) == (win[i] == 1)) else dn[i],
                marker="*", ms=20, color="#0a6b2c" if win[i] == 1 else "#a01c1c", zorder=5)
        lab = "돌파" if win[i] == 1 else "되돌림"
        col = "#0a6b2c" if win[i] == 1 else "#a01c1c"
        ax.set_title(f"{lab} · {'상방' if sgn[i]>0 else '하방'} 발현 · {int(tmin[i])}분 만에 해소\n"
                     f"{pd.Timestamp(d['timestamp'].iloc[i]):%Y-%m-%d %H:%M} · "
                     f"{d['split'].iloc[i][:4]} · ATR {T[i]*100:.2f}%",
                     color=col, fontsize=12, fontweight="bold")
        ax.set_xlabel("트리거 기준 분", fontsize=10)
        ax.grid(alpha=.25)
    fig.suptitle("돌파 / 되돌림 라벨 예시  ·  H = 1시간(12봉) · 배리어 ±0.30%\n"
                 "파란 세로선 = 트리거 분(결정 시점) │ 파란 음영 = 피쳐 구간(트리거 직전까지) │ "
                 "주황 음영 = 라벨 구간 │ 파란 가로선 = 진입(트리거 레벨) │ 점선 = 앵커 기준가 │ ★ = 첫 터치",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"저장 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
