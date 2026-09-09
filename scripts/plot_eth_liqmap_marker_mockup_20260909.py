#!/usr/bin/env python3
"""청산맵에 증거신호·이벤트 트리거를 얹는 세 가지 안 목업 (2026-09-09, 사용자 요청).

실제 청산맵 기하를 그대로 재현한다: 72봉(6시간) · viewBox 1200x400 · ml45/mr112/mt20/mb40
→ 컬럼 피치 14.5px · 캔들폭 11.6px. 기존 매매 저널 마커는 12px 삼각형 + 25px 스택.

A안 전부 봉 밀착 (사용자가 말한 그대로)
B안 전부 고정 레인
C안 하이브리드 -- 증거신호는 레인, 이벤트 트리거만 봉 밀착
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import live_evidence_signal_dashboard_20260823 as EV
import build_eth_anchor_label_dataset_20260907 as B
CACHE = ROOT / "tmp/eth_signal_map_20260909/klcache"
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 17, "axes.titlesize": 20,
                     "figure.facecolor": "white"})
G, R, N, EDGE, INK = "#1F9D55", "#D94141", "#9AA0A6", "#7F8794", "#1F2430"


def newest(sym):
    best, bn = None, 0
    for f in glob.glob(str(CACHE / f"{sym}_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    kl, btc = newest("ETHUSDT"), newest("BTCUSDT")
    kl = kl[kl.timestamp >= kl.timestamp.max() - pd.Timedelta(days=30)].reset_index(drop=True)
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    n = len(sig)
    bot = np.zeros(n, int); top = np.zeros(n, int)
    for s in B.SIGNALS:
        bot += sig[f"bottom_{s}"].fillna(False).to_numpy(bool).astype(int)
        top += sig[f"top_{s}"].fillna(False).to_numpy(bool).astype(int)
    W = 72
    marks = np.array([((bot[i:i+W] > 0).sum() + (top[i:i+W] > 0).sum()) for i in range(900, n - W)])
    starts = np.arange(900, n - W)
    med_i = int(starts[np.argmin(np.abs(marks - np.median(marks)))])
    hi_i = int(starts[np.argmin(np.abs(marks - np.percentile(marks, 90)))])
    rng = np.random.default_rng(20260909)

    fig, axes = plt.subplots(3, 2, figsize=(34, 22), dpi=120)
    for col, (s0, lab) in enumerate([(med_i, "중앙 밀도 창"), (hi_i, "90분위 밀도 창")]):
        b = sig.iloc[s0:s0+W].reset_index(drop=True)
        bo, to = bot[s0:s0+W], top[s0:s0+W]
        # 이벤트 트리거는 실제 건수 비율대로 흩뿌린다(목업 -- 모델 재계산 없이 배치만 본다)
        ev_b = rng.choice(W, size=max(int(round(3.9/2 + 5.5/2 + 0.7/2)), 1), replace=False)
        ev_t = rng.choice(W, size=max(int(round(3.9/2 + 5.5/2 + 0.7/2)), 1), replace=False)
        lo_, hi_ = b.low.min(), b.high.max()
        pad = (hi_ - lo_) * 0.15
        for row, name in enumerate(["A안 · 전부 봉 밀착", "B안 · 전부 고정 레인",
                                    "C안 · 하이브리드(증거=레인 · 이벤트=봉 밀착)"]):
            ax = axes[row][col]
            for i, r in b.iterrows():
                ax.vlines(i, r.low, r.high, color=EDGE, lw=1.0, zorder=2)
                y0, hgt = min(r.open, r.close), max(abs(r.close - r.open), 1e-9)
                ax.add_patch(Rectangle((i-0.4, y0), 0.8, hgt,
                                       facecolor="white" if r.close >= r.open else EDGE,
                                       edgecolor=EDGE, lw=1.0, zorder=3))
            ax.set_ylim(lo_ - pad, hi_ + pad)
            span = (hi_ + pad) - (lo_ - pad); off = span * 0.030
            lane_t, lane_b = hi_ + pad * 0.90, lo_ - pad * 0.90
            if row in (1, 2):        # 레인 배경
                ax.axhspan(lane_t - span*0.018, lane_t + span*0.018, color="#EEF0F3", zorder=0)
                ax.axhspan(lane_b - span*0.018, lane_b + span*0.018, color="#EEF0F3", zorder=0)
            for i in range(W):
                if row == 0:         # A안: 증거신호도 봉 밀착
                    if bo[i]: ax.scatter(i, b.low[i]-off, marker="^", s=150, color=G, ec="white", lw=0.8, zorder=6)
                    if to[i]: ax.scatter(i, b.high[i]+off, marker="v", s=150, color=R, ec="white", lw=0.8, zorder=6)
                else:                # B/C안: 증거신호는 레인(종수=불투명도)
                    if bo[i]: ax.add_patch(Rectangle((i-0.4, lane_b-span*0.014), 0.8, span*0.028,
                                                     color=G, alpha=0.35+0.15*min(bo[i],4), zorder=4))
                    if to[i]: ax.add_patch(Rectangle((i-0.4, lane_t-span*0.014), 0.8, span*0.028,
                                                     color=R, alpha=0.35+0.15*min(to[i],4), zorder=4))
            for i in ev_b:           # 이벤트 트리거
                if row == 1: ax.add_patch(Rectangle((i-0.4, lane_b-span*0.014), 0.8, span*0.028, color=G, zorder=5))
                else: ax.scatter(i, b.low[i]-off*2.2, marker="^", s=330, color=G, ec="white", lw=1.4, zorder=7)
            for i in ev_t:
                if row == 1: ax.add_patch(Rectangle((i-0.4, lane_t-span*0.014), 0.8, span*0.028, color=R, zorder=5))
                else: ax.scatter(i, b.high[i]+off*2.2, marker="v", s=330, color=R, ec="white", lw=1.4, zorder=7)
            nb, nt = int((bo>0).sum()), int((to>0).sum())
            if row == 0:
                ax.set_title(f"{name}   ({lab}: 증거 {nb+nt}개 + 이벤트 {len(ev_b)+len(ev_t)}개)",
                             loc="left", color=INK, pad=8)
            else:
                ax.set_title(name, loc="left", color=INK, pad=8)
            ax.set_xlim(-1.5, W+0.5); ax.set_xticks([]); ax.tick_params(labelsize=14)
            for sp in ax.spines.values(): sp.set_color("#C9CDD4")
    fig.suptitle("청산맵 신호 표시 3안 비교 — 실제 기하(72봉·6시간) · 실제 증거신호 발동", fontsize=27, color=INK, y=0.995)
    fig.text(0.5, 0.012, "레인 진하기 = 동시발동 종수 · 큰 삼각형 = 이벤트 트리거 · 작은 삼각형 = 증거신호",
             ha="center", fontsize=18, color="#5B616E")
    out = ROOT / "tmp/eth_signal_map_20260909/liqmap_marker_mockup.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.975]); fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"→ {out} · 중앙창 마크 {marks[med_i-900]} · 90분위창 {marks[hi_i-900]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
