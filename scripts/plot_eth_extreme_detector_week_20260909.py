#!/usr/bin/env python3
"""극점 탐지기 주간 차트 — 사람이 보는 용도 (2026-09-09).

마크의 뜻: "이 봉이 ±12봉(±60분) 국소 극점일 확률" 등급.
표본외 실측 정밀도  강 62.3% · 중 51.4% · 약 34.4%  (증거신호 발동봉 기저 24.2% · 무작위 봉 2.9%)
채운 마커=실제 극점이었음 · 빈 마커=빗나감 · 회색 테두리=아직 미해소(12봉 안 지남)
⚠️매매 신호가 아니다 — 이 등급으로 매매하면 비용 뒤 +2.4~3.9bp 로 여유가 없다.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/eth_signal_map_20260909"; CACHE = D / "klcache"
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 19, "axes.titlesize": 23,
                     "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 18,
                     "legend.fontsize": 18, "figure.facecolor": "white"})
G, R, N, EDGE, INK = "#1F9D55", "#D94141", "#9AA0A6", "#7F8794", "#1F2430"
SZ = {"강": 900, "중": 460, "약": 190}
LBL = {"demarker_extreme": "DeM", "orthogonal_combo": "오실", "short_term_return_z": "급변",
       "taker_delta_z_climax": "체결", "smt_divergence": "SMT", "liquidity_sweep": "스윕",
       "kalman_deviation_meanrev": "칼만", "fib_extension_exhaustion": "확장"}


def newest(sym):
    best, bn = None, 0
    for f in glob.glob(str(CACHE / f"{sym}_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()
    kl = newest("ETHUSDT")
    kl["t"] = kl["timestamp"] + pd.Timedelta(hours=9)          # KST
    # 추세 분위(7일 롤링) — 게이트 구간 음영용. 탐지기와 같은 정의.
    c = kl["close"]; hi = kl["high"]; lo = kl["low"]
    tr = np.maximum(hi - lo, np.maximum((hi - c.shift(1)).abs(), (lo - c.shift(1)).abs()))
    atrp = tr.rolling(14).mean() / c
    r144 = (c / c.shift(144) - 1) / atrp.clip(lower=1e-9)
    kl["tq"] = r144.rolling(2016, min_periods=500).rank(pct=True)
    W = pd.read_csv(D / "extreme_detector_week.csv")
    W["t"] = pd.to_datetime(W["_ts"]) + pd.Timedelta(hours=9)
    meta = json.load(open(D / "extreme_detector_meta.json"))
    prec = meta["precision"]
    # itertuples 는 밑줄로 시작하는 컬럼명을 _1,_2 로 바꿔버린다 -> 미리 개명한다
    W = W.rename(columns={"_long": "is_long", "_y": "hit", "_names": "names"})
    if "gated" not in W.columns: W["gated"] = False
    W = W[W.grade != "-"].copy()

    last = kl.t.max().normalize()
    days = [last - pd.Timedelta(days=k) for k in range(a.days - 1, -1, -1)]
    Hh = 6.6 * a.days + 3.2
    fig = plt.figure(figsize=(34, Hh), dpi=115)
    gs = fig.add_gridspec(a.days, 1, hspace=0.30, top=1 - 3.0 / Hh, bottom=0.9 / Hh,
                          left=0.055, right=0.995)
    for r, day in enumerate(days):
        ax = fig.add_subplot(gs[r])
        b = kl[(kl.t >= day) & (kl.t < day + pd.Timedelta(days=1))].reset_index(drop=True)
        if b.empty: continue
        pos = {t: i for i, t in enumerate(b.t)}
        for i, row in b.iterrows():
            ax.vlines(i, row.low, row.high, color=EDGE, lw=1.0, zorder=2)
            y0, h = min(row.open, row.close), max(abs(row.close - row.open), 1e-9)
            ax.add_patch(Rectangle((i - 0.34, y0), 0.68, h,
                                   facecolor="white" if row.close >= row.open else EDGE,
                                   edgecolor=EDGE, lw=1.0, zorder=3))
        # 강한 추세 구간 음영 — 여기서는 콜을 억제한다
        gm = ((b.tq >= 0.80) | (b.tq <= 0.20)).to_numpy()
        st = None
        for i2, v in enumerate(list(gm) + [False]):
            if v and st is None: st = i2
            elif not v and st is not None:
                ax.axvspan(st - 0.5, i2 - 0.5, color="#F0A500", alpha=0.10, zorder=0); st = None
        span = b.high.max() - b.low.min()
        ax.set_ylim(b.low.min() - span * 0.16, b.high.max() + span * (0.34 if r == 0 else 0.20))
        off = span * 0.030
        q = W[(W.t >= day) & (W.t < day + pd.Timedelta(days=1))].copy()
        q["x"] = q["t"].map(pos); q = q.dropna(subset=["x"])
        nh = {}
        for row in q.itertuples():
            x = int(row.x); long = bool(row.is_long)
            col = G if long else R
            y = b.low.iloc[x] - off if long else b.high.iloc[x] + off
            hit = int(row.hit)
            if bool(row.gated):                      # 추세 구간 = 억제된 콜(참고 표시만)
                ax.scatter(x, y, marker="x", s=SZ[row.grade] * 0.45, color=N, lw=2.2,
                           alpha=0.55, zorder=5)
                continue
            face = col if hit == 1 else "none"
            ec = col if hit >= 0 else N
            ax.scatter(x, y, marker="^" if long else "v", s=SZ[row.grade], facecolor=face,
                       edgecolor=ec, lw=2.6 if hit != 1 else 1.4, zorder=6)
            if row.grade == "강":
                ax.text(x, y + (-off * 1.5 if long else off * 1.5), f"{row.p:.2f}", ha="center",
                        va="top" if long else "bottom", fontsize=15, color=col, zorder=7)
            nh[row.grade] = nh.get(row.grade, [0, 0])
            nh[row.grade][0] += 1; nh[row.grade][1] += (hit == 1)
        head = "  ·  ".join(f"{g} {nh[g][0]}건(적중 {nh[g][1]})" for g in ("강", "중", "약") if g in nh)
        ax.set_title(f"{day:%m-%d (%a)}   {head if head else '후보 없음'}", loc="left",
                     color=INK, pad=9)
        ax.set_ylabel("가격", color=INK)
        ax.grid(alpha=0.16, lw=0.8)
        ticks = [i for i, t in enumerate(b.t) if t.minute == 0 and t.hour % 2 == 0]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{b.t.iloc[i]:%H:%M}" for i in ticks])
        ax.set_xlim(-1.5, len(b) + 0.5)
        for sp in ax.spines.values(): sp.set_color("#C9CDD4")
        if r == 0:
            ax.legend(handles=[
                Line2D([], [], marker="^", ls="", ms=22, color=G, label="바닥 후보(강)"),
                Line2D([], [], marker="^", ls="", ms=15, color=G, label="중"),
                Line2D([], [], marker="^", ls="", ms=10, color=G, label="약"),
                Line2D([], [], marker="v", ls="", ms=22, color=R, label="천장 후보(강)"),
                Line2D([], [], marker="^", ls="", ms=18, mfc="none", mec=G, mew=2.4, label="빈 마커 = 빗나감"),
                Line2D([], [], marker="^", ls="", ms=18, mfc="none", mec=N, mew=2.4, label="회색 = 미해소"),
                Line2D([], [], marker="x", ls="", ms=15, color=N, mew=2.2, label="× = 추세구간 억제"),
            ], loc="upper left", ncol=6, framealpha=0.93, columnspacing=1.2, handletextpad=0.4)
        if r == a.days - 1:
            ax.set_xlabel("시각 (KST)", color=INK)
    fig.text(0.5, 1 - 0.4 / Hh, "ETH 극점 탐지기 — 증거신호 8종을 피쳐로 쓴 극점 확률 모델 "
             f"(최근 {a.days}일, KST)", ha="center", va="top", fontsize=29, color=INK)
    fig.text(0.5, 1 - 1.05 / Hh,
             f"마크 = 이 봉이 ±60분 국소 극점일 확률 등급   |   추세 게이트 적용 후 표본외 정밀도  "
             f"강 {prec.get('강',0)*100:.1f}% (하루 1.5건) · 중 {prec.get('중',0)*100:.1f}% · "
             f"약 {prec.get('약',0)*100:.1f}%   ·   증거신호 발동봉 기저 {meta['base']*100:.1f}% · 무작위 봉 2.9%",
             ha="center", va="top", fontsize=21, color="#5B616E")
    fig.text(0.5, 1 - 1.60 / Hh,
             "채운 마커=실제 극점이었음 · 빈 마커=빗나감 · 회색 테두리=미해소   |   "
             "주황 음영 = 강한 추세 구간(7일 롤링 상하 20%) — 여기선 콜을 억제한다"
             " · 근거: 강한상승 천장 콜은 적중 +10.8 / 빗나감 -60.7bp 로 완전 비대칭(순 -23.8bp)",
             ha="center", va="top", fontsize=20, color="#5B616E")
    out = D / f"extreme_detector_{a.days}days_KST.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"→ {out} ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
