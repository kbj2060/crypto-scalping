"""**배리어 라벨 차트** — 라벨이 실제로 무엇을 말하는지 눈으로 확인 (2026-09-14).

사용자: *"라벨도 차트를 한 번 보여줘. 하기 전에 검사를 받고 진행해."*

라벨은 **재구현하지 않는다** -- `research_eth_direction_barrier_label_20260914.leg` 를 그대로 부른다.
4단:
  ① 한 달 가격 + 봉마다의 라벨 부호 띠(초록=롱이 먼저 익절 · 빨강=숏이 먼저 · 회색=양쪽 손절)
  ② 앵커 하나 확대: 두 배리어선과 롱·숏 각각의 **우선도달 지점**
  ③ 해결시간 분포(롱/숏/양쪽손절)
  ④ 변동성 분위별 쌍당 손익(= 스트래들 항 m) -- 게이트가 무엇을 고르는지
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402
from plot_korean_font_20260909 import use_korean_font  # noqa: E402

GREEN, RED, GREY = "#1a9850", "#d73027", "#9e9e9e"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="tmp/eth_barrier_label_20260914.png")
    a = ap.parse_args()
    use_korean_font()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ts = d.timestamp.to_numpy()
    i0 = int(np.searchsorted(ts, np.datetime64(a.start)))
    i1 = min(i0 + a.days * 288, len(c) - 1)

    # ── 라벨 (배리어 스크립트를 그대로 호출) ──────────────────────────────
    idx, sign, tl, ts_, both = [], [], [], [], []
    for i in range(i0, i1):
        L = B.leg(c, hi, lo, i, 1, a.up, a.down); Sg = B.leg(c, hi, lo, i, 2, a.up, a.down)
        if L is None or Sg is None:
            continue
        idx.append(i); tl.append(L[1]); ts_.append(Sg[1])
        both.append(not L[2] and not Sg[2])
        sign.append(1 if L[2] else (-1 if Sg[2] else 0))
    idx = np.array(idx); sign = np.array(sign); tl = np.array(tl); ts_ = np.array(ts_)
    both = np.array(both)
    t = ts[idx]

    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.1, 1.0], hspace=0.33, wspace=0.16)

    # ① 가격 + 라벨 띠
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, c[idx], lw=1.1, color="#222", zorder=3)
    y0, y1 = np.nanmin(c[idx]), np.nanmax(c[idx]); pad = (y1 - y0) * 0.10
    band0, band1 = y0 - pad, y0 - pad * 0.35
    for v, col in ((1, GREEN), (-1, RED), (0, GREY)):
        m = sign == v
        if m.any():
            ax.fill_between(t, band0, band1, where=m, color=col, step="mid", lw=0)
    ax.set_ylim(band0 - pad * 0.2, y1 + pad * 0.3)
    ax.set_title(f"① 봉마다의 라벨 — 익절 +{a.up*100:g}% / 손절 −{a.down*100:g}% 우선도달"
                 f"   (초록=롱이 먼저 익절 {100*(sign==1).mean():.0f}% · "
                 f"빨강=숏이 먼저 {100*(sign==-1).mean():.0f}% · "
                 f"회색=양쪽 손절 {100*(sign==0).mean():.0f}%)", fontsize=15, pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_ylabel("ETH 종가(USDT)", fontsize=12); ax.grid(alpha=.25)

    # ② 앵커 하나 확대 — 배리어와 우선도달
    k = int(np.argmax((sign == 1) & (tl < 2000)))       # 롱이 이긴 앵커 하나
    i = int(idx[k]); span = int(max(tl[k], ts_[k])) + 60
    j1 = min(i + span, len(c) - 1)
    ax = fig.add_subplot(gs[1, 0])
    tt = ts[i:j1]
    ax.plot(tt, c[i:j1], lw=1.2, color="#222", zorder=3)
    ax.fill_between(tt, lo[i:j1], hi[i:j1], color="#888", alpha=.28, lw=0, zorder=2)
    px = c[i]
    for lv, col, lab in ((px * (1 + a.up), GREEN, f"롱 익절 +{a.up*100:g}%"),
                         (px * (1 - a.down), RED, f"롱 손절 −{a.down*100:g}%"),
                         (px * (1 - a.up), "#1a6f9e", f"숏 익절 −{a.up*100:g}%"),
                         (px * (1 + a.down), "#e08214", f"숏 손절 +{a.down*100:g}%")):
        ax.axhline(lv, color=col, ls="--", lw=1.6, alpha=.9)
        ax.text(tt[-1], lv, " " + lab, color=col, va="center", fontsize=11)
    ax.axvline(ts[i], color="#333", lw=1.4)
    ax.scatter([ts[i]], [px], s=90, color="#333", zorder=5, label="진입(봉 종가)")
    ax.scatter([ts[i + int(tl[k])]], [c[i + int(tl[k])]], s=130, marker="^", color=GREEN,
               zorder=5, label=f"롱 해결 {tl[k]*5/60:.1f}h")
    ax.scatter([ts[i + int(ts_[k])]], [c[i + int(ts_[k])]], s=130, marker="v", color=RED,
               zorder=5, label=f"숏 해결 {ts_[k]*5/60:.1f}h")
    ax.set_title("② 앵커 하나 — 두 다리를 각각 굴린다 (회색 = 봉 고저, 배리어 판정 기준)", fontsize=14)
    ax.legend(fontsize=11, loc="upper left"); ax.grid(alpha=.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))

    # ③ 해결시간 분포
    ax = fig.add_subplot(gs[1, 1])
    res = np.maximum(tl, ts_) * 5 / 60
    bins = np.logspace(np.log10(max(res.min(), .05)), np.log10(res.max()), 44)
    ax.hist(res[sign == 1], bins=bins, color=GREEN, alpha=.62, label="롱 승")
    ax.hist(res[sign == -1], bins=bins, color=RED, alpha=.62, label="숏 승")
    ax.hist(res[sign == 0], bins=bins, color=GREY, alpha=.62, label="양쪽 손절")
    ax.set_xscale("log"); ax.axvline(np.median(res), color="#333", ls="--", lw=1.6)
    ax.text(np.median(res), ax.get_ylim()[1] * .92, f" 중앙 {np.median(res):.0f}h", fontsize=12)
    ax.set_title("③ 해결까지 걸린 시간 — 이게 회전율을 정한다", fontsize=14)
    ax.set_xlabel("시간(로그)", fontsize=12); ax.legend(fontsize=11); ax.grid(alpha=.25)

    # ④ 변동성 분위별 쌍당 손익(스트래들 항 m) — 전 창
    ax = fig.add_subplot(gs[2, :])
    rv = d["rv48"].to_numpy(float)
    tl_, th_ = win["TRAIN"]; edges = np.nanpercentile(rv[tl_:th_], [20, 40, 60, 80])
    W = ("BACK22_23", "TRAIN", "VAL", "OOS", "TEST")
    wd = 0.15
    for wi, w in enumerate(W):
        L = B.label_window(c, hi, lo, sm["ok"], *win[w], 6, a.up, a.down)
        g = np.digitize(rv[L["idx"]], edges)
        vals = [L["m"][g == q].mean() if (g == q).sum() > 30 else np.nan for q in range(5)]
        ax.bar(np.arange(5) + (wi - 2) * wd, vals, width=wd, label=w,
               color=plt.cm.viridis(wi / 4), edgecolor="#333", lw=.5)
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"분위{q}\n{'변동성 최저' if q == 0 else ('변동성 최고' if q == 4 else '')}"
                        for q in range(5)], fontsize=12)
    ax.set_title("④ 양측 동시진입(스트래들) 쌍당 손익 — rv48 5분위별 · 경계는 TRAIN 에서만 자름",
                 fontsize=14)
    ax.set_ylabel("쌍당 bp", fontsize=12); ax.legend(fontsize=11, ncol=5); ax.grid(alpha=.25, axis="y")

    fig.suptitle(f"ETH 배리어 라벨 점검 — {a.start} 부터 {a.days}일 · 익절 +{a.up*100:g}% / "
                 f"손절 −{a.down*100:g}% · 앵커 {len(idx):,}개", fontsize=18, y=0.995)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
    print(f"저장: {out}  ({out.stat().st_size/1e6:.1f}MB)")
    print(f"라벨 구성: 롱승 {100*(sign==1).mean():.1f}% · 숏승 {100*(sign==-1).mean():.1f}% · "
          f"양쪽손절 {100*(sign==0).mean():.1f}% · 해결 중앙 {np.median(res):.1f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
