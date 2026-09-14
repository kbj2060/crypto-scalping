"""**스트래들 1주일 확대 차트** — 한 쌍이 어떻게 끝나는지 (2026-09-14, 사용자 요청).

OOS(2026-01~03) 안에서 7일 창을 잘라 **쌍 하나의 전 과정**을 본다: 진입 → 한쪽이 손절 →
남은 쪽이 익절(또는 양쪽 손절). 판정은 재구현하지 않고 `research_eth_direction_barrier_label`의
`leg`/`first_touch` 를 그대로 부른다.

①  가격 + 네 배리어선 + 두 다리의 해결 지점 + 게이트 자격 구간
②  두 다리의 미실현 경주 — 누가 먼저 자기 선에 닿는가(봉 고저 기준)
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
import research_eth_straddle_tighten_20260914 as T  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402
from plot_korean_font_20260909 import use_korean_font  # noqa: E402

GREEN, RED, BLUE, ORANGE = "#1a9850", "#d73027", "#3f6fb5", "#e08214"
WEEK = 7 * 288


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--pair", type=int, default=0, help="OOS 순차 쌍 중 몇 번째(0부터)")
    ap.add_argument("--out", default="tmp/eth_straddle_week_20260914.png")
    a = ap.parse_args()
    use_korean_font()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]; ts = d.timestamp.to_numpy()
    g = T.gate_series(d, c, hi, lo)
    tl, th = win["TRAIN"]; edge = float(np.nanpercentile(g[tl:th], 20))
    i0, i1 = win["OOS"]

    # 순차 쌍을 뽑고 **7일 안에 끝나는** 것만 후보로
    pairs, free = [], -1
    for i in range(i0, i1):
        if i <= free or not (ok[i] and np.isfinite(g[i]) and g[i] <= edge):
            continue
        L = B.leg(c, hi, lo, i, 1, a.up, a.down); Sg = B.leg(c, hi, lo, i, 2, a.up, a.down)
        if L is None or Sg is None:
            continue
        span = int(max(L[1], Sg[1]))
        pairs.append({"i": i, "span": span, "L": L, "S": Sg, "bp": 0.5e4 * (L[0] + Sg[0])})
        free = i + span
    fit = [p for p in pairs if p["span"] <= WEEK]
    print(f"OOS 순차 쌍 {len(pairs)}개 · 7일 안에 끝난 것 {len(fit)}개")
    for k, p in enumerate(fit):
        print(f"  [{k}] {str(ts[p['i']])[:16]} · {p['span']*5/60:5.1f}h · {p['bp']:+8.1f}bp"
              f" · 롱 {'승' if p['L'][2] else '패'} / 숏 {'승' if p['S'][2] else '패'}")
    if not fit:
        print("7일 안에 끝나는 쌍이 없다"); return 1
    p = fit[min(a.pair, len(fit) - 1)]
    i = p["i"]; span = p["span"]
    lo_i = max(i0, i - 288); hi_i = min(i1 - 1, i + span + 288)
    if hi_i - lo_i > WEEK + 576:
        hi_i = lo_i + WEEK + 576
    sl = slice(lo_i, hi_i)
    px = c[i]

    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.24)

    # ① 가격 + 배리어
    ax = fig.add_subplot(gs[0])
    ax.plot(ts[sl], c[sl], lw=1.3, color="#222", zorder=4)
    ax.fill_between(ts[sl], lo[sl], hi[sl], color="#999", alpha=.30, lw=0, zorder=2)
    y0, y1 = np.nanmin(lo[sl]), np.nanmax(hi[sl]); pad = (y1 - y0) * 0.12
    elig = (g[sl] <= edge) & ok[sl]
    ax.fill_between(ts[sl], y0 - pad, y0 - pad * 0.6, where=elig, color=BLUE, alpha=.45, step="mid", lw=0)
    for lv, col, lab, ls in ((px * (1 + a.up), GREEN, f"롱 익절 +{a.up*100:g}%", "--"),
                             (px * (1 - a.down), RED, f"롱 손절 −{a.down*100:g}%", "--"),
                             (px * (1 - a.up), "#1a6f9e", f"숏 익절 −{a.up*100:g}%", ":"),
                             (px * (1 + a.down), ORANGE, f"숏 손절 +{a.down*100:g}%", ":")):
        ax.axhline(lv, color=col, ls=ls, lw=1.8, alpha=.9)
        ax.text(ts[hi_i - 1], lv, "  " + lab, color=col, va="center", fontsize=12)
    ax.axvline(ts[i], color="#333", lw=1.6)
    ax.scatter([ts[i]], [px], s=140, color="#333", zorder=6, label="진입(양측 동시)")
    for legname, leg, col, mk in (("롱", p["L"], GREEN, "^"), ("숏", p["S"], RED, "v")):
        j = i + int(leg[1])
        ax.scatter([ts[j]], [c[j]], s=190, marker=mk, color=col, zorder=6, edgecolor="#222", lw=.8,
                   label=f"{legname} {'익절' if leg[2] else '손절'} · {leg[1]*5/60:.1f}h · {1e4*leg[0]:+.0f}bp")
    ax.set_ylim(y0 - pad * 1.05, y1 + pad * 0.3)
    ax.set_title(f"① {str(ts[i])[:16]} 진입 — 익절 +{a.up*100:g}% / 손절 −{a.down*100:g}% 양측 동시"
                 f"   (회색 = 봉 고저 · 파랑 띠 = 게이트 자격 · 쌍 손익 {p['bp']:+.0f}bp)",
                 fontsize=16, pad=12)
    ax.set_ylabel("ETH 종가(USDT)", fontsize=13); ax.grid(alpha=.25)
    ax.legend(fontsize=12, loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))

    # ② 두 다리의 경주 (봉 고저 기준 미실현)
    ax = fig.add_subplot(gs[1])
    jj = np.arange(i, min(i + span + 288, hi_i))
    up_move = 1e4 * (hi[jj] / px - 1); dn_move = 1e4 * (lo[jj] / px - 1)
    ax.fill_between(ts[jj], dn_move, up_move, color="#bbb", alpha=.45, lw=0, label="봉 고저 범위")
    ax.plot(ts[jj], 1e4 * (c[jj] / px - 1), lw=1.4, color="#222", label="종가 기준 이동")
    for lv, col, ls in ((a.up * 1e4, GREEN, "--"), (-a.down * 1e4, RED, "--"),
                        (-a.up * 1e4, "#1a6f9e", ":"), (a.down * 1e4, ORANGE, ":")):
        ax.axhline(lv, color=col, ls=ls, lw=1.6, alpha=.9)
    for legname, leg, col, mk in (("롱", p["L"], GREEN, "^"), ("숏", p["S"], RED, "v")):
        j = i + int(leg[1])
        ax.axvline(ts[j], color=col, lw=1.4, alpha=.6)
        ax.scatter([ts[j]], [1e4 * (c[j] / px - 1)], s=170, marker=mk, color=col, zorder=6,
                   edgecolor="#222", lw=.8)
        ax.text(ts[j], a.up * 1e4 * 0.82, f" {legname} 해결\n {leg[1]*5/60:.1f}h", color=col, fontsize=12)
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_title("② 두 다리의 경주 — 어느 선에 먼저 닿는가 (배리어 판정은 봉 고저 기준, 라이브 컨벤션)",
                 fontsize=15)
    ax.set_ylabel("진입가 대비 이동(bp)", fontsize=13); ax.grid(alpha=.25)
    ax.legend(fontsize=12, loc="lower left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))

    fig.suptitle(f"ETH 저변동 스트래들 — OOS 쌍 #{min(a.pair, len(fit)-1)} 확대 "
                 f"(게이트: 예측변동성 최저 20%)", fontsize=19, y=0.975)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
