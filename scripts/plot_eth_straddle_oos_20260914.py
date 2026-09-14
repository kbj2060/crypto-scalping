"""**ETH 스트래들 OOS 거래 차트** (2026-09-14, 사용자 *"스트래들의 oos 거래 차트를 보여줘"*).

라벨·체결은 재구현하지 않는다 -- `research_eth_direction_barrier_label_20260914.leg` 를 그대로 부른다.
4단: ① OOS 가격 + 순차 쌍(진입→해결) + 게이트 자격 구간 ② 쌍별 손익 ③ 자산곡선 4종
(순차 stride1 · 순차 stride3 · 동시 4쌍 · 동시 32쌍) ④ 겹침 앵커 전체 분포 vs 실제로 잡힌 쌍.
③④ 가 이 축이 죽은 이유다 -- 같은 규칙인데 **어느 자격 봉에 들어가느냐**로 부호가 바뀐다.
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

GREEN, RED, BLUE, GREY = "#1a9850", "#d73027", "#3f6fb5", "#9e9e9e"


def walk(c, hi, lo, ok, g, edge, lo_i, hi_i, u, dn, kmax: int, stride: int):
    """동시 최대 kmax 쌍. (진입봉, 종료봉, 쌍수익bp) 목록과 (종료봉, 자산배수) 곡선."""
    eq, open_until, pend, pairs, curve = 1.0, [], {}, [], []
    for i in range(lo_i, hi_i, stride):
        for e in sorted([k for k in pend if k <= i]):
            for m in pend.pop(e):
                eq *= (1.0 + m / kmax); curve.append((e, eq))
        open_until = [e for e in open_until if e > i]
        if len(open_until) >= kmax or not (ok[i] and np.isfinite(g[i]) and g[i] <= edge):
            continue
        a = B.leg(c, hi, lo, i, 1, u, dn); b = B.leg(c, hi, lo, i, 2, u, dn)
        if a is None or b is None:
            continue
        m = 0.5 * (a[0] + b[0]); end = i + int(max(a[1], b[1]))
        open_until.append(end); pend.setdefault(end, []).append(m)
        pairs.append((i, end, 1e4 * m))
    for e in sorted(pend):
        for m in pend[e]:
            eq *= (1.0 + m / kmax); curve.append((e, eq))
    return pairs, curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--down", type=float, default=0.03)
    ap.add_argument("--window", default="OOS")
    ap.add_argument("--out", default="tmp/eth_straddle_oos_20260914.png")
    a = ap.parse_args()
    use_korean_font()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]; ts = d.timestamp.to_numpy()
    g = T.gate_series(d, c, hi, lo)
    tl, th = win["TRAIN"]; edge = float(np.nanpercentile(g[tl:th], 20))
    i0, i1 = win[a.window]

    arms = {"순차(stride1)": (1, 1), "순차(stride3)": (1, 3), "동시 4쌍": (4, 3), "동시 32쌍": (32, 3)}
    out = {k: walk(c, hi, lo, ok, g, edge, i0, i1, a.up, a.down, km, st)
           for k, (km, st) in arms.items()}
    base_pairs = out["순차(stride1)"][0]

    cmp_w = "VAL" if a.window != "VAL" else "TEST"      # 대조로 다른 창도 같이 그린다
    j0, j1 = win[cmp_w]
    out2 = {k: walk(c, hi, lo, ok, g, edge, j0, j1, a.up, a.down, km, st)
            for k, (km, st) in arms.items()}
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.0], hspace=0.33, wspace=0.18)

    # ① 가격 + 순차 쌍 + 게이트 자격 구간
    ax = fig.add_subplot(gs[0, :])
    sl = slice(i0, i1)
    ax.plot(ts[sl], c[sl], lw=1.0, color="#222", zorder=3)
    y0, y1 = np.nanmin(c[sl]), np.nanmax(c[sl]); pad = (y1 - y0) * 0.12
    elig = (g[sl] <= edge) & ok[sl]
    ax.fill_between(ts[sl], y0 - pad, y0 - pad * 0.55, where=elig, color=BLUE, alpha=.5, step="mid", lw=0)
    ax.text(ts[i0], y0 - pad * 0.77, "  파랑 = 게이트 자격(예측변동성 최저 20%)", color=BLUE, fontsize=12)
    for (i, e, bp) in base_pairs:
        col = GREEN if bp > 0 else RED
        ax.plot([ts[i], ts[min(e, i1 - 1)]], [c[i], c[i]], lw=3.2, color=col, alpha=.85, zorder=4)
        ax.scatter([ts[i]], [c[i]], s=55, color=col, zorder=5, edgecolor="#222", lw=.6)
        ax.plot([ts[i], ts[i]], [c[i] * (1 - a.down), c[i] * (1 + a.up)], lw=.9, color=col, alpha=.45, zorder=2)
    ax.set_ylim(y0 - pad * 1.05, y1 + pad * 0.35)
    ax.set_title(f"① {a.window}({str(ts[i0])[:10]}~{str(ts[i1-1])[:10]}) 스트래들 순차 진입 "
                 f"{len(base_pairs)}쌍 — 익절 +{a.up*100:g}% / 손절 −{a.down*100:g}%"
                 f"   (가로선 = 보유구간 · 세로선 = 두 배리어 · 초록=쌍 이익 · 빨강=쌍 손실)",
                 fontsize=15, pad=12)
    ax.set_ylabel("ETH 종가(USDT)", fontsize=12); ax.grid(alpha=.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ② 쌍별 손익
    ax = fig.add_subplot(gs[1, 0])
    bps = [p[2] for p in base_pairs]
    ax.bar(range(len(bps)), bps, color=[GREEN if b > 0 else RED for b in bps], edgecolor="#333", lw=.6)
    ax.axhline(0, color="#333", lw=1.2)
    ax.plot(range(len(bps)), np.cumsum(bps), color=BLUE, lw=2.2, marker="o", ms=4, label="누적")
    ax.set_title(f"② 쌍별 손익 — 평균 {np.mean(bps):+.1f}bp · 승률 {np.mean(np.array(bps) > 0):.0%} · "
                 f"최악 {np.min(bps):+.0f}bp", fontsize=14)
    ax.set_xlabel("쌍 번호", fontsize=12); ax.set_ylabel("bp", fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=.25, axis="y")

    # ③ 자산곡선 4종 — 같은 규칙, 다른 진입 시점
    ax = fig.add_subplot(gs[1, 1])
    for (name, (pairs, curve)), col in zip(out.items(), (GREEN, RED, "#8c6bb1", "#e08214")):
        if not curve:
            continue
        x = [ts[min(e, i1 - 1)] for e, _ in curve]; y = [v for _, v in curve]
        ax.step([ts[i0]] + x, [1.0] + y, where="post", lw=2.2, color=col,
                label=f"{name}  n={len(pairs)} → x{y[-1]:.3f}")
    ax.axhline(1.0, color="#333", lw=1.2, ls="--")
    ax.set_title(f"③ {a.window} 자산배수 — 같은 규칙, 진입 봉만 다르다", fontsize=14)
    ax.legend(fontsize=11, loc="upper left"); ax.grid(alpha=.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ④ 대조 창의 자산곡선 — 여기서 부호가 갈린다
    ax = fig.add_subplot(gs[2, 0])
    for (name, (pairs, curve)), col in zip(out2.items(), (GREEN, RED, "#8c6bb1", "#e08214")):
        if not curve:
            continue
        x = [ts[min(e, j1 - 1)] for e, _ in curve]; y = [v for _, v in curve]
        ax.step([ts[j0]] + x, [1.0] + y, where="post", lw=2.2, color=col,
                label=f"{name}  n={len(pairs)} → x{y[-1]:.3f}")
    ax.axhline(1.0, color="#333", lw=1.2, ls="--")
    ax.set_title(f"④ 대조 창 {cmp_w} — 같은 규칙인데 진입 봉에 따라 부호가 갈린다", fontsize=14)
    ax.legend(fontsize=11, loc="lower left"); ax.grid(alpha=.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # ⑤ 겹침 앵커 전체 분포 vs 실제로 잡힌 쌍
    ax = fig.add_subplot(gs[2, 1])
    L = B.label_window(c, hi, lo, ok, i0, i1, 6, a.up, a.down)
    sel = g[L["idx"]] <= edge
    allm = L["m"][sel]          # label_window 가 이미 bp 로 준다(1e4 를 또 곱하면 안 된다)
    bins = np.linspace(min(allm.min(), min(bps)) - 20, max(allm.max(), max(bps)) + 20, 60)
    ax.hist(allm, bins=bins, color=GREY, alpha=.75,
            label=f"겹침 앵커 전체 {sel.sum():,}개 · 평균 {allm.mean():+.1f}bp")
    ax2 = ax.twinx()
    ax2.hist(bps, bins=bins, color=BLUE, alpha=.85,
             label=f"실제로 잡힌 순차 {len(bps)}쌍 · 평균 {np.mean(bps):+.1f}bp")
    ax.axvline(allm.mean(), color="#333", ls="--", lw=1.8)
    ax.axvline(np.mean(bps), color=BLUE, ls="--", lw=1.8)
    ax.set_title(f"⑤ {a.window} 쌍당 손익 분포 — 겹침 앵커(회색) vs 실제로 잡힌 쌍(파랑)", fontsize=14)
    ax.set_xlabel("쌍당 손익(bp)", fontsize=12); ax.set_ylabel("겹침 앵커 수", fontsize=12)
    ax2.set_ylabel("실제 쌍 수", fontsize=12, color=BLUE)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12, loc="upper left"); ax.grid(alpha=.25, axis="y")

    fig.suptitle(f"ETH 저변동 스트래들 {a.window} 거래 — 게이트: 홀드아웃 변동성모델 예측 최저 20%",
                 fontsize=18, y=0.995)
    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=105, bbox_inches="tight", facecolor="white")
    print(f"저장: {p} ({p.stat().st_size/1e6:.1f}MB)")
    for w_, o_ in ((a.window, out), (cmp_w, out2)):
        print(f"  [{w_}]")
        for name, (pairs, curve) in o_.items():
            v = [x[2] for x in pairs]
            print(f"    {name:<14} n={len(pairs):>3} · 평균 {np.mean(v) if v else 0:>+7.1f}bp · "
                  f"자산배수 {curve[-1][1] if curve else 1:.3f}")
    for name, (pairs, curve) in ():
        v = [x[2] for x in pairs]
        print(f"  {name:<14} n={len(pairs):>3} · 평균 {np.mean(v):>+7.1f}bp · "
              f"자산배수 {curve[-1][1] if curve else 1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
