#!/usr/bin/env python3
"""「내 계좌」 최종안 목업 — C(핵심 숫자) x E(좌우 2분할 + 성과) 혼합 (2026-09-11, 사용자 선택).

왼쪽 = C안: 큰 숫자 셋(청산까지·미실현·증거금) + 노출 막대 + 포지션 한 줄
오른쪽 = E안: 왕복 손익 막대 + 누적선 + "이 한 건" 강조

⚠️대시보드 패널 비율(가로로 넓고 낮음)에 맞춰 그린다. 실제 구현도 이 배치를 그대로 따른다.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False})
D = "\\$"
BG, PANEL, INK, MUTED = "#0b0d13", "#15171f", "#eef0f6", "#8b91a6"
GOOD, BAD, WARN, LINE, TRACK = "#6bab84", "#cf6a5c", "#dc8f4a", "#2a2e3a", "#20242e"


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8787/api/binance-account"
    with urllib.request.urlopen(url, timeout=25) as r:
        d = json.loads(r.read().decode())
    b, p = d["balance"], d["positions"][0]
    wallet, margin, upnl = float(b["wallet"]), float(b["margin"]), float(b["unrealized"])
    entry, mark, liq = float(p["entry_price"]), float(p["mark_price"]), float(p["liquidation_price"])
    lev, notional = float(p["leverage"]), float(p["notional"])
    liq_pct = abs(mark - liq) / mark * 100
    used = margin / wallet * 100
    upnl_pct = upnl / wallet * 100
    expo = notional / wallet
    closed = [t for t in d["trades"] if t["closed"]]
    net = [float(t["net_pnl"]) for t in closed]
    cum = np.cumsum(net); wins = sum(1 for x in net if x > 0); wi = int(np.argmin(net))

    fig, ax = plt.subplots(figsize=(14, 4.6), facecolor=BG)
    ax.set_facecolor(PANEL); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
    ax.plot([54, 54], [0.8, 9.2], color=LINE, lw=1.4)

    # ── 왼쪽 (C안) ───────────────────────────────────────────────────────────
    cols = ((3.0, f"{liq_pct:.2f}%", "청산까지", BAD if liq_pct < 3 else WARN if liq_pct < 6 else GOOD),
            (20.0, f"{upnl_pct:+.1f}%", f"미실현 ({D}{upnl:,.0f})", BAD if upnl < 0 else GOOD),
            (38.0, f"{used:.0f}%", "증거금 사용", BAD if used > 80 else WARN if used > 60 else GOOD))
    for x, big, lab, col in cols:
        ax.text(x, 6.6, big, color=col, fontsize=34, fontweight="bold", va="center")
        ax.text(x, 5.1, lab, color=MUTED, fontsize=11.5, va="center")
    # 노출 막대
    ax.text(3, 3.85, "노출", color=MUTED, fontsize=11, va="center")
    ax.add_patch(Rectangle((10, 3.5), 40, 0.72, color=TRACK))
    ax.add_patch(Rectangle((10, 3.5), 40 * min(expo / 30, 1), 0.72,
                           color=BAD if expo > 15 else WARN))
    ax.text(51, 3.86, f"{expo:.1f}배", color=BAD if expo > 15 else WARN, fontsize=13,
            fontweight="bold", va="center")
    ax.text(3, 2.0, f"ETH {'롱' if p['side']=='LONG' else '숏'} ×{lev:.0f} · 수량 {p['qty']}",
            color=INK, fontsize=12, va="center")
    ax.text(3, 1.0, f"진입 {D}{entry:,.0f} → 현재 {D}{mark:,.0f} · 청산 {D}{liq:,.0f} "
                    f"· 지갑 {D}{wallet:,.0f}", color=MUTED, fontsize=11, va="center")

    # ── 오른쪽 (E안) ─────────────────────────────────────────────────────────
    x0, x1 = 57.5, 97.5
    ax.text(x0, 9.0, f"왕복 {len(net)}건 · 승률 {wins/len(net)*100:.0f}% · 누적 {sum(net):+,.0f}",
            color=MUTED, fontsize=12, va="center")
    zero = 5.4
    bw = (x1 - x0) / max(len(net), 1)
    sc = 2.9 / max(abs(min(net)), abs(max(net)))
    for i, v in enumerate(net):
        ax.add_patch(Rectangle((x0 + i * bw, zero), bw * 0.72, v * sc,
                               color=BAD if v < 0 else GOOD))
    ax.plot([x0, x1], [zero, zero], color=LINE, lw=1.2)
    ax.plot(x0 + (np.arange(len(net)) + 0.36) * bw, zero + cum * sc,
            color=WARN, lw=1.8, marker="o", ms=2.6)
    ax.annotate(f"이 한 건 {net[wi]:+,.0f}",
                xy=(x0 + (wi + 0.36) * bw, zero + net[wi] * sc * 0.55),
                xytext=(x0 + 9, 1.55), color=BAD, fontsize=11.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.5))
    ax.text(x1, 1.55, f"나머지 {len(net)-1}건 {sum(net)-net[wi]:+,.0f}",
            color=GOOD, fontsize=11.5, ha="right", va="center")

    out = ROOT / "docs/charts/account_panel_mockup_ce_20260911.png"
    fig.tight_layout(); fig.savefig(out, dpi=118, facecolor=BG)
    print(f"저장 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
