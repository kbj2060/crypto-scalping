#!/usr/bin/env python3
"""「내 계좌」 **전체 요약** 시각화 3안 (2026-09-11, 사용자 재요청).

1차 3안(숫자 크게/막대/가격축)은 "포지션 하나"만 봤다. 사용자: "전체 내 계좌를 한 번에
정리해서 한 눈에 들어오는 방향으로". 그래서 자본·노출·리스크·성과를 한 장에 담는다.

⭐실측에서 드러난 것: 닫힌 왕복 12건 승률 58%(7/12)인데 순손익 -$286.82 다.
   그런데 **한 건이 -$535.96** 이고 나머지 11건 합은 +$249. 텍스트로는 안 보이는 이야기라
   세 안 모두 이걸 드러내도록 했다.
⚠️가격 축은 **% 공간**에 그린다 -- 1차 A안은 가격 공간에 여백을 넣어 1.99% 가 화면 절반을
   가로지르는 왜곡이 있었다. % 공간이면 청산 -2.09% / 진입 +0.60% 가 실제 비율로 놓인다.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False})
D = "\\$"   # matplotlib mathtext 가 $...$ 를 수식으로 먹는다 -- 라벨용 달러기호
BG, PANEL, INK, MUTED = "#0b0d13", "#15171f", "#eef0f6", "#8b91a6"
GOOD, BAD, WARN, LINE, TRACK = "#6bab84", "#cf6a5c", "#dc8f4a", "#2a2e3a", "#20242e"


def panel(ax, title):
    ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
    ax.set_title(title, color=INK, fontsize=15, loc="left", pad=9, fontweight="bold")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8787/api/binance-account"
    with urllib.request.urlopen(url, timeout=25) as r:
        d = json.loads(r.read().decode())
    b, p = d["balance"], d["positions"][0]
    wallet, margin, avail, upnl = (float(b[k]) for k in ("wallet", "margin", "available", "unrealized"))
    eq = wallet + upnl
    entry, mark, liq = float(p["entry_price"]), float(p["mark_price"]), float(p["liquidation_price"])
    lev, notional = float(p["leverage"]), float(p["notional"])
    liq_pct = (liq - mark) / mark * 100          # 롱이면 음수
    ent_pct = (entry - mark) / mark * 100
    expo = notional / wallet
    closed = [t for t in d["trades"] if t["closed"]]
    net = [float(t["net_pnl"]) for t in closed]
    cum = np.cumsum(net)
    wins = sum(1 for x in net if x > 0)
    worst_i = int(np.argmin(net))
    fees = sum(float(t["commission"]) for t in d["trades"])
    realized = sum(net)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 15), facecolor=BG)
    fig.suptitle("「내 계좌」 전체 요약 3안 — 실제 계좌 수치", color=INK, fontsize=20, y=0.988)

    # ── D안: 한 장 요약 카드(계층 4단) ────────────────────────────────────────
    ax = axes[0]; panel(ax, "D안  한 장 요약 — 순자산 → 자본 배치 → 포지션 리스크 → 성과")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.text(3, 8.0, f"{D}{eq:,.0f}", color=INK, fontsize=40, fontweight="bold")
    ax.text(3, 7.1, "순자산", color=MUTED, fontsize=12)
    ax.text(22, 8.3, f"{upnl:+,.0f}", color=BAD if upnl < 0 else GOOD, fontsize=22, fontweight="bold")
    ax.text(22, 7.6, f"미실현 ({upnl/wallet*100:+.1f}%)", color=MUTED, fontsize=11)
    # 자본 배치 스택바
    ax.text(3, 6.3, "자본 배치", color=MUTED, fontsize=11)
    seg = [("증거금", margin, WARN), ("가용", avail, GOOD)]
    x0 = 3
    for lab, v, col in seg:
        wdt = 94 * v / wallet
        ax.add_patch(Rectangle((x0, 5.2), wdt, 0.85, color=col))
        if wdt > 8: ax.text(x0 + wdt / 2, 5.62, f"{lab} {v/wallet*100:.0f}%", color="#0b0d13",
                            ha="center", va="center", fontsize=11, fontweight="bold")
        x0 += wdt
    ax.text(97, 6.3, f"노출 {expo:.1f}배 (명목 {D}{notional:,.0f})", color=WARN, ha="right", fontsize=12)
    # 포지션 리스크 축 (% 공간 = 실제 비율)
    span = max(abs(liq_pct), abs(ent_pct)) * 1.45
    px = lambda v: 50 + v / span * 47
    ax.plot([3, 97], [3.6, 3.6], color=LINE, lw=2)
    ax.add_patch(Rectangle((3, 3.25), px(liq_pct) - 3, 0.7, color=BAD, alpha=0.25))
    for v, c, lab, dy in ((liq_pct, BAD, f"청산 {liq_pct:+.2f}%", 0.75),
                          (ent_pct, MUTED, f"진입 {ent_pct:+.2f}%", -0.95),
                          (0.0, INK, "현재", 0.75)):
        ax.plot([px(v)], [3.6], "o", ms=13 if v == 0 else 9, color=c, zorder=3)
        ax.text(px(v), 3.6 + dy, lab, color=c, ha="center", fontsize=11, fontweight="bold")
    # 성과 스파크
    ax.text(3, 1.9, f"닫힌 왕복 {len(net)}건 · 승 {wins}/{len(net)} · 실현 {realized:+,.0f} "
                    f"(수수료 {fees:,.0f} 포함)", color=MUTED, fontsize=11)
    bw = 94 / max(len(net), 1)
    for i, v in enumerate(net):
        hgt = 1.1 * v / max(abs(min(net)), abs(max(net)))
        ax.add_patch(Rectangle((3 + i * bw, 0.9), bw * 0.8, hgt,
                               color=BAD if v < 0 else GOOD))
    ax.plot([3, 97], [0.9, 0.9], color=LINE, lw=1)

    # ── E안: 좌우 2분할 (상태 | 성과) ────────────────────────────────────────
    ax = axes[1]; panel(ax, "E안  좌우 2분할 — 왼쪽 지금 상태 / 오른쪽 지금까지 성과")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.plot([49, 49], [0.6, 8.4], color=LINE, lw=1.5)
    for i, (lab, val, cap, col, unit) in enumerate((
            ("청산까지", abs(liq_pct), 8.0, BAD if abs(liq_pct) < 3 else WARN, "%"),
            ("증거금 사용", margin / wallet * 100, 100.0, WARN, "%"),
            ("노출 배수", expo, 30.0, BAD if expo > 15 else WARN, "배"))):
        y = 6.6 - i * 2.1
        ax.text(3, y + 0.42, lab, color=MUTED, fontsize=12)
        ax.add_patch(FancyBboxPatch((16, y), 22, 0.85, boxstyle="round,pad=0.01,rounding_size=0.2",
                                    fc=TRACK, ec="none"))
        ax.add_patch(FancyBboxPatch((16, y), max(22 * min(val / cap, 1), 0.5), 0.85,
                                    boxstyle="round,pad=0.01,rounding_size=0.2", fc=col, ec="none"))
        ax.text(39.5, y + 0.42, f"{val:.1f}{unit}", color=col, va="center", fontsize=14, fontweight="bold")
    ax.text(3, 1.4, f"순자산 {D}{eq:,.0f}   미실현 {upnl:+,.0f}", color=INK, fontsize=15, fontweight="bold")
    # 오른쪽: 왕복 손익 + 누적
    x = np.arange(len(net)); bwd = 45 / max(len(net), 1)
    sc = 3.0 / max(abs(min(net)), abs(max(net)))
    for i, v in enumerate(net):
        ax.add_patch(Rectangle((52 + i * bwd, 5.0), bwd * 0.75, v * sc,
                               color=BAD if v < 0 else GOOD))
    ax.plot([52, 97], [5.0, 5.0], color=LINE, lw=1)
    ax.plot(52 + (x + 0.4) * bwd, 5.0 + cum * sc, color=WARN, lw=2, marker="o", ms=3)
    ax.text(52, 8.0, f"왕복 {len(net)}건 · 승률 {wins/len(net)*100:.0f}%  ·  누적 {realized:+,.0f}",
            color=MUTED, fontsize=12)
    ax.annotate(f"이 한 건 {net[worst_i]:+,.0f}", xy=(52 + (worst_i + 0.4) * bwd, 5.0 + net[worst_i] * sc),
                xytext=(62, 1.6), color=BAD, fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1.8))
    ax.text(52, 0.7, f"나머지 {len(net)-1}건 합 {sum(net)-net[worst_i]:+,.0f}", color=GOOD, fontsize=12)

    # ── F안: 워터폴 (돈이 어디로 갔나) ───────────────────────────────────────
    ax = axes[2]; panel(ax, "F안  워터폴 — 지갑에서 순자산까지, 돈이 어디로 갔나")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    gross = realized + fees
    steps = [("초기 자본", wallet - realized, None), ("실현 손익", gross, gross >= 0),
             ("수수료", -fees, False), ("미실현", upnl, upnl >= 0)]
    base = 0.0; xs = 6; bw2 = 15
    vmax = max(wallet, wallet - realized) * 1.12
    sy = lambda v: 1.6 + v / vmax * 6.4
    for i, (lab, v, pos) in enumerate(steps):
        if i == 0:
            ax.add_patch(Rectangle((xs, sy(0)), bw2, sy(v) - sy(0), color=MUTED)); base = v
        else:
            y0, y1 = sy(base), sy(base + v)
            ax.add_patch(Rectangle((xs, min(y0, y1)), bw2, abs(y1 - y0),
                                   color=GOOD if pos else BAD)); base += v
        ax.text(xs + bw2 / 2, 1.0, lab, color=MUTED, ha="center", fontsize=11)
        ax.text(xs + bw2 / 2, sy(base) + 0.25, f"{v:+,.0f}" if i else f"{v:,.0f}",
                color=INK, ha="center", fontsize=12, fontweight="bold")
        xs += bw2 + 4
    ax.add_patch(Rectangle((xs, sy(0)), bw2, sy(base) - sy(0), color=WARN))
    ax.text(xs + bw2 / 2, 1.0, "순자산", color=MUTED, ha="center", fontsize=11)
    ax.text(xs + bw2 / 2, sy(base) + 0.25, f"{base:,.0f}", color=INK, ha="center",
            fontsize=13, fontweight="bold")
    ax.text(6, 9.0, f"승률 {wins/len(net)*100:.0f}% 인데 실현이 {realized:+,.0f} 인 이유 = "
                    f"한 건 {net[worst_i]:+,.0f} (나머지 {len(net)-1}건 {sum(net)-net[worst_i]:+,.0f})",
            color=BAD, fontsize=12)

    out = ROOT / "docs/charts/account_panel_mockup_v2_20260911.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.972]); fig.savefig(out, dpi=112, facecolor=BG)
    print(f"저장 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
