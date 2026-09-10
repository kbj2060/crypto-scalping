#!/usr/bin/env python3
"""「내 계좌」 패널 시각화 3안 목업 (2026-09-11, 사용자 요청).

사용자: "내 계좌 정보 컴포넌트를 텍스트 말고 그래프나 그림으로 보면 바로 알 수 있게끔".
현재는 여섯 숫자가 한 줄에 나열돼 가장 중요한 **청산가까지 거리**가 묻힌다.

⚠️실계좌 실측값을 그대로 쓴다(`/api/binance-account`). 대시보드 팔레트로 그려 실제 화면에
   가깝게 미리 본다 -- Playwright 크로미움이 dev·서버 어디에도 없어서 스크린샷을 못 찍는다
   (2026-09-09 청산맵 목업과 같은 방식).
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
KF = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KF.exists():
    fm.fontManager.addfont(str(KF)); plt.rcParams["font.family"] = fm.FontProperties(fname=str(KF)).get_name()
plt.rcParams.update({"axes.unicode_minus": False})
# ⚠️matplotlib mathtext 는 $...$ 를 수식으로 파싱한다 -- 라벨의 달러기호는 반드시 이스케이프.
D = "\\$"   # 라벨에 찍히는 달러기호

# 대시보드 팔레트 (styles.css :root)
BG, PANEL, INK, MUTED = "#0b0d13", "#15171f", "#eef0f6", "#8b91a6"
GOOD, BAD, WARN, AMBER, LINE = "#6bab84", "#cf6a5c", "#dc8f4a", "#f2b84b", "#2a2e3a"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=25) as r:
        return json.loads(r.read().decode())


def panel(ax, title):
    ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
    ax.set_title(title, color=INK, fontsize=15, loc="left", pad=10, fontweight="bold")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8787/api/binance-account"
    d = fetch(url)
    b, p = d["balance"], d["positions"][0]
    entry, mark, liq = float(p["entry_price"]), float(p["mark_price"]), float(p["liquidation_price"])
    upnl, wallet = float(p["unrealized_pnl"]), float(b["wallet"])
    liq_pct = abs(mark - liq) / mark * 100
    used_pct = float(b["margin"]) / wallet * 100
    upnl_pct = upnl / wallet * 100
    long = p["side"] == "LONG"
    print(f"실측: 지갑 ${wallet:.2f} · 청산까지 {liq_pct:.2f}% · 증거금 사용 {used_pct:.1f}% · 미실현 {upnl_pct:+.1f}%")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12.5), facecolor=BG)
    fig.suptitle("「내 계좌」 시각화 3안 — 실제 계좌 수치 그대로", color=INK, fontsize=20, y=0.985)

    # ── A안: 가격 축 리스크 게이지 ────────────────────────────────────────────
    ax = axes[0]; panel(ax, "A안  가격 축 게이지 — 청산·현재·진입을 한 축 위에")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    lo, hi = min(liq, entry, mark), max(liq, entry, mark)
    pad = (hi - lo) * 0.35
    x = lambda v: (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * 100
    ax.add_patch(Rectangle((0, 4.3), x(liq), 1.4, color=BAD, alpha=0.30))      # 청산 구역
    ax.plot([0, 100], [5, 5], color=LINE, lw=2, zorder=1)
    for v, c, lab, ha in ((liq, BAD, f"청산 {D}{liq:,.0f}", "center"),
                          (entry, MUTED, f"진입 {D}{entry:,.0f}", "center"),
                          (mark, GOOD if upnl >= 0 else BAD, f"현재 {D}{mark:,.0f}", "center")):
        ax.plot([x(v)], [5], "o", ms=14 if v == mark else 9, color=c, zorder=3)
        ax.text(x(v), 6.3 if v != entry else 3.0, lab, color=c, ha=ha, fontsize=13, fontweight="bold")
    ax.annotate("", xy=(x(liq), 7.6), xytext=(x(mark), 7.6),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=2))
    ax.text((x(liq) + x(mark)) / 2, 8.1, f"청산까지 {liq_pct:.2f}%", color=WARN,
            ha="center", fontsize=15, fontweight="bold")
    ax.text(1, 1.2, f"ETH {'롱' if long else '숏'} ×{p['leverage']:.0f}  ·  미실현 "
                    f"{D}{upnl:,.2f} ({upnl_pct:+.1f}% of 지갑)",
            color=BAD if upnl < 0 else GOOD, fontsize=13)

    # ── B안: 이중 막대 게이지 ────────────────────────────────────────────────
    ax = axes[1]; panel(ax, "B안  이중 막대 — 청산 여유 · 증거금 사용률")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    for i, (lab, val, cap, col) in enumerate((
            ("청산까지 여유", liq_pct, 10.0, BAD if liq_pct < 3 else WARN if liq_pct < 6 else GOOD),
            ("증거금 사용률", used_pct, 100.0, BAD if used_pct > 80 else WARN if used_pct > 60 else GOOD))):
        y = 6.4 - i * 3.1
        ax.add_patch(FancyBboxPatch((12, y), 76, 1.5, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc="#20242e", ec="none"))
        ax.add_patch(FancyBboxPatch((12, y), max(76 * min(val / cap, 1), 0.6), 1.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.3", fc=col, ec="none"))
        ax.text(11, y + 0.75, lab, color=MUTED, ha="right", va="center", fontsize=13)
        ax.text(89, y + 0.75, f"{val:.1f}%", color=col, ha="left", va="center",
                fontsize=15, fontweight="bold")
    ax.text(50, 1.0, f"미실현 {D}{upnl:,.2f}  ({upnl_pct:+.1f}% of 지갑 {D}{wallet:,.0f})",
            color=BAD if upnl < 0 else GOOD, ha="center", fontsize=14, fontweight="bold")

    # ── C안: 큰 숫자 + 미니 막대 ─────────────────────────────────────────────
    ax = axes[2]; panel(ax, "C안  핵심 숫자 크게 + 보조 막대")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    ax.text(3, 6.2, f"{liq_pct:.2f}%", color=BAD if liq_pct < 3 else WARN, fontsize=44, fontweight="bold")
    ax.text(3, 4.4, "청산까지", color=MUTED, fontsize=13)
    ax.text(34, 6.2, f"{upnl_pct:+.1f}%", color=BAD if upnl < 0 else GOOD, fontsize=44, fontweight="bold")
    ax.text(34, 4.4, f"미실현 ({D}{upnl:,.0f})", color=MUTED, fontsize=13)
    ax.text(66, 6.2, f"{used_pct:.0f}%", color=WARN if used_pct > 60 else GOOD, fontsize=44, fontweight="bold")
    ax.text(66, 4.4, "증거금 사용", color=MUTED, fontsize=13)
    ax.add_patch(Rectangle((3, 2.2), 94, 0.9, color="#20242e"))
    ax.add_patch(Rectangle((3, 2.2), 94 * min(used_pct / 100, 1), 0.9, color=WARN))
    ax.text(3, 1.0, f"ETH {'롱' if long else '숏'} ×{p['leverage']:.0f} · 진입 {D}{entry:,.0f} → 현재 "
                    f"{D}{mark:,.0f} · 청산 {D}{liq:,.0f}", color=MUTED, fontsize=12)

    out = ROOT / "docs/charts/account_panel_mockup_20260911.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(out, dpi=115, facecolor=BG)
    print(f"저장 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
