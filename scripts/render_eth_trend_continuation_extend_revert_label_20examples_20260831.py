#!/usr/bin/env python3
"""Visual verification for the trend-continuation EXTEND vs REVERT pure-direction label
(scripts/research_eth_trend_continuation_head_phase1_20260831.py section [B]/[2], the exact
construction whose GBM-proxy AUC was 0.49-0.52 and is now being escalated to TabPFN).
Per project convention (feedback_visual_verification_chart_gate_explain_before_proceed), this
MUST be reviewed and explicitly approved by the user before any TabPFN training runs.

Label, reproduced exactly from research_eth_trend_continuation_head_phase1_20260831.py:
  * Candidate events = union of the 8 evidence-signal bottom/top fires
    (live_evidence_signal_dashboard_20260823.SIGNAL_ORDER), cluster-anchored GAP=12 bars per
    side (keep the first fire in any run of same-side fires closer than 12 bars apart).
  * At H=24 bars (120min) past the fire bar, measure the intrabar MFE with the move
    (continuation) and against the move (revert), both in ATR units, off the fire bar's close.
  * K = the median continuation/ATR ratio at this H, recomputed from data (not hardcoded) --
    the same K is applied to both classes so the split is close to 50/50 by construction.
  * EXTEND = continuation excursion >= K*ATR reached first. REVERT = the opposite-direction
    excursion >= K*ATR reached first. Events where NEITHER or BOTH cross K*ATR within the
    window are dropped (the "pure direction" filter that removes the volatility confound) --
    every event shown below is exactly one class.
  * A bottom fire's continuation direction is DOWN (the move that triggered it keeps going); a
    top fire's continuation direction is UP -- the mirror image of the reversal evidence-signal
    chips: EXTEND here is what those chips call a miss.

HOLDOUT (2026-04..08) is fully excluded (fire bar + H-bar resolution window both < HOLDOUT_START).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

for _p in ("/mnt/c/Windows/Fonts/malgun.ttf",):
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
        plt.rcParams["axes.unicode_minus"] = False
        break

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_trend_continuation_20260831"
START = pd.Timestamp("2024-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
H, GAP = 24, 12
N_PER_GROUP, SEED = 5, 20260831
BACK, FWD = 12, H + 6
KST = pd.Timedelta(hours=9)
TINT = {"EXTEND": "#EAF6EC", "REVERT": "#FBEDEC"}
CONT_COLOR, REV_COLOR = "#1E8449", "#C0392B"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close, high, low, h):
    fh = pd.Series(high).shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy()
    fl = pd.Series(low).shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy()
    return (fh - close) / close, (close - fl) / close


def draw(ax, sub, fire_pos, hit_pos, entry_px, cont_px, rev_px):
    lows, highs = [], []
    for i, (_, b) in enumerate(sub.iterrows()):
        col = "#2E86AB" if b["close"] >= b["open"] else "#C73E1D"
        ax.plot([i, i], [b["low"], b["high"]], color=col, linewidth=1.3, zorder=2)
        lo, hi = sorted([b["open"], b["close"]])
        ax.add_patch(Rectangle((i - 0.32, lo), 0.64, max(hi - lo, (b["high"] - b["low"]) * 0.03),
                               facecolor=col, edgecolor=col, zorder=3))
        lows.append(b["low"]); highs.append(b["high"])
    ax.axhline(entry_px, color="dimgray", linestyle="-.", linewidth=1.1, zorder=1)
    ax.axhline(cont_px, color=CONT_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(rev_px, color=REV_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(fire_pos, color="dimgray", linestyle=":", linewidth=1.2, zorder=1)
    if hit_pos is not None:
        ax.axvline(hit_pos, color="#7A0EBF", linestyle="-", linewidth=1.6, alpha=0.75, zorder=1)
    lo_, hi_ = min(lows + [cont_px, rev_px]), max(highs + [cont_px, rev_px])
    pad = (hi_ - lo_) * 0.08 or 1.0
    ax.set_ylim(lo_ - pad, hi_ + pad); ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    eth, btc = load("ETHUSDT"), load("BTCUSDT")
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)
    assert len(kl) == len(sig) == len(ind)
    assert (kl["timestamp"].to_numpy() == sig["timestamp"].to_numpy()).all()
    assert (kl["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()

    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        bot |= sig[f"bottom_{n}"].to_numpy(); top |= sig[f"top_{n}"].to_numpy()

    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i; rows.append((i, side == "bottom"))
    ev = pd.DataFrame(rows, columns=["i", "is_bottom"]).sort_values("i").reset_index(drop=True)

    ts = kl["timestamp"]; o, hi, lo, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr = ind["atr_pct"].to_numpy()
    up, dn = forward_extremes(c, hi, lo, H)

    iu = ev["i"].to_numpy(); isb = ev["is_bottom"].to_numpy()
    cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
    ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr[iu] > 0)
    k50 = float(np.median(cont[ok] / atr[iu][ok]))
    y_ext = ok & (cont >= k50 * atr[iu]); y_rev = ok & (rev >= k50 * atr[iu])
    pure = y_ext ^ y_rev

    holdout_pos = int((ts < HOLDOUT_START).sum())
    safe = pure & (iu >= BACK) & (iu + H < holdout_pos)

    tab = pd.DataFrame({
        "i": iu[safe], "side": np.where(isb[safe], "bottom", "top"),
        "cls": np.where(y_ext[safe], "EXTEND", "REVERT"),
        "entry_px": c[iu[safe]], "atr": atr[iu[safe]],
    })
    print(f"population after pure-direction filter (H={H}, K={k50:.3f}, GAP={GAP}, "
          f"pre-HOLDOUT only): n={len(tab)}")
    print(tab.groupby(["side", "cls"]).size().to_string())

    rng = np.random.default_rng(SEED)
    groups = [("bottom", "EXTEND"), ("bottom", "REVERT"), ("top", "EXTEND"), ("top", "REVERT")]
    picks = []
    for side, cls in groups:
        pool = tab[(tab["side"] == side) & (tab["cls"] == cls)]
        take = pool.iloc[rng.choice(len(pool), size=min(N_PER_GROUP, len(pool)), replace=False)]
        picks.append(take)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(4, 5, figsize=(40, 24), dpi=145)
    fig.suptitle(
        "ETH 5m — 추세지속(EXTEND) vs 반전(REVERT) 순수방향 라벨 20예시 | "
        "후보=8개 증거신호 발동(union), 클러스터앵커 GAP=12봉 | H=24봉(120분) 안에 지속방향/"
        f"반대방향 중 K×ATR(K={k50:.2f}, 데이터에서 재계산)을 먼저 넘는 쪽으로 판정, 둘 다/둘 다 "
        "아님인 사건은 제외(변동성 교란 제거) | 점선세로=발동봉  회색-.=발동봉 종가(기준가)  "
        "초록--=지속방향 K×ATR선  빨강--=반전방향 K×ATR선  보라|=먼저 닿은 시점 | "
        "1행=bottom발동/EXTEND(하락지속) 2행=bottom발동/REVERT(반등) 3행=top발동/EXTEND(상승지속) "
        "4행=top발동/REVERT(눌림) | HOLDOUT 미사용",
        fontsize=15.5, y=0.997)

    for r, (side, cls) in enumerate(groups):
        take = picks[r]
        for k, (_, t) in enumerate(take.iterrows()):
            ax = axes[r][k]
            i = int(t.i)
            s0, s1 = i - BACK, min(i + FWD, len(kl) - 1)
            sub = kl.iloc[s0:s1 + 1].reset_index(drop=True)
            down_side = t.side == "bottom"
            cont_px = t.entry_px * (1 - k50 * t.atr) if down_side else t.entry_px * (1 + k50 * t.atr)
            rev_px = t.entry_px * (1 + k50 * t.atr) if down_side else t.entry_px * (1 - k50 * t.atr)
            target_px = cont_px if t.cls == "EXTEND" else rev_px
            target_is_down = down_side if t.cls == "EXTEND" else not down_side
            hit_pos = None
            for j in range(i + 1, min(i + H, len(kl) - 1) + 1):
                touched = (lo[j] <= target_px) if target_is_down else (hi[j] >= target_px)
                if touched:
                    hit_pos = j - s0
                    break
            draw(ax, sub, i - s0, hit_pos, t.entry_px, cont_px, rev_px)
            ax.set_facecolor(TINT[t.cls])
            ticks = list(range(0, len(sub), 6))
            ax.set_xticks(ticks); ax.set_xticklabels([f"{(x-(i-s0))*5:+d}" for x in ticks], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_title(
                f"{t.cls} | {t.side}발동 | {(pd.Timestamp(ts.iloc[i])+KST):%Y-%m-%d %H:%M} KST\n"
                f"ATR={t.atr*100:.2f}% | K×ATR={k50*t.atr*100:.2f}%",
                fontsize=11)
            ax.grid(alpha=0.25)
    for r_ in axes[:, 0]:
        r_.set_ylabel("price", fontsize=12)
    for r_ in axes[-1]:
        r_.set_xlabel("발동봉 기준 분", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "extend_revert_label_20examples.png"
    fig.savefig(out); plt.close(fig)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
