#!/usr/bin/env python3
"""Visual verification (docs/homer/README.md §2 item 6 / the standing user gate) for the
TREND-CONTINUATION trade, before the one-shot HOLDOUT is spent.

What is being verified here is NOT a label -- it is the actual trade. Each panel is one real
trade from core.causal_futures_backtest at the confirmed cell (SL=3.5 / ARM=0.5 / Trail=0.1,
H=24), drawn on the raw 5m candles:

  * dotted vertical  = the FIRE bar (evidence-signal trigger; decision uses info through this bar)
  * gray dash-dot    = ENTRY, taken at the NEXT bar's open (the engine's causal convention)
  * red dashed       = initial stop, 3.5xATR AGAINST the traded direction
  * green dotted     = arm level, 0.5xATR IN FAVOR -- once touched, the stop starts trailing 0.1xATR
  * solid vertical   = EXIT bar

Direction convention (the whole point of this experiment): the trade goes WITH the move, i.e.
the opposite of what the evidence-signal chip implies. A BOTTOM fire (price just made a low ->
the chip says "bounce") is traded SHORT. A TOP fire is traded LONG.

10 winners + 10 losers, sampled from VAL+OOS only (HOLDOUT 2026-04..08 untouched).
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

# WSL has no Linux Korean font; borrow the Windows one so the Korean annotations render.
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_trend_continuation_20260831"
START = pd.Timestamp("2024-01-01")
VAL_START, OOS_START, HOLDOUT_START = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
H, SL, ARM, TRAIL, GAP = 24, 3.5, 0.5, 0.1, 12
N_PER_CLASS, SEED = 10, 20260831
BACK, FWD = 12, 30
KST = pd.Timedelta(hours=9)
TINT = {"WIN": "#EAF6EC", "LOSS": "#FBEDEC"}


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def draw(ax, sub, fire_i, entry_i, exit_i, entry_px, sl_px, arm_px):
    lows, highs = [], []
    for i, (_, b) in enumerate(sub.iterrows()):
        col = "#2E86AB" if b["close"] >= b["open"] else "#C73E1D"
        ax.plot([i, i], [b["low"], b["high"]], color=col, linewidth=1.3, zorder=2)
        lo, hi = sorted([b["open"], b["close"]])
        ax.add_patch(Rectangle((i - 0.32, lo), 0.64, max(hi - lo, (b["high"] - b["low"]) * 0.03),
                               facecolor=col, edgecolor=col, zorder=3))
        lows.append(b["low"]); highs.append(b["high"])
    ax.axhline(entry_px, color="dimgray", linestyle="-.", linewidth=1.1, zorder=1)
    ax.axhline(sl_px, color="#C0392B", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(arm_px, color="#1E8449", linestyle=":", linewidth=1.2, zorder=1)
    ax.axvline(fire_i, color="dimgray", linestyle=":", linewidth=1.2, zorder=1)
    ax.axvline(exit_i, color="#7A0EBF", linestyle="-", linewidth=1.6, alpha=0.75, zorder=1)
    lo_, hi_ = min(lows + [sl_px]), max(highs + [sl_px])
    pad = (hi_ - lo_) * 0.08 or 1.0
    ax.set_ylim(lo_ - pad, hi_ + pad); ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    eth, btc = load("ETHUSDT"), load("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)

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
            last = i; rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)

    ts = kl["timestamp"]
    o, hi, lo, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr_pct = ind["atr_pct"].to_numpy()
    dec = ev["pos"].to_numpy(np.int64)
    scores = np.where(ev["side"].to_numpy() == "bottom", -1.0, 1.0)   # WITH the move

    ledgers = []
    for s, e in ((VAL_START, OOS_START), (OOS_START, HOLDOUT_START)):
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        m = np.array([d in el for d in dec])
        a = atr_pct[dec][m]
        r = simulate_single_position(
            timestamps=ts, open_px=o, high=hi, low=lo, close=c, decision_indices=dec[m],
            scores=scores[m], tp_moves=np.full(int(m.sum()), 999.0), sl_moves=SL * a,
            upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H, margin_fraction=0.30,
            leverage=3.0, roundtrip_cost_rate=0.001, arm_moves=ARM * a, trail_moves=TRAIL * a)
        ledgers.append(r.ledger)
    led = pd.concat(ledgers, ignore_index=True)
    pos_of = pd.Series(np.arange(len(kl)), index=ts.to_numpy())
    led["fire_i"] = led["decision_timestamp"].map(pos_of)
    led["entry_i"] = led["entry_timestamp"].map(pos_of)
    led["exit_i"] = led["exit_timestamp"].map(pos_of)
    led["atr"] = atr_pct[led["fire_i"].to_numpy()]
    led["entry_px"] = o[led["entry_i"].to_numpy()]
    led["fired"] = [", ".join(n for n in names
                              if sig[f"{'bottom' if s < 0 else 'top'}_{n}"].iloc[int(i)])
                    for i, s in zip(led["fire_i"], led["score"])]
    led["fire_side"] = np.where(led["score"] < 0, "bottom", "top")
    led["cls"] = np.where(led["trade_return"] > 0, "WIN", "LOSS")
    led = led[(led["fire_i"] >= BACK) & (led["exit_i"] < len(kl) - 2)]
    print(f"trades VAL+OOS: {len(led)}  WIN={int((led.cls=='WIN').sum())} "
          f"LOSS={int((led.cls=='LOSS').sum())}  mean={led.trade_return.mean()*1e4:+.2f}bp")

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(4, 5, figsize=(40, 24), dpi=145)
    fig.suptitle(
        "ETH 5m — 추세 지속(CONTINUATION) 트레이드 20예시 | 증거신호 발동 시 움직임을 '따라' 진입"
        "(bottom 발동→SHORT, top 발동→LONG, 칩이 가리키는 반대) | "
        f"SL={SL}×ATR / ARM={ARM}×ATR / Trail={TRAIL}×ATR, 최대 {H}봉({H*5}분), 10bp 비용 | "
        "점선세로=발동봉  회색=진입(다음봉 시가)  빨강--=초기손절  초록:=트레일링 개시선  보라|=청산 | "
        "위 2행=수익, 아래 2행=손실 | VAL+OOS only (HOLDOUT 미사용)",
        fontsize=19, y=0.995)
    rng = np.random.default_rng(SEED)
    for cls, off in (("WIN", 0), ("LOSS", 2)):
        pool = led[led["cls"] == cls]
        take = pool.iloc[rng.choice(len(pool), size=min(N_PER_CLASS, len(pool)), replace=False)]
        for k, (_, t) in enumerate(take.iterrows()):
            ax = axes[off + k // 5][k % 5]
            f, en, ex = int(t.fire_i), int(t.entry_i), int(t.exit_i)
            s0, s1 = f - BACK, min(f + FWD, len(kl) - 1)
            sub = kl.iloc[s0:s1 + 1].reset_index(drop=True)
            short = t.score < 0
            sl_px = t.entry_px * (1 + SL * t.atr) if short else t.entry_px * (1 - SL * t.atr)
            arm_px = t.entry_px * (1 - ARM * t.atr) if short else t.entry_px * (1 + ARM * t.atr)
            draw(ax, sub, f - s0, en - s0, ex - s0, t.entry_px, sl_px, arm_px)
            ax.set_facecolor(TINT[cls])
            ticks = list(range(0, len(sub), 6))
            ax.set_xticks(ticks); ax.set_xticklabels([f"{(x-(f-s0))*5:+d}" for x in ticks], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            fired = t.fired if len(t.fired) <= 46 else t.fired[:44] + ".."
            ax.set_title(
                f"{cls} | {'SHORT' if short else 'LONG'} ({t.fire_side} 발동) | "
                f"{(pd.Timestamp(t.decision_timestamp)+KST):%Y-%m-%d %H:%M} KST\n"
                f"{t.trade_return*1e4:+.0f}bp | {t.reason} | {int(t.bars_held)}봉 보유 | ATR={t.atr*100:.2f}%\n"
                f"{fired}", fontsize=11)
            ax.grid(alpha=0.25)
    for r_ in axes[:, 0]:
        r_.set_ylabel("price", fontsize=12)
    for r_ in axes[-1]:
        r_.set_xlabel("발동봉 기준 분", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "continuation_trade_examples.png"
    fig.savefig(out); plt.close(fig)
    print(f"saved -> {out}")
    print("\nexit reason mix (all VAL+OOS trades):")
    print(led.groupby(["cls", "reason"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
