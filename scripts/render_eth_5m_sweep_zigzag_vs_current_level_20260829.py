#!/usr/bin/env python3
"""Chart illustrating the zigzag-vs-current-level anchor comparison (research_eth_sweep_zigzag_
anchor_comparison_20260829.py). Real historical event: current 48-bar rolling-max level treats a
stale 4h-old high as "the" level, while a fully causal confirmed zigzag pivot (must reverse by
>=1.0-1.8% before confirming, verbatim algorithm from build_zigzag_action_labels_v2_20260604.py)
finds a much more recent, tighter high just 8 bars (40min) before the sweep bar -- the kind of
"stale level -> late-looking trigger" case the user flagged from eyeballing a 1h chart.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_PATH = ROOT / "tmp/eth_liquidity_sweep_frequency_20260829/zigzag_vs_current_level.png"
KST_OFFSET = pd.Timedelta(hours=9)

EVENT_TS = "2025-02-26T22:00:00+00:00"
ZIGZAG_LEVEL = 2343.36
ZIGZAG_PIVOT_BARS_AGO = 8
CURRENT_LEVEL = 2381.93


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_zzchart_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)
    event = labels[labels["timestamp"] == EVENT_TS].iloc[0]
    idx = int(event["candidate_index"])

    lookback, lookforward = 20, 4
    sub = frame.iloc[idx - lookback: idx + lookforward + 1].reset_index(drop=True)
    sweep_pos = lookback

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(26, 12), dpi=150)

    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=2.4, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.34, body_low), 0.68, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])

    zigzag_pos = sweep_pos - ZIGZAG_PIVOT_BARS_AGO
    ax.axhline(CURRENT_LEVEL, color="dimgray", linestyle="--", linewidth=2.2, zorder=1)
    ax.text(-0.5, CURRENT_LEVEL, f"현재 정의(48봉/4시간 롤링최고) 레벨: {CURRENT_LEVEL:.2f}",
            va="bottom", ha="left", fontsize=15, color="dimgray", fontweight="bold")
    ax.axhline(ZIGZAG_LEVEL, color="#8E44AD", linestyle="--", linewidth=2.2, zorder=1)
    ax.text(-0.5, ZIGZAG_LEVEL, f"zigzag 확정 고점(causal): {ZIGZAG_LEVEL:.2f} ({ZIGZAG_PIVOT_BARS_AGO}봉={ZIGZAG_PIVOT_BARS_AGO*5}분 전)",
            va="top", ha="left", fontsize=15, color="#8E44AD", fontweight="bold")
    ax.axvline(zigzag_pos, color="#8E44AD", linestyle=":", linewidth=1.8, zorder=1)
    ax.axvline(sweep_pos, color="black", linestyle=":", linewidth=1.8, zorder=1)
    ax.text(sweep_pos, max(highs), "스윕봉(0분)", ha="center", va="bottom", fontsize=14)
    ax.text(zigzag_pos, max(highs), "zigzag 고점\n확정 지점", ha="center", va="bottom", fontsize=13, color="#8E44AD")

    pad = (max(highs) - min(lows)) * 0.15 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad * 1.3)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.set_xticks(range(0, len(sub), 2))
    ax.set_xticklabels([f"{(i - sweep_pos) * 5:+d}분" for i in range(0, len(sub), 2)], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(alpha=0.25)
    ax.set_ylabel("가격(USDT)", fontsize=15)
    ax.set_xlabel("스윕봉 기준 경과 시간", fontsize=15)

    kst_ts = pd.Timestamp(EVENT_TS) + KST_OFFSET
    ax.set_title(
        f"{kst_ts:%Y-%m-%d %H:%M} KST (upside 스윕) — 현재 정의가 쓰는 레벨(4시간 전 고점)보다\n"
        f"zigzag로 확정한 진짜 최근 고점(40분 전)이 훨씬 더 가깝고 관련성 높음",
        fontsize=19, pad=16,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
