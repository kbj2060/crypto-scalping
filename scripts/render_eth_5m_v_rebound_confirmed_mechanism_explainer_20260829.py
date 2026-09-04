#!/usr/bin/env python3
"""User asked for a chart explaining the "confirmed" (sustained-close) condition in the
V_REBOUND label: why does reaching the 1.5x ATR magnitude target NOT guarantee label=1 -- one
bar closing back past the swept level within the 6-bar window fails it, even after a big move.

Two real historical events side by side (not synthetic examples): one where the magnitude
target was cleared by a wide margin (2.55x ATR) but label=0 because bar 5 rolled over and closed
back below the level (and stayed there in bar 6); one clean label=1 where the move (2.06x ATR)
held above the level on every one of the 6 bars' closes. Both downside sweeps, chosen from the
real label population (research_eth_sweep_v_rebound... inspection, 2026-08-29) for a clean,
unambiguous illustration -- not cherry-picked for a misleading effect, just for legibility.
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
OUT_PATH = ROOT / "tmp/eth_liquidity_sweep_frequency_20260829/v_rebound_confirmed_mechanism_explainer.png"
KST_OFFSET = pd.Timedelta(hours=9)

EXAMPLES = [
    {"timestamp": "2025-06-05T06:15:00+00:00", "tag": "reb_atr 2.55x, but label = 0 (NO_V_REBOUND)",
     "note": "봉5(25분 후)에서 다시 레벨 아래로 종가 마감 -> confirmed 깨짐"},
    {"timestamp": "2026-02-05T15:30:00+00:00", "tag": "reb_atr 2.06x, and label = 1 (V_REBOUND)",
     "note": "6봉 내내 종가가 레벨 위 유지 -> confirmed 통과"},
]


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_explainer_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_panel(ax, sub: pd.DataFrame, level: float, sweep_pos: int) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=2.2, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.34, body_low), 0.68, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])
        # explicit close-price marker, colored by pass/fail of the sustain check (skip the sweep
        # bar itself at i==sweep_pos -- the check only applies to the 6 FUTURE bars). Label is
        # anchored to this bar's own high/low (not the close point itself) so it always lands in
        # clear space above/below that bar's wick instead of overlapping a neighboring candle.
        if i > sweep_pos:
            ok = bar["close"] > level
            color = "#1B9E4B" if ok else "#D62728"
            ax.scatter([i], [bar["close"]], s=220, zorder=5, facecolor=color, edgecolor="black",
                       linewidth=1.5, marker=("o" if ok else "X"))
            anchor_y = bar["high"] if ok else bar["low"]
            va = "bottom" if ok else "top"
            offset = 12 if ok else -12
            ax.annotate(f"{'종가>레벨' if ok else '종가<레벨!'} {bar['close']:.1f}",
                        (i, anchor_y), textcoords="offset points", xytext=(0, offset),
                        ha="center", va=va, fontsize=12.5, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

    ax.axhline(level, color="dimgray", linestyle="--", linewidth=2.0, zorder=1)
    ax.text(len(sub) - 0.5, level, f"스윕된 레벨 {level:.1f}", va="bottom", ha="right",
            fontsize=14, color="dimgray", fontweight="bold")
    ax.axvline(sweep_pos, color="black", linestyle=":", linewidth=1.6, zorder=1)
    ax.text(sweep_pos, max(highs), "스윕봉(0분)", ha="center", va="bottom", fontsize=13)
    pad = (max(highs) - min(lows)) * 0.15 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad * 1.6)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([f"{(i - sweep_pos) * 5:+d}분" for i in range(len(sub))], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(alpha=0.25)


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(32, 14), dpi=150)
    fig.suptitle(
        "'유지' 조건 설명: 30분 반등폭(1.5x ATR)을 넘겨도, 6봉 중 단 하나라도 종가가\n"
        "레벨을 다시 넘으면(왼쪽) 실패, 6봉 전부 종가가 레벨을 지키면(오른쪽) 성공",
        fontsize=22, y=1.02,
    )

    for ax, ex in zip(axes, EXAMPLES):
        event = labels[labels["timestamp"] == ex["timestamp"]].iloc[0]
        idx = int(event["candidate_index"])
        level = float(event["sweep_level"])
        sub = frame.iloc[idx - 2: idx + 7].reset_index(drop=True)
        sweep_pos = 2  # 2 bars of pre-sweep context, then the sweep bar itself
        draw_panel(ax, sub, level, sweep_pos)
        kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
        ax.set_title(f"{kst_ts:%Y-%m-%d %H:%M} KST (downside 스윕)\n{ex['tag']}\n{ex['note']}",
                     fontsize=17, pad=14)
        ax.set_ylabel("가격(USDT)", fontsize=14)
        ax.set_xlabel("스윕봉 기준 경과 시간", fontsize=14)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
