#!/usr/bin/env python3
"""Supplementary visual check for JUST the 2 newly-added triggers (demarker_extreme,
kalman_deviation_meanrev) -- the original 7-trigger chart (multitrigger_v_rebound_21examples.png)
was already reviewed and approved; this only covers what's new since then. 5 examples each (more
than the standard 3, since kalman_deviation_meanrev's v7b-formula hit rate (9.5%) is the lowest of
all 9 triggers and specifically warrants a closer look) at giveback percentile 10/30/50/70/90.

Reuses draw_candles/pick_per_trigger from render_eth_5m_v_rebound_multitrigger_20examples_20260831.py
verbatim (imported, not copy-pasted).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/kbj20/crypto-scalping")
RENDER_SCRIPT = ROOT / "scripts/render_eth_5m_v_rebound_multitrigger_20examples_20260831.py"
_spec = importlib.util.spec_from_file_location("render_multitrigger_20260831", RENDER_SCRIPT)
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
draw_candles = _render.draw_candles

ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABELS_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/kalman_demarker_supplement_10examples.png"
WINDOW_BARS = 12

TRIGGERS = ["demarker_extreme", "kalman_deviation_meanrev"]
TRIGGER_LABEL_KR = {
    "demarker_extreme": "demarker_extreme (신규추가, 호메로스 후보풀·TabPFN미검증)",
    "kalman_deviation_meanrev": "kalman_deviation_meanrev (신규추가, 호메로스 후보풀·TabPFN미검증, v7b적중률9.5%최저)",
}


def pick_per_trigger(labels: pd.DataFrame, trigger: str, n: int = 5) -> pd.DataFrame:
    pool = labels[(labels["triggers"].str.contains(trigger)) & (labels["outcome"] == "V자반등")]
    pool = pool.sort_values("giveback_ratio").reset_index(drop=True)
    if len(pool) <= n:
        return pool
    pcts = [0.10, 0.30, 0.50, 0.70, 0.90][:n]
    idxs = sorted({min(int(p * (len(pool) - 1)), len(pool) - 1) for p in pcts})
    return pool.iloc[idxs]


def main() -> int:
    frame = pd.read_csv(ETH_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    labels = pd.read_csv(LABELS_CSV)

    n_cols, n_rows = 5, len(TRIGGERS)
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.4 * n_rows), dpi=145)
    fig.suptitle(
        "신규 추가 2개 트리거 보충 육안검증 (트리거별 5건, giveback percentile 10/30/50/70/90) — "
        "라벨식은 v7b 그대로, 원시트리거 정의만 research_eth_kalman_demarker_gridscreen_20260831.py에서 재사용",
        fontsize=16, y=0.998,
    )

    for r, trig in enumerate(TRIGGERS):
        picked = pick_per_trigger(labels, trig, n=5)
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(picked):
                ax.axis("off")
                continue
            event = picked.iloc[c]
            idx = int(event["idx"])
            is_down = event["direction"] == "downside"
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            level = float(frame["low"].iloc[idx]) if is_down else float(frame["high"].iloc[idx])
            draw_candles(ax, sub, WINDOW_BARS, level)
            ax.set_facecolor("#eef8f0")
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            dir_kr = "상승반등" if is_down else "하락반전"
            title = (f"{TRIGGER_LABEL_KR[trig] if c == 0 else trig}\n"
                     f"{dir_kr} | fast={event['fast_move_atr_mult']:.2f}x | giveback={event['giveback_ratio']:.3f}\n"
                     f"other_triggers={event['triggers']}")
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
