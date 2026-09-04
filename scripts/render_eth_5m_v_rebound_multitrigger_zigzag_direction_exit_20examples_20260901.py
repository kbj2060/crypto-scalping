#!/usr/bin/env python3
"""ZDC exit(두 번째 피벗) 라벨 육안검증 -- HIT 이벤트 중 20건, entry+exit 둘 다 표시.

entry(주황 실선=confirm_idx)까지는 기존 wick-앵커 렌더 스크립트와 동일. 여기에 exit(두 번째
피벗 확정봉=exit_confirm_idx)을 청록 실선으로 추가 표시 -- "진입 방향으로 확정된 움직임이
어디서 끝나는지" 확인이 목적이므로 HIT(라벨=1) 이벤트만 샘플링한다(MISS는애초에 확정된
스윙이 없어 exit 개념이 성립하지 않음).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

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
ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/eth_5m_v_rebound_multitrigger_zigzag_direction_labels_with_exit.csv"
OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/exit_label_20examples.png"
LOOKBACK_BARS = 8
LOOKFORWARD_PAD_BARS = 8
N_SAMPLES = 20
SEED = 20260901


def draw_candles(ax, sub: pd.DataFrame, event_pos: int, event_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    print("[1/2] klines+라벨(exit포함) 로딩...", flush=True)
    df = pd.read_csv(ETH_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    low_arr, high_arr = df["low"].to_numpy(), df["high"].to_numpy()

    labels = pd.read_csv(LABEL_CSV)
    labels["hit_bool"] = labels["hit"].astype(str).map({"True": True, "False": False})
    pool = labels[(labels["hit_bool"] == True) & labels["exit_confirm_idx"].notna()].reset_index(drop=True)  # noqa: E712
    print(f"  HIT+exit해상 pool n={len(pool)}", flush=True)

    rng = np.random.default_rng(SEED)
    sample = pool.iloc[rng.choice(len(pool), size=min(N_SAMPLES, len(pool)), replace=False)]

    print("[2/2] 차트 렌더링...", flush=True)
    n_cols, n_rows = 5, 4
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.4 * n_rows), dpi=145)
    fig.suptitle(
        "ZDC exit(두 번째 피벗) 육안검증 -- HIT 이벤트 20건 무작위(seed=20260901), entry+exit 둘 다 표시\n"
        "회색 점선=트리거idx, 회색 가로선=wick 앵커. 굵은 주황선=entry 확정봉(confirm_idx, 첫 피벗). "
        "굵은 청록선=exit 확정봉(exit_confirm_idx, 두 번째 피벗=확정된 스윙이 반대로 꺾이는 지점).\n"
        "확인할 것: entry 이후 가격이 트리거 방향으로 실제 진행하다가, exit 선 부근에서 반대 방향 임계치 반전이 실제로 보이는지.",
        fontsize=13, y=0.999,
    )

    for i, (_, event) in enumerate(sample.iterrows()):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        idx = int(event["idx"])
        confirm_idx = int(event["confirm_idx"])
        exit_confirm_idx = int(event["exit_confirm_idx"])
        is_bottom = event["direction"] == "upside"
        anchor = float(low_arr[idx]) if is_bottom else float(high_arr[idx])
        start = max(0, idx - LOOKBACK_BARS)
        end = min(len(df), exit_confirm_idx + LOOKFORWARD_PAD_BARS + 1)
        sub = df.iloc[start:end].reset_index(drop=True)
        event_pos = idx - start
        confirm_pos = confirm_idx - start
        exit_pos = exit_confirm_idx - start
        draw_candles(ax, sub, event_pos, anchor)
        ax.axvline(confirm_pos, color="#E8871E", linestyle="-", linewidth=2.0, zorder=2)
        ax.axvline(exit_pos, color="#0D9488", linestyle="-", linewidth=2.2, zorder=2)
        ax.set_facecolor("#eef8f0")
        ticks = list(range(0, len(sub), max(1, len(sub) // 8)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - event_pos) * 5:+d}" for t in ticks], fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        dir_kr = "bottom(상승기대)" if is_bottom else "top(하락기대)"
        bars_to_confirm = confirm_idx - idx
        bars_entry_to_exit = exit_confirm_idx - confirm_idx
        title = (f"{dir_kr} | entry={bars_to_confirm}봉 exit={bars_entry_to_exit}봉({bars_entry_to_exit*5}분)\n"
                 f"첫피벗={event['pivot_type']} exit피벗={event['exit_pivot_type']}")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH} ({len(sample)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
