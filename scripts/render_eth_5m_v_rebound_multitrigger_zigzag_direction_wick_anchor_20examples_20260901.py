#!/usr/bin/env python3
"""ZDC(wick-앵커) 저장된 라벨 파일 기반 육안검증 -- HIT 10건 + MISS 10건, 한 이미지.

종가앵커판 렌더 스크립트(render_eth_5m_v_rebound_multitrigger_zigzag_direction_20examples_
20260901.py)와 동일한 레이아웃/색상 관례를 재사용하되, 이번엔 매번 ZDC를 재계산하지 않고
build_eth_5m_v_rebound_multitrigger_zigzag_direction_labels_20260901.py가 이미 저장해둔
라벨 CSV(pivot_type/extreme_idx/confirm_idx/hit)를 그대로 읽는다 -- TabPFN 학습에 실제로
들어갈 그 라벨 파일 자체를 검증하는 것이 목적.
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
LABEL_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/eth_5m_v_rebound_multitrigger_zigzag_direction_labels.csv"
OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/wick_anchor_hit_miss_20examples.png"
LOOKBACK_BARS = 12
LOOKFORWARD_PAD_BARS = 8
N_PER_CLASS = 10
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
    print("[1/2] klines+라벨 로딩...", flush=True)
    df = pd.read_csv(ETH_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    low_arr, high_arr = df["low"].to_numpy(), df["high"].to_numpy()

    labels = pd.read_csv(LABEL_CSV)
    resolved = labels[labels["hit"].isin([True, False]) | labels["hit"].isin(["True", "False"])].copy()
    resolved["hit"] = resolved["hit"].astype(str).map({"True": True, "False": False}) if resolved["hit"].dtype == object else resolved["hit"]
    print(f"  resolved n={len(resolved)}, hit={int(resolved['hit'].sum())}, miss={int((~resolved['hit']).sum())}", flush=True)

    rng = np.random.default_rng(SEED)
    hit_pool = resolved[resolved["hit"]].reset_index(drop=True)
    miss_pool = resolved[~resolved["hit"]].reset_index(drop=True)
    hit_sample = hit_pool.iloc[rng.choice(len(hit_pool), size=min(N_PER_CLASS, len(hit_pool)), replace=False)]
    miss_sample = miss_pool.iloc[rng.choice(len(miss_pool), size=min(N_PER_CLASS, len(miss_pool)), replace=False)]
    samples = [("HIT(라벨=1)", r) for _, r in hit_sample.iterrows()] + [("MISS(라벨=0)", r) for _, r in miss_sample.iterrows()]

    print("[2/2] 차트 렌더링...", flush=True)
    n_cols, n_rows = 5, 4
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.4 * n_rows), dpi=145)
    fig.suptitle(
        "ZDC wick-앵커 라벨 육안검증(저장된 라벨 파일 기준, TabPFN 학습에 실제로 쓰일 그 파일) -- "
        "HIT 10건(위 2줄)/MISS 10건(아래 2줄), 무작위 샘플(seed=20260901)\n"
        "회색 점선=트리거 idx, 회색 가로선=wick 앵커(bottom→low[idx], top→high[idx], giveback과 동일 관례). "
        "굵은 주황선=실제 확정봉. 가는 보라 점선=피벗 기록봉(idx와 다를 때만 표시). "
        "Step A wick-앵커 raw-lift: VAL/OOS 4칸 lift 1.01~1.05x — TabPFN이 후보군 안에서 더 가릴 수 있는지 확인 중.",
        fontsize=14, y=0.999,
    )

    for i, (group_label, event) in enumerate(samples):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        idx = int(event["idx"])
        extreme_idx = int(event["extreme_idx"])
        confirm_idx = int(event["confirm_idx"])
        is_bottom = event["direction"] == "upside"
        anchor = float(low_arr[idx]) if is_bottom else float(high_arr[idx])
        start = max(0, idx - LOOKBACK_BARS)
        end = min(len(df), max(confirm_idx + LOOKFORWARD_PAD_BARS + 1, idx + 1))
        sub = df.iloc[start:end].reset_index(drop=True)
        event_pos = idx - start
        confirm_pos = confirm_idx - start
        extreme_pos = extreme_idx - start
        draw_candles(ax, sub, event_pos, anchor)
        ax.axvline(confirm_pos, color="#E8871E", linestyle="-", linewidth=2.0, zorder=2)
        if extreme_idx != idx:
            ax.axvline(extreme_pos, color="#7B2FBE", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_facecolor("#eef8f0" if event["hit"] else "#fdeeee")
        ticks = list(range(0, len(sub), max(1, len(sub) // 8)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - event_pos) * 5:+d}" for t in ticks], fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        dir_kr = "bottom(상승기대)" if is_bottom else "top(하락기대)"
        bars_to_confirm = confirm_idx - idx
        title = (f"{group_label if c == 0 else ''}\n"
                 f"{dir_kr} | 첫피벗={event['pivot_type']} | 확정={bars_to_confirm}봉({bars_to_confirm*5}분)\n"
                 f"triggers={event['triggers']}")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH} ({len(samples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
