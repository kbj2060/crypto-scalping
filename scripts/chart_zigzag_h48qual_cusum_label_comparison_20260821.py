#!/usr/bin/env python3
"""zigzag(zig075 direction)/h48qual(h48_conservative quality label)/cusum(dense-cashfill) 3개
라벨의 실제 LONG/SHORT 발동 위치를 같은 가격창에 나란히 그려 비교. 5-way 라벨로직 비교
(`docs/experiments/eth_tabm_label_logic_5way_comparison_20260820.md`)에서 이 3개만 OOS
상대적으로 양호(zigzag 2/3, h48qual 3/3, cusum 3/3)해 "좀 더 깊게 연구"할 후보로 남았다.

윈도우는 기존 라벨 차트 관례(`chart_h48qual_existing_quality_label_20260728.py`,
`chart_eth_triple_barrier_label_ground_truth_20260728.py`)와 동일한 2025-01-06..01-20 --
h48qual 패널은 그 스크립트를 그대로 재사용(경로만 현재 유저로 수정), 세 패널이 같은 가격구간을
공유해 직접 비교 가능.

각 패널이 charting하는 대상:
  - zigzag: zigzag_action_labels_20260531의 zigzag_action(=zig075 production direction 소스,
    5-way 비교에서 zigzag/h48qual 둘 다의 direction 입력)
  - h48qual: h48_conservative 자체의 독립 tb_action(quality 게이트 원본 라벨) -- 5-way
    학습에서 실제 쓰인 "zigzag direction ∩ h48 quality" 결합판이 아니라 h48_conservative
    단독 신호. 기존 chart_h48qual_existing_quality_label_20260728.py와 동일 대상.
  - cusum: eth_cusum_triple_barrier_labels_dense_cashfill_20260820의 zigzag_action(5-way
    학습에서 실제 쓰인 CUSUM dense-cashfill direction 라벨 그 자체)

가격선은 zigzag_action_labels_2025.csv의 close(세 패널 공통, 같은 기초자산가격)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"
OUT_PNG = ROOT / "tmp/research_20260821/chart_zigzag_h48qual_cusum_label_comparison.png"

ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
H48_RAW = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/train_triple_barrier_labels.csv"
CUSUM_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820"
H48_CFG = "h48_conservative"

COLOR_PRICE, COLOR_LONG, COLOR_SHORT = "#9AA5B1", "#2C6FBB", "#B5651D"
ACTION_NAME = {0: "CASH", 1: "LONG", 2: "SHORT"}


def _load_zigzag_style(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "close", "zigzag_action"], parse_dates=["timestamp"])
    return df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)].reset_index(drop=True)


def _load_h48_conservative(price: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", f"entry_timestamp_{H48_CFG}", f"tb_action_{H48_CFG}"]
    chunks = []
    for chunk in pd.read_csv(H48_RAW, usecols=cols, parse_dates=["timestamp", f"entry_timestamp_{H48_CFG}"], chunksize=500_000):
        sub = chunk[(chunk["timestamp"] >= WINDOW_START) & (chunk["timestamp"] <= WINDOW_END)]
        if len(sub):
            chunks.append(sub)
        if chunk["timestamp"].iloc[-1] > pd.Timestamp(WINDOW_END):
            break
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    df = df.drop(columns=["timestamp"]).rename(
        columns={f"entry_timestamp_{H48_CFG}": "timestamp", f"tb_action_{H48_CFG}": "zigzag_action"})
    df = df.merge(price[["timestamp", "close"]], on="timestamp", how="inner")
    return df[["timestamp", "close", "zigzag_action"]]


def _load_cusum_dense(price: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(CUSUM_DIR / "zigzag_action_labels_2025.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    df = df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    df = df.merge(price[["timestamp", "close"]], on="timestamp", how="left")
    return df[["timestamp", "close", "zigzag_action"]]


def _plot_panel(ax, price: pd.DataFrame, labels: pd.DataFrame, title: str) -> None:
    ax.plot(price["timestamp"], price["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close")
    long_t = labels[labels["zigzag_action"] == 1]
    short_t = labels[labels["zigzag_action"] == 2]
    n_long, n_short = len(long_t), len(short_t)
    ax.scatter(long_t["timestamp"], long_t["close"], marker="^", s=26, color=COLOR_LONG, alpha=0.75, zorder=3, linewidth=0, label=f"LONG (n={n_long})")
    ax.scatter(short_t["timestamp"], short_t["close"], marker="v", s=26, color=COLOR_SHORT, alpha=0.75, zorder=3, linewidth=0, label=f"SHORT (n={n_short})")
    ax.set_title(f"{title} -- LONG={n_long} SHORT={n_short}", fontsize=10.5, loc="left")
    ax.set_ylabel("ETH price (USDT)")
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, markerscale=1.5)


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    price = _load_zigzag_style(ZIGZAG_DIR / "zigzag_action_labels_2025.csv")
    print(f"price window rows: {len(price)} [{price['timestamp'].min()}..{price['timestamp'].max()}]", flush=True)

    zigzag_labels = price[["timestamp", "close", "zigzag_action"]]
    h48_labels = _load_h48_conservative(price)
    cusum_labels = _load_cusum_dense(price)

    for name, df in [("zigzag", zigzag_labels), ("h48qual(h48_conservative)", h48_labels), ("cusum", cusum_labels)]:
        vc = df["zigzag_action"].map(ACTION_NAME).value_counts()
        print(f"{name}: {vc.to_dict()}", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 15), dpi=150, sharex=True)
    _plot_panel(axes[0], price, zigzag_labels, "zigzag (zig075 production direction, zigzag_action_labels_20260531)")
    _plot_panel(axes[1], price, h48_labels, "h48qual (h48_conservative quality label, own independent tb_action)")
    _plot_panel(axes[2], price, cusum_labels, "cusum (CUSUM+TB dense-cashfill, used in 5-way TabM training)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(f"Label comparison: zigzag vs h48qual vs cusum -- TRAIN {WINDOW_START}..{WINDOW_END}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
