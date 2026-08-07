"""Pure label-quality report, NO modeling: for a given zigzag label (1h or 5m), if a trader had
PERFECT foresight of the label (traded every completed wave exactly as labeled), what would OOS
total return / stats / flip-rate look like? This measures the label's own economic ceiling and
continuity BEFORE asking any model to learn it.

OOS = 2026-01-01..2026-03-31 (Fresh-Forward convention). Cost applied ONCE per completed wave
segment (one round-trip trade per wave), not per bar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOS_START, OOS_END = "2026-01-01", "2026-04-01"
ROUND_TRIP_COST = 0.0010


def analyze(label_path: Path, tag: str) -> None:
    df = pd.read_parquet(label_path)
    oos = df[(df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)].reset_index(drop=True)

    print(f"\n{'='*20} {tag} (OOS {OOS_START}..{OOS_END}, n_bars={len(oos)}) {'='*20}")

    counts = oos["zigzag_action_name"].value_counts()
    print("class balance (bars):")
    print((counts / len(oos) * 100).round(1).astype(str) + "%")

    # one row per completed segment (segment fully inside OOS window)
    seg = oos[oos["zigzag_segment_id"] >= 0].groupby("zigzag_segment_id").agg(
        action=("zigzag_action_name", "first"),
        wave_return=("zigzag_wave_return", "first"),
        wave_bars=("zigzag_wave_bars", "first"),
        start_ts=("timestamp", "first"),
        end_ts=("timestamp", "last"),
    ).reset_index()

    net_ret = seg["wave_return"] - ROUND_TRIP_COST
    n_seg = len(seg)
    win_rate = (net_ret > 0).mean() if n_seg else float("nan")
    total_ret = net_ret.sum() * 100
    print(f"\ncompleted wave segments in OOS: {n_seg}")
    print(f"oracle total return (sum of net wave returns, 1 trade/wave, cost={ROUND_TRIP_COST*100:.2f}%/trade): {total_ret:.2f}%")
    print(f"oracle win rate: {win_rate:.4f}")
    print(f"mean net return/wave: {net_ret.mean()*100:.4f}%   median: {net_ret.median()*100:.4f}%")
    print(f"wave duration (bars) mean/median: {seg['wave_bars'].mean():.1f} / {seg['wave_bars'].median():.1f}")
    bar_unit_min = 60 if "1h" in tag else 5
    print(f"wave duration (hours) mean/median: {seg['wave_bars'].mean()*bar_unit_min/60:.2f} / {seg['wave_bars'].median()*bar_unit_min/60:.2f}")

    # flip rate: how often does the ACTIVE action change (LONG<->SHORT), ignoring CASH-only bars,
    # measured on the compressed segment sequence (this is the label's OWN choppiness, not any
    # model's prediction noise)
    active_actions = seg["action"].tolist()
    flips = sum(1 for a, b in zip(active_actions, active_actions[1:]) if a != b and "CASH" not in (a, b))
    print(f"direction flips between consecutive active waves: {flips} (out of {max(n_seg-1,0)} consecutive wave-pairs)")

    # bar-level flip rate on the raw per-bar action series (includes CASH transitions)
    raw_action = oos["zigzag_action_name"].to_numpy()
    bar_flips = sum(1 for a, b in zip(raw_action, raw_action[1:]) if a != b)
    print(f"bar-level label changes (any transition, incl. CASH buffer): {bar_flips} over {len(oos)-1} bar-pairs "
          f"({bar_flips/max(len(oos)-1,1)*100:.2f}% of bars are a transition)")


def main() -> int:
    analyze(ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet", "BTC 1h zigzag")
    analyze(ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet", "BTC 5m zigzag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
