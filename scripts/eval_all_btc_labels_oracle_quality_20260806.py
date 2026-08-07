"""Pure label-quality report (no modeling) for EVERY label design built this session, not just
zigzag: vol-regime (direction-agnostic) and pivot-transition (Layer A, binary gate) as well.
These two are NOT standalone tradeable action labels (no direction), so "oracle total return"
isn't well-defined for them the way it is for zigzag -- reported honestly as such, with the stats
that ARE meaningful for a gate/filter label (class balance, persistence, flip/transition rate).

OOS = 2026-01-01..2026-03-31 (Fresh-Forward convention).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOS_START, OOS_END = "2026-01-01", "2026-04-01"


def analyze_volregime(path: Path, tag: str) -> None:
    df = pd.read_parquet(path)
    oos = df[(df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)].dropna(subset=["label_3class"]).reset_index(drop=True)
    print(f"\n{'='*20} {tag} (NOT directional -- gate/filter label only) {'='*20}")
    print(f"n_bars={len(oos)}")
    counts = oos["label_3class"].value_counts(normalize=True).sort_index() * 100
    print("class balance: " + ", ".join(f"{k:+.0f}={v:.1f}%" for k, v in counts.items()))
    action = oos["label_3class"].to_numpy()
    flips = sum(1 for a, b in zip(action, action[1:]) if a != b)
    print(f"bar-level label changes: {flips}/{len(oos)-1} ({flips/max(len(oos)-1,1)*100:.2f}% of bars)")
    # regime run-length (persistence): consecutive-same-label streak lengths
    runs, cur = [], 1
    for a, b in zip(action, action[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    import numpy as np
    print(f"regime run-length (bars) mean/median: {np.mean(runs):.1f} / {np.median(runs):.1f}")
    print("NOTE: no direction -> oracle 'total return' not computable without an external direction rule.")


def analyze_pivot(path: Path, tag: str, bar_unit_min: int) -> None:
    df = pd.read_parquet(path)
    oos = df[(df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)].dropna(subset=["transition_soon"]).reset_index(drop=True)
    print(f"\n{'='*20} {tag} (NOT directional -- binary transition gate only) {'='*20}")
    print(f"n_bars={len(oos)}")
    pos_rate = oos["transition_soon"].mean() * 100
    print(f"positive (transition-soon) rate: {pos_rate:.2f}%")
    n_pivots = (oos["is_pivot"] == 1).sum()
    print(f"actual pivots in OOS: {n_pivots} (H={(oos['pivot_type']=='H').sum()}, L={(oos['pivot_type']=='L').sum()})")
    avg_gap_bars = len(oos) / max(n_pivots, 1)
    print(f"avg bars between pivots: {avg_gap_bars:.1f} ({avg_gap_bars*bar_unit_min/60:.2f}h)")
    action = oos["transition_soon"].to_numpy()
    flips = sum(1 for a, b in zip(action, action[1:]) if a != b)
    print(f"bar-level transition_soon flips: {flips}/{len(oos)-1} ({flips/max(len(oos)-1,1)*100:.2f}% of bars)")
    print("NOTE: no direction -> oracle 'total return' not computable standalone (see combined-with-zigzag backtests).")


def main() -> int:
    analyze_volregime(ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet", "BTC 1h vol-regime")
    analyze_pivot(ROOT / "data/splits/year_oos/btc_1h_pivot_transition_labels_20260805.parquet", "BTC 1h pivot-transition (Layer A)", 60)
    analyze_pivot(ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet", "BTC 5m pivot-transition (Layer A)", 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
