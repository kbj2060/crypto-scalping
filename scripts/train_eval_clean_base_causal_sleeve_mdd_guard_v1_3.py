#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_plus_causal_conviction_sleeve_v1_1 as base  # noqa: E402


def _guard_grid() -> list[base.CausalSleeveConfig]:
    rows: dict[str, base.CausalSleeveConfig] = {}
    for same_thr in (0.0015, 0.0025, 0.0035):
        for frac in (0.10, 0.15, 0.25):
            for bars in (3, 6):
                for acct in (0.04, 0.06, 0.08):
                    for day in (0.006, 0.010, 0.015):
                        name = (
                            f"guard_same{same_thr:.4f}_frac{frac:.2f}_bars{bars}_"
                            f"acct{acct:.3f}_day{day:.3f}"
                        )
                        rows[name] = base.CausalSleeveConfig(
                            name=name,
                            same_threshold=float(same_thr),
                            hedge_threshold=0.0025,
                            max_sleeve_frac=float(frac),
                            max_sleeve_bars=int(bars),
                            same_enabled=True,
                            hedge_enabled=False,
                            account_dd_disable=float(acct),
                            daily_dd_disable=float(day),
                        )
    return list(rows.values())


def _guard_score(metrics: dict, cost3: dict) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    cost3_pnl = float(cost3["pnl"])
    sleeve_frac = float(metrics["sleeve_fraction"])
    score = pnl + 0.10 * cost3_pnl
    score -= 160.0 * max(0.0, abs(mdd) - 17.76)
    score -= 20.0 * max(0.0, sleeve_frac - 0.20)
    return float(score)


def main() -> int:
    base.DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_causal_sleeve_mdd_guard_v1_3"
    base.DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_2026.json"
    base.DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_grid.csv"
    base.DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_ledger.csv"
    base.DEFAULT_DOC = ROOT / "docs/experiments/clean_base_causal_sleeve_mdd_guard_v1_3.md"
    base._grid = _guard_grid
    base._score = _guard_score
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
