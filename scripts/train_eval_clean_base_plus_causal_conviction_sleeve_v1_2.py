#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_plus_causal_conviction_sleeve_v1_1 as base  # noqa: E402


def _grid_v1_2() -> list[base.CausalSleeveConfig]:
    rows: dict[str, base.CausalSleeveConfig] = {}
    for same_thr in (0.0015, 0.0025, 0.0035):
        for hedge_thr in (0.0025,):
            for frac in (0.15, 0.25, 0.35):
                for bars in (3, 6):
                    for same_enabled in (True,):
                        for hedge_enabled in (False,):
                            for acct in (0.08,):
                                for day in (0.015,):
                                    name = (
                                        f"v12_same{same_thr:.4f}_hedge{hedge_thr:.4f}_frac{frac:.2f}_"
                                        f"bars{bars}_same{int(same_enabled)}_hedge{int(hedge_enabled)}_"
                                        f"acct{acct:.3f}_day{day:.3f}"
                                    )
                                    rows[name] = base.CausalSleeveConfig(
                                        name=name,
                                        same_threshold=float(same_thr),
                                        hedge_threshold=float(hedge_thr),
                                        max_sleeve_frac=float(frac),
                                        max_sleeve_bars=int(bars),
                                        same_enabled=bool(same_enabled),
                                        hedge_enabled=bool(hedge_enabled),
                                        account_dd_disable=float(acct),
                                        daily_dd_disable=float(day),
                                    )
    return list(rows.values())


def _score_v1_2(metrics: dict, cost3: dict) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    tpd = float(metrics["core_trades_per_day"])
    sleeve_frac = float(metrics["sleeve_fraction"])
    cost3_pnl = float(cost3["pnl"])
    score = pnl + 0.18 * cost3_pnl
    score -= 20.0 * max(0.0, abs(mdd) - 14.0)
    score -= 35.0 * max(0.0, 5.8 - tpd)
    score -= 45.0 * max(0.0, sleeve_frac - 0.35)
    return float(score)


def main() -> int:
    base.DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_plus_causal_conviction_sleeve_v1_2"
    base.DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_2_2026.json"
    base.DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_2_grid.csv"
    base.DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_2_ledger.csv"
    base.DEFAULT_DOC = ROOT / "docs/experiments/clean_base_plus_causal_conviction_sleeve_v1_2.md"
    base._grid = _grid_v1_2
    base._score = _score_v1_2
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
