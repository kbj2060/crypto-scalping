#!/usr/bin/env python3
"""Causal confirmation-policy follow-up for the validation-selected 1h Regime3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/causal_regen_20260516/regime3_1h_deep_research_20260728/selected_fresh_forward_equity_curves.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_confirmation_policy_20260728"
GRID = OUT_DIR / "validation_confirmation_grid.csv"
REPORT = OUT_DIR / "confirmation_policy_report.json"
COST_BPS = 2


def _apply(frame: pd.DataFrame, confirmation_hours: int, probability_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    regime = out["regime"].astype(str)
    group = regime.ne(regime.shift()).cumsum()
    run = regime.groupby(group).cumcount() + 1
    probability = np.select(
        [regime.eq("bull"), regime.eq("bear")],
        [out["bull_prob"], out["bear_prob"]],
        default=out["chop_prob"],
    )
    signal = np.where(
        regime.eq("bull") & run.ge(confirmation_hours) & (probability >= probability_threshold),
        1.0,
        np.where(
            regime.eq("bear") & run.ge(confirmation_hours) & (probability >= probability_threshold),
            -1.0,
            0.0,
        ),
    )
    out["policy_position"] = pd.Series(signal, index=out.index).shift(1).fillna(0.0)
    out["policy_turnover"] = out["policy_position"].diff().abs().fillna(out["policy_position"].abs())
    out["policy_return"] = (
        out["policy_position"] * out["next_open_return"]
        - out["policy_turnover"] * COST_BPS / 10_000.0
    )
    return out


def _metrics(frame: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    part = frame[
        (frame["timestamp"] > start)
        & (frame["timestamp"] <= end)
        & frame["next_open_return"].notna()
    ]
    equity = (1.0 + part["policy_return"]).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min() * 100.0)
    pnl = float((equity.iloc[-1] - 1.0) * 100.0)
    return {
        "pnl_pct": pnl,
        "mdd_pct": mdd,
        "turnover_units": float(part["policy_turnover"].sum()),
        "active_bar_share": float(part["policy_position"].ne(0.0).mean()),
        "selection_score": pnl + 0.5 * mdd,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    rows = []
    policies = {}
    for confirmation in (1, 2, 3, 4, 6, 8, 12, 18, 24):
        for threshold in (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            parts = [
                _apply(part, confirmation, threshold)
                for _, part in source.groupby("source_year", sort=False)
            ]
            policy = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
            validation = _metrics(policy, "2025-09-01", "2026-01-01")
            name = f"confirm{confirmation}_prob{threshold:.2f}"
            policies[name] = policy
            rows.append({
                "policy": name,
                "confirmation_hours": confirmation,
                "probability_threshold": threshold,
                **validation,
            })
    grid = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID, index=False)
    selected = grid.iloc[0]
    selected_name = str(selected["policy"])
    selected_policy = policies[selected_name]
    validation = _metrics(selected_policy, "2025-09-01", "2026-01-01")
    oos = _metrics(selected_policy, "2026-01-01", "2026-04-01")
    latest = _metrics(selected_policy, "2026-06-01", "2026-07-20")
    report = {
        "research_id": "eth_regime3_1h_confirmation_policy_20260728",
        "parent_prediction_source": str(SOURCE),
        "parent_predictions_are_fresh_forward": True,
        "trade_ledgers_used_as_input": False,
        "future_rows_used_for_entry": False,
        "selection_window": ["2025-09-01", "2026-01-01"],
        "oos_used_for_selection": False,
        "latest_used_for_selection": False,
        "selected_policy": {
            "name": selected_name,
            "confirmation_hours": int(selected["confirmation_hours"]),
            "probability_threshold": float(selected["probability_threshold"]),
        },
        "validation": validation,
        "oos": oos,
        "latest_diagnostic": latest,
        "verdict": "REJECT_NO_VALIDATION_EDGE" if validation["pnl_pct"] <= 0.0 else "CONTINUE_RESEARCH",
        "grid": str(GRID),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
