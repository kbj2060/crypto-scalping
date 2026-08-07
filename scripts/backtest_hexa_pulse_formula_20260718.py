#!/usr/bin/env python3
"""Causal diagnostic for the immutable HexaPulse-R v1 threshold formula.

The historical tail-risk stream is known-invalid (all values are zero).  This script therefore
runs an explicitly labelled five-input diagnostic with tail risk fixed to zero.  It is not a
promotion test and never consumes saved trade ledgers or parent exits.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_bot_modules.hexa_pulse_formula import (  # noqa: E402
    AVAIL_SHIFT_MIN,
    FORMULA_ID,
    HexaPulseConfig,
    HexaPulseState,
    compute_formula_values,
    reconstruct_whale_position_score,
    step_formula,
)


MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
REPORT = ROOT / "data/ensemble/reports/hexa_pulse_r_v1_diagnostic_20260718.json"
LEDGER = ROOT / "data/ensemble/reports/hexa_pulse_r_v1_diagnostic_20260718_bars.csv"


def _load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    con = duckdb.connect(str(MICRO_DB), read_only=True)
    micro = con.execute(
        """
        SELECT ts, nif_whale, obi, eai, oi_delta_pct, shadow_toxicity_score,
               data_stale, valid_nif, warmup_30m_ready
        FROM microstructure_1m
        ORDER BY ts
        """
    ).fetchdf()
    con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    micro = micro.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    micro["whale_position_score"] = reconstruct_whale_position_score(
        micro["nif_whale"], micro["oi_delta_pct"]
    )

    tail_con = duckdb.connect(str(TAIL_DB), read_only=True)
    tail_stats = tail_con.execute(
        """
        SELECT count(*) AS rows,
               sum(CASE WHEN long_usd_1m > 0 OR short_usd_1m > 0 THEN 1 ELSE 0 END) AS active_rows,
               min(shadow_aftershock_prob) AS min_prob,
               max(shadow_aftershock_prob) AS max_prob
        FROM tail_risk_1m
        """
    ).fetchone()
    tail_con.close()
    tail_valid = bool(tail_stats[1] and float(tail_stats[3] or 0.0) > 0.0)
    if tail_valid:
        raise RuntimeError("historical tail stream unexpectedly became valid; run a strict six-input evaluator")

    # Diagnostic-only substitution.  Production uses prepare_live_formula_frame(), which
    # requires tail schema v3 and never performs this substitution.
    micro["shadow_aftershock_prob"] = 0.0
    formula = compute_formula_values(micro)
    formula["available"] = (
        np.isfinite(formula["score"])
        & ~micro["data_stale"].astype(bool)
        & micro["valid_nif"].astype(bool)
        & micro["warmup_30m_ready"].astype(bool)
    )
    formula.index = formula.index + pd.Timedelta(minutes=AVAIL_SHIFT_MIN)

    price = pd.read_csv(KLINES, usecols=["timestamp", "open"], parse_dates=["timestamp"])
    price = price.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    start = formula.index.min()
    end = min(formula.index.max(), price.index.max())
    grid = price.loc[start:end].copy()
    grid["next_return"] = grid["open"].shift(-1) / grid["open"] - 1.0
    joined = grid.join(formula, how="left")
    joined["available"] = joined["available"].eq(True)
    metadata = {
        "micro_start": str(micro.index.min()),
        "micro_end": str(micro.index.max()),
        "price_start": str(price.index.min()),
        "price_end": str(price.index.max()),
        "tail_rows": int(tail_stats[0]),
        "tail_active_rows": int(tail_stats[1] or 0),
        "tail_probability_min": float(tail_stats[2] or 0.0),
        "tail_probability_max": float(tail_stats[3] or 0.0),
    }
    return joined.iloc[:-1].copy(), metadata


def _run_segment(frame: pd.DataFrame, config: HexaPulseConfig, fee: float) -> tuple[dict[str, Any], pd.DataFrame]:
    state = HexaPulseState()
    rows: list[dict[str, Any]] = []
    previous = 0
    for ts, row in frame.iterrows():
        decision = step_formula(
            state,
            score=float(row["score"]) if pd.notna(row["score"]) else float("nan"),
            toxicity=float(row["toxicity"]) if pd.notna(row["toxicity"]) else 1.0,
            tail_risk=float(row["tail_risk"]) if pd.notna(row["tail_risk"]) else 1.0,
            available=bool(row["available"]),
            config=config,
        )
        position = int(decision.position)
        turnover = abs(position - previous)
        gross = position * float(row["next_return"])
        cost = fee * turnover
        rows.append(
            {
                "timestamp": ts,
                "score": row["score"],
                "available": bool(row["available"]),
                "action": decision.action,
                "reason": decision.reason,
                "previous_position": previous,
                "position": position,
                "turnover": turnover,
                "next_return": float(row["next_return"]),
                "gross_return": gross,
                "cost": cost,
                "net_return": gross - cost,
            }
        )
        previous = position

    ledger = pd.DataFrame(rows)
    if len(ledger) and int(ledger.iloc[-1]["position"]) != 0:
        ledger.loc[ledger.index[-1], "turnover"] += 1
        ledger.loc[ledger.index[-1], "cost"] += fee
        ledger.loc[ledger.index[-1], "net_return"] -= fee
    net = ledger["net_return"].to_numpy(dtype=float)
    gross = ledger["gross_return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + net)
    curve = np.r_[1.0, equity]
    peak = np.maximum.accumulate(curve)
    drawdown = 1.0 - curve / peak
    ledger["equity"] = equity

    entry_mask = (ledger["previous_position"] == 0) & (ledger["position"] != 0)
    durations: list[int] = []
    trade_returns: list[float] = []
    active_start: int | None = None
    active_net = 0.0
    for idx, row in ledger.iterrows():
        if row["previous_position"] == 0 and row["position"] != 0:
            active_start = int(idx)
            active_net = float(row["net_return"])
        elif active_start is not None:
            active_net = (1.0 + active_net) * (1.0 + float(row["net_return"])) - 1.0
            if row["position"] == 0:
                durations.append(int(idx) - active_start)
                trade_returns.append(active_net)
                active_start = None
                active_net = 0.0
    if active_start is not None:
        durations.append(len(ledger) - active_start)
        trade_returns.append(active_net)

    metrics = {
        "bars": int(len(ledger)),
        "available_fraction": float(ledger["available"].mean()) if len(ledger) else 0.0,
        "trades": int(entry_mask.sum()),
        "win_rate": float(np.mean(np.asarray(trade_returns) > 0.0)) if trade_returns else 0.0,
        "compounded_return_pct": float((equity[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "additive_cost_pct": float(ledger["cost"].sum() * 100.0),
        "max_drawdown_pct": float(drawdown.max() * 100.0) if len(drawdown) else 0.0,
        "exposure_fraction": float((ledger["position"] != 0).mean()) if len(ledger) else 0.0,
        "long_entries": int(((ledger["previous_position"] == 0) & (ledger["position"] > 0)).sum()),
        "short_entries": int(((ledger["previous_position"] == 0) & (ledger["position"] < 0)).sum()),
        "holding_minutes": {
            "median": float(np.median(durations)) if durations else 0.0,
            "p95": float(np.quantile(durations, 0.95)) if durations else 0.0,
            "max": int(max(durations)) if durations else 0,
        },
    }
    return metrics, ledger


def main() -> None:
    frame, source = _load_inputs()
    config = HexaPulseConfig()
    split_specs = {
        "may_research": ("2026-05-03", "2026-06-01"),
        "june_validation_diagnostic": ("2026-06-01", "2026-07-01"),
        "july_reused_diagnostic": ("2026-07-01", "2026-07-16"),
        "all_available": ("2026-05-03", "2026-07-16"),
    }
    results: dict[str, Any] = {}
    ledgers: list[pd.DataFrame] = []
    for name, (start, end) in split_specs.items():
        segment = frame[(frame.index >= start) & (frame.index < end)]
        metrics, ledger = _run_segment(segment, config, fee=0.00045)
        results[name] = metrics
        if name == "all_available":
            ledger.insert(0, "split", name)
            ledgers.append(ledger)

    fee_stress: dict[str, Any] = {}
    all_segment = frame[(frame.index >= "2026-05-03") & (frame.index < "2026-07-16")]
    for bps in (2.0, 3.25, 4.5, 5.5):
        fee_stress[f"{bps:.2f}_bps_per_notional_change"], _ = _run_segment(
            all_segment, config, fee=bps / 10_000.0
        )

    report = {
        "formula_id": FORMULA_ID,
        "evaluation_class": "research_diagnostic_only",
        "formula_config": asdict(config),
        "source": source,
        "splits": results,
        "fee_stress": fee_stress,
        "data_contract": {
            "decision_uses_micro_row_at_or_before_d_minus_2m": True,
            "tail_risk_historical_contract_valid": False,
            "tail_risk_diagnostic_substitution": "shadow_aftershock_prob=0.0",
            "whale_position_reconstructed_from": ["nif_whale", "oi_delta_pct"],
            "thresholds_tuned": False,
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "fixed_holding_period_used": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": "historical tail-risk input is invalid and July was reused during prior research",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.concat(ledgers, ignore_index=True).to_csv(LEDGER, index=False)
    print(json.dumps({"report": str(REPORT), "splits": results, "fee_stress": fee_stress}, indent=2))


if __name__ == "__main__":
    main()
