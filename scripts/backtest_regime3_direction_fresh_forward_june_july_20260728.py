#!/usr/bin/env python3
"""Fresh-forward test: bull=long, bear=short, chop=cash on ETH 5-minute bars."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import chart_regime3_fresh_forward_june_july_20260728 as regime_chart  # noqa: E402
import experiment_regime3_current_hmm_wide24_20260529 as regime_builder  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_direction_backtest_20260728"
REPORT = OUT_DIR / "eth_regime3_direction_fresh_forward_report.json"
CURVES = OUT_DIR / "eth_regime3_direction_fresh_forward_equity.csv"
MONTHLY = OUT_DIR / "eth_regime3_direction_fresh_forward_monthly.csv"
CHART = OUT_DIR / "eth_regime3_direction_fresh_forward_equity.png"

START = pd.Timestamp("2026-06-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-08-01 00:00:00")
ONE_WAY_COST_BPS = (0.0, 2.0, 5.0, 10.0)
NOTIONAL = 1.0
LEVERAGE = 1.0
MARGIN_FRACTION = NOTIONAL / LEVERAGE
POSITION = {"bull": 1.0, "bear": -1.0, "chop": 0.0}


def _drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min() * 100.0)


def _metrics(frame: pd.DataFrame, return_col: str) -> dict[str, float | int]:
    returns = frame[return_col]
    equity = (1.0 + returns).cumprod()
    active = frame["position"].ne(0.0)
    return {
        "bars": int(len(frame)),
        "pnl_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "mdd_pct": _drawdown(equity),
        "turnover_units": float(frame["turnover"].sum()),
        "position_changes": int(frame["turnover"].gt(0.0).sum()),
        "long_bar_share": float(frame["position"].gt(0.0).mean()),
        "short_bar_share": float(frame["position"].lt(0.0).mean()),
        "cash_bar_share": float(frame["position"].eq(0.0).mean()),
        "active_bar_hit_rate": float((returns[active] > 0.0).mean()) if active.any() else 0.0,
    }


def _build_frame() -> tuple[pd.DataFrame, dict]:
    source = regime_builder._read(regime_chart.FEATURES)
    payload = joblib.load(regime_chart.MODEL)
    fresh = regime_chart._fresh_forward(payload, source)
    frame = source[["timestamp", "open", "close"]].merge(
        fresh[["timestamp", "regime", "bull_prob", "bear_prob", "chop_prob"]],
        on="timestamp",
        validate="one_to_one",
    )

    # Signal on bar t close becomes the target position at bar t+1 open.
    frame["signal_position"] = frame["regime"].map(POSITION).astype(float)
    frame["position"] = frame["signal_position"].shift(1).fillna(0.0)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["next_open_return"] = frame["open"].shift(-1) / frame["open"] - 1.0
    frame["gross_return"] = frame["position"] * frame["next_open_return"] * NOTIONAL
    for cost_bps in ONE_WAY_COST_BPS:
        tag = int(cost_bps)
        frame[f"net_return_{tag}bps"] = frame["gross_return"] - frame["turnover"] * cost_bps / 10_000.0

    tested = frame[
        (frame["timestamp"] >= START)
        & (frame["timestamp"] < END_EXCLUSIVE)
        & frame["next_open_return"].notna()
    ].copy()
    audit = {
        "model_id": payload["model_id"],
        "source_range": [str(source["timestamp"].iloc[0]), str(source["timestamp"].iloc[-1])],
        "test_range": [str(tested["timestamp"].iloc[0]), str(tested["timestamp"].iloc[-1])],
        "fresh_forward_bar_by_bar": True,
        "future_rows_used_for_entry": False,
        "signal_timing": "regime from bar t close executes at bar t+1 open",
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "saved_regime_sidecar_used_as_input": False,
        "notional": NOTIONAL,
        "leverage": LEVERAGE,
        "margin_fraction": MARGIN_FRACTION,
    }
    return tested, audit


def _plot(frame: pd.DataFrame) -> None:
    plt.style.use("dark_background")
    fig, (ax_price, ax_equity, ax_position) = plt.subplots(
        3,
        1,
        figsize=(20, 10.5),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 2.1, 0.75], "hspace": 0.09},
    )
    fig.patch.set_facecolor("#0b1018")
    for ax in (ax_price, ax_equity, ax_position):
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.14, lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#263449")

    ax_price.plot(frame["timestamp"], frame["close"], color="#e7edf6", lw=0.85)
    ax_price.set_ylabel("ETH close")
    ax_price.set_title(
        "ETH Regime3 Direction Test — Bull Long / Bear Short / Chop Cash",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    ax_price.text(
        0.0,
        1.01,
        "Causal 5m regime at close → execution at next open · notional 1.0 · no TP/SL",
        transform=ax_price.transAxes,
        color="#93a4b8",
        fontsize=9,
    )

    benchmark = (1.0 + frame["next_open_return"]).cumprod()
    ax_equity.plot(frame["timestamp"], benchmark, color="#dbe4ef", lw=1.1, alpha=0.8, label="ETH buy & hold")
    colors = {0: "#20c997", 2: "#76a9fa", 5: "#f4c95d", 10: "#ff5c77"}
    for cost_bps in ONE_WAY_COST_BPS:
        tag = int(cost_bps)
        equity = (1.0 + frame[f"net_return_{tag}bps"]).cumprod()
        ax_equity.plot(
            frame["timestamp"],
            equity,
            color=colors[tag],
            lw=1.15,
            label=f"regime strategy · {tag}bp/side",
        )
    ax_equity.axhline(1.0, color="#8190a5", lw=0.7, alpha=0.6)
    ax_equity.set_ylabel("Equity (start = 1.0)")
    ax_equity.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)

    ax_position.fill_between(
        frame["timestamp"], 0.0, frame["position"], where=frame["position"].gt(0.0),
        step="post", color=regime_chart.COLORS["bull"], alpha=0.85, label="LONG",
    )
    ax_position.fill_between(
        frame["timestamp"], 0.0, frame["position"], where=frame["position"].lt(0.0),
        step="post", color=regime_chart.COLORS["bear"], alpha=0.85, label="SHORT",
    )
    ax_position.set_ylim(-1.15, 1.15)
    ax_position.set_yticks([-1.0, 0.0, 1.0], ["SHORT", "CASH", "LONG"])
    ax_position.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_position.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_position.set_xlabel("2026 (UTC timestamps from feature tape)")

    fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.065)
    fig.savefig(CHART, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, audit = _build_frame()

    benchmark_equity = (1.0 + frame["next_open_return"]).cumprod()
    full_metrics = {
        f"cost_{int(cost_bps)}bps_per_side": _metrics(frame, f"net_return_{int(cost_bps)}bps")
        for cost_bps in ONE_WAY_COST_BPS
    }
    full_metrics["buy_and_hold"] = {
        "pnl_pct": float((benchmark_equity.iloc[-1] - 1.0) * 100.0),
        "mdd_pct": _drawdown(benchmark_equity),
    }

    monthly_rows = []
    for month, part in frame.groupby(frame["timestamp"].dt.to_period("M")):
        row: dict[str, float | int | str] = {"month": str(month), "bars": int(len(part))}
        for cost_bps in ONE_WAY_COST_BPS:
            tag = int(cost_bps)
            one = _metrics(part, f"net_return_{tag}bps")
            row[f"pnl_{tag}bps_pct"] = float(one["pnl_pct"])
            row[f"mdd_{tag}bps_pct"] = float(one["mdd_pct"])
        row["buy_hold_pct"] = float(((1.0 + part["next_open_return"]).prod() - 1.0) * 100.0)
        row["turnover_units"] = float(part["turnover"].sum())
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(MONTHLY, index=False)

    curve_cols = [
        "timestamp", "open", "close", "regime", "position", "turnover", "gross_return",
        *[f"net_return_{int(cost_bps)}bps" for cost_bps in ONE_WAY_COST_BPS],
    ]
    curves = frame[curve_cols].copy()
    for cost_bps in ONE_WAY_COST_BPS:
        tag = int(cost_bps)
        curves[f"equity_{tag}bps"] = (1.0 + curves[f"net_return_{tag}bps"]).cumprod()
    curves["buy_hold_equity"] = (1.0 + frame["next_open_return"]).cumprod().to_numpy()
    curves.to_csv(CURVES, index=False)

    report = {
        **audit,
        "position_contract": POSITION,
        "one_way_cost_bps_tested": list(ONE_WAY_COST_BPS),
        "full_period": full_metrics,
        "monthly": monthly_rows,
        "outputs": {"equity_csv": str(CURVES), "monthly_csv": str(MONTHLY), "chart": str(CHART)},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(frame)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
