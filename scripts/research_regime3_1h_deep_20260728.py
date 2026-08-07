#!/usr/bin/env python3
"""Validation-selected deep research for the causal ETH 1h Regime3 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import chart_regime3_1h_fresh_forward_latest_week_20260728 as base  # noqa: E402
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402


SOURCE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_deep_research_20260728"
GRID_OUT = OUT_DIR / "validation_candidate_grid.csv"
REPORT_OUT = OUT_DIR / "regime3_1h_deep_research_report.json"
MODEL_OUT = OUT_DIR / "selected_validation_only_regime3_1h_model.joblib"
CURVES_OUT = OUT_DIR / "selected_fresh_forward_equity_curves.csv"
CHART_OUT = OUT_DIR / "selected_fresh_forward_equity_chart.png"

VAL_START = pd.Timestamp("2025-09-01 00:00:00")
VAL_END = pd.Timestamp("2026-01-01 00:00:00")
OOS_START = pd.Timestamp("2026-01-01 00:00:00")
OOS_END = pd.Timestamp("2026-04-01 00:00:00")
LATEST_START = pd.Timestamp("2026-06-01 00:00:00")
LATEST_END = pd.Timestamp("2026-07-20 00:00:00")
COSTS_BPS = (0, 2, 5, 10)
CLASS_TO_POSITION = {"bull": 1.0, "bear": -1.0, "chop": 0.0}

CORE_TREND_FEATURES = [
    "log_return_1h", "return_3h", "return_6h", "return_12h", "return_24h",
    "ema_slope_5h", "trend_efficiency_12h", "trend_efficiency_24h",
    "realized_vol_12h", "realized_vol_24h", "bb_width_20h", "bb_width_z_168h",
    "rsi_14h", "macd_hist_1h", "adx_14h",
]
FEATURE_SETS = {
    "all21": list(base.FEATURE_COLS),
    "core15": CORE_TREND_FEATURES,
    "no_crossasset19": [c for c in base.FEATURE_COLS if c not in {"btc_corr_24h", "eth_btc_spread_6h"}],
    "no_flow19": [c for c in base.FEATURE_COLS if c not in {"net_taker_ratio_1h", "oi_change_1h"}],
}


def _fit(train: pd.DataFrame, feature_cols: list[str], states: int, sticky: float, seed: int) -> dict[str, Any]:
    raw = train[feature_cols].copy()
    medians = raw.median(numeric_only=True).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    observations = scaler.fit_transform(raw.fillna(medians).fillna(0.0))
    model = GaussianStateModel(states, 24, seed, sticky=sticky).fit(observations)
    state_probability = model.filter_proba(observations)
    state_class = base._state_class_matrix(state_probability, base._labels(train))
    return {
        "model_id": f"eth_regime3_1h_{len(feature_cols)}f_s{states}_sticky{sticky:.2f}_seed{seed}",
        "classes": base.CLASSES,
        "feature_cols": feature_cols,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "model": model,
        "state_class_matrix": state_class,
        "state_count": states,
        "sticky_initialization": sticky,
        "seed": seed,
        "fit_period": "2024",
    }


def _filter(payload: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    cols = payload["feature_cols"]
    medians = pd.Series(payload["feature_medians"])
    observations = payload["scaler"].transform(frame[cols].fillna(medians).fillna(0.0))
    model = payload["model"]
    log_emission = model._log_emission(observations)
    log_transition = np.log(model.A_ + 1e-300)
    state_probability = np.empty((len(frame), model.n_states), dtype=np.float64)
    previous: np.ndarray | None = None
    for bar_index in range(len(frame)):
        if previous is None:
            current = np.log(model.pi_ + 1e-300) + log_emission[bar_index]
        else:
            current = log_emission[bar_index] + model._logsumexp(
                previous[:, None] + log_transition, axis=0
            )
        current -= model._logsumexp(current, axis=0)
        state_probability[bar_index] = np.exp(current)
        previous = current
    probability = base._class_probability(state_probability, payload["state_class_matrix"])
    out = frame[["timestamp", "open", "close"]].copy()
    for class_index, name in enumerate(base.CLASSES):
        out[f"{name}_prob"] = probability[:, class_index]
    out["regime_id"] = np.argmax(probability, axis=1)
    out["regime"] = np.asarray(base.CLASSES, dtype=object)[out["regime_id"]]
    out["target_id"] = base._labels(frame)
    out["target_regime"] = np.asarray(base.CLASSES, dtype=object)[out["target_id"]]
    out["signal_position"] = out["regime"].map(CLASS_TO_POSITION).astype(float)
    out["position"] = out["signal_position"].shift(1).fillna(0.0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["next_open_return"] = out["open"].shift(-1) / out["open"] - 1.0
    out["gross_return"] = out["position"] * out["next_open_return"]
    for cost in COSTS_BPS:
        out[f"net_return_{cost}bps"] = out["gross_return"] - out["turnover"] * cost / 10_000.0
    for horizon in (6, 12, 24):
        out[f"forward_close_{horizon}h"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return out


def _drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min() * 100.0)


def _runs(regime: pd.Series) -> np.ndarray:
    values = regime.to_numpy()
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return np.diff(np.r_[starts, len(values)])


def _evaluate(filtered: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    frame = filtered[
        (filtered["timestamp"] > start)
        & (filtered["timestamp"] <= end)
        & filtered["next_open_return"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError(f"empty evaluation window: {start}..{end}")
    runs = _runs(frame["regime"])
    metrics: dict[str, Any] = {
        "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        "bars": int(len(frame)),
        "balanced_accuracy_current_label": float(balanced_accuracy_score(frame["target_id"], frame["regime_id"])),
        "macro_precision_current_label": float(precision_score(frame["target_id"], frame["regime_id"], average="macro", zero_division=0)),
        "macro_recall_current_label": float(recall_score(frame["target_id"], frame["regime_id"], average="macro", zero_division=0)),
        "regime_share": {name: float((frame["regime"] == name).mean()) for name in base.CLASSES},
        "flips": int((frame["regime"].iloc[1:].to_numpy() != frame["regime"].iloc[:-1].to_numpy()).sum()),
        "mean_run_hours": float(runs.mean()),
        "median_run_hours": float(np.median(runs)),
        "turnover_units": float(frame["turnover"].sum()),
    }
    for regime in ("bull", "bear"):
        subset = frame[frame["regime"] == regime]
        metrics[f"{regime}_precision_current_label"] = float((subset["target_regime"] == regime).mean()) if len(subset) else 0.0
        for horizon in (6, 12, 24):
            values = subset[f"forward_close_{horizon}h"].dropna()
            hit = values.gt(0.0) if regime == "bull" else values.lt(0.0)
            metrics[f"{regime}_{horizon}h_direction_hit"] = float(hit.mean()) if len(values) else 0.0
            metrics[f"{regime}_{horizon}h_mean_forward_pct"] = float(values.mean() * 100.0) if len(values) else 0.0
    metrics["costs"] = {}
    for cost in COSTS_BPS:
        equity = (1.0 + frame[f"net_return_{cost}bps"]).cumprod()
        metrics["costs"][f"{cost}bps_per_side"] = {
            "pnl_pct": float((equity.iloc[-1] - 1.0) * 100.0),
            "mdd_pct": _drawdown(equity),
        }
    benchmark = (1.0 + frame["next_open_return"]).cumprod()
    metrics["buy_hold"] = {
        "pnl_pct": float((benchmark.iloc[-1] - 1.0) * 100.0),
        "mdd_pct": _drawdown(benchmark),
    }
    metrics["selection_score_cost2_pnl_plus_half_mdd"] = float(
        metrics["costs"]["2bps_per_side"]["pnl_pct"]
        + 0.5 * metrics["costs"]["2bps_per_side"]["mdd_pct"]
    )
    return metrics


def _candidate_row(name: str, feature_set: str, states: int, sticky: float, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": name,
        "feature_set": feature_set,
        "feature_count": len(FEATURE_SETS[feature_set]),
        "states": states,
        "sticky": sticky,
        "selection_score": metrics["selection_score_cost2_pnl_plus_half_mdd"],
        "val_cost0_pnl": metrics["costs"]["0bps_per_side"]["pnl_pct"],
        "val_cost2_pnl": metrics["costs"]["2bps_per_side"]["pnl_pct"],
        "val_cost2_mdd": metrics["costs"]["2bps_per_side"]["mdd_pct"],
        "val_cost5_pnl": metrics["costs"]["5bps_per_side"]["pnl_pct"],
        "val_bacc": metrics["balanced_accuracy_current_label"],
        "val_bull_12h_hit": metrics["bull_12h_direction_hit"],
        "val_bear_12h_hit": metrics["bear_12h_direction_hit"],
        "val_flips": metrics["flips"],
        "val_median_run_hours": metrics["median_run_hours"],
        "val_turnover": metrics["turnover_units"],
    }


def _plot_curves(curves: pd.DataFrame, windows: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), dpi=150, gridspec_kw={"hspace": 0.28})
    fig.patch.set_facecolor("#0b1018")
    for ax, (name, start, end) in zip(axes, windows, strict=True):
        part = curves[(curves["timestamp"] > start) & (curves["timestamp"] <= end)].copy()
        for cost, color in ((0, "#20c997"), (2, "#76a9fa"), (5, "#f4c95d"), (10, "#ff5c77")):
            equity = (1.0 + part[f"net_return_{cost}bps"]).cumprod()
            ax.plot(part["timestamp"], equity, color=color, lw=1.15, label=f"{cost}bp/side")
        benchmark = (1.0 + part["next_open_return"]).cumprod()
        ax.plot(part["timestamp"], benchmark, color="#e7edf6", lw=1.0, alpha=0.8, label="buy & hold")
        ax.axhline(1.0, color="#8190a5", lw=0.7, alpha=0.6)
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.14, lw=0.6)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_ylabel("Equity")
        ax.legend(loc="upper left", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Selected ETH 1h Regime3 — Validation Lock, OOS, Latest Diagnostic", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.055)
    fig.savefig(CHART_OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = base._with_1h_features(base._aggregate_completed_hours(base._read_source(base.TRAIN_2024)))
    frame_2025 = base._with_1h_features(base._aggregate_completed_hours(base._read_source(SOURCE_2025)))
    frame_2026 = base._with_1h_features(base._aggregate_completed_hours(base._read_source(base.FORWARD_2026)))

    candidates: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for states in (4, 5, 6, 8):
        for sticky in (0.90, 0.93, 0.97):
            name = f"all21_s{states}_sticky{sticky:.2f}"
            payload = _fit(train, FEATURE_SETS["all21"], states, sticky, base.SEED + states)
            val_metrics = _evaluate(_filter(payload, frame_2025), VAL_START, VAL_END)
            payloads[name] = payload
            validations[name] = val_metrics
            candidates.append(_candidate_row(name, "all21", states, sticky, val_metrics))
            print(f"[topology] {name} score={candidates[-1]['selection_score']:.3f}", flush=True)

    topology_best = max(candidates, key=lambda row: float(row["selection_score"]))
    best_states = int(topology_best["states"])
    best_sticky = float(topology_best["sticky"])
    for feature_set in ("core15", "no_crossasset19", "no_flow19"):
        name = f"{feature_set}_s{best_states}_sticky{best_sticky:.2f}"
        payload = _fit(train, FEATURE_SETS[feature_set], best_states, best_sticky, base.SEED + best_states)
        val_metrics = _evaluate(_filter(payload, frame_2025), VAL_START, VAL_END)
        payloads[name] = payload
        validations[name] = val_metrics
        candidates.append(_candidate_row(name, feature_set, best_states, best_sticky, val_metrics))
        print(f"[ablation] {name} score={candidates[-1]['selection_score']:.3f}", flush=True)

    grid = pd.DataFrame(candidates).sort_values("selection_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    selected_name = str(grid.iloc[0]["candidate"])
    selected_payload = payloads[selected_name]
    joblib.dump(selected_payload, MODEL_OUT)

    selected_2025 = _filter(selected_payload, frame_2025)
    selected_2026 = _filter(selected_payload, frame_2026)
    validation = validations[selected_name]
    oos = _evaluate(selected_2026, OOS_START, OOS_END)
    latest = _evaluate(selected_2026, LATEST_START, LATEST_END)
    curves = pd.concat(
        [selected_2025.assign(source_year=2025), selected_2026.assign(source_year=2026)],
        ignore_index=True,
    )
    curves.to_csv(CURVES_OUT, index=False)
    _plot_curves(
        curves,
        [
            ("Validation · selected here", VAL_START, VAL_END),
            ("2026 OOS · evaluated only after lock", OOS_START, OOS_END),
            ("June–July · latest diagnostic only", LATEST_START, LATEST_END),
        ],
    )

    val_positive = validation["costs"]["2bps_per_side"]["pnl_pct"] > 0.0
    oos_positive = oos["costs"]["2bps_per_side"]["pnl_pct"] > 0.0
    directional = (
        oos["bull_12h_direction_hit"] > 0.5
        and oos["bear_12h_direction_hit"] > 0.5
    )
    verdict = "CONTINUE_RESEARCH" if val_positive and oos_positive and directional else "REJECT_AS_DIRECTION_OWNER"
    report = {
        "research_id": "eth_regime3_1h_deep_research_20260728",
        "status": "RESEARCH_ONLY_NOT_PROMOTION_ELIGIBLE",
        "selection_policy": {
            "train": "2024 completed 1h bars",
            "validation": [str(VAL_START), str(VAL_END)],
            "oos": [str(OOS_START), str(OOS_END)],
            "latest_diagnostic": [str(LATEST_START), str(LATEST_END)],
            "selection_score": "validation cost2 pnl_pct + 0.5 * validation cost2 mdd_pct",
            "oos_used_for_selection": False,
            "latest_used_for_selection": False,
        },
        "selected_candidate": selected_name,
        "selected_config": {
            "feature_cols": selected_payload["feature_cols"],
            "state_count": selected_payload["state_count"],
            "sticky_initialization": selected_payload["sticky_initialization"],
            "seed": selected_payload["seed"],
        },
        "validation": validation,
        "oos": oos,
        "latest_diagnostic": latest,
        "verdict": verdict,
        "audit": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "saved_5m_regime_probabilities_used": False,
            "signal_timing": "completed 1h bar regime executes at next 1h open",
            "notional": 1.0,
            "leverage": 1.0,
            "margin_fraction": 1.0,
        },
        "outputs": {
            "grid": str(GRID_OUT),
            "model": str(MODEL_OUT),
            "curves": str(CURVES_OUT),
            "chart": str(CHART_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_candidate": selected_name,
        "validation_cost2": validation["costs"]["2bps_per_side"],
        "oos_cost2": oos["costs"]["2bps_per_side"],
        "latest_cost2": latest["costs"]["2bps_per_side"],
        "verdict": verdict,
        "report": str(REPORT_OUT),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
