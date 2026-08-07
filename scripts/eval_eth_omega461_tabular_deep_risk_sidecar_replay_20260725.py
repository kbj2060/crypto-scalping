#!/usr/bin/env python3
"""Research-only replay for the ETH Omega4.6.1 tabular deep risk sidecar.

This attaches the trained tabular MLP candidate only to the offline research replay harness.
It does not import or modify trading_bot.py, live adapters, runtime config, or environment
settings. All OOS/extension results here are diagnostic-only because those intervals were already
consumed by prior research.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_eth_omega461_live_chop_hazard_composition_20260724 as composition  # noqa: E402
import research_eth_omega461_censored_stopping_value_20260724 as stopping  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_tabular_deep_lifecycle_sidecar_20260725 as sidecar  # noqa: E402


MODEL_ID = "eth_omega461_tabular_deep_risk_sidecar_replay_20260725"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SIDECAR_MODEL = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "eth_omega461_tabular_deep_lifecycle_sidecar_20260725"
    / "model.pt"
)
EXTENSION_START, EXTENSION_END = "2026-04-01", "2026-07-12 09:00:00"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _load_sidecar() -> tuple[sidecar.LifecycleMLP, dict[str, Any]]:
    if not SIDECAR_MODEL.exists():
        raise RuntimeError(f"missing sidecar model: {SIDECAR_MODEL}")
    payload = torch.load(SIDECAR_MODEL, map_location="cpu", weights_only=False)
    feature_cols = list(payload["feature_columns"])
    model = sidecar.LifecycleMLP(len(feature_cols))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {
        "feature_columns": feature_cols,
        "scaler_mean": np.asarray(payload["scaler_mean"], dtype=np.float64),
        "scaler_scale": np.asarray(payload["scaler_scale"], dtype=np.float64),
        "horizon": int(payload.get("horizon", 96)),
        "source_model": str(SIDECAR_MODEL),
    }


def _predict_sl_probability(model: sidecar.LifecycleMLP, meta: dict[str, Any], feature_rows: list[dict[str, float]]) -> np.ndarray:
    cols = list(meta["feature_columns"])
    x = np.asarray([[float(row.get(col, 0.0)) for col in cols] for row in feature_rows], dtype=np.float64)
    scale = np.asarray(meta["scaler_scale"], dtype=np.float64)
    scale = np.where(np.abs(scale) <= 1.0e-12, 1.0, scale)
    x = ((x - np.asarray(meta["scaler_mean"], dtype=np.float64)) / scale).astype(np.float32)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.tensor(x[start : start + 4096], dtype=torch.float32)
            _, logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy()[:, 2])
    return np.concatenate(probs)


def _raw_move(price: float, *, side: int, entry_price: float) -> float:
    return float(side * (price - entry_price) / max(entry_price, 1.0e-12))


def replay(
    frame: pd.DataFrame,
    components: dict[str, dict[str, Any]],
    model: sidecar.LifecycleMLP | None,
    meta: dict[str, Any] | None,
    *,
    sl_probability_min: float,
    min_hold_bars: int,
    persistence: int,
    cost_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = stopping._arrays(frame)
    fee, slip = stopping.hazard.omega._load_fee_slip()
    fee_eff, slip_eff = float(fee) * cost_mult, float(slip) * cost_mult
    active_masks = {name: stopping.hazard.omega._active(comp["dec"]) for name, comp in components.items()}
    cash = 1.0
    peak = 1.0
    close_mdd = 0.0
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    i = 0
    last_i = len(frame) - 2
    while i <= last_i:
        option = stopping._entry_option(components, active_masks, arrays, i, fee_eff=fee_eff, slip_eff=slip_eff)
        if option is None:
            i += 1
            continue
        entry_equity = cash
        cash *= 1.0 - float(option["entry_fee"]) * float(option["notional"])
        indices, feature_rows = stopping._position_feature_rows(frame, arrays, components, option, end_i=last_i)
        if model is None or meta is None:
            p_sl = np.zeros(len(indices), dtype=np.float64)
        else:
            p_sl = _predict_sl_probability(model, meta, feature_rows)
        streak = 0
        mfe = 0.0
        mae = 0.0
        reason = "end_censored"
        exit_signal_i = indices[-1]
        decision_p_sl = float("nan")
        for path_idx, state_i in enumerate(indices):
            move, best, worst = stopping._bar_moves(
                arrays, state_i, side=int(option["side"]), entry_price=float(option["entry_price"])
            )
            mfe = max(mfe, best)
            mae = min(mae, worst)
            equity = cash * (1.0 + move * float(option["notional"]))
            peak = max(peak, equity)
            close_mdd = min(close_mdd, equity / max(peak, 1.0e-12) - 1.0)
            if option["stop_loss"] > 0.0 and worst <= -abs(float(option["stop_loss"])):
                reason, exit_signal_i = "stop_loss", state_i
                break
            if option["take_profit"] > 0.0 and best >= float(option["take_profit"]):
                reason, exit_signal_i = "take_profit", state_i
                break
            hold_bars = int(state_i) - int(option["entry_i"])
            hit = bool(hold_bars >= int(min_hold_bars) and float(p_sl[path_idx]) >= float(sl_probability_min))
            streak = streak + 1 if hit else 0
            if model is not None and streak >= int(persistence):
                reason, exit_signal_i = "deep_risk_exit", state_i
                decision_p_sl = float(p_sl[path_idx])
                break
        filled, exit_price, exit_fee, exit_route = stopping.hazard.omega._try_execution(
            arrays, exit_signal_i, int(option["side"]), entry=False, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            raise RuntimeError("exit execution unexpectedly missed")
        raw_exit = _raw_move(float(exit_price), side=int(option["side"]), entry_price=float(option["entry_price"]))
        before_exit = cash
        cash *= 1.0 + raw_exit * float(option["notional"])
        cash -= before_exit * float(exit_fee) * float(option["notional"])
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        reasons[reason] += 1
        rows.append({
            "entry_signal_i": option["signal_i"],
            "entry_i": option["entry_i"],
            "exit_i": exit_signal_i,
            "entry_timestamp": str(frame["timestamp"].iloc[int(option["signal_i"])]),
            "exit_timestamp": str(frame["timestamp"].iloc[exit_signal_i]),
            "side": option["side"],
            "source_component": option["source_component"],
            "reason": reason,
            "trade_return": trade_return,
            "net_per_notional": trade_return / max(float(option["notional"]), 1.0e-12),
            "mae_price_move": mae,
            "mfe_price_move": mfe,
            "notional": option["notional"],
            "margin_fraction": option["margin_fraction"],
            "leverage": option["leverage"],
            "entry_chop_probability": option["chop_probability"],
            "entry_sizing_multiplier": option["sizing_multiplier"],
            "deep_p_sl_h96": decision_p_sl,
            "entry_route": option["entry_route"],
            "exit_route": exit_route,
        })
        i = exit_signal_i + 1
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("empty replay ledger")
    returns = ledger["trade_return"].to_numpy(dtype=np.float64)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "close_mark_to_market_mdd": float(close_mdd * 100.0),
        "realized_mdd": composition._realized_mdd(returns),
        "trades": int(len(ledger)),
        "wr": float(np.mean(returns > 0.0)),
        "exit_reasons": dict(reasons),
        "avg_notional": float(ledger["notional"].mean()),
    }
    return metrics, ledger


def _passes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    pnl_floor = 0.90 * baseline["pnl"] if baseline["pnl"] >= 0.0 else 1.10 * baseline["pnl"]
    return bool(
        candidate["pnl"] >= pnl_floor
        and candidate["close_mark_to_market_mdd"] >= baseline["close_mark_to_market_mdd"]
        and candidate["realized_mdd"] >= baseline["realized_mdd"]
    )


def _prepare_split(start: str, end: str, *, base_csv: Path, wide24_csv: Path, prediction_split: str, oof: bool):
    return stopping._prepare_split(
        start,
        end,
        base_csv=base_csv,
        wide24_csv=wide24_csv,
        prediction_split=prediction_split,
        oof=oof,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = _load_sidecar()
    val_frame, val_components = _prepare_split(
        sweep.VAL_START,
        sweep.VAL_END,
        base_csv=sweep.BASE_2025,
        wide24_csv=sweep.WIDE24_2025,
        prediction_split="validation",
        oof=True,
    )
    baseline, baseline_ledger = replay(
        val_frame, val_components, None, None,
        sl_probability_min=1.0, min_hold_bars=10**9, persistence=1, cost_mult=1.0,
    )
    baseline_ledger.to_csv(OUT_DIR / "validation_live_baseline_ledger.csv", index=False)
    grid_rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[float, int, int], pd.DataFrame] = {}
    for sl_probability_min in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for min_hold_bars in (12, 24, 48, 96):
            for persistence in (1, 3):
                metrics, ledger = replay(
                    val_frame,
                    val_components,
                    model,
                    meta,
                    sl_probability_min=sl_probability_min,
                    min_hold_bars=min_hold_bars,
                    persistence=persistence,
                    cost_mult=1.0,
                )
                key = (sl_probability_min, min_hold_bars, persistence)
                ledgers[key] = ledger
                grid_rows.append({
                    "sl_probability_min": sl_probability_min,
                    "min_hold_bars": min_hold_bars,
                    "persistence": persistence,
                    **metrics,
                    "deep_exit_count": int(metrics["exit_reasons"].get("deep_risk_exit", 0)),
                    "passes_dev_gate": bool(metrics["exit_reasons"].get("deep_risk_exit", 0) > 0 and _passes(metrics, baseline)),
                })
    ranking = pd.DataFrame(grid_rows).sort_values(
        ["passes_dev_gate", "close_mark_to_market_mdd", "realized_mdd", "pnl"],
        ascending=[False, False, False, False],
    )
    ranking.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    passing = [row for row in grid_rows if row["passes_dev_gate"]]
    selected = max(
        passing,
        key=lambda row: (row["close_mark_to_market_mdd"], row["realized_mdd"], row["pnl"]),
    ) if passing else None
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "development_rejected" if selected is None else "development_selected_diagnostic_only",
        "deployment_verdict": "do_not_apply_to_live",
        "sidecar": meta,
        "validation": {
            "live_baseline": baseline,
            "selected": selected,
            "best_rows_head": ranking.head(10).to_dict(orient="records"),
        },
        "protocol": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "model_wired_to_live": False,
            "candidate_action": "research replay exits when p_sl_h96 clears validation-selected threshold",
            "selection_split": "validation_only",
            "diagnostic_oos_warning": "OOS and extension intervals were already consumed by prior research; not promotion evidence.",
        },
    }
    if selected is not None:
        key = (
            float(selected["sl_probability_min"]),
            int(selected["min_hold_bars"]),
            int(selected["persistence"]),
        )
        ledgers[key].to_csv(OUT_DIR / "validation_selected_ledger.csv", index=False)
        evaluations = (
            ("oos", sweep.OOS_START, sweep.OOS_END, sweep.BASE_2026, sweep.WIDE24_2026, "oos", False),
            ("extension", EXTENSION_START, EXTENSION_END, sweep.BASE_2026, sweep.WIDE24_2026, "oos", False),
        )
        for name, start, end, base_csv, wide24_csv, prediction_split, oof in evaluations:
            frame, components = _prepare_split(
                start,
                end,
                base_csv=base_csv,
                wide24_csv=wide24_csv,
                prediction_split=prediction_split,
                oof=oof,
            )
            split_rows: dict[str, Any] = {}
            for cost_mult in (1.0, 2.0, 3.0):
                base_metrics, base_ledger = replay(
                    frame, components, None, None,
                    sl_probability_min=1.0, min_hold_bars=10**9, persistence=1, cost_mult=cost_mult,
                )
                candidate_metrics, candidate_ledger = replay(
                    frame, components, model, meta,
                    sl_probability_min=key[0],
                    min_hold_bars=key[1],
                    persistence=key[2],
                    cost_mult=cost_mult,
                )
                tag = f"cost{int(cost_mult)}"
                split_rows[tag] = {
                    "live_baseline": base_metrics,
                    "candidate": candidate_metrics,
                    "passes_same_cost_baseline": _passes(candidate_metrics, base_metrics),
                }
                if cost_mult == 1.0:
                    base_ledger.to_csv(OUT_DIR / f"{name}_live_baseline_ledger.csv", index=False)
                    candidate_ledger.to_csv(OUT_DIR / f"{name}_candidate_ledger.csv", index=False)
            report[f"{name}_diagnostic"] = split_rows
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({
        "report": str(OUT_DIR / "report.json"),
        "status": report["status"],
        "validation_baseline": baseline,
        "selected": selected,
    }, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
