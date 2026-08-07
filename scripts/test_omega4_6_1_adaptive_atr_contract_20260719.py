#!/usr/bin/env python3
"""Validation-selected adaptive ATR SL/TP contract experiment for ETH Omega4.6.1.

The frozen parent, quality gates, risk sidecars, duration gate, and exit head are unchanged.
Nine SL/TP contracts are ranked on the clean 2025-10-01..12-31 OOF validation window. Only the
single validation-selected adaptive candidate and the fixed live baseline are then replayed on
the exact per-bar 2026 OOS prediction range.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_adaptive_atr_contract_20260719"
PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
DEVICE = torch.device("cpu")

BASELINE = {
    "name": "baseline_tp075_sl040_m12_6",
    "min_tp": 0.075,
    "min_sl": 0.040,
    "tp_mult": 12.0,
    "sl_mult": 6.0,
}
ADAPTIVE_CANDIDATES = [
    {
        "name": f"adaptive_tp{int(tp_floor * 10000):04d}_sl{int(sl_floor * 10000):04d}_m{tp_mult}_{sl_mult}",
        "min_tp": tp_floor,
        "min_sl": sl_floor,
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
    }
    for tp_floor, sl_floor in ((0.050, 0.025), (0.055, 0.0275), (0.060, 0.030))
    for tp_mult, sl_mult in ((24, 12), (27, 14), (30, 16))
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


@contextmanager
def _contract_override(contract: dict[str, Any]):
    original = atr_eval._apply_atr_safety_sltp

    def wrapped(decisions: pd.DataFrame, frame: pd.DataFrame, **kwargs: Any):
        kwargs.update(
            min_tp=float(contract["min_tp"]),
            min_sl=float(contract["min_sl"]),
            tp_mult=float(contract["tp_mult"]),
            sl_mult=float(contract["sl_mult"]),
        )
        return original(decisions, frame, **kwargs)

    atr_eval._apply_atr_safety_sltp = wrapped
    try:
        yield
    finally:
        atr_eval._apply_atr_safety_sltp = original


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    returns = ledger["trade_return"].to_numpy(dtype=np.float64)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(curve / np.maximum(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(ledger)),
        "wr": float(np.mean(returns > 0.0)),
    }


def _gate_ledger(ledger: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    market = frame[["timestamp", "ou_halflife"]].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    out = ledger.copy()
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"])
    out = out.merge(
        market.rename(columns={"timestamp": "entry_timestamp"}),
        on="entry_timestamp",
        how="left",
        validate="one_to_one",
    )
    return out.loc[out["ou_halflife"] > greedy.DURATION_THRESHOLD].reset_index(drop=True)


def _monthly_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {}
    work = ledger.copy()
    work["month"] = pd.to_datetime(work["entry_timestamp"]).dt.to_period("M").astype(str)
    return {month: _compound_metrics(rows) for month, rows in work.groupby("month", sort=True)}


def _dynamic_share(frame: pd.DataFrame, contract: dict[str, Any]) -> dict[str, float]:
    atr = atr_eval._atr_pct(frame, 192)
    atr = atr[np.isfinite(atr)]
    return {
        "tp_dynamic_fraction": float(np.mean(atr * float(contract["tp_mult"]) > float(contract["min_tp"]))),
        "sl_dynamic_fraction": float(np.mean(atr * float(contract["sl_mult"]) > float(contract["min_sl"]))),
    }


def _aligned_validation_frame() -> pd.DataFrame:
    frame = valmod.load_val_frame()
    common: set[pd.Timestamp] | None = None
    for pred_path in valmod.VAL_PRED.values():
        pred = pd.read_csv(pred_path, usecols=["timestamp"])
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred_ts = set(pred["timestamp"])
        common = pred_ts if common is None else common.intersection(pred_ts)
    if common is None:
        raise RuntimeError("validation predictions unavailable")
    return frame.loc[frame["timestamp"].isin(common)].reset_index(drop=True)


def _prepare_validation_components(frame: pd.DataFrame) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name, cfg in retest.COMPONENTS.items():
        pred = pd.read_csv(valmod.VAL_PRED[name])
        pred = pred.rename(columns={col: col.replace("_expertdq_oof_", "_expertdq_") for col in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred.loc[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        tmp_pred = OUT_DIR / f"_{name}_validation_aligned.csv"
        pred.to_csv(tmp_pred, index=False)
        components[name] = greedy.prepare_component(frame, tmp_pred, cfg, DEVICE)
    return components


def _oos_frame() -> pd.DataFrame:
    prediction_sets: list[set[pd.Timestamp]] = []
    for name, cfg in retest.COMPONENTS.items():
        pred_path = PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_path, usecols=["timestamp"])
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        prediction_sets.append(set(pred["timestamp"]))
    common = set.intersection(*prediction_sets)
    if not common:
        raise RuntimeError("OOS prediction timestamp intersection is empty")
    start, end = min(common), max(common)
    frame = retest.load_frame_current(str(start), str(end))
    frame = frame.loc[frame["timestamp"].isin(common)].reset_index(drop=True)
    if set(frame["timestamp"]) != common:
        raise RuntimeError("OOS feature/prediction timestamp contract mismatch")
    return frame


def _prepare_oos_components(frame: pd.DataFrame) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_path = PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        components[name] = greedy.prepare_component(frame, pred_path, cfg, DEVICE)
    return components


def _replay(frame: pd.DataFrame, components: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee, slip = omega._load_fee_slip()
    _, ledger = greedy.greedy_replay(
        frame,
        components,
        fee=fee,
        slip=slip,
        cost_mult=retest.COST_MULT,
        device=DEVICE,
    )
    return ledger, _gate_ledger(ledger, frame)


def _validation_result(contract: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    with _contract_override(contract):
        components = _prepare_validation_components(frame)
        raw, gated = _replay(frame, components)
    monthly = _monthly_metrics(gated)
    positive_months = sum(int(metrics["pnl"] > 0.0) for metrics in monthly.values())
    metrics = _compound_metrics(gated)
    return {
        "contract": contract,
        "metrics": metrics,
        "no_duration_gate_metrics": _compound_metrics(raw),
        "monthly": monthly,
        "positive_months": positive_months,
        "calmar_like": float(metrics["pnl"] / max(abs(metrics["mdd"]), 1e-12)),
        "exit_reasons": gated["reason"].value_counts().to_dict(),
        **_dynamic_share(frame, contract),
    }


def _select(validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive = [row for row in validation_rows if row["contract"]["name"] != BASELINE["name"]]
    eligible = [
        row for row in adaptive
        if row["metrics"]["pnl"] > 0.0
        and row["metrics"]["trades"] >= 15
        and row["positive_months"] >= 2
    ]
    if not eligible:
        eligible = [row for row in adaptive if row["metrics"]["pnl"] > 0.0 and row["metrics"]["trades"] >= 15]
    if not eligible:
        raise RuntimeError("no adaptive candidate passed the validation eligibility rules")
    return max(eligible, key=lambda row: (row["calmar_like"], row["metrics"]["pnl"]))


def _oos_result(contract: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    with _contract_override(contract):
        components = _prepare_oos_components(frame)
        raw, gated = _replay(frame, components)
    return {
        "contract": contract,
        "metrics": _compound_metrics(gated),
        "no_duration_gate_metrics": _compound_metrics(raw),
        "exit_reasons": gated["reason"].value_counts().to_dict(),
        **_dynamic_share(frame, contract),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = _aligned_validation_frame()
    validation_rows: list[dict[str, Any]] = []
    for contract in [BASELINE, *ADAPTIVE_CANDIDATES]:
        print(f"validation start {contract['name']}", flush=True)
        result = _validation_result(contract, val_frame)
        validation_rows.append(result)
        print(json.dumps(result, default=_json_default), flush=True)

    selected = _select(validation_rows)
    selected_contract = selected["contract"]
    print(f"validation selected {selected_contract['name']}", flush=True)

    oos_frame = _oos_frame()
    oos_results = {}
    for contract in (BASELINE, selected_contract):
        print(f"oos start {contract['name']}", flush=True)
        oos_results[contract["name"]] = _oos_result(contract, oos_frame)
        print(json.dumps(oos_results[contract["name"]], default=_json_default), flush=True)

    baseline_val = next(row for row in validation_rows if row["contract"]["name"] == BASELINE["name"])
    selected_oos = oos_results[selected_contract["name"]]
    baseline_oos = oos_results[BASELINE["name"]]
    promotion_pass = bool(
        selected["metrics"]["pnl"] >= baseline_val["metrics"]["pnl"]
        and selected["metrics"]["mdd"] >= baseline_val["metrics"]["mdd"]
        and selected_oos["metrics"]["pnl"] >= baseline_oos["metrics"]["pnl"]
        and selected_oos["metrics"]["mdd"] >= baseline_oos["metrics"]["mdd"]
    )
    report = {
        "experiment": "omega4_6_1_adaptive_atr_contract_20260719",
        "selection_policy": {
            "window": [str(val_frame["timestamp"].iloc[0]), str(val_frame["timestamp"].iloc[-1])],
            "rule": "adaptive only; pnl>0, trades>=15, >=2 positive months; maximize pnl/abs(mdd)",
            "oos_opened_after_selection": True,
        },
        "oos_window": [str(oos_frame["timestamp"].iloc[0]), str(oos_frame["timestamp"].iloc[-1])],
        "validation_results": validation_rows,
        "selected_contract": selected_contract,
        "oos_results": oos_results,
        "promotion_pass": promotion_pass,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note": "Research-only SLTP contract experiment; live runtime is unchanged.",
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for row in validation_rows:
        flat_rows.append({
            **row["contract"],
            **row["metrics"],
            "positive_months": row["positive_months"],
            "calmar_like": row["calmar_like"],
            "tp_dynamic_fraction": row["tp_dynamic_fraction"],
            "sl_dynamic_fraction": row["sl_dynamic_fraction"],
            "selected": row["contract"]["name"] == selected_contract["name"],
        })
    pd.DataFrame(flat_rows).to_csv(OUT_DIR / "validation_ranking.csv", index=False)
    print(f"wrote {OUT_DIR / 'report.json'}", flush=True)
    print(f"promotion_pass={promotion_pass}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
