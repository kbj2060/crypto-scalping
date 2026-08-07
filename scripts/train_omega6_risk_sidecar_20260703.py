#!/usr/bin/env python3
"""Refit the Omega6 L4 risk sizing sidecar against Omega6's own L2 decision trace.

Fixes contamination-audit Finding 2 in
docs/model_contracts/omega6_synthesis_v1_20260703_contract.md: the previously-reused
omega4_4_v18-lineage sidecar was selected on a window overlapping Omega6's fresh-forward
validation period (2025-10-01..12-31). This script builds risk features and realized trade
outcomes purely from Omega6 L2's own train split (timestamp < SPLIT_TS = 2025-10-01, same
boundary as scripts/train_eval_omega6_tabm_3head_20260703.py) and fits + selects the sidecar
mapping without ever reading validation or OOS rows -- matching the target-construction
convention of scripts/train_eval_omega4_2_risk_sidecar_20260622.py (net_per_notional realized
return, target_mae_penalty=0.0 default).

Output artifact matches the exact schema trading_bot_modules/omega6_live.py::_validate_sidecar
and ::_sidecar_sizing expect: {model: {-1: hgb, 1: hgb}, feature_columns, train_score_q50,
train_score_iqr, selected_mapping, risk_feature_mode="parent_outputs", side_split_model=True,
dynamic_leverage=True}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega6_tabm_3head_20260703 as omega6_tabm  # noqa: E402
from trading_bot_modules.omega6_live import (  # noqa: E402
    L4_BASELINE_LEVERAGE,
    L4_BASELINE_NOTIONAL,
    L5_BASE_SL_PRICE_MOVE,
    L5_BASE_TP_PRICE_MOVE,
    L5_MAX_HOLD_BARS,
    Omega6LiveAdapter,
)

MODEL_ID = "omega6_risk_sidecar_20260703"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = omega6_tabm.SPLIT_TS  # 2025-10-01, identical boundary to the L2 trainer
CONTEXT_BARS = 260  # matches scripts/backtest_omega6_synthesis_fresh_forward_20260703.py

MAPPING_DEFAULTS = {
    "min_scale": 1.0,
    "max_scale": 2.0,
    "long_scale": 1.0,
    "short_scale": 1.0,
    "leverage_min": 1.5,
    "leverage_max": 2.5,
    "leverage_temp": 1.5,
    "leverage_floor": 1.0,
    "leverage_cap": 3.0,
    "long_leverage_scale": 1.0,
    "short_leverage_scale": 1.0,
}
GRID_TEMP = (1.0, 1.5, 2.0, 2.5)
GRID_FLOOR = (0.15, 0.20, 0.30)
GRID_CAP = (0.30, 0.40, 0.50, 0.60)
# Hard MDD cap on the train-only mapping-selection replay: candidates whose train MDD is worse
# than this are rejected outright, not just deprioritized. Added per user request after the
# uncapped run (Fixes Applied #1 in the contract doc) selected cap=0.6 and pushed val MDD from
# -16.5% to -28.5% -- log_growth_sum was previously the primary key with MDD only a tiebreaker.
MDD_CAP_PCT = -20.0


def _simulate_baseline_trades(
    frame: pd.DataFrame,
    adapter: Omega6LiveAdapter,
    arrays: dict[str, np.ndarray],
    *,
    start_idx: int,
    end_idx: int,
    fee: float,
    slip: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Walk the given index range bar-by-bar with Omega6's own primary/fallback parent,
    sizing every trade at the fixed L4 baseline (no sidecar involved) and exiting via the
    L5 barrier/time-stop rule. Records (risk_features_row, net_per_notional, side) for
    every closed trade -- this is the train-only dataset the sidecar regressor learns from."""
    rows: list[pd.DataFrame] = []
    targets: list[float] = []
    sides: list[int] = []
    pos = 0
    entry_price = 0.0
    hold_start = 0
    pending_features: pd.DataFrame | None = None
    take_profit = 0.0
    stop_loss = 0.0
    i = start_idx
    while i < end_idx:
        if pos == 0:
            window = frame.iloc[max(0, i - CONTEXT_BARS + 1) : i + 1]
            primary_out = adapter._predict_parent(adapter.primary, window)
            if primary_out["side"] != 0:
                parent_out = primary_out
            else:
                parent_out = adapter._predict_parent(adapter.fallback, window)
            if parent_out["side"] == 0:
                i += 1
                continue
            side = int(parent_out["side"])
            atr_pct = adapter._atr_pct(window, adapter.atr_window)
            features = adapter._risk_features(parent_out, atr_pct)
            take_profit = float(L5_BASE_TP_PRICE_MOVE) * float(L4_BASELINE_LEVERAGE)
            stop_loss = float(L5_BASE_SL_PRICE_MOVE) * float(L4_BASELINE_LEVERAGE)
            filled, px, entry_fee, _route = omega._try_execution(arrays, i, side, entry=True, fee_base=fee, slip_base=slip)
            if not filled:
                i += 1
                continue
            pos = side
            entry_price = float(px)
            hold_start = i
            pending_features = features
            i += 1
            continue
        px = float(arrays["close"][i])
        raw = (
            (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
            if pos > 0
            else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        )
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and raw >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and raw <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= L5_MAX_HOLD_BARS:
            reason = "time_stop"
        if reason or i == end_idx - 1:
            filled, exit_px, exit_fee, _route = omega._try_execution(arrays, i, pos, entry=False, fee_base=fee, slip_base=slip)
            if filled:
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                net_per_notional = float(raw_exit - fee - exit_fee)
                rows.append(pending_features)
                targets.append(net_per_notional)
                sides.append(int(pos))
                pos = 0
        i += 1
    x = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return x, np.asarray(targets, dtype=np.float64), np.asarray(sides, dtype=np.int64)


def _fit_side_split(x: pd.DataFrame, y: np.ndarray, side: np.ndarray, *, seed: int) -> dict[int, HistGradientBoostingRegressor]:
    models: dict[int, HistGradientBoostingRegressor] = {}
    for s in (-1, 1):
        mask = side == s
        if int(mask.sum()) < 20:
            raise RuntimeError(f"Omega6 risk sidecar: too few side={s} train trades ({int(mask.sum())}) to fit")
        model = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.05, random_state=int(seed))
        model.fit(x.loc[mask], y[mask])
        models[s] = model
    return models


def _replay_with_mapping(
    frame: pd.DataFrame,
    adapter_features: list[tuple[pd.DataFrame, int, int, int]],
    arrays: dict[str, np.ndarray],
    models: dict[int, HistGradientBoostingRegressor],
    mapping: dict[str, float],
    q50: float,
    iqr: float,
    *,
    fee: float,
    slip: float,
) -> dict[str, float]:
    """Re-simulate the SAME train-split entries (fixed entry_i/side from the baseline pass)
    but size each with the candidate mapping applied to the fitted model's score -- used only
    to rank mapping candidates, never touching validation/OOS rows."""
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    log_growth = 0.0
    for features, side, entry_i, exit_i in adapter_features:
        score = float(models[side].predict(features)[0])
        z = float(np.clip((score - q50) / max(iqr, 1e-8), -8.0, 8.0))
        unit = 1.0 / (1.0 + np.exp(-float(mapping["temp"]) * z))
        scale = float(mapping["min_scale"]) + (float(mapping["max_scale"]) - float(mapping["min_scale"])) * unit
        base_margin = float(L4_BASELINE_NOTIONAL) / max(float(L4_BASELINE_LEVERAGE), 1e-8)
        margin = float(np.clip(base_margin * scale, float(mapping["floor"]), float(mapping["cap"])))
        margin *= float(mapping["long_scale"]) if side > 0 else float(mapping["short_scale"])
        margin = float(np.clip(margin, float(mapping["floor"]), float(mapping["cap"])))
        unit_lev = 1.0 / (1.0 + np.exp(-float(mapping["leverage_temp"]) * z))
        leverage = float(mapping["leverage_min"]) + (float(mapping["leverage_max"]) - float(mapping["leverage_min"])) * unit_lev
        leverage = float(np.clip(leverage, float(mapping["leverage_floor"]), float(mapping["leverage_cap"])))
        notional = margin * leverage
        entry_price = float(arrays["open"][min(entry_i + 1, len(arrays["open"]) - 1)])
        exit_price = float(arrays["close"][exit_i])
        raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        net = raw_exit - fee - fee
        before = cash
        cash = cash * (1.0 + net * notional)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        log_growth += float(np.log(max(cash / max(before, 1e-12), 1e-6)))
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "log_growth_sum": float(log_growth), "trades": len(adapter_features)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=260703)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument(
        "--primary-bundle",
        default=str(ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_primary/true_3head_tabm_bundle.pt"),
    )
    ap.add_argument(
        "--fallback-bundle",
        default=str(ROOT / "tmp/causal_regen_20260516/omega6_true_3head_tabm_20260703_fallback/true_3head_tabm_bundle.pt"),
    )
    ap.add_argument(
        "--tcn-gate",
        default=str(ROOT / "tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_20260703/tcn_seq_gate_L24_flat.pt"),
    )
    ap.add_argument(
        "--seed-risk-sidecar",
        default=str(
            ROOT
            / "tmp/causal_regen_20260516"
            / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
            / "risk_sidecar.pkl"
        ),
        help="Only used to bootstrap Omega6LiveAdapter construction before this script overwrites its sidecar; not used as a training input.",
    )
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    adapter = Omega6LiveAdapter(
        primary_bundle_path=args.primary_bundle,
        fallback_bundle_path=args.fallback_bundle,
        tcn_gate_path=args.tcn_gate,
        risk_sidecar_path=args.seed_risk_sidecar,
        device=device,
    )

    train, eval_df, _overlay = omega._load_omega_frames()
    combined = pd.concat([train, eval_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    fee, slip = omega._load_fee_slip()
    arrays = {c: pd.to_numeric(combined[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}

    train_start_idx = CONTEXT_BARS
    train_end_idx = int(combined["timestamp"].searchsorted(SPLIT_TS, side="left"))
    if train_end_idx <= train_start_idx:
        raise RuntimeError("Omega6 risk sidecar: empty train range")

    x, y, side = _simulate_baseline_trades(combined, adapter, arrays, start_idx=train_start_idx, end_idx=train_end_idx, fee=fee, slip=slip)
    if len(x) < 60:
        raise RuntimeError(f"Omega6 risk sidecar: too few train trades ({len(x)}) to fit a regressor")

    models = _fit_side_split(x, y, side, seed=int(args.seed))
    all_scores = np.concatenate([models[int(s)].predict(x.loc[[i]]) for i, s in enumerate(side)])
    q50 = float(np.median(all_scores))
    iqr = float(np.subtract(*np.percentile(all_scores, [75, 25])))
    iqr = abs(iqr) if abs(iqr) > 1e-8 else 1.0

    # Re-derive (features, side, entry_i, exit_i) pairs for mapping-selection replay by
    # re-running the same baseline pass, this time keeping bar indices instead of just targets.
    replay_rows: list[tuple[pd.DataFrame, int, int, int]] = []
    pos = 0
    entry_price = 0.0
    hold_start = 0
    pending_features = None
    take_profit = stop_loss = 0.0
    i = train_start_idx
    while i < train_end_idx:
        if pos == 0:
            window = combined.iloc[max(0, i - CONTEXT_BARS + 1) : i + 1]
            primary_out = adapter._predict_parent(adapter.primary, window)
            parent_out = primary_out if primary_out["side"] != 0 else adapter._predict_parent(adapter.fallback, window)
            if parent_out["side"] == 0:
                i += 1
                continue
            side_i = int(parent_out["side"])
            atr_pct = adapter._atr_pct(window, adapter.atr_window)
            features = adapter._risk_features(parent_out, atr_pct)
            take_profit = float(L5_BASE_TP_PRICE_MOVE) * float(L4_BASELINE_LEVERAGE)
            stop_loss = float(L5_BASE_SL_PRICE_MOVE) * float(L4_BASELINE_LEVERAGE)
            filled, px, _f, _r = omega._try_execution(arrays, i, side_i, entry=True, fee_base=fee, slip_base=slip)
            if not filled:
                i += 1
                continue
            pos = side_i
            entry_price = float(px)
            hold_start = i
            pending_features = features
            i += 1
            continue
        px = float(arrays["close"][i])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and raw >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and raw <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= L5_MAX_HOLD_BARS:
            reason = "time_stop"
        if reason or i == train_end_idx - 1:
            replay_rows.append((pending_features, pos, hold_start, i))
            pos = 0
        i += 1

    candidates: list[tuple[dict[str, float], dict[str, Any]]] = []
    for temp, floor, cap in itertools.product(GRID_TEMP, GRID_FLOOR, GRID_CAP):
        if floor >= cap:
            continue
        mapping = {**MAPPING_DEFAULTS, "temp": temp, "floor": floor, "cap": cap}
        metrics = _replay_with_mapping(combined, replay_rows, arrays, models, mapping, q50, iqr, fee=fee, slip=slip)
        candidates.append((mapping, metrics))

    # Hard MDD cap: only rank by log_growth_sum among candidates whose train-only MDD is no
    # worse than MDD_CAP_PCT. If nothing qualifies, fall back to the least-bad-MDD candidate
    # (log_growth_sum as tiebreaker) rather than silently accepting an uncapped MDD.
    eligible = [(m, r) for m, r in candidates if r["mdd"] >= MDD_CAP_PCT]
    pool = eligible if eligible else candidates
    key_fn = (lambda pair: (pair[1]["log_growth_sum"], -pair[1]["mdd"])) if eligible else (lambda pair: (-pair[1]["mdd"], pair[1]["log_growth_sum"]))
    best_mapping, best_metrics = max(pool, key=key_fn)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "model": models,
        "feature_columns": list(x.columns),
        "train_score_q50": q50,
        "train_score_iqr": iqr,
        "selected_mapping": best_mapping,
        "risk_feature_mode": "parent_outputs",
        "side_split_model": True,
        "dynamic_leverage": True,
        "model_id": MODEL_ID,
        "lineage": "Refit from scratch against Omega6's own L2 (primary/fallback) decision trace, "
        "train-split only (timestamp < 2025-10-01), fixing contamination-audit Finding 2 in "
        "docs/model_contracts/omega6_synthesis_v1_20260703_contract.md.",
        "train_window": {"start": str(combined.iloc[train_start_idx]["timestamp"]), "end": str(combined.iloc[train_end_idx - 1]["timestamp"])},
        "grid_search": {
            "temp": GRID_TEMP,
            "floor": GRID_FLOOR,
            "cap": GRID_CAP,
            "selection_scope": "train_only",
            "mdd_cap_pct": MDD_CAP_PCT,
            "n_candidates": len(candidates),
            "n_eligible_under_mdd_cap": len(eligible),
            "selected": best_metrics,
        },
    }
    out_path = OUT_DIR / "risk_sidecar.pkl"
    with out_path.open("wb") as f:
        pickle.dump(sidecar, f)
    report = {
        "model_id": MODEL_ID,
        "train_window": sidecar["train_window"],
        "n_train_trades": int(len(x)),
        "selected_mapping": best_mapping,
        "selected_metrics_train_only": best_metrics,
        "artifact": str(out_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
