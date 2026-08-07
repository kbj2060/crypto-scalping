#!/usr/bin/env python3
"""Priority-2 test: retrain the L4 risk sizing sidecar against the frozen v2 winner's OWN entry
set (persistence=3, quality_threshold=0.58, ATR tp=15x/sl=5x barriers, cooldown=12) instead of
the v1 fixed-barrier baseline trade set the existing sidecar
(tmp/causal_regen_20260516/omega6_risk_sidecar_20260703/risk_sidecar.pkl) was trained on.

Root cause this addresses (documented in the contract doc's L4/L6 test section): the existing
sidecar is out-of-distribution for the v2 policy's trade set (different entry filter, different
barrier widths, different decision-context features), and empirically made the v2 policy WORSE
when tested (cost3 +10.68% -> +2.72%). Train-only (2025-01-02..09-30, before SPLIT_TS), same
boundary as L2's own train/val split.
"""

from __future__ import annotations

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

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import replay_omega6_v2_l4l6_20260704 as l4l6  # noqa: E402

MODEL_ID = "omega6_risk_sidecar_v2_20260704"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_TAPE_PATH = ROOT / "tmp/causal_regen_20260516/omega6_train_period_decision_tape_20260704/tape.parquet"

FROZEN_KW = dict(
    persistence_bars=3,
    quality_threshold=0.58,
    tp_atr_mult=15.0,
    sl_atr_mult=5.0,
    cooldown_bars=12,
    fixed_margin=0.30,
    fixed_leverage=2.0,
)
FEE = 0.00020
SLIP = 0.00050

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
MDD_CAP_PCT = -20.0


def load_train_tape() -> pd.DataFrame:
    tape = pd.read_parquet(TRAIN_TAPE_PATH)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    return tape.sort_values("i").reset_index(drop=True)


def get_v2_trades(tape: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the frozen v2 winner's entry/exit logic over the train tape and return the trade
    list plus the tape (with quality threshold applied) for feature reconstruction."""
    tape_qt = v2.apply_quality_threshold(tape, FROZEN_KW["quality_threshold"])
    cfg = v2.VariantConfig(
        name="v2_train_trades",
        tp_mode="atr_scaled",
        tp_atr_mult=FROZEN_KW["tp_atr_mult"],
        sl_atr_mult=FROZEN_KW["sl_atr_mult"],
        sizing_mode="fixed",
        fixed_margin=FROZEN_KW["fixed_margin"],
        fixed_leverage=FROZEN_KW["fixed_leverage"],
        cooldown_bars=FROZEN_KW["cooldown_bars"],
        quality_threshold=FROZEN_KW["quality_threshold"],
        persistence_bars=FROZEN_KW["persistence_bars"],
    )
    start, end = tape_qt["timestamp"].min(), tape_qt["timestamp"].max()
    result = v2.run_variant(tape_qt, cfg, start=start, end=end)
    return tape_qt, result["_trade_list"]


def build_features_and_targets(tape_qt: pd.DataFrame, trades: list[dict[str, Any]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    close = tape_qt["close"].to_numpy(dtype=np.float64)
    open_ = tape_qt["open"].to_numpy(dtype=np.float64)
    n = len(tape_qt)
    feature_rows = []
    targets = []
    sides = []
    for t in trades:
        entry_i, exit_i, side = t["entry_i"], t["exit_i"], t["side"]
        row_features = l4l6.build_l4_features(tape_qt.iloc[[entry_i]], FROZEN_KW["quality_threshold"])
        entry_price = open_[min(entry_i + 1, n - 1)] * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
        exit_price = close[exit_i] * (1.0 - SLIP if side > 0 else 1.0 + SLIP)
        raw_exit = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        net_per_notional = raw_exit - FEE - FEE
        feature_rows.append(row_features)
        targets.append(net_per_notional)
        sides.append(side)
    x = pd.concat(feature_rows, ignore_index=True)
    return x, np.asarray(targets, dtype=np.float64), np.asarray(sides, dtype=np.int64)


def fit_side_split(x: pd.DataFrame, y: np.ndarray, side: np.ndarray, *, seed: int) -> dict[int, HistGradientBoostingRegressor]:
    models = {}
    for s in (-1, 1):
        mask = side == s
        if int(mask.sum()) < 20:
            raise RuntimeError(f"too few side={s} train trades ({int(mask.sum())}) to fit")
        model = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.05, random_state=int(seed))
        model.fit(x.loc[mask], y[mask])
        models[s] = model
    return models


def replay_with_mapping(x: pd.DataFrame, trades: list[dict[str, Any]], tape_qt: pd.DataFrame, models: dict, mapping: dict, q50: float, iqr: float) -> dict[str, float]:
    close = tape_qt["close"].to_numpy(dtype=np.float64)
    open_ = tape_qt["open"].to_numpy(dtype=np.float64)
    n = len(tape_qt)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    log_growth = 0.0
    for idx, t in enumerate(trades):
        side = t["side"]
        features_row = x.iloc[[idx]]
        score = float(models[side].predict(features_row)[0])
        z = float(np.clip((score - q50) / max(iqr, 1e-8), -8.0, 8.0))
        unit = 1.0 / (1.0 + np.exp(-float(mapping["temp"]) * z))
        scale = float(mapping["min_scale"]) + (float(mapping["max_scale"]) - float(mapping["min_scale"])) * unit
        base_margin = FROZEN_KW["fixed_margin"]
        margin = float(np.clip(base_margin * scale, float(mapping["floor"]), float(mapping["cap"])))
        margin *= float(mapping["long_scale"]) if side > 0 else float(mapping["short_scale"])
        margin = float(np.clip(margin, float(mapping["floor"]), float(mapping["cap"])))
        unit_lev = 1.0 / (1.0 + np.exp(-float(mapping["leverage_temp"]) * z))
        leverage = float(mapping["leverage_min"]) + (float(mapping["leverage_max"]) - float(mapping["leverage_min"])) * unit_lev
        leverage = float(np.clip(leverage, float(mapping["leverage_floor"]), float(mapping["leverage_cap"])))
        notional = margin * leverage
        entry_i, exit_i = t["entry_i"], t["exit_i"]
        entry_price = open_[min(entry_i + 1, n - 1)] * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
        exit_price = close[exit_i] * (1.0 - SLIP if side > 0 else 1.0 + SLIP)
        raw_exit = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        net = raw_exit - FEE - FEE
        before = cash
        cash = cash * (1.0 + net * notional)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        log_growth += float(np.log(max(cash / max(before, 1e-12), 1e-6)))
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "log_growth_sum": float(log_growth), "trades": len(trades)}


def main() -> int:
    tape = load_train_tape()
    tape_qt, trades = get_v2_trades(tape)
    print(f"train trades: {len(trades)}", flush=True)
    if len(trades) < 60:
        raise RuntimeError(f"too few train trades ({len(trades)}) to fit a sidecar")

    x, y, side = build_features_and_targets(tape_qt, trades)
    models = fit_side_split(x, y, side, seed=260704)
    all_scores = np.concatenate([models[int(s)].predict(x.iloc[[i]]) for i, s in enumerate(side)])
    q50 = float(np.median(all_scores))
    iqr = float(np.subtract(*np.percentile(all_scores, [75, 25])))
    iqr = abs(iqr) if abs(iqr) > 1e-8 else 1.0

    candidates = []
    for temp, floor, cap in itertools.product(GRID_TEMP, GRID_FLOOR, GRID_CAP):
        if floor >= cap:
            continue
        mapping = {**MAPPING_DEFAULTS, "temp": temp, "floor": floor, "cap": cap}
        metrics = replay_with_mapping(x, trades, tape_qt, models, mapping, q50, iqr)
        candidates.append((mapping, metrics))

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
        "lineage": "Refit against the frozen v2 winner's own entry set (persistence=3, "
        "quality_threshold=0.58, ATR tp=15x/sl=5x, cooldown=12), train-only (2025-01-02..09-30), "
        "to fix out-of-distribution mismatch vs the v1-baseline-trained sidecar documented in "
        "docs/model_contracts/omega6_synthesis_v1_20260703_contract.md L4/L6 test section.",
        "train_window": {"start": str(tape_qt["timestamp"].min()), "end": str(tape_qt["timestamp"].max())},
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
