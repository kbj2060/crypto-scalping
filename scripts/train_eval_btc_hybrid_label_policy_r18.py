#!/usr/bin/env python3
"""BTC 5m policy with trajectory primary labels and independent barrier/structure auxiliaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_independent_catboost_r15 import (  # noqa: E402
    DEV_END, HALF_LIFE_DAYS, TRAIN_DATA, VAL_DATA, feature_sets, labels_for, market,
    recency_weights, simulate, target_values, values,
)
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

TRIPLE = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
ZIGZAG = ROOT / "data/splits/year_oos/btc_5m_zigzag_correctedvol_labels_20260806.parquet"
OUT = ROOT / "tmp/btc_hybrid_label_policy_r18"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
ENTRIES, LARGES, BARRIER_GATES = (.55, .60, .65, .70), (.75, .80, .85), (.40, .50, .60)
MIN_EVENTS = 15


def auxiliary_labels(start: str, end: str) -> pd.DataFrame:
    triple = pd.read_parquet(TRIPLE)
    zigzag = pd.read_parquet(ZIGZAG)
    for frame in (triple, zigzag):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    columns = {"trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"}
    if not columns <= set(triple):
        raise ValueError("triple-barrier soft-label contract mismatch")
    action_col = "zigzag_correctedvol_action"
    if action_col not in zigzag:
        raise ValueError("corrected-vol Zigzag contract mismatch")
    labels = triple.merge(zigzag[["timestamp", action_col]], on="timestamp", how="inner", validate="one_to_one")
    return labels.loc[labels["timestamp"].between(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True))].reset_index(drop=True)


def model(**extra: object) -> CatBoostRegressor:
    return CatBoostRegressor(iterations=350, depth=7, learning_rate=.05, loss_function="RMSE", random_seed=270705, verbose=False, thread_count=-1, **extra)


def fit_primary(train: pd.DataFrame, direction_features: list[str], quality_features: list[str]) -> tuple[CatBoostRegressor, CatBoostRegressor]:
    direction, quality = target_values(train); weight = recency_weights(train)
    return (model().fit(values(train, direction_features), direction, sample_weight=weight), model().fit(values(train, quality_features), quality, sample_weight=weight))


def fit_auxiliary(base: pd.DataFrame, labels: pd.DataFrame, direction_features: list[str], quality_features: list[str]) -> tuple[CatBoostRegressor, CatBoostRegressor, CatBoostRegressor]:
    frame = base.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    if len(frame) < .98 * len(base):
        raise ValueError("auxiliary labels do not cover the causal feature frame")
    weight = np.exp2(-(pd.to_datetime(frame["timestamp"], utc=True).max() - pd.to_datetime(frame["timestamp"], utc=True)).dt.total_seconds().to_numpy() / 86400.0 / HALF_LIFE_DAYS).astype(np.float32); weight /= weight.mean()
    barrier_direction = (frame["trade_outcome_soft_long"] - frame["trade_outcome_soft_short"]).to_numpy(dtype=np.float32)
    barrier_quality = (1.0 - frame["trade_outcome_soft_cash"]).to_numpy(dtype=np.float32)
    raw = frame["zigzag_correctedvol_action"].to_numpy()
    unique = set(np.unique(raw).tolist())
    if not unique <= {0, 1, 2}:
        raise ValueError(f"unexpected corrected-vol Zigzag actions: {sorted(unique)}")
    structure_direction = np.where(raw == 1, 1.0, np.where(raw == 2, -1.0, 0.0)).astype(np.float32)
    return (model().fit(base.loc[frame.index, direction_features].replace([np.inf, -np.inf], np.nan).fillna(0.0), barrier_direction, sample_weight=weight), model().fit(base.loc[frame.index, quality_features].replace([np.inf, -np.inf], np.nan).fillna(0.0), barrier_quality, sample_weight=weight), model().fit(base.loc[frame.index, direction_features].replace([np.inf, -np.inf], np.nan).fillna(0.0), structure_direction, sample_weight=weight))


def predict(primary_direction: CatBoostRegressor, primary_quality: CatBoostRegressor, barrier_direction: CatBoostRegressor, barrier_quality: CatBoostRegressor, structure: CatBoostRegressor, base: pd.DataFrame, direction_features: list[str], quality_features: list[str], entry: float, large: float, barrier_gate: float) -> tuple[np.ndarray, np.ndarray]:
    states = np.array([-.30, -.15, .0, .15, .30], dtype=np.float32)
    dv = base[direction_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    qv = base[quality_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    bd, bq, zs = barrier_direction.predict(dv), np.clip(barrier_quality.predict(qv), 0, 1), structure.predict(dv)
    current = 0.; margins=[]; qualities=[]
    for i, (drow, qrow) in enumerate(zip(dv, qv)):
        state = states[np.argmin(np.abs(states-current))]
        primary_d = float(primary_direction.predict(np.append(drow, state)[None, :])[0]); primary_q = float(np.clip(primary_quality.predict(np.append(qrow, state)[None, :])[0], 0, 1))
        agree_barrier = np.sign(primary_d) == np.sign(bd[i])
        agree_structure = abs(zs[i]) < .10 or np.sign(primary_d) == np.sign(zs[i])
        magnitude = .30 if primary_q >= large else .15 if primary_q >= entry else .0
        current = float(np.sign(primary_d) * magnitude) if bq[i] >= barrier_gate and agree_barrier and agree_structure else 0.
        margins.append(current); qualities.append(primary_q)
    return np.asarray(margins), np.asarray(qualities)


def main() -> int:
    direction_features, quality_features = feature_sets(); all_features = list(dict.fromkeys([*direction_features, *quality_features]))
    base24 = read_window(TRAIN_DATA, all_features, "2024-01-01", "2024-12-31 23:59:59+00:00"); base25 = read_window(VAL_DATA, all_features, "2025-01-01", DEV_END)
    base = pd.concat([base24, base25], ignore_index=True); teacher = pd.concat([labels_for(base24), labels_for(base25)], ignore_index=True); state = base.merge(teacher, left_on="timestamp", right_on="decision_timestamp", how="inner")
    aux = auxiliary_labels("2024-01-01", DEV_END)
    pdir, pqual = fit_primary(state, direction_features, quality_features); bdir, bqual, structure = fit_auxiliary(base, aux, direction_features, quality_features)
    validation, returns = market(read_window(VAL_DATA, all_features, VAL_START, VAL_END)); rows=[]
    for entry in ENTRIES:
        for large in LARGES:
            for gate in BARRIER_GATES:
                if large <= entry: continue
                margin, quality = predict(pdir,pqual,bdir,bqual,structure,validation,direction_features,quality_features,entry,large,gate); metrics=simulate(margin,returns)
                rows.append({"entry":entry,"large":large,"barrier_gate":gate,**metrics,"mean_primary_quality":float(quality.mean()),"selection_eligible":bool(metrics["action_events"]>=MIN_EVENTS and metrics["pnl_pct"]>0)})
    grid=pd.DataFrame(rows); candidates=grid.loc[grid.selection_eligible]; selected=None if candidates.empty else candidates.sort_values(["action_events","pnl_pct"],ascending=[True,False]).iloc[0].to_dict()
    OUT.mkdir(parents=True,exist_ok=True); grid.to_csv(OUT/"validation_grid.csv",index=False)
    report={"diagnostic_only":True,"architecture":"independent trajectory primary, triple-barrier agreement gate, corrected-vol Zigzag structure veto","train_period":["2024-01-01",DEV_END],"validation_period":[VAL_START,VAL_END],"future_prices_used_only_for_labels":True,"auxiliary_labels_not_execution_inputs":True,"selected":selected,"oos_opened":False,"promotion_eligible":False}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
