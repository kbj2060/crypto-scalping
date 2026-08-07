#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, FullyLearnedGovernorConfig, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_deep_tabular_parent_mdd_20260514 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_selective_hazard_pruning_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_selective_hazard_pruning_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_selective_hazard_pruning_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_selective_hazard_pruning_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_selective_hazard_pruning_20260514_grid.csv"

warnings.simplefilter("ignore", PerformanceWarning)


MICRO_COLS = (
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "trade_intensity",
    "big_trade_ratio",
    "whale_retail_ratio",
    "smart_money_flow",
)
RISK_COLS = (
    "volatility_z",
    "garch_vol_z",
    "rogers_satchell_vol",
    "amihud_illiquidity_z",
    "liquidity_vacuum",
    "execution_quality",
    "jump_z",
    "evt_tail_flag",
    "evt_excess_z",
    "funding_pressure",
    "funding_price_divergence",
    "long_squeeze_risk",
    "crowding_pressure",
)
TREND_COLS = (
    "mom_1d",
    "mom_3d",
    "mom_21d",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "m7_expected_ret",
    "m7_composite_score",
    "ai_dir_edge",
)
DELTA_COLS = tuple(dict.fromkeys(MICRO_COLS + RISK_COLS + TREND_COLS))


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    threshold: float
    max_cut: float
    min_scale: float
    sim_weight: float
    cost_weight: float
    super_q: float
    super_cap: float


def _safe_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(float(default))
        .to_numpy(dtype=np.float64)
    )


def _risk_cost_components(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    risk = (
        0.18 * np.clip(np.abs(_safe_array(df, "volatility_z")) / 3.0, 0.0, 1.5)
        + 0.16 * np.clip(np.maximum(_safe_array(df, "garch_vol_z") - 0.5, 0.0) / 2.5, 0.0, 1.5)
        + 0.14 * np.clip(np.abs(_safe_array(df, "jump_z")) / 4.0, 0.0, 1.5)
        + 0.12 * np.clip(_safe_array(df, "evt_tail_flag"), 0.0, 1.0)
        + 0.12 * np.clip(_safe_array(df, "long_squeeze_risk"), 0.0, 1.5)
        + 0.10 * np.clip(_safe_array(df, "crowding_pressure"), 0.0, 1.5)
        + 0.10 * np.clip(_safe_array(df, "liquidity_vacuum"), 0.0, 1.5)
        + 0.08 * np.clip(np.abs(_safe_array(df, "funding_pressure")) / 3.0, 0.0, 1.5)
    )
    cost = (
        0.30 * np.clip(np.abs(_safe_array(df, "amihud_illiquidity_z")) / 3.0, 0.0, 1.5)
        + 0.25 * np.clip(_safe_array(df, "liquidity_vacuum"), 0.0, 1.5)
        + 0.20 * np.clip(np.maximum(-_safe_array(df, "execution_quality"), 0.0), 0.0, 1.5)
        + 0.15 * np.clip(np.abs(_safe_array(df, "trade_intensity")) / 4.0, 0.0, 1.5)
        + 0.10 * np.clip(np.abs(_safe_array(df, "big_trade_ratio")) / 4.0, 0.0, 1.5)
    )
    return np.clip(risk, 0.0, 1.5), np.clip(cost, 0.0, 1.5)


def _base_feature_frame(df: pd.DataFrame, decisions: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if set(feature_cols).issubset(df.columns):
        base = df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
        if "side_hint" in base.columns:
            base["side_hint"] = 0.0
    else:
        base = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
    out = base.copy()
    side = decisions["side"].astype(float).to_numpy()
    out["parent_side"] = side
    out["parent_abs_notional"] = np.abs(pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    out["parent_leverage"] = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    out["parent_tp"] = pd.to_numeric(decisions["take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["parent_sl"] = pd.to_numeric(decisions["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["parent_quality"] = pd.to_numeric(decisions.get("quality_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["parent_confidence"] = pd.to_numeric(decisions.get("confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    risk, cost = _risk_cost_components(df)
    out["hazard_risk_pressure"] = risk
    out["hazard_cost_pressure"] = cost

    for col in DELTA_COLS:
        arr = _safe_array(df, col, 0.0)
        if col in MICRO_COLS or col in TREND_COLS:
            out[f"side_{col}"] = side * arr
        for lag in (1, 3, 6, 12):
            shifted = np.roll(arr, lag)
            shifted[:lag] = arr[:lag]
            out[f"d{lag}_{col}"] = arr - shifted
            if col in MICRO_COLS or col in TREND_COLS:
                out[f"side_d{lag}_{col}"] = side * (arr - shifted)
    return out.replace([np.inf, -np.inf], np.nan)


def _variant() -> v45.LayerVariant:
    return v45.LayerVariant("alpha1_selective_hazard_pruning", "hazard_pruning", alpha1._overlay_alpha1())


def _backtest(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    *,
    cost_mult: float,
    record: bool = False,
) -> dict[str, Any]:
    cfg = dict(parent["config"])
    return v45.backtest_variant(
        df,
        parent,
        jackpot_model,
        add_cfg,
        q,
        _variant(),
        fee=float(cfg["fee"]),
        slip=float(cfg["slip"]),
        cost_mult=float(cost_mult),
        decisions=decisions,
        record=record,
    )


def _records_to_entry_set(df: pd.DataFrame, decisions: pd.DataFrame, records: list[dict[str, Any]], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.DataFrame(records)
    if ledger.empty:
        return pd.DataFrame(), ledger
    ledger = ledger[ledger["owner"].astype(str) == "v21_2"].copy()
    ts = pd.to_datetime(df["timestamp"])
    index_by_ts = pd.Series(np.arange(len(df), dtype=np.int64), index=ts)
    entry_ts = pd.to_datetime(ledger["entry_signal_timestamp"])
    idx = entry_ts.map(index_by_ts).dropna().astype(int)
    ledger = ledger.loc[idx.index].reset_index(drop=True)
    idx_arr = idx.to_numpy(dtype=np.int64)
    x_all = _base_feature_frame(df, decisions, feature_cols)
    x = x_all.iloc[idx_arr].reset_index(drop=True)
    ledger["row_idx"] = idx_arr
    return x, ledger


def _label_hazards(ledger: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    realized = pd.to_numeric(ledger["realized_net_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mae = pd.to_numeric(ledger["mae_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fee = pd.to_numeric(ledger["fee_entry_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) + pd.to_numeric(ledger["fee_exit_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    exit_reason = ledger.get("exit_reason", pd.Series("", index=ledger.index)).astype(str).to_numpy()
    if len(realized) == 0:
        raise ValueError("no parent entry ledger rows")
    realized_cut = float(np.quantile(realized, 0.08))
    mae_cut = float(np.quantile(mae, 0.08))
    fee_cut = float(np.quantile(fee, 0.75))
    hazard = (
        (realized <= min(realized_cut, -2.8))
        | (mae <= min(mae_cut, -3.2))
        | ((np.char.find(exit_reason.astype(str), "stop_loss") >= 0) & (realized <= -2.2))
    )
    severity = np.clip((-realized / max(abs(realized_cut), 1.0)) + (-mae / max(abs(mae_cut), 1.0)) + 0.25 * (fee / max(fee_cut, 1e-6)), 0.2, 6.0)
    weights = np.where(hazard, 2.0 + severity, 0.45 + 0.20 * np.clip(realized, 0.0, 8.0))
    meta = {
        "n": int(len(realized)),
        "hazards": int(np.sum(hazard)),
        "hazard_rate": float(np.mean(hazard)),
        "realized_cut_pct": realized_cut,
        "mae_cut_pct": mae_cut,
        "fee_cut_pct": fee_cut,
        "realized_mean_pct": float(np.mean(realized)),
        "realized_min_pct": float(np.min(realized)),
    }
    return hazard.astype(np.int64), weights.astype(np.float64), meta


def _delta_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith("d") or c.startswith("side_d")]


def _fit_hazard_model(x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    cols = list(x.columns)
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x)
    clf = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.18,
        early_stopping=False,
        random_state=20260514,
    )
    if np.unique(y).size < 2:
        raise ValueError("hazard labels contain one class only")
    clf.fit(x_imp, y, sample_weight=weights)
    dcols = _delta_cols(cols)
    d_idx = np.asarray([cols.index(c) for c in dcols], dtype=np.int64)
    d_train = x_imp[:, d_idx] if len(d_idx) else np.zeros((len(x_imp), 1), dtype=np.float64)
    d_mean = np.nanmean(d_train, axis=0)
    d_std = np.nanstd(d_train, axis=0) + 1e-8
    z = (d_train - d_mean) / d_std
    bad = z[y.astype(bool)]
    centroid = np.nanmean(bad, axis=0) if len(bad) else np.zeros(z.shape[1], dtype=np.float64)
    centroid_norm = float(np.linalg.norm(centroid) + 1e-8)
    return {
        "cols": cols,
        "imputer": imputer,
        "classifier": clf,
        "delta_idx": d_idx,
        "delta_mean": d_mean,
        "delta_std": d_std,
        "bad_centroid": centroid,
        "bad_centroid_norm": centroid_norm,
    }


def _hazard_components(model: dict[str, Any], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = list(model["cols"])
    x_in = x.reindex(columns=cols)
    x_imp = model["imputer"].transform(x_in)
    proba = model["classifier"].predict_proba(x_imp)
    classes = list(model["classifier"].classes_)
    p_bad = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(x_imp), dtype=np.float64)
    d_idx = np.asarray(model["delta_idx"], dtype=np.int64)
    if len(d_idx):
        z = (x_imp[:, d_idx] - np.asarray(model["delta_mean"])) / np.asarray(model["delta_std"])
        centroid = np.asarray(model["bad_centroid"])
        sim = (z @ centroid) / ((np.linalg.norm(z, axis=1) + 1e-8) * float(model["bad_centroid_norm"]))
        sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
    else:
        sim = np.zeros(len(x_imp), dtype=np.float64)
    return np.asarray(p_bad, dtype=np.float64), np.asarray(sim, dtype=np.float64)


def _apply_runtime(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    x_all: pd.DataFrame,
    model: dict[str, Any],
    rt: RuntimeConfig,
    super_quality: float,
    super_conf: float,
) -> pd.DataFrame:
    out = decisions.copy()
    p_bad, sim = _hazard_components(model, x_all)
    _, cost = _risk_cost_components(df)
    score = np.clip((1.0 - rt.sim_weight - rt.cost_weight) * p_bad + rt.sim_weight * sim + rt.cost_weight * np.clip(cost / 1.5, 0.0, 1.0), 0.0, 1.0)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    severity = np.clip((score - rt.threshold) / max(1.0 - rt.threshold, 1e-8), 0.0, 1.0)
    scale = 1.0 - rt.max_cut * severity
    scale = np.clip(scale, rt.min_scale, 1.0)
    super_trade = (
        pd.to_numeric(out.get("quality_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= super_quality
    ) & (
        pd.to_numeric(out.get("confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= super_conf
    )
    pass_super = active & super_trade & (score <= rt.super_cap)
    scale = np.where(pass_super, 1.0, scale)
    out.loc[active, "notional_exposure"] = out.loc[active, "notional_exposure"].to_numpy(dtype=np.float64) * scale[active]
    out.loc[active, "position_fraction"] = out.loc[active, "position_fraction"].to_numpy(dtype=np.float64) * scale[active]
    out.loc[:, "hazard_score"] = score
    out.loc[:, "hazard_p_bad"] = p_bad
    out.loc[:, "hazard_delta_similarity"] = sim
    out.loc[:, "hazard_scale"] = scale
    out.loc[:, "hazard_super_pass"] = pass_super
    return out


def _grid() -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for threshold in (0.45, 0.55):
        for max_cut in (0.08, 0.15):
            for min_scale in (0.85, 0.92):
                rows.append(RuntimeConfig(f"micro_hazard_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.0, 0.0, 0.90, 0.78))
                rows.append(RuntimeConfig(f"micro_hazard_sim_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.20, 0.0, 0.90, 0.80))
                rows.append(RuntimeConfig(f"micro_hazard_cost_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.12, 0.08, 0.90, 0.82))
    for threshold in (0.65, 0.75, 0.85, 0.90):
        for max_cut in (0.15, 0.25, 0.35):
            for min_scale in (0.65, 0.75, 0.85):
                rows.append(RuntimeConfig(f"hazard_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.0, 0.0, 0.90, 0.82))
                rows.append(RuntimeConfig(f"hazard_sim_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.25, 0.0, 0.90, 0.84))
                rows.append(RuntimeConfig(f"hazard_cost_p{threshold:.2f}_cut{max_cut:.2f}_min{min_scale:.2f}", threshold, max_cut, min_scale, 0.15, 0.10, 0.90, 0.86))
    return rows


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    # Preserve PnL first, then reward MDD relief and cost durability.
    return float(c1["pnl"] + 0.32 * c2["pnl"] + 0.18 * c3["pnl"] - 1.8 * abs(c1["mdd"]))


def _metrics(df: pd.DataFrame, parent: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, q: np.ndarray, decisions: pd.DataFrame) -> dict[str, Any]:
    return {f"cost{m}": _backtest(df, parent, jackpot_model, add_cfg, q, decisions, cost_mult=float(m)) for m in (1, 2, 3)}


def main() -> int:
    p = argparse.ArgumentParser(description="Selective hazard pruning for Alpha1: trade-level bad-cluster soft notional scaling.")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    feature_cols = list(parent.get("feature_cols") or [])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    print(f"[{MODEL_ID}] data rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)}", flush=True)

    # Hazard labels are parent-entry labels. Disabling deep scout on the fit window keeps
    # this step fast and avoids teaching the hazard model about a different owner.
    train_q = np.full((len(train_df), 2), -1e9, dtype=np.float32)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    print(f"[{MODEL_ID}] building trade-level hazard labels from 2025-01..09", flush=True)
    train_bt = _backtest(train_df, parent, jackpot_model, add_cfg, train_q, train_dec, cost_mult=1.0, record=True)
    train_x, train_ledger = _records_to_entry_set(train_df, train_dec, train_bt.get("trade_records", []), feature_cols)
    y, weights, label_meta = _label_hazards(train_ledger)
    hazard_model = _fit_hazard_model(train_x, y, weights)
    joblib.dump({"model_id": MODEL_ID, "hazard_model": hazard_model, "label_meta": label_meta}, OUT_DIR / "hazard_sniper.pkl")
    super_quality = float(np.quantile(train_x["parent_quality"], 0.90)) if "parent_quality" in train_x else 0.0
    super_conf = float(np.quantile(train_x["parent_confidence"], 0.90)) if "parent_confidence" in train_x else 0.0
    print(f"[{MODEL_ID}] hazard labels {label_meta} super_quality={super_quality:.5f} super_conf={super_conf:.5f}", flush=True)

    x_val_all = _base_feature_frame(val_df, val_dec, feature_cols)
    x_eval_all = _base_feature_frame(eval_df, eval_dec, feature_cols)
    grid = _grid()
    if args.quick:
        grid = [g for g in grid if g.name.startswith("micro_") or (g.threshold in (0.75, 0.85) and g.max_cut in (0.15,) and g.min_scale in (0.75, 0.85))]

    rows: list[dict[str, Any]] = []
    selected: RuntimeConfig | None = None
    best_score = -1e18
    print(f"[{MODEL_ID}] selecting runtime on 2025-Q4 configs={len(grid)}", flush=True)
    for rt in grid:
        dec = _apply_runtime(val_df, val_dec, x_val_all, hazard_model, rt, super_quality, super_conf)
        metrics = _metrics(val_df, parent, jackpot_model, add_cfg, val_q, dec)
        score = _score(metrics)
        row = {
            **asdict(rt),
            "score": score,
            "val_pnl": metrics["cost1"]["pnl"],
            "val_mdd": metrics["cost1"]["mdd"],
            "val_trades": metrics["cost1"]["trades"],
            "val_cost2_pnl": metrics["cost2"]["pnl"],
            "val_cost3_pnl": metrics["cost3"]["pnl"],
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            selected = rt
            print(f"[{MODEL_ID}] new val best {rt.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f} c2={row['val_cost2_pnl']:.2f} c3={row['val_cost3_pnl']:.2f}", flush=True)
    assert selected is not None
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] evaluating selected runtime on fixed 2026 OOS", flush=True)
    baseline_metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_dec)
    selected_dec = _apply_runtime(eval_df, eval_dec, x_eval_all, hazard_model, selected, super_quality, super_conf)
    selected_metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, selected_dec)
    ledger_result = _backtest(eval_df, parent, jackpot_model, add_cfg, eval_q, selected_dec, cost_mult=1.0, record=True)
    ledger = pd.DataFrame(ledger_result.pop("trade_records", []))
    ledger_path = REPORT_OUT.with_name(f"{REPORT_OUT.stem}_cost1_ledger.csv")
    ledger.to_csv(ledger_path, index=False)
    selected_metrics["cost1_ledger_recorded"] = ledger_result

    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if selected_metrics["cost1"]["pnl"] < baseline_metrics["cost1"]["pnl"]:
        warnings.append("hazard_pruning_cost1_below_alpha1")
    if selected_metrics["cost1"]["mdd"] <= baseline_metrics["cost1"]["mdd"]:
        warnings.append("hazard_pruning_did_not_improve_mdd")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and selected_metrics["cost1"]["mdd"] > baseline_metrics["cost1"]["mdd"] and selected_metrics["cost1"]["pnl"] >= 0.85 * baseline_metrics["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "train_window": "2025-01-01..2025-09-30 trade ledger only",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "label_meta": label_meta,
        "design": "Selective Hazard Pruning: hazard-centric features, delta-cluster similarity, cost-sensitive score, probabilistic soft notional scaling, and veto-the-veto super-trade pass-through.",
        "base_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "selected_runtime": asdict(selected),
        "super_quality_threshold": super_quality,
        "super_confidence_threshold": super_conf,
        "baseline": {"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)},
        "selected": {"name": selected.name, "metrics": selected_metrics, "score": _score(selected_metrics)},
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "ledger_path": str(ledger_path),
        "artifact_path": str(OUT_DIR / "hazard_sniper.pkl"),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] OOS baseline cost1={baseline_metrics['cost1']['pnl']:.2f} mdd={baseline_metrics['cost1']['mdd']:.2f} cost2={baseline_metrics['cost2']['pnl']:.2f} cost3={baseline_metrics['cost3']['pnl']:.2f}", flush=True)
    print(f"[{MODEL_ID}] OOS selected {selected.name} cost1={selected_metrics['cost1']['pnl']:.2f} mdd={selected_metrics['cost1']['mdd']:.2f} cost2={selected_metrics['cost2']['pnl']:.2f} cost3={selected_metrics['cost3']['pnl']:.2f}", flush=True)
    print(f"[{MODEL_ID}] report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
