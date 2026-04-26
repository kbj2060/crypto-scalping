#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_m7_signal_only import run_backtest

CSV_PATH = ROOT / "data" / "splits" / "year_oos" / "rl_training_2025_m7_contract_highorder.csv"
OUT_JSON = ROOT / "data" / "ensemble" / "reports" / "explore_unsup_redesign_m7_2025.json"
OUT_CSV = ROOT / "data" / "splits" / "year_oos" / "rl_training_2025_m7_experimental_unsup.csv"


@dataclass
class Config:
    name: str
    soft_q: float
    hard_q: float
    vol_penalty: float
    exec_bonus: float
    gate_shrink: float
    crowd_weight: float
    flow_weight: float
    persist_weight: float
    direction_aware: bool
    min_conf_base: float


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Explore experimental unsupervised redesign on existing M7 CSV.")
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    return ap.parse_args()


def _robust_z(s: pd.Series, window: int = 288, min_periods: int = 48) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    med = x.rolling(window=window, min_periods=min_periods).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median()
    z = (x - med) / (1.4826 * mad.replace(0.0, np.nan) + 1e-6)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax3(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -50.0, 50.0))
    s = ez.sum(axis=1, keepdims=True)
    s = np.where(s <= 1e-12, 1.0, s)
    return ez / s


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    numeric_cols = [
        "open", "high", "low", "close",
        "m7_trend_xgb_dn", "m7_trend_xgb_fl", "m7_trend_xgb_up",
        "m7_mtl_dn", "m7_mtl_fl", "m7_mtl_up",
        "m7_quant_dn", "m7_quant_fl", "m7_quant_up",
        "m7_q10", "m7_q50", "m7_q90", "m7_qwidth",
        "m7_quality_pred", "m7_hold_pred", "m7_target_hold",
        "m7_entry_long_offset", "m7_entry_short_offset", "m7_entry_long_price", "m7_entry_short_price",
        "m7_tp_offset", "m7_sl_offset", "m7_tp_price", "m7_sl_price",
        "smart_money_flow", "whale_retail_ratio", "whale_conviction", "funding_price_divergence",
        "cvp_volume_imbalance", "cvp_poc_dist", "cvp_cluster_position",
        "trade_intensity", "taker_acceleration", "net_taker_ratio",
        "amihud_illiquidity_z", "garman_klass_vol", "rogers_satchell_vol",
        "regime_persistence", "cross_scale_curvature", "liquidity_vacuum", "crowding_pressure", "execution_quality",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _baseline_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_xgb = df[["m7_trend_xgb_dn", "m7_trend_xgb_fl", "m7_trend_xgb_up"]].to_numpy(np.float64)
    p_mtl = df[["m7_mtl_dn", "m7_mtl_fl", "m7_mtl_up"]].to_numpy(np.float64)
    p_q = df[["m7_quant_dn", "m7_quant_fl", "m7_quant_up"]].to_numpy(np.float64)
    probs = 0.45 * p_xgb + 0.35 * p_mtl + 0.20 * p_q
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    prior = np.clip(np.mean(probs, axis=0), 1e-6, 1.0)
    target_prior = np.array([0.42, 0.16, 0.42], dtype=np.float64)
    prior_scale = np.clip(target_prior / prior, 0.75, 1.35)
    probs = probs * prior_scale[None, :]
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    sort_p = np.sort(probs, axis=1)
    conf = np.clip((sort_p[:, 2] - 1.0 / 3.0) * 1.5 + (sort_p[:, 2] - sort_p[:, 1]) * 0.6, 0.0, 1.0)
    return probs, conf


def _experimental_unsup(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    rs = _robust_z(_safe_series(df, "rogers_satchell_vol"))
    gk = _robust_z(_safe_series(df, "garman_klass_vol"))
    amihud = _robust_z(_safe_series(df, "amihud_illiquidity_z"))
    lv = _robust_z(_safe_series(df, "liquidity_vacuum"))
    curvature = _robust_z(_safe_series(df, "cross_scale_curvature").abs())
    execq = _safe_series(df, "execution_quality").clip(-1.5, 1.5)
    crowd = _safe_series(df, "crowding_pressure").clip(-3.0, 3.0)
    whale = _robust_z(_safe_series(df, "whale_conviction"))
    funding_div = _robust_z(_safe_series(df, "funding_price_divergence").abs())
    smf = _robust_z(_safe_series(df, "smart_money_flow"))
    cvp_imb = _robust_z(_safe_series(df, "cvp_volume_imbalance").abs())
    poc = _robust_z(_safe_series(df, "cvp_poc_dist").abs())
    taker = _robust_z(_safe_series(df, "taker_acceleration").abs())
    nti = _robust_z(_safe_series(df, "net_taker_ratio").abs())
    regime_persist = _safe_series(df, "regime_persistence").clip(0.0, 1.5)
    trade_int = _robust_z(_safe_series(df, "trade_intensity"))

    vol_surface = 0.26 * rs + 0.22 * gk + 0.18 * amihud + 0.20 * lv + 0.14 * curvature
    vol_rank = _sigmoid(1.1 * vol_surface.to_numpy(np.float64))
    gmm_conf = np.clip(1.0 - 0.55 * vol_rank + 0.18 * np.clip(execq.to_numpy(np.float64), -1.0, 1.0), 0.0, 1.0)
    gmm_cluster = np.digitize(vol_rank, bins=[0.22, 0.45, 0.68]).astype(np.float64)

    flow_dislocation = cfg.flow_weight * (
        0.24 * cvp_imb + 0.20 * poc + 0.18 * nti + 0.18 * taker + 0.20 * lv
    )
    crowd_pressure = cfg.crowd_weight * (
        0.38 * crowd.abs() + 0.22 * whale.abs() + 0.20 * funding_div + 0.20 * smf.abs()
    )
    persistence_stress = cfg.persist_weight * (
        0.55 * _robust_z(1.0 - regime_persist) + 0.45 * curvature
    )
    execution_fragility = (
        0.45 * _robust_z(-execq) + 0.20 * lv + 0.15 * trade_int.abs() + 0.20 * cvp_imb
    )

    iso_score = np.maximum(
        0.0,
        (0.42 * crowd_pressure + 0.33 * flow_dislocation + 0.25 * execution_fragility).to_numpy(np.float64),
    )
    vae_error = np.maximum(
        0.0,
        (0.36 * execution_fragility + 0.34 * persistence_stress + 0.30 * vol_surface).to_numpy(np.float64),
    )

    return {
        "gmm_cluster": gmm_cluster,
        "gmm_conf": gmm_conf,
        "vol_rank": vol_rank,
        "iso_score": iso_score,
        "vae_error": vae_error,
        "crowd_raw": crowd.to_numpy(np.float64),
        "execq_raw": execq.to_numpy(np.float64),
    }


def _apply_variant(base_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = base_df.copy()
    probs, confidence = _baseline_probs(df)
    direction = np.argmax(probs, axis=1).astype(np.int64)  # 0 down 1 flat 2 up
    side = np.where(direction == 2, 1.0, np.where(direction == 0, -1.0, 0.0))

    uns = _experimental_unsup(df, cfg)
    vol_rank = uns["vol_rank"]
    iso_score = uns["iso_score"]
    vae_error = uns["vae_error"]
    crowd_raw = uns["crowd_raw"]
    execq_raw = uns["execq_raw"]

    if cfg.direction_aware:
        directional_crowd = np.maximum(side * crowd_raw, 0.0)
        iso_score = iso_score + 0.35 * directional_crowd
        vae_error = vae_error + 0.20 * directional_crowd

    soft_th = float(np.quantile(iso_score + 0.7 * vae_error, cfg.soft_q))
    hard_th = float(np.quantile(iso_score + 0.7 * vae_error, cfg.hard_q))
    gate_energy = iso_score + 0.7 * vae_error
    iso_anom = (gate_energy >= soft_th).astype(np.float32)
    vae_anom = (gate_energy >= hard_th).astype(np.float32)
    gate_block = (gate_energy >= hard_th).astype(np.float32)

    q10 = pd.to_numeric(df["m7_q10"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q50 = pd.to_numeric(df["m7_q50"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q90 = pd.to_numeric(df["m7_q90"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q_width = np.maximum(pd.to_numeric(df["m7_qwidth"], errors="coerce").fillna(1e-6).to_numpy(np.float64), 1e-6)
    quality = pd.to_numeric(df["m7_quality_pred"], errors="coerce").fillna(0.0).to_numpy(np.float64)

    rr_sym = np.abs(q50) / q_width
    dir_gap = np.abs(probs[:, 2] - probs[:, 0])
    conf_mix = np.clip(0.65 * confidence + 0.35 * dir_gap, 0.0, 1.0)
    rr = np.where(direction == 1, 0.0, rr_sym)
    base_size = np.tanh(rr * 1.1) * conf_mix
    quality_scale = np.clip(0.8 + quality * 80.0, 0.25, 1.25)
    size = base_size * quality_scale

    route_scale = np.clip(
        1.0
        - cfg.vol_penalty * vol_rank
        - cfg.gate_shrink * np.clip((gate_energy - soft_th) / max(hard_th - soft_th, 1e-6), 0.0, 1.0)
        + cfg.exec_bonus * np.clip(execq_raw, -1.0, 1.0),
        0.15,
        1.25,
    )
    size = np.clip(size * route_scale, 0.0, 1.0)
    size = np.where(gate_block == 1.0, 0.0, size)

    min_conf = np.clip(cfg.min_conf_base + 0.08 * vol_rank + 0.10 * (gate_energy >= soft_th), 0.42, 0.78)
    action = np.zeros(len(df), dtype=np.float32)
    long_cond = (direction == 2) & (confidence >= min_conf) & (gate_block == 0.0)
    short_cond = (direction == 0) & (confidence >= min_conf) & (gate_block == 0.0)
    action[long_cond] = 1.0
    action[short_cond] = -1.0
    action[size < 0.07] = 0.0

    hold_raw = pd.to_numeric(df["m7_hold_pred"], errors="coerce").fillna(12.0).to_numpy(np.float64)
    target_hold = np.clip(np.round(hold_raw), 1, 48).astype(np.int64)
    target_hold = np.where(action == 0.0, 0, target_hold)
    target_hold = np.where(gate_energy >= soft_th, np.minimum(target_hold, 6), target_hold)
    target_hold = np.where(gate_energy >= hard_th, np.minimum(target_hold, 3), target_hold)

    expected_ret = np.where(action == 1.0, q50, np.where(action == -1.0, -q50, 0.0))
    tail_risk = np.where(action == 1.0, np.minimum(q10, 0.0), np.where(action == -1.0, -np.maximum(q90, 0.0), 0.0))
    composite = np.clip(expected_ret * (0.5 + confidence) * (1.0 - 0.55 * gate_block), -1.0, 1.0)

    out = df.copy()
    out["m7_gmm_cluster"] = uns["gmm_cluster"].astype(np.float32)
    out["m7_gmm_conf"] = uns["gmm_conf"].astype(np.float32)
    out["m7_gmm_vol_rank"] = vol_rank.astype(np.float32)
    out["m7_iso_pred"] = np.where(iso_anom > 0.0, -1.0, 1.0).astype(np.float32)
    out["m7_iso_score"] = iso_score.astype(np.float32)
    out["m7_vae_error"] = vae_error.astype(np.float32)
    out["m7_iso_anom"] = iso_anom.astype(np.float32)
    out["m7_vae_anom"] = vae_anom.astype(np.float32)
    out["m7_gate_block"] = gate_block.astype(np.float32)
    out["m7_confidence"] = confidence.astype(np.float32)
    out["m7_action"] = action.astype(np.float32)
    out["m7_size"] = size.astype(np.float32)
    out["m7_target_hold"] = target_hold.astype(np.float32)
    out["m7_expected_ret"] = expected_ret.astype(np.float32)
    out["m7_tail_risk"] = tail_risk.astype(np.float32)
    out["m7_composite_score"] = composite.astype(np.float32)
    return out


def _configs() -> list[Config]:
    return [
        Config("tensor_soft_barrier", 0.93, 0.985, 0.28, 0.08, 0.38, 1.10, 1.00, 0.90, True, 0.46),
        Config("execution_first", 0.92, 0.980, 0.24, 0.16, 0.34, 0.90, 0.95, 1.00, False, 0.45),
        Config("crowding_barrier", 0.94, 0.987, 0.26, 0.10, 0.42, 1.25, 0.90, 0.85, True, 0.47),
        Config("persistence_tensor", 0.93, 0.983, 0.30, 0.12, 0.40, 0.95, 0.95, 1.25, False, 0.46),
        Config("convex_flow", 0.95, 0.989, 0.22, 0.14, 0.46, 1.05, 1.15, 0.85, True, 0.48),
        Config("shock_absorber", 0.91, 0.977, 0.34, 0.06, 0.48, 0.85, 1.00, 1.15, False, 0.44),
        Config("directional_crowding", 0.94, 0.986, 0.25, 0.10, 0.39, 1.20, 1.05, 0.90, True, 0.46),
        Config("liquidity_lens", 0.92, 0.981, 0.31, 0.12, 0.36, 0.92, 1.18, 0.95, False, 0.45),
        Config("execution_convex", 0.93, 0.984, 0.27, 0.18, 0.35, 1.00, 0.98, 1.00, True, 0.47),
        Config("balanced_tensor", 0.93, 0.985, 0.27, 0.12, 0.38, 1.00, 1.00, 1.00, False, 0.46),
    ]


def _score(result: dict) -> float:
    return (
        float(result["pnl_pct"])
        + 0.05 * float(result["sharpe"])
        + 0.01 * float(result["win_rate_pct"])
        + 0.5 * min(float(result["profit_factor"]), 2.0)
        - 0.0015 * abs(float(result["trades"]) - 24000.0)
    )


def main() -> int:
    args = _parse_args()
    df = _load(args.csv)

    baseline_res = asdict(run_backtest(df, fee_bps=2.0, slip_bps=1.0))
    rows: list[dict] = [{"name": "current_contract_highorder", "score": _score(baseline_res), **baseline_res}]
    best_name = "current_contract_highorder"
    best_score = rows[0]["score"]
    best_df = df

    for cfg in tqdm(_configs(), desc="experimental-unsup", unit="cfg"):
        trial_df = _apply_variant(df, cfg)
        res = asdict(run_backtest(trial_df, fee_bps=2.0, slip_bps=1.0))
        row = {
            "name": cfg.name,
            "config": asdict(cfg),
            "score": _score(res),
            **res,
        }
        rows.append(row)
        if row["score"] > best_score:
            best_score = row["score"]
            best_name = cfg.name
            best_df = trial_df

    historical = {}
    hist_path = ROOT / "data" / "ensemble" / "reports" / "backtest_m7_signal_only_ablation4_nohdb_2025.json"
    if hist_path.exists():
        historical = json.loads(hist_path.read_text(encoding="utf-8")).get("result", {})

    payload = {
        "source_csv": args.csv,
        "baseline_current_contract": rows[0],
        "historical_best_known": historical,
        "best_name": best_name,
        "best_score": best_score,
        "results": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_name != "current_contract_highorder":
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        best_df.to_csv(args.out_csv, index=False)

    print(json.dumps({
        "best_name": best_name,
        "best_score": best_score,
        "best_result": next(r for r in rows if r["name"] == best_name),
        "historical_best_known": historical,
        "out_json": args.out_json,
        "out_csv": args.out_csv if best_name != "current_contract_highorder" else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
