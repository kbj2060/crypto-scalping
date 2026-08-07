#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_day_opportunity_meta_deep_stop_cd18_20260529"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
MODEL_OUT = OUT_DIR / "day_opportunity_meta.pkl"
FEATURES_OUT = OUT_DIR / "feature_cols.json"
GRID_OUT = OUT_DIR / "grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
TRAIN_CANDIDATES_OUT = OUT_DIR / "train_candidates.parquet"
VAL_CANDIDATES_OUT = OUT_DIR / "val_candidates.parquet"
OOS_CANDIDATES_OUT = OUT_DIR / "oos_candidates.parquet"

FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")
REGIME_COLS = [
    "clean_regime4_state24_sticky090_v2_bull_prob",
    "clean_regime4_state24_sticky090_v2_bear_prob",
    "clean_regime4_state24_sticky090_v2_chop_prob",
    "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    "clean_regime4_state24_sticky090_v2_confidence",
    "clean_regime4_state24_sticky090_v2_entropy",
    "clean_regime4_state24_sticky090_v2_directional_bias",
    "clean_regime4_state24_sticky090_v2_factor_trend",
    "clean_regime4_state24_sticky090_v2_factor_vol",
    "regime4_pred_bull_prob",
    "regime4_pred_bear_prob",
    "regime4_pred_chop_prob",
    "regime4_pred_whipsaw_prob",
    "regime4_pred_confidence",
    "regime4_pred_entropy",
    "regime4_pred_directional_bias",
]
MARKET_COLS = [
    "tp_sl_action_score",
    "log_return",
    "volatility_z",
    "garch_vol_z",
    "bb_width_z",
    "funding_abs",
    "funding_pressure",
    "funding_price_divergence",
    "jump_z",
    "jump_flag",
    "evt_tail_flag",
    "evt_excess_z",
    "liquidity_vacuum",
    "execution_quality",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "whale_conviction",
    "smart_money_flow",
    "m7_expected_ret",
    "m7_composite_score",
    "m7_confidence",
    "m7_tail_risk",
    "m7_qwidth",
    "ai_dir_edge",
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rsi",
    "trade_intensity",
    "big_trade_ratio",
    "whale_retail_ratio",
    "squeeze_power",
    "breakout_strength",
]
CANDIDATE_COLS = [
    "source_id",
    "side_num",
    "notional",
    "leverage",
    "take_profit",
    "stop_loss",
    "max_hold",
    "cooldown",
    "quality_score",
    "confidence",
    "q_long",
    "q_short",
    "q_edge",
    "q_margin",
]


def _assert_clean_frame(df: pd.DataFrame, *, name: str) -> None:
    missing = [c for c in ["timestamp", "close", "tp_sl_action_score"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def _days(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.notna().sum() < 2:
        return 1.0
    return max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)


def _close(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy(dtype=np.float64)


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = float(row.get(col, default))
    except Exception:
        return float(default)
    return val if np.isfinite(val) else float(default)


def _candidate_return(
    close: np.ndarray,
    i: int,
    *,
    side: int,
    notional: float,
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    entry_i = min(int(i) + 1, len(close) - 1)
    entry_px = float(close[entry_i]) * (1.0 + slip if side > 0 else 1.0 - slip)
    max_h = max(1, int(max_hold))
    mfe = 0.0
    mae = 0.0
    exit_i = min(entry_i + max_h, len(close) - 1)
    reason = "max_hold"
    for j in range(entry_i + 1, min(entry_i + max_h, len(close) - 1) + 1):
        px = float(close[j])
        raw = (px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px) / max(entry_px, 1e-12)
        pnl = raw * float(notional)
        mfe = max(mfe, pnl)
        mae = min(mae, pnl)
        if take_profit > 0.0 and pnl >= take_profit:
            exit_i = int(j)
            reason = "take_profit"
            break
        if stop_loss > 0.0 and pnl <= -abs(stop_loss):
            exit_i = int(j)
            reason = "stop_loss"
            break
    exit_px = float(close[exit_i]) * (1.0 - slip if side > 0 else 1.0 + slip)
    raw = (exit_px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - exit_px) / max(entry_px, 1e-12)
    gross = raw * float(notional)
    net = gross - 2.0 * float(fee) * float(notional)
    return {
        "realized_return": float(net),
        "gross_return": float(gross),
        "mfe": float(mfe),
        "mae": float(mae),
        "hold_bars": int(max(1, exit_i - entry_i)),
        "exit_reason": str(reason),
        "stopped_out": int(reason == "stop_loss"),
    }


def _candidate_row(
    df: pd.DataFrame,
    close: np.ndarray,
    i: int,
    *,
    source: str,
    side: int,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    cooldown: int,
    quality_score: float,
    confidence: float,
    q_long: float,
    q_short: float,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    row = df.iloc[int(i)]
    out: dict[str, Any] = {
        "idx": int(i),
        "timestamp": str(row["timestamp"]),
        "source": str(source),
        "source_id": 1.0 if source == "parent" else 2.0,
        "side_num": float(side),
        "notional": float(notional),
        "leverage": float(leverage),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "max_hold": float(max_hold),
        "cooldown": float(cooldown),
        "quality_score": float(quality_score),
        "confidence": float(confidence),
        "q_long": float(q_long),
        "q_short": float(q_short),
        "q_edge": float(max(q_long, q_short)),
        "q_margin": float(abs(q_long - q_short)),
    }
    for col in REGIME_COLS + MARKET_COLS:
        out[col] = _safe(row, col, 0.0)
    outcome = _candidate_return(
        close,
        i,
        side=int(side),
        notional=float(notional),
        take_profit=float(take_profit),
        stop_loss=float(stop_loss),
        max_hold=int(max_hold),
        fee=float(fee),
        slip=float(slip),
    )
    adverse = abs(min(float(outcome["mae"]), 0.0))
    short_churn = int(outcome["hold_bars"] <= 6 and abs(float(outcome["realized_return"])) < 0.004)
    out.update(outcome)
    out["entry_utility"] = float(
        outcome["realized_return"]
        - 0.35 * adverse
        - 0.010 * int(outcome["stopped_out"])
        - 0.004 * short_churn
    )
    out["pass_label"] = int(out["entry_utility"] > 0.0015)
    return out


def _build_candidates(
    *,
    name: str,
    df: pd.DataFrame,
    dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
) -> pd.DataFrame:
    _assert_clean_frame(df, name=name)
    close = _close(df)
    decisions = precision._apply_decision_mods(dec, cfg).reset_index(drop=True)
    base_overlay = precision._overlay(stack["overlay"], cfg)
    overlay = sweep.replace(
        base_overlay,
        name=f"{base_overlay.name}_{variant.name}",
        notional=float(base_overlay.notional * variant.deep_notional_mult),
        edge_th=float(base_overlay.edge_th * variant.deep_edge_mult),
        margin_th=float(base_overlay.margin_th * variant.deep_margin_mult),
    )
    fee = float(stack["fee"]) * 3.0
    slip = float(stack["slip"]) * 3.0
    rows: list[dict[str, Any]] = []
    n = min(len(df) - 2, len(decisions), len(q))
    for i in range(0, max(0, n)):
        dec_row = decisions.iloc[i]
        if int(dec_row.action) != ACTION_CASH and int(dec_row.side) != 0:
            rows.append(
                _candidate_row(
                    df,
                    close,
                    i,
                    source="parent",
                    side=int(dec_row.side),
                    notional=float(dec_row.notional_exposure),
                    leverage=float(dec_row.leverage),
                    take_profit=float(dec_row.take_profit),
                    stop_loss=float(dec_row.stop_loss),
                    max_hold=int(dec_row.max_hold_bars),
                    cooldown=int(dec_row.cooldown_bars),
                    quality_score=float(dec_row.quality_score),
                    confidence=float(dec_row.confidence),
                    q_long=float(q[i, 0]),
                    q_short=float(q[i, 1]),
                    fee=fee,
                    slip=slip,
                )
            )
        if i >= 60:
            ql, qs = float(q[i, 0]), float(q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            side_edge_th = float(overlay.edge_th * (variant.deep_long_edge_mult if side > 0 else variant.deep_short_edge_mult))
            side_margin_th = float(overlay.margin_th * (variant.deep_long_margin_mult if side > 0 else variant.deep_short_margin_mult))
            if edge >= side_edge_th and margin >= side_margin_th:
                rows.append(
                    _candidate_row(
                        df,
                        close,
                        i,
                        source="deep_alpha",
                        side=int(side),
                        notional=float(overlay.notional),
                        leverage=float(max(overlay.notional, 1.0)),
                        take_profit=float(overlay.base_tp),
                        stop_loss=float(overlay.base_sl),
                        max_hold=int(overlay.base_hold),
                        cooldown=int(overlay.cooldown),
                        quality_score=float(edge),
                        confidence=float(margin),
                        q_long=ql,
                        q_short=qs,
                        fee=fee,
                        slip=slip,
                    )
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"{name} candidate dataset is empty")
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_cols(candidates: pd.DataFrame) -> list[str]:
    cols = CANDIDATE_COLS + [c for c in REGIME_COLS + MARKET_COLS if c in candidates.columns]
    bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"forbidden feature columns in day opportunity meta: {bad[:20]}")
    missing = [c for c in CANDIDATE_COLS if c not in candidates.columns]
    if missing:
        raise RuntimeError(f"candidate dataset missing generated feature columns: {missing}")
    return cols


def _train_meta(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    x = train[feature_cols]
    y_cls = train["pass_label"].astype(int).to_numpy()
    y_reg = train["entry_utility"].astype(float).to_numpy()
    weights = np.clip(np.abs(y_reg) * 80.0, 0.5, 8.0)
    cls = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=529001,
        ),
    )
    reg = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.10,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=529002,
        ),
    )
    cls.fit(x, y_cls, histgradientboostingclassifier__sample_weight=weights)
    reg.fit(x, y_reg, histgradientboostingregressor__sample_weight=weights)
    return {"classifier": cls, "regressor": reg, "feature_cols": feature_cols}


def _alpha7_combo_decisions(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = joblib.load(decontam.CANDIDATE_DIR / "primary_parent.pkl")
    fallback = joblib.load(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl")
    loop._assert_parent_contract(primary, left_df, name="alpha7_primary_left")
    loop._assert_parent_contract(primary, right_df, name="alpha7_primary_right")
    loop._assert_parent_contract(fallback, left_df, name="alpha7_fallback_left")
    loop._assert_parent_contract(fallback, right_df, name="alpha7_fallback_right")
    p_rt = loop._load_best_scale_runtime(decontam.CANDIDATE_DIR / "primary_summary.json")
    f_rt = loop._load_best_scale_runtime(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_summary.json")
    p_left = loop._predict_scaled(primary, left_df, p_rt)
    p_right = loop._predict_scaled(primary, right_df, p_rt)
    f_left = loop._predict_scaled(fallback, left_df, f_rt)
    f_right = loop._predict_scaled(fallback, right_df, f_rt)
    return loop._combine_primary_fallback(p_left, f_left), loop._combine_primary_fallback(p_right, f_right)


def _load_augmented_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = loop._merge_state24(loop._read(loop.v31.DEFAULT_TRAIN), loop.alpha3_full.SIDE_CLEAN4_2025)
    eval_df = loop._merge_state24(loop._read(loop.v31.DEFAULT_EVAL), loop.alpha3_full.SIDE_CLEAN4_2026)
    a7_train = loop._rename_clean4_v2(loop._read(decontam.TRAIN_CSV))
    a7_eval = loop._rename_clean4_v2(loop._read(decontam.EVAL_CSV))
    train_all = loop._augment_with_alpha7_features(train_all, a7_train)
    eval_df = loop._augment_with_alpha7_features(eval_df, a7_eval)
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    return train_df, val_df, eval_df.reset_index(drop=True)


def _predict_meta(bundle: dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(bundle["feature_cols"])
    out = candidates.copy()
    proba = bundle["classifier"].predict_proba(out[feature_cols])
    classes = np.asarray(bundle["classifier"].classes_, dtype=int)
    if 1 in classes:
        pass_prob = proba[:, int(np.where(classes == 1)[0][0])]
    else:
        pass_prob = np.zeros(len(out), dtype=np.float64)
    out["pred_pass_prob"] = pass_prob
    out["pred_utility"] = bundle["regressor"].predict(out[feature_cols])
    out["pred_margin"] = out["pred_utility"] - 0.0015
    return out


def _allow_maps(pred: pd.DataFrame, *, prob_th: float, utility_th: float) -> tuple[set[int], set[int]]:
    ok = (pd.to_numeric(pred["pred_pass_prob"], errors="coerce").fillna(0.0) >= float(prob_th)) & (
        pd.to_numeric(pred["pred_utility"], errors="coerce").fillna(-999.0) >= float(utility_th)
    )
    parent = set(int(v) for v in pred.loc[ok & pred["source"].eq("parent"), "idx"].tolist())
    deep = set(int(v) for v in pred.loc[ok & pred["source"].eq("deep_alpha"), "idx"].tolist())
    return parent, deep


def _filter_parent_decisions(dec: pd.DataFrame, allowed_parent: set[int]) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = (pd.to_numeric(out["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int) != 0
    )
    if active.any():
        idx = pd.Series(np.arange(len(out)), index=out.index)
        block = active & ~idx.isin(allowed_parent)
        out.loc[block, ["action", "side"]] = 0
    return out


def _day_opportunity_score(res: dict[str, Any], *, target_tpd: float = 2.5) -> float:
    if int(res.get("trades", 0)) < 20:
        return -1e9 + float(res.get("pnl", 0.0))
    pnl = float(res["pnl"])
    mdd = abs(float(res["mdd"]))
    wr = float(res["wr"])
    tpd = float(res.get("trades_per_day", 0.0))
    sl = float(sweep._sl_ratio(res))
    return pnl - 1.15 * mdd + 25.0 * wr - 12.0 * abs(tpd - target_tpd) - 70.0 * sl


def _eval_gate(
    *,
    name: str,
    df: pd.DataFrame,
    dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    allowed_parent: set[int],
    allowed_deep: set[int],
    gate_deep: bool,
    record: bool,
) -> dict[str, Any]:
    dec2 = _filter_parent_decisions(precision._apply_decision_mods(dec, cfg), allowed_parent)

    def deep_gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        return (int(i) in allowed_deep), "day_opportunity"

    res = sweep._backtest_variant(
        df=df,
        q=q,
        dec=dec2,
        stack=stack,
        cfg=cfg,
        variant=sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18),
        cost_mult=3,
        record=record,
        deep_gate=deep_gate if bool(gate_deep) else None,
    )
    records = list(res.pop("trade_records", [])) if record else []
    return {
        "name": str(name),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(_day_opportunity_score(res)),
        "exits": res.get("exits", {}),
        "records": records,
    }


def _baseline(
    *,
    df: pd.DataFrame,
    dec: pd.DataFrame,
    q: np.ndarray,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    record: bool,
) -> dict[str, Any]:
    res = sweep._backtest_variant(
        df=df,
        q=q,
        dec=dec,
        stack=stack,
        cfg=cfg,
        variant=sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18),
        cost_mult=3,
        record=record,
    )
    records = list(res.pop("trade_records", [])) if record else []
    return {
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(_day_opportunity_score(res)),
        "exits": res.get("exits", {}),
        "records": records,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    train_df, val_df, eval_df = _load_augmented_frames()
    for name, frame in {"train": train_df, "val": val_df, "oos": eval_df}.items():
        _assert_clean_frame(frame, name=name)

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    train_dec, _ = _alpha7_combo_decisions(train_df, val_df)
    val_dec, oos_dec = _alpha7_combo_decisions(val_df, eval_df)
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    oos_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    train_cand = _build_candidates(name="train", df=train_df, dec=train_dec, q=train_q, stack=stack, cfg=cfg, variant=variant)
    val_cand = _build_candidates(name="val", df=val_df, dec=val_dec, q=val_q, stack=stack, cfg=cfg, variant=variant)
    oos_cand = _build_candidates(name="oos", df=eval_df, dec=oos_dec, q=oos_q, stack=stack, cfg=cfg, variant=variant)
    train_cand.to_parquet(TRAIN_CANDIDATES_OUT, index=False)
    val_cand.to_parquet(VAL_CANDIDATES_OUT, index=False)
    oos_cand.to_parquet(OOS_CANDIDATES_OUT, index=False)

    feature_cols = _feature_cols(train_cand)
    bundle = _train_meta(train_cand, feature_cols)
    joblib.dump(bundle, MODEL_OUT)
    FEATURES_OUT.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    val_pred = _predict_meta(bundle, val_cand)
    oos_pred = _predict_meta(bundle, oos_cand)
    val_pred.to_csv(OUT_DIR / "val_candidates_scored.csv", index=False)
    oos_pred.to_csv(OUT_DIR / "oos_candidates_scored.csv", index=False)

    base_val = _baseline(df=val_df, dec=val_dec, q=val_q, stack=stack, cfg=cfg, record=False)
    base_oos = _baseline(df=eval_df, dec=oos_dec, q=oos_q, stack=stack, cfg=cfg, record=True)
    pd.DataFrame(base_oos.pop("records", [])).to_csv(OUT_DIR / "baseline_oos_cost3_ledger.csv", index=False)

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    prob_grid = [0.42, 0.48, 0.54, 0.60, 0.66, 0.72]
    util_grid = [-0.002, 0.0, 0.0015, 0.003, 0.005]
    for gate_scope in ("parent_only", "parent_deep"):
        for prob_th in prob_grid:
            for utility_th in util_grid:
                variant_name = f"{gate_scope}_prob{prob_th:.2f}_util{utility_th:.4f}"
                gate_deep = gate_scope == "parent_deep"
                val_parent, val_deep = _allow_maps(val_pred, prob_th=prob_th, utility_th=utility_th)
                oos_parent, oos_deep = _allow_maps(oos_pred, prob_th=prob_th, utility_th=utility_th)
                val_res = _eval_gate(
                    name=variant_name,
                    df=val_df,
                    dec=val_dec,
                    q=val_q,
                    stack=stack,
                    cfg=cfg,
                    allowed_parent=val_parent,
                    allowed_deep=val_deep,
                    gate_deep=gate_deep,
                    record=False,
                )
                oos_res = _eval_gate(
                    name=variant_name,
                    df=eval_df,
                    dec=oos_dec,
                    q=oos_q,
                    stack=stack,
                    cfg=cfg,
                    allowed_parent=oos_parent,
                    allowed_deep=oos_deep,
                    gate_deep=gate_deep,
                    record=True,
                )
                ledger_path = OUT_DIR / f"{variant_name}_oos_cost3_ledger.csv"
                pd.DataFrame(oos_res.pop("records", [])).to_csv(ledger_path, index=False)
                reports[variant_name] = {"val": val_res, "oos": oos_res, "oos_ledger": str(ledger_path)}
                rows.append(
                    {
                        "name": variant_name,
                        "gate_scope": str(gate_scope),
                        "prob_th": float(prob_th),
                        "utility_th": float(utility_th),
                        "val_pnl": val_res["pnl"],
                        "val_mdd": val_res["mdd"],
                        "val_wr": val_res["wr"],
                        "val_trades": val_res["trades"],
                        "val_trades_per_day": val_res["trades_per_day"],
                        "val_sl_ratio": val_res["sl_ratio"],
                        "val_score": val_res["score"],
                        "oos_pnl": oos_res["pnl"],
                        "oos_mdd": oos_res["mdd"],
                        "oos_wr": oos_res["wr"],
                        "oos_trades": oos_res["trades"],
                        "oos_trades_per_day": oos_res["trades_per_day"],
                        "oos_deep_entries": oos_res["deep_entries"],
                        "oos_long_entries": oos_res["long_entries"],
                        "oos_short_entries": oos_res["short_entries"],
                        "oos_sl_ratio": oos_res["sl_ratio"],
                        "oos_score": oos_res["score"],
                        "oos_ledger": str(ledger_path),
                    }
                )

    grid = pd.DataFrame(rows).sort_values(["val_score", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    best = grid.iloc[0].to_dict() if not grid.empty else {}
    summary = {
        "model_id": MODEL_ID,
        "contract": str(ROOT / "docs/model_contracts/alpha7_day_opportunity_deep_stop_cd18_20260529_contract.md"),
        "base_model": "alpha7_submodel_01965_decontam_deep_stop_cd18_20260528",
        "scope": "Research-only candidate utility meta layer. No runtime hard trade cap and no active/live injection.",
        "feature_count": int(len(feature_cols)),
        "candidate_counts": {
            "train": int(len(train_cand)),
            "val": int(len(val_cand)),
            "oos": int(len(oos_cand)),
            "train_pass_rate": float(train_cand["pass_label"].mean()),
            "val_pass_rate": float(val_cand["pass_label"].mean()),
            "oos_pass_rate": float(oos_cand["pass_label"].mean()),
        },
        "artifacts": {
            "model": str(MODEL_OUT),
            "feature_cols": str(FEATURES_OUT),
            "grid": str(GRID_OUT),
            "train_candidates": str(TRAIN_CANDIDATES_OUT),
            "val_candidates": str(VAL_CANDIDATES_OUT),
            "oos_candidates": str(OOS_CANDIDATES_OUT),
        },
        "baseline": {
            "val": {k: v for k, v in base_val.items() if k != "records"},
            "oos": {k: v for k, v in base_oos.items() if k != "records"},
        },
        "selection_policy": "Select thresholds by validation day-opportunity score only; OOS is reported with fixed selected threshold.",
        "selected_by_val_score": best,
        "reports": reports,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "selected": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
