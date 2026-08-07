#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_omega1_direction_head_direction_only_20260602 as base
import train_omega1_direction_head_tsfm_chronos_20260602 as confirmed


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_direction_head_raw_context_groups_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_direction_head_raw_context_groups_20260602"
YEAR_OOS_DIR = ROOT / "data/splits/year_oos"

BASE_COLS = confirmed.VARIANTS["core_plus_tsfm_chronos"]

GROUP_CANDIDATES: dict[str, list[str]] = {
    "raw_market_ohlcv": [
        "open",
        "high",
        "low",
        "close",
        "close_btc",
    ],
    "volume_flow": [
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "volume_btc",
        "quote_volume_btc",
        "trade_intensity",
        "big_trade_ratio",
        "taker_acceleration",
        "net_taker_ratio",
        "ofi_acceleration",
        "ofti",
        "cvd_12",
        "cvd_48",
        "cvd_288",
        "cvd_slope_12",
        "cvd_slope_48",
        "price_cvd_divergence",
        "cvd_breakout_z",
        "smart_money_flow",
        "whale_retail_ratio",
        "whale_conviction",
    ],
    "liquidity_execution_spread_proxy": [
        "spread",
        "bid_ask_spread",
        "amihud_illiquidity_z",
        "cvp_poc_dist",
        "cvp_vah_val_width",
        "cvp_cluster_position",
        "cvp_volume_imbalance",
        "cvp_regime",
        "liquidity_vacuum",
        "execution_quality",
        "distance_to_day_high_low_pct",
        "upper_wick_z",
        "lower_wick_z",
        "sweep_prev_high_reclaim",
        "sweep_prev_low_reclaim",
        "failed_breakout_up",
        "failed_breakout_down",
    ],
    "funding_context": [
        "last_funding_rate",
        "funding_roc_12",
        "funding_roc_48",
        "funding_roc_288",
        "funding_z_score",
        "funding_abs",
        "funding_pressure",
        "funding_price_divergence",
        "funding_oi_divergence",
        "funding_flip_signal",
        "mta_funding",
        "ou_funding_z",
    ],
    "session_context": [
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "session_europe",
        "session_us",
        "is_hour_open",
    ],
    "volatility_context": [
        "log_return",
        "volatility_z",
        "bb_width",
        "bb_width_z",
        "garman_klass_vol",
        "realized_vol_ratio",
        "rogers_satchell_vol",
        "parkinson_vol",
        "bb_width_pct_rank_288",
        "atr_pct_rank_288",
        "compression_score",
        "compression_release_up",
        "compression_release_down",
        "garch_vol_z",
        "jump_flag",
        "jump_z",
        "evt_tail_flag",
        "evt_excess_z",
        "squeeze_power",
        "long_squeeze_risk",
        "short_squeeze_risk",
        "crowding_pressure",
        "crowded_long_unwind_risk",
        "crowded_short_squeeze_risk",
    ],
}

FORBIDDEN_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "a5dir_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
)
FORBIDDEN_TOKENS = (
    "label",
    "target",
    "future",
    "pnl",
    "action_score",
    "wave3",
    "zigzag_soft",
)

BASELINE = {
    "variant": "core_plus_tsfm_chronos",
    "feature_count": 55,
    "oos_bacc": 0.5974048650,
    "oos_auc": 0.7907205158,
    "oos_proxy_wr": 0.6579421029,
    "oos_proxy_trades": 13334,
}


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _year_oos_path(year: int) -> Path:
    if int(year) == 2025:
        return YEAR_OOS_DIR / "training_features_2025.csv"
    if int(year) == 2026:
        return YEAR_OOS_DIR / "training_features_2026_rebuilt.csv"
    raise ValueError(f"unsupported year: {year}")


def _validate_context_cols(cols: list[str], frame: pd.DataFrame) -> None:
    seen: set[str] = set()
    dups = [c for c in cols if c in seen or seen.add(c)]
    if dups:
        raise ValueError(f"duplicate context columns: {dups}")
    for col in cols:
        lower = col.lower()
        if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise ValueError(f"forbidden context prefix selected: {col}")
        if any(token in lower for token in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden context token selected: {col}")
        if "future" in lower:
            raise ValueError(f"forbidden future token selected: {col}")
        if col not in frame.columns:
            raise ValueError(f"context column missing from frame: {col}")
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise TypeError(f"context column must be numeric: {col}")


def _assert_finite(frame: pd.DataFrame, cols: list[str], label: str) -> None:
    arr = frame[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = {c: int((~np.isfinite(frame[c].to_numpy(dtype=np.float64))).sum()) for c in cols}
        bad = {k: v for k, v in bad.items() if v}
        raise ValueError(f"{label} contains non-finite values: {bad}")


def _read_year_oos(year: int) -> pd.DataFrame:
    frame = base._read_csv(_year_oos_path(year))
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(year)]:
        raise RuntimeError(f"year_oos year guard failed for {year}: {years}")
    return frame


def _available_group_cols(year_frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    available: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for group, candidates in GROUP_CANDIDATES.items():
        available[group] = [c for c in candidates if c in year_frame.columns]
        missing[group] = [c for c in candidates if c not in year_frame.columns]
    return available, missing


def _build_frame(year: int) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    frame = confirmed._build_frame(year, include_core=True)
    year_oos = _read_year_oos(year)
    available, missing = _available_group_cols(year_oos)
    context_cols = sorted({c for cols in available.values() for c in cols})
    frame = base._exact_join(frame, year_oos[["timestamp", *context_cols]], context_cols, f"year_oos_context {year}")
    return frame, available, missing


def _dedupe(cols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _oof_proba(train: pd.DataFrame, feature_cols: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(train)
    starts = [int(n * 0.35), int(n * 0.50), int(n * 0.65), int(n * 0.80)]
    ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    proba = np.full((n, 3), np.nan, dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    folds: list[dict[str, Any]] = []
    y = train["zigzag_action"].to_numpy(dtype=np.int64)
    for fold, (start, end) in enumerate(zip(starts, ends), start=1):
        model = base._fit_catboost(train.iloc[:start][feature_cols], y[:start], seed=seed + fold, iterations=500)
        pred = base._proba3(model, train.iloc[start:end][feature_cols])
        proba[start:end] = pred
        covered[start:end] = True
        folds.append(
            {
                "fold": fold,
                "train_rows": int(start),
                "predict_start": int(start),
                "predict_end": int(end),
                "metrics": base._metrics(y[start:end], pred),
            }
        )
    return proba, covered, folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.DROP_EVENTS.clear()
    train, groups_train, missing_train = _build_frame(2025)
    oos, groups_oos, missing_oos = _build_frame(2026)
    if groups_train != groups_oos:
        raise RuntimeError(f"context group contract mismatch 2025 vs 2026: {groups_train} != {groups_oos}")
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_oos = oos["zigzag_action"].to_numpy(dtype=np.int64)

    group_variants: dict[str, list[str]] = {f"add_{group}": cols for group, cols in groups_train.items() if cols}
    all_group_cols = _dedupe([col for group in GROUP_CANDIDATES for col in groups_train[group]])
    group_variants["add_all_requested_context"] = all_group_cols

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "label_source": "zigzag_action",
        "base_direction_head": "core_plus_tsfm_chronos",
        "baseline": BASELINE,
        "group_candidates": GROUP_CANDIDATES,
        "available_groups": groups_train,
        "missing_candidates_2025": missing_train,
        "missing_candidates_2026": missing_oos,
        "variants": {},
        "drop_events": base.DROP_EVENTS,
        "artifacts": {"out_dir": str(OUT_DIR)},
    }
    ranking: list[dict[str, Any]] = []
    for idx, (variant, add_cols) in enumerate(group_variants.items(), start=1):
        feature_cols = _dedupe(BASE_COLS + add_cols)
        base._validate_features(BASE_COLS, train)
        base._validate_features(BASE_COLS, oos)
        _validate_context_cols(add_cols, train)
        _validate_context_cols(add_cols, oos)
        _assert_finite(train, feature_cols, f"{variant} train")
        _assert_finite(oos, feature_cols, f"{variant} oos")

        variant_dir = OUT_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        oof, covered, folds = _oof_proba(train, feature_cols, seed=20260602 + idx * 100)
        oof_metrics = base._metrics(y_train[covered], oof[covered])
        final_model = base._fit_catboost(train[feature_cols], y_train, seed=20260602 + idx, iterations=800)
        oos_proba = base._proba3(final_model, oos[feature_cols])
        oos_metrics = base._metrics(y_oos, oos_proba)

        oof_out = base._outputs(train.loc[covered].reset_index(drop=True), oof[covered], prefix="omega1_dir_ctx_oof")
        oos_out = base._outputs(oos, oos_proba, prefix="omega1_dir_ctx")
        oof_path = variant_dir / f"training_features_2025_{variant}_omega1_direction_context_oof_20260602.csv"
        oos_path = variant_dir / f"training_features_2026_rebuilt_{variant}_omega1_direction_context_20260602.csv"
        oof_out.to_csv(oof_path, index=False)
        oos_out.to_csv(oos_path, index=False)
        model_path = variant_dir / f"{variant}_omega1_direction_context.cbm"
        final_model.save_model(str(model_path))
        contract_path = variant_dir / f"{variant}_omega1_direction_context_contract.joblib"
        joblib.dump(
            {
                "variant": variant,
                "label_source": "zigzag_action",
                "base_cols": BASE_COLS,
                "added_cols": add_cols,
                "feature_cols": feature_cols,
            },
            contract_path,
        )
        delta = {
            "oos_bacc": float(oos_metrics["balanced_accuracy"] - BASELINE["oos_bacc"]),
            "oos_auc": None if oos_metrics["ovr_auc"] is None else float(oos_metrics["ovr_auc"] - BASELINE["oos_auc"]),
            "oos_proxy_wr": None if oos_metrics["proxy_wr"] is None else float(oos_metrics["proxy_wr"] - BASELINE["oos_proxy_wr"]),
            "oos_proxy_trades": int(oos_metrics["proxy_trades"] - BASELINE["oos_proxy_trades"]),
        }
        payload = {
            "variant": variant,
            "feature_count": int(len(feature_cols)),
            "added_feature_count": int(len(add_cols)),
            "added_cols": add_cols,
            "oof_metrics": oof_metrics,
            "oos_metrics": oos_metrics,
            "delta_vs_core_plus_tsfm_chronos": delta,
            "folds": folds,
            "artifacts": {
                "oof_2025": str(oof_path),
                "oos_2026": str(oos_path),
                "model": str(model_path),
                "contract": str(contract_path),
            },
        }
        report["variants"][variant] = payload
        ranking.append(
            {
                "variant": variant,
                "feature_count": int(len(feature_cols)),
                "added_feature_count": int(len(add_cols)),
                "oof_bacc": oof_metrics["balanced_accuracy"],
                "oof_auc": oof_metrics["ovr_auc"],
                "oof_proxy_wr": oof_metrics["proxy_wr"],
                "oos_bacc": oos_metrics["balanced_accuracy"],
                "oos_auc": oos_metrics["ovr_auc"],
                "oos_proxy_wr": oos_metrics["proxy_wr"],
                "oos_proxy_trades": oos_metrics["proxy_trades"],
                "delta_oos_bacc_vs_baseline": delta["oos_bacc"],
                "delta_oos_auc_vs_baseline": delta["oos_auc"],
                "delta_oos_proxy_wr_vs_baseline": delta["oos_proxy_wr"],
                "delta_oos_proxy_trades_vs_baseline": delta["oos_proxy_trades"],
            }
        )

    ranking.sort(key=lambda r: (float(r["oos_bacc"]), float(r["oos_auc"] or 0.0)), reverse=True)
    report["ranking"] = ranking
    report["selected_by_oos_bacc"] = ranking[0]["variant"]
    pd.DataFrame(ranking).to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": ranking}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
