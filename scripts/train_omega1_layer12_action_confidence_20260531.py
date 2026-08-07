#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_layer12_action_confidence_20260531"

DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_AI_DIR = ROOT / "tmp/causal_regen_20260516/ai_role_specific_eval_20260530"
DEFAULT_CHRONOS_DIR = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"
DEFAULT_REGIME3_STABILITY_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_REGIME3_CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
DEFAULT_DIR3_PATCH_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_patch_full_20260531"
DEFAULT_DIR3_VSNLSTM_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_layer12_action_confidence_20260531"
DROP_EVENTS: list[dict[str, Any]] = []

L1_CORE64 = [
    "log_return",
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "garman_klass_vol",
    "realized_vol_ratio",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "amihud_illiquidity_z",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "chop_index",
    "cvp_poc_dist",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "cvp_regime",
    "breakout_strength",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "funding_price_divergence",
    "regime_trending",
    "ofi_acceleration",
    "kalman_velocity",
    "ofti",
    "kel",
    "funding_abs",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_slope_12",
    "cvd_slope_48",
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_6",
    "eth_btc_ret_spread_12",
    "compression_score",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "funding_oi_divergence",
    "oi_up_price_down",
    "oi_up_price_up",
    "upper_wick_z",
    "lower_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "garch_vol_z",
    "jump_z",
    "evt_excess_z",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
    "regime_persistence",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]

AI_LAYER2 = [
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_zscore",
]

CHRONOS_LAYER2 = [
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
]

REGIME3_CURRENT_LAYER2 = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

REGIME3_STABILITY_LAYER2 = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]

REGIME3_CMAMBA_LAYER2 = [
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
]

M7_LAYER2 = [
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]

DIR3_PATCH_LAYER2 = [
    "dir3_patch_h6_fl_prob",
    "dir3_patch_h6_up_prob",
    "dir3_patch_h6_dn_prob",
    "dir3_patch_h6_confidence",
    "dir3_patch_h6_side_edge",
    "dir3_patch_h6_trade_prob",
]

DIR3_VSNLSTM_LAYER2 = [
    "dir3_vsnlstm_h6_fl_prob",
    "dir3_vsnlstm_h6_up_prob",
    "dir3_vsnlstm_h6_dn_prob",
    "dir3_vsnlstm_h6_confidence",
    "dir3_vsnlstm_h6_side_edge",
    "dir3_vsnlstm_h6_trade_prob",
]

RAW_LEVEL_BLOCK = {
    "open",
    "high",
    "low",
    "close",
    "close_btc",
    "volume_btc",
    "quote_volume_btc",
}

FORBIDDEN_PREFIXES = (
    "teacher_",
    "a5dir_",
    "clean_regime_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
)
FORBIDDEN_SUBSTRINGS = ("label", "target", "future", "pnl", "action_score", "wave3")
FORBIDDEN_EXACT = {
    "timestamp",
    "zigzag_action",
    "zigzag_action_name",
    "zigzag_segment_id",
    "zigzag_wave_return",
    "zigzag_wave_bars",
    "zigzag_transition_buffer",
    "zigzag_atr_pct",
    "zigzag_path_return",
    "zigzag_path_mae",
    "zigzag_path_mfe",
    "zigzag_path_calmar",
    "zigzag_path_edge",
    "zigzag_soft_cash",
    "zigzag_soft_long",
    "zigzag_soft_short",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_zigzag_cat_action",
    "m7_zigzag_xgb_action",
    "pred_patchtst",
    "conf_patchtst",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_csv(path: Path, *, parse_dates: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    kwargs: dict[str, Any] = {"low_memory": False}
    if parse_dates:
        kwargs["parse_dates"] = ["timestamp"]
    frame = pd.read_csv(path, **kwargs)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
    source: str,
    *,
    allow_tail_drop: bool = False,
    allow_head_drop: bool = False,
    allow_sparse_drop: bool = False,
) -> pd.DataFrame:
    missing_cols = sorted(set(cols) - set(right.columns))
    if missing_cols:
        raise ValueError(f"{source} missing required columns: {missing_cols}")
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    missing = {c: int(merged[c].isna().sum()) for c in cols if int(merged[c].isna().sum()) > 0}
    if missing:
        miss_any = merged[cols].isna().any(axis=1).to_numpy()
        miss_idx = np.flatnonzero(miss_any)
        tail_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(len(merged) - miss_idx.size, len(merged)))
        head_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(miss_idx.size))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        if allow_head_drop and head_only:
            DROP_EVENTS.append({"source": source, "drop_type": "head", "rows": int(miss_idx.size), "missing": missing})
            return merged.iloc[miss_idx.size :].reset_index(drop=True)
        if allow_sparse_drop:
            DROP_EVENTS.append({"source": source, "drop_type": "sparse", "rows": int(miss_idx.size), "missing": missing})
            return merged.loc[~miss_any].reset_index(drop=True)
        raise RuntimeError(f"{source} exact timestamp join has missing values: {missing}")
    return merged


def _chronos_features(path: Path, prefix: str) -> pd.DataFrame:
    frame = _read_csv(path)
    required = ["q10", "q50", "q90", "width"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing Chronos columns: {missing}")
    width = pd.to_numeric(frame["width"], errors="raise").clip(lower=0.0)
    q50 = pd.to_numeric(frame["q50"], errors="raise")
    q90 = pd.to_numeric(frame["q90"], errors="raise").clip(lower=0.0)
    out = pd.DataFrame({"timestamp": frame["timestamp"]})
    out[f"chronos_{prefix}_width"] = width
    out[f"chronos_{prefix}_large_move_score"] = width * (1.0 + q50.abs())
    out[f"chronos_{prefix}_upside_band_ewm3"] = q90.ewm(span=3, adjust=False, min_periods=1).mean()
    out[f"chronos_{prefix}_width_ewm6"] = width.ewm(span=6, adjust=False, min_periods=1).mean()
    return out.drop_duplicates("timestamp", keep="last")


def _add_layer2(frame: pd.DataFrame, year: int, args: argparse.Namespace) -> pd.DataFrame:
    tag = "val2025" if int(year) == 2025 else "oos2026"

    ai_name = "tsfm_role_features_2025_exact.csv" if int(year) == 2025 else "tsfm_role_features_2026_exact.csv"
    frame = _exact_join(frame, _read_csv(args.ai_dir / ai_name), AI_LAYER2, f"AI role features {year}")

    atr = _chronos_features(args.chronos_dir / f"atr14_pct_{tag}_chronos.csv", "atr14")
    rv = _chronos_features(args.chronos_dir / f"realized_vol_24_{tag}_chronos.csv", "realized_vol24")
    frame = _exact_join(
        frame,
        atr,
        [
            "chronos_atr14_upside_band_ewm3",
            "chronos_atr14_width_ewm6",
            "chronos_atr14_width",
            "chronos_atr14_large_move_score",
        ],
        f"Chronos atr14 {year}",
    )
    frame = _exact_join(frame, rv, ["chronos_realized_vol24_width", "chronos_realized_vol24_large_move_score"], f"Chronos rv24 {year}")

    stability_name = (
        "training_features_2025_regime3_stability_risk_h6.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
    )
    frame = _exact_join(
        frame,
        _read_csv(args.regime3_stability_dir / stability_name),
        REGIME3_STABILITY_LAYER2,
        f"Regime3 stability {year}",
        allow_tail_drop=True,
    )

    current_name = (
        "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
    )
    frame = _exact_join(frame, _read_csv(args.regime3_current_dir / current_name), REGIME3_CURRENT_LAYER2, f"Regime3 current {year}")

    cmamba_name = (
        "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
    )
    frame = _exact_join(
        frame,
        _read_csv(args.regime3_cmamba_dir / cmamba_name),
        REGIME3_CMAMBA_LAYER2,
        f"Regime3 CryptoMamba {year}",
        allow_head_drop=True,
    )

    m7_name = "rl_training_2025_m7_zigzag_direction.csv" if int(year) == 2025 else "rl_training_2026_m7_zigzag_direction.csv"
    frame = _exact_join(
        frame,
        _read_csv(args.split_dir / m7_name),
        M7_LAYER2,
        f"M7 ZigZag {year}",
        allow_head_drop=True,
        allow_tail_drop=True,
        allow_sparse_drop=True,
    )

    patch_name = (
        "training_features_2025_omega1_dir3_patch_full_20260531.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_omega1_dir3_patch_full_20260531.csv"
    )
    frame = _exact_join(frame, _read_csv(args.dir3_patch_dir / patch_name), DIR3_PATCH_LAYER2, f"dir3 patch full {year}", allow_head_drop=True)

    vsn_name = (
        "training_features_2025_omega1_dir3_vsnlstm_full_20260531.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_omega1_dir3_vsnlstm_full_20260531.csv"
    )
    frame = _exact_join(frame, _read_csv(args.dir3_vsnlstm_dir / vsn_name), DIR3_VSNLSTM_LAYER2, f"dir3 VSN-LSTM full {year}", allow_head_drop=True)
    return frame


def _add_labels(frame: pd.DataFrame, year: int, label_dir: Path) -> pd.DataFrame:
    labels = _read_csv(label_dir / f"zigzag_action_labels_{int(year)}.csv")
    cols = [
        "zigzag_action",
        "zigzag_soft_cash",
        "zigzag_soft_long",
        "zigzag_soft_short",
        "zigzag_path_edge",
    ]
    return _exact_join(frame, labels, cols, f"ZigZag labels {year}")


def _blocked_feature(col: str) -> bool:
    if col in FORBIDDEN_EXACT:
        return True
    if col in RAW_LEVEL_BLOCK:
        return True
    if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return True
    low = col.lower()
    return any(token in low for token in FORBIDDEN_SUBSTRINGS)


def _feature_sets(train: pd.DataFrame, oos: pd.DataFrame) -> dict[str, list[str]]:
    layer2 = (
        AI_LAYER2
        + CHRONOS_LAYER2
        + REGIME3_CURRENT_LAYER2
        + REGIME3_STABILITY_LAYER2
        + REGIME3_CMAMBA_LAYER2
        + M7_LAYER2
        + DIR3_PATCH_LAYER2
        + DIR3_VSNLSTM_LAYER2
    )
    architect_strict_layer2 = (
        AI_LAYER2
        + CHRONOS_LAYER2
        + REGIME3_CURRENT_LAYER2
        + [c for c in M7_LAYER2 if c not in {"m7_q10", "m7_q90", "m7_qwidth"}]
        + DIR3_PATCH_LAYER2
        + DIR3_VSNLSTM_LAYER2
    )
    l1_core = [c for c in L1_CORE64 if c in train.columns and c in oos.columns and not _blocked_feature(c)]
    common = [c for c in train.columns if c in oos.columns]
    l1_all = [
        c
        for c in common
        if c not in layer2
        and not _blocked_feature(c)
        and pd.api.types.is_numeric_dtype(train[c])
        and pd.api.types.is_numeric_dtype(oos[c])
    ]
    sets = {
        "layer2_only": layer2,
        "l1core64_layer2": l1_core + layer2,
        "l1all_safe_layer2": l1_all + layer2,
        "architect_strict_l1all_layer2": l1_all + architect_strict_layer2,
    }
    for name, cols in sets.items():
        missing = sorted(c for c in cols if c not in train.columns or c not in oos.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if not cols:
            raise ValueError(f"{name} feature set is empty")
    return sets


def _fit_model(x: pd.DataFrame, y: np.ndarray, *, seed: int, cfg: dict[str, Any]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=float(cfg["learning_rate"]),
                    max_iter=int(cfg["max_iter"]),
                    max_leaf_nodes=int(cfg["max_leaf_nodes"]),
                    l2_regularization=float(cfg["l2_regularization"]),
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=30,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(x, y, hgb__sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    return model


def _proba3(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.named_steps["hgb"].classes_)
    full = np.zeros((len(x), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]
    return full


def _classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    raw = proba.argmax(axis=1).astype(np.int64)
    conf = proba.max(axis=1)
    pred = np.where(conf >= float(threshold), raw, 0).astype(np.int64)
    trade = pred != 0
    return {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ovr_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2])),
        "raw_balanced_accuracy": float(balanced_accuracy_score(y, raw)),
        "raw_macro_f1": float(f1_score(y, raw, average="macro")),
        "proxy_trades": int(trade.sum()),
        "proxy_long_trades": int((pred == 1).sum()),
        "proxy_short_trades": int((pred == 2).sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((pred[trade] == y[trade]).mean()) if trade.any() else None,
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "mean_confidence": float(np.mean(conf)),
    }


def _decisions(frame: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    raw = proba.argmax(axis=1).astype(np.int64)
    conf = proba.max(axis=1)
    action = np.where(conf >= float(threshold), raw, 0).astype(np.int64)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "action": action,
            "confidence": conf,
            "raw_action": raw,
            "p_cash": proba[:, 0],
            "p_long": proba[:, 1],
            "p_short": proba[:, 2],
        }
    )


def _days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    return float(max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0))


def _backtest(frame: pd.DataFrame, actions: np.ndarray, *, fee: float, slip: float, tp_pct: float, sl_pct: float, max_hold_bars: int, exposure: float) -> dict[str, Any]:
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    hold = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    exits: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * float(exposure))

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, cash, hold, long_entries, short_entries
        fill_i = min(i + 1, len(open_px) - 1)
        side = int(new_side)
        entry = open_px[fill_i] * (1.0 + float(slip) if side > 0 else 1.0 - float(slip))
        entry_equity = cash
        cash -= cash * float(fee) * float(exposure)
        hold = 0
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, trades, wins
        if fill_px is None:
            fill_i = min(i + 1, len(open_px) - 1)
            fill_px = open_px[fill_i] * (1.0 - float(slip) if side > 0 else 1.0 + float(slip))
        before_fee = cash
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        cash = cash * (1.0 + raw * float(exposure))
        cash -= before_fee * float(fee) * float(exposure)
        pnl = cash / max(entry_equity, 1e-12) - 1.0
        trades += 1
        wins += int(pnl > 0.0)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        if side != 0:
            hold += 1
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + float(tp_pct))
                sl_hit = low[i] <= entry * (1.0 - float(sl_pct))
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_sl_first", entry * (1.0 - float(sl_pct)) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + float(tp_pct)) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - float(sl_pct)) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - float(tp_pct))
                sl_hit = high[i] >= entry * (1.0 + float(sl_pct))
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_sl_first", entry * (1.0 + float(sl_pct)) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - float(tp_pct)) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + float(sl_pct)) * (1.0 + float(slip)))
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side != 0 and hold >= int(max_hold_bars):
            exit_pos(i, "max_hold")
        elif side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
    }


def _cost_metrics(frame: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    actions = decisions["action"].to_numpy(dtype=np.int64)
    for mult in (1, 2, 3):
        out[f"cost{mult}"] = _backtest(
            frame,
            actions,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            max_hold_bars=int(args.max_hold_bars),
            exposure=float(args.exposure),
        )
    return out


def _score(m: dict[str, Any]) -> float:
    return float(m["balanced_accuracy"]) + 0.20 * float(m["ovr_auc"]) + 0.05 * float(m["proxy_wr"] or 0.0)


def _select_threshold(y: np.ndarray, proba: np.ndarray, grid: list[float]) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for t in grid:
        m = _classification_metrics(y, proba, t)
        rows.append({"threshold": float(t), "score": _score(m), "metrics": m})
    best = max(rows, key=lambda r: float(r["score"]))
    return float(best["threshold"]), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Omega1 Layer1+Layer2 action/confidence models.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--ai-dir", type=Path, default=DEFAULT_AI_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--regime3-stability-dir", type=Path, default=DEFAULT_REGIME3_STABILITY_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--dir3-patch-dir", type=Path, default=DEFAULT_DIR3_PATCH_DIR)
    parser.add_argument("--dir3-vsnlstm-dir", type=Path, default=DEFAULT_DIR3_VSNLSTM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-start", default="2025-10-01")
    parser.add_argument("--confidence-grid", default="0.34,0.38,0.42,0.46,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--tp-pct", type=float, default=0.018)
    parser.add_argument("--sl-pct", type=float, default=0.010)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slip", type=float, default=0.00015)
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = _read_csv(args.split_dir / "training_features_2025.csv")
    oos = _read_csv(args.split_dir / "training_features_2026_rebuilt.csv")
    train = _add_layer2(train, 2025, args)
    oos = _add_layer2(oos, 2026, args)
    train = _add_labels(train, 2025, args.label_dir)
    oos = _add_labels(oos, 2026, args.label_dir)

    y_all = train["zigzag_action"].astype(int).to_numpy()
    y_oos = oos["zigzag_action"].astype(int).to_numpy()
    val_mask = pd.to_datetime(train["timestamp"]) >= pd.Timestamp(args.val_start)
    fit_idx = np.flatnonzero(~val_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    if len(fit_idx) < 1000 or len(val_idx) < 1000:
        raise RuntimeError(f"bad 2025 fit/validation split: fit={len(fit_idx)} val={len(val_idx)}")

    grids = [
        {"max_iter": 260, "learning_rate": 0.035, "max_leaf_nodes": 31, "l2_regularization": 0.10, "min_samples_leaf": 45},
        {"max_iter": 380, "learning_rate": 0.025, "max_leaf_nodes": 31, "l2_regularization": 0.12, "min_samples_leaf": 60},
    ]
    thresholds = [float(x.strip()) for x in str(args.confidence_grid).split(",") if x.strip()]
    feature_sets = _feature_sets(train, oos)
    runs: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for set_name, cols in feature_sets.items():
        for cfg in grids:
            model = _fit_model(train.iloc[fit_idx][cols], y_all[fit_idx], seed=int(args.seed), cfg=cfg)
            val_proba = _proba3(model, train.iloc[val_idx][cols])
            threshold, threshold_grid = _select_threshold(y_all[val_idx], val_proba, thresholds)
            val_metrics = _classification_metrics(y_all[val_idx], val_proba, threshold)
            run = {
                "feature_set": set_name,
                "feature_count": int(len(cols)),
                "config": cfg,
                "threshold": float(threshold),
                "validation": val_metrics,
                "threshold_grid": threshold_grid,
                "selection_score": _score(val_metrics),
            }
            runs.append(run)
            print(json.dumps({"run": run}, ensure_ascii=False, default=_json_default), flush=True)
            if best is None or float(run["selection_score"]) > float(best["selection_score"]):
                best = {**run, "feature_cols": cols}
    assert best is not None

    final_model = _fit_model(train[best["feature_cols"]], y_all, seed=int(args.seed), cfg=best["config"])
    val_final_proba = _proba3(final_model, train.iloc[val_idx][best["feature_cols"]])
    oos_proba = _proba3(final_model, oos[best["feature_cols"]])
    val_dec = _decisions(train.iloc[val_idx].reset_index(drop=True), val_final_proba, float(best["threshold"]))
    oos_dec = _decisions(oos, oos_proba, float(best["threshold"]))
    val_class = _classification_metrics(y_all[val_idx], val_final_proba, float(best["threshold"]))
    oos_class = _classification_metrics(y_oos, oos_proba, float(best["threshold"]))
    val_cost = _cost_metrics(train.iloc[val_idx].reset_index(drop=True), val_dec, args)
    oos_cost = _cost_metrics(oos, oos_dec, args)

    val_dec.to_csv(args.out_dir / "validation_decisions.csv", index=False)
    oos_dec.to_csv(args.out_dir / "oos_2026_decisions.csv", index=False)
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": best["feature_cols"],
            "confidence_threshold": float(best["threshold"]),
            "feature_set": best["feature_set"],
            "model_id": MODEL_ID,
        },
        args.out_dir / "model.joblib",
    )
    summary = {
        "model_id": MODEL_ID,
        "layer_contract": "Layer1 + Layer2 inputs only; teacher_* and other Layer3 outputs forbidden",
        "label_source": "zigzag_action",
        "train_window": "2025",
        "validation_start": str(args.val_start),
        "oos_window": "2026",
        "best": {
            "feature_set": best["feature_set"],
            "feature_count": int(len(best["feature_cols"])),
            "config": best["config"],
            "confidence_threshold": float(best["threshold"]),
            "validation_selection": best["validation"],
            "validation_final": val_class,
            "oos_2026": oos_class,
            "validation_backtest": val_cost,
            "oos_2026_backtest": oos_cost,
        },
        "all_runs": runs,
        "feature_sets": {name: cols for name, cols in feature_sets.items()},
        "row_drop_events": DROP_EVENTS,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "model": str(args.out_dir / "model.joblib"),
            "validation_decisions": str(args.out_dir / "validation_decisions.csv"),
            "oos_2026_decisions": str(args.out_dir / "oos_2026_decisions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    (args.out_dir / "selected_features.json").write_text(json.dumps(best["feature_cols"], indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
