from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENSEMBLE_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_ENSEMBLE_DIR)
for _p in (_ROOT_DIR, _ENSEMBLE_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.common import (
    load_feature_frame,
    select_feature_columns,
    time_split_indices,
    median_fill_by_train,
    DEFAULT_DATA_PATH,
    DEFAULT_RL_DATA_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MISSING_WARN_RATIO = 0.30
SAVE_PATH = "data/ensemble/supervised/entry_price_model.json"


def _require_lightgbm():
    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except ImportError as e:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm") from e
    return LGBMRegressor


def _combine_pred_conf(df: pd.DataFrame) -> pd.DataFrame:
    pred_conf_map = {
        "pred_chronos": "conf_chronos",
        "pred_patchtst": "conf_patchtst",
        "pred_tide": "conf_tide",
    }
    for pred_col, conf_col in pred_conf_map.items():
        sig_col = pred_col.replace("pred_", "signal_")
        if sig_col in df.columns:
            continue
        if pred_col in df.columns and conf_col in df.columns:
            df[sig_col] = pd.to_numeric(df[pred_col], errors="coerce") * pd.to_numeric(df[conf_col], errors="coerce")
        elif pred_col in df.columns:
            df[sig_col] = pd.to_numeric(df[pred_col], errors="coerce")
    return df


def _add_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else c
    l = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else c

    if "ret_12" not in df.columns:
        df["ret_12"] = np.tanh(c.pct_change(12) * 10)
    if "ret_24" not in df.columns:
        df["ret_24"] = np.tanh(c.pct_change(24) * 10)
    if "ret_48" not in df.columns:
        df["ret_48"] = np.tanh(c.pct_change(48) * 10)
    if "hh_count_24" not in df.columns:
        df["hh_count_24"] = (h > h.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
    if "hl_count_24" not in df.columns:
        df["hl_count_24"] = (l > l.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
    if "trend_accel" not in df.columns:
        df["trend_accel"] = np.tanh((c.pct_change(12) - c.pct_change(48) / 4) * 20)
    return df


def _future_extrema_offsets(df: pd.DataFrame, horizon: int, clip_pct: float) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").fillna(pd.Series(close)).to_numpy(dtype=np.float64) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").fillna(pd.Series(close)).to_numpy(dtype=np.float64) if "low" in df.columns else close

    n = len(df)
    long_offset = np.full(n, np.nan, dtype=np.float64)
    short_offset = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        c = max(close[i], 1e-8)
        future_low = float(np.min(low[i + 1 : i + horizon + 1]))
        future_high = float(np.max(high[i + 1 : i + horizon + 1]))
        long_offset[i] = np.clip(future_low / c - 1.0, -clip_pct, 0.0)
        short_offset[i] = np.clip(future_high / c - 1.0, 0.0, clip_pct)
    return long_offset, short_offset


class EntryPriceBrain:
    def __init__(self):
        self.long_model = None
        self.short_model = None
        self.feature_cols: list[str] = []
        self.horizon = 3
        self.long_clip = 0.02
        self.short_clip = 0.02
        self._last_missing_ratio = 0.0

    def _prepare_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        df_w = df.copy()
        if timestamp_col in df_w.columns:
            df_w[timestamp_col] = pd.to_datetime(df_w[timestamp_col], errors="coerce")
            df_w = df_w.set_index(timestamp_col).sort_index()
        df_w = _combine_pred_conf(df_w)
        df_w = _add_trend_structure_features(df_w)

        missing_cols = [c for c in self.feature_cols if c not in df_w.columns]
        for col in missing_cols:
            df_w[col] = np.nan
        self._last_missing_ratio = len(missing_cols) / max(len(self.feature_cols), 1)
        if missing_cols:
            if self._last_missing_ratio >= MISSING_WARN_RATIO:
                logger.warning(
                    "EntryPrice 입력 피처 누락률 높음: %d/%d (%.1f%%)",
                    len(missing_cols), len(self.feature_cols), self._last_missing_ratio * 100.0,
                )

        x = df_w[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        return x.astype(np.float32)

    def predict_batch_from_df(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        n = len(df)
        if n == 0:
            return pd.DataFrame(index=df.index)

        out = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float64)

        if self.long_model is None or self.short_model is None or not self.feature_cols:
            long_offset = np.zeros(n, dtype=np.float64)
            short_offset = np.zeros(n, dtype=np.float64)
        else:
            x = self._prepare_features(df, timestamp_col=timestamp_col)
            if self._last_missing_ratio >= 0.35:
                # 학습-추론 피처 스키마 불일치가 크면 안전하게 오프셋 비활성화
                long_offset = np.zeros(n, dtype=np.float64)
                short_offset = np.zeros(n, dtype=np.float64)
            else:
                x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
                long_offset = np.asarray(self.long_model.predict(x), dtype=np.float64).reshape(-1)
                short_offset = np.asarray(self.short_model.predict(x), dtype=np.float64).reshape(-1)
                long_offset = np.clip(long_offset, -self.long_clip, 0.0)
                short_offset = np.clip(short_offset, 0.0, self.short_clip)

        out["entry_long_offset"] = long_offset.astype(np.float32)
        out["entry_short_offset"] = short_offset.astype(np.float32)
        out["entry_long_price"] = (close * (1.0 + long_offset)).astype(np.float32)
        out["entry_short_price"] = (close * (1.0 + short_offset)).astype(np.float32)
        return out

    def predict_from_df(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> Optional[dict]:
        pred = self.predict_batch_from_df(df.tail(1), timestamp_col=timestamp_col)
        if pred.empty:
            return None
        row = pred.iloc[-1]
        return {k: float(v) for k, v in row.to_dict().items()}

    @classmethod
    def load(cls, meta_path: str = SAVE_PATH) -> "EntryPriceBrain":
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"entry price meta file missing: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_ref = meta.get("model_path", "")
        model_path = model_ref if os.path.isabs(model_ref) else os.path.join(os.path.dirname(meta_path), model_ref)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"entry price model file missing: {model_path}")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        inst = cls()
        inst.long_model = payload.get("long_model")
        inst.short_model = payload.get("short_model")
        inst.feature_cols = list(payload.get("feature_cols", meta.get("feature_cols", [])))
        inst.horizon = int(payload.get("horizon", meta.get("horizon", 3)))
        inst.long_clip = float(payload.get("long_clip", meta.get("long_clip", 0.02)))
        inst.short_clip = float(payload.get("short_clip", meta.get("short_clip", 0.02)))
        if not inst.feature_cols or inst.long_model is None or inst.short_model is None:
            raise ValueError("entry price model payload incomplete")
        logger.info("✅ EntryPriceBrain 로드 완료: %s (%d개 피처)", model_path, len(inst.feature_cols))
        return inst


def train(args: argparse.Namespace) -> dict:
    LGBMRegressor = _require_lightgbm()

    df = load_feature_frame(args.data_path, args.rl_path)
    df = _add_trend_structure_features(df)
    long_target, short_target = _future_extrema_offsets(df, horizon=args.horizon, clip_pct=args.clip_pct)
    valid = np.isfinite(long_target) & np.isfinite(short_target)
    df = df.loc[valid].reset_index(drop=True)
    y_long = long_target[valid]
    y_short = short_target[valid]

    feature_cols = select_feature_columns(df)
    if args.max_features > 0:
        feature_cols = feature_cols[: args.max_features]
    x_all = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    tr_idx, va_idx, te_idx = time_split_indices(len(df), args.train_ratio, args.val_ratio)
    x_train = x_all.iloc[tr_idx].copy()
    x_val = x_all.iloc[va_idx].copy()
    x_test = x_all.iloc[te_idx].copy()
    x_train, x_val = median_fill_by_train(x_train, x_val)
    x_train, x_test = median_fill_by_train(x_train, x_test)

    params = dict(
        objective="quantile",
        alpha=0.5,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )

    long_model = LGBMRegressor(**params)
    short_model = LGBMRegressor(**params)
    long_model.fit(x_train, y_long[tr_idx])
    short_model.fit(x_train, y_short[tr_idx])

    long_pred = np.asarray(long_model.predict(x_test), dtype=np.float64)
    short_pred = np.asarray(short_model.predict(x_test), dtype=np.float64)
    long_mae = float(np.mean(np.abs(long_pred - y_long[te_idx])))
    short_mae = float(np.mean(np.abs(short_pred - y_short[te_idx])))

    save_path = args.save_path or SAVE_PATH
    model_path = os.path.splitext(save_path)[0] + ".pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "long_model": long_model,
        "short_model": short_model,
        "feature_cols": feature_cols,
        "horizon": int(args.horizon),
        "long_clip": float(args.clip_pct),
        "short_clip": float(args.clip_pct),
        "metrics": {"long_mae": long_mae, "short_mae": short_mae},
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "model_path": os.path.basename(model_path),
        "feature_cols": feature_cols,
        "horizon": int(args.horizon),
        "long_clip": float(args.clip_pct),
        "short_clip": float(args.clip_pct),
        "metrics": {"long_mae": long_mae, "short_mae": short_mae},
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("✅ Entry price model saved: %s", save_path)
    logger.info("test_long_mae=%.6f | test_short_mae=%.6f", long_mae, short_mae)
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train entry price recommendation model")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--rl-path", default=DEFAULT_RL_DATA_PATH)
    p.add_argument("--save-path", default=SAVE_PATH)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--clip-pct", type=float, default=0.02)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-samples", type=int, default=40)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_entry_price_model")
        raise SystemExit(0)
    train(args)
