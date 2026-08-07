from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from ensemble.supervised.train_trend_xgb import XGBTrendBrain
from ensemble.supervised.train_entry_price_model import EntryPriceBrain

logger = logging.getLogger(__name__)


class MultiTargetLGBMBrain:
    """multi_target_lgbm.pkl 로드 후 Brain B 형식 추세 신호를 생성."""

    MISSING_WARN_RATIO = 0.30

    def __init__(self):
        self.available = False
        self.feature_cols: list[str] = []
        self.direction_model = None
        self.quality_model = None
        self.hold_model = None

    @staticmethod
    def _resolve_model_path(meta_path: str) -> tuple[str, list[str]]:
        feature_cols = []
        model_path = ""
        base_dir = os.path.dirname(meta_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            feature_cols = list(data.get("feature_cols", []))
            model_ref = data.get("model_path", "")
            if model_ref:
                model_path = model_ref if os.path.isabs(model_ref) else os.path.join(base_dir, model_ref)
        else:
            prefix = os.path.splitext(meta_path)[0]
            model_path = f"{prefix}.pkl"

        return model_path, feature_cols

    @classmethod
    def load(cls, meta_path: str = "data/ensemble/supervised/multi_target_lgbm.json") -> "MultiTargetLGBMBrain":
        instance = cls()
        model_path, feature_cols = cls._resolve_model_path(meta_path)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"MultiTarget LGBM 모델 파일 누락: {model_path}")

        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        instance.direction_model = payload["direction_model"]
        instance.quality_model = payload["quality_model"]
        instance.hold_model = payload["hold_model"]
        instance.feature_cols = feature_cols or list(payload.get("feature_cols", []))
        if not instance.feature_cols:
            raise ValueError("feature_cols 복원 실패")

        instance.available = True
        logger.info(
            "✅ MultiTargetLGBMBrain 로드 완료: %s (%d개 피처)",
            model_path,
            len(instance.feature_cols),
        )
        return instance

    def _prepare_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        df_w = df.copy()
        if timestamp_col in df_w.columns:
            df_w[timestamp_col] = pd.to_datetime(df_w[timestamp_col])
            df_w = df_w.set_index(timestamp_col).sort_index()

        pred_conf_map = {
            "pred_chronos": "conf_chronos",
            "pred_patchtst": "conf_patchtst",
            "pred_tide": "conf_tide",
        }
        for pred_col, conf_col in pred_conf_map.items():
            sig_col = pred_col.replace("pred_", "signal_")
            if sig_col not in df_w.columns:
                if pred_col in df_w.columns and conf_col in df_w.columns:
                    df_w[sig_col] = df_w[pred_col] * df_w[conf_col]
                elif pred_col in df_w.columns:
                    df_w[sig_col] = df_w[pred_col]

        trend_feats = ["ret_12", "ret_24", "ret_48", "hh_count_24", "hl_count_24", "trend_accel"]
        if any(f in self.feature_cols and f not in df_w.columns for f in trend_feats):
            c = df_w["close"]
            h = df_w["high"] if "high" in df_w.columns else c
            l = df_w["low"] if "low" in df_w.columns else c
            if "ret_12" not in df_w.columns:
                df_w["ret_12"] = np.tanh(c.pct_change(12) * 10)
            if "ret_24" not in df_w.columns:
                df_w["ret_24"] = np.tanh(c.pct_change(24) * 10)
            if "ret_48" not in df_w.columns:
                df_w["ret_48"] = np.tanh(c.pct_change(48) * 10)
            if "hh_count_24" not in df_w.columns:
                df_w["hh_count_24"] = (h > h.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
            if "hl_count_24" not in df_w.columns:
                df_w["hl_count_24"] = (l > l.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
            if "trend_accel" not in df_w.columns:
                df_w["trend_accel"] = np.tanh((c.pct_change(12) - c.pct_change(48) / 4) * 20)

        missing_cols = [col for col in self.feature_cols if col not in df_w.columns]
        for col in missing_cols:
            df_w[col] = np.nan

        if missing_cols:
            miss_ratio = len(missing_cols) / max(len(self.feature_cols), 1)
            if miss_ratio >= self.MISSING_WARN_RATIO:
                sample = ", ".join(missing_cols[:6])
                logger.warning(
                    "MultiTarget 입력 피처 누락률 높음: %d/%d (%.1f%%) | sample=[%s]",
                    len(missing_cols),
                    len(self.feature_cols),
                    miss_ratio * 100.0,
                    sample,
                )

        return df_w

    def predict_from_df(self, df: pd.DataFrame, timestamp_col: str = "timestamp", min_candles: int = 1) -> dict | None:
        if not self.available or self.direction_model is None:
            return None
        if len(df) < min_candles:
            return None

        df_w = self._prepare_features(df, timestamp_col=timestamp_col)
        last_row = df_w[self.feature_cols].iloc[[-1]].astype(np.float32)
        last_row = last_row.replace([np.inf, -np.inf], np.nan)

        probs_arr = np.asarray(self.direction_model.predict(last_row), dtype=np.float64)
        probs = probs_arr.reshape(-1)
        if probs.size != 2:
            raise ValueError(f"MULTITARGET_LGBM direction model must emit 2-class [DOWN, UP], got shape={probs_arr.shape}")
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        denom = float(probs.sum())
        if denom <= 1e-12:
            probs = np.array([0.5, 0.5], dtype=np.float64)
        else:
            probs = probs / denom

        trend_dir = 2 if probs[1] >= probs[0] else 0
        strength = float(np.clip(abs(probs[1] - probs[0]), 0.0, 1.0))
        rev_prob = float(probs[0] if trend_dir == 2 else probs[1])

        quality_pred = float(np.asarray(self.quality_model.predict(last_row), dtype=np.float64).reshape(-1)[0])
        hold_pred = float(np.asarray(self.hold_model.predict(last_row), dtype=np.float64).reshape(-1)[0])

        p_down, p_up = (float(probs[0]), float(probs[1]))
        return {
            "trend_dir": trend_dir,
            "strength": strength,
            "rev_prob": rev_prob,
            "probs": [p_down, p_up],
            "p_down": p_down,
            "p_up": p_up,
            "prob_dn": p_down,
            "prob_up": p_up,
            "quality_pred": quality_pred,
            "hold_pred": hold_pred,
            "trend_model": "MULTITARGET_LGBM",
        }


def trend_signal_to_dict(signal, default_model: str) -> dict | None:
    if signal is None:
        return None
    if isinstance(signal, dict):
        out = dict(signal)
    elif hasattr(signal, "to_arbiter_dict"):
        out = signal.to_arbiter_dict()
    else:
        return None

    probs = out.get("probs", [])
    if not isinstance(probs, (list, tuple)) or len(probs) < 2:
        p_dn = float(out.get("p_down", out.get("prob_dn", 0.5)))
        p_up = float(out.get("p_up", out.get("prob_up", 0.5)))
        probs = [p_dn, p_up]
    probs = np.asarray(probs[:2], dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(probs.sum())
    if denom <= 1e-12:
        probs = np.array([0.5, 0.5], dtype=np.float64)
    else:
        probs = probs / denom

    out["probs"] = [float(probs[0]), float(probs[1])]
    out["p_down"] = float(probs[0])
    out["p_up"] = float(probs[1])
    out["prob_dn"] = float(probs[0])
    out["prob_up"] = float(probs[1])
    out["trend_dir"] = int(out.get("trend_dir", 2 if probs[1] >= probs[0] else 0))
    out["strength"] = float(out.get("strength", np.clip(abs(probs[1] - probs[0]), 0.0, 1.0)))
    if "rev_prob" not in out:
        if out["trend_dir"] == 2:
            out["rev_prob"] = float(probs[0])
        else:
            out["rev_prob"] = float(probs[1])
    out.setdefault("trend_model", default_model)
    return out


def blend_trend_signals(sig_a: dict, sig_b: dict, w_a: float = 0.5, w_b: float = 0.5) -> dict:
    pa = np.asarray(sig_a.get("probs", [0.5, 0.5]), dtype=np.float64)
    pb = np.asarray(sig_b.get("probs", [0.5, 0.5]), dtype=np.float64)
    if pa.size < 2:
        pa = np.pad(pa, (0, 2 - pa.size), constant_values=0.0)
    if pb.size < 2:
        pb = np.pad(pb, (0, 2 - pb.size), constant_values=0.0)
    pa = np.nan_to_num(pa[:2], nan=0.0, posinf=0.0, neginf=0.0)
    pb = np.nan_to_num(pb[:2], nan=0.0, posinf=0.0, neginf=0.0)
    p = (float(w_a) * pa) + (float(w_b) * pb)
    denom = float(p.sum())
    if denom <= 1e-12:
        p = np.array([0.5, 0.5], dtype=np.float64)
    else:
        p = p / denom

    trend_dir = 2 if p[1] >= p[0] else 0
    strength = float(np.clip(abs(p[1] - p[0]), 0.0, 1.0))
    rev_prob = float(p[0] if trend_dir == 2 else p[1])

    q_vals = [sig_a.get("quality_pred"), sig_b.get("quality_pred")]
    q_vals = [float(v) for v in q_vals if v is not None and np.isfinite(v)]
    h_vals = [sig_a.get("hold_pred"), sig_b.get("hold_pred")]
    h_vals = [float(v) for v in h_vals if v is not None and np.isfinite(v)]

    return {
        "trend_dir": trend_dir,
        "strength": strength,
        "rev_prob": rev_prob,
        "probs": [float(p[0]), float(p[1])],
        "p_down": float(p[0]),
        "p_up": float(p[1]),
        "prob_dn": float(p[0]),
        "prob_up": float(p[1]),
        "quality_pred": float(np.mean(q_vals)) if q_vals else None,
        "hold_pred": float(np.mean(h_vals)) if h_vals else None,
        "trend_model": f"{sig_a.get('trend_model', 'A')}+{sig_b.get('trend_model', 'B')}",
        "sub_signals": {"a": sig_a, "b": sig_b},
    }


class SupervisedTrendHub:
    """XGB + MultiTarget 추세 모델을 로드하고 단일 Brain B 시그널로 반환."""

    def __init__(
        self,
        xgb_meta_path: str = "data/ensemble/supervised/trend_xgb.json",
        multitarget_meta_path: str = "data/ensemble/supervised/multi_target_lgbm.json",
        blend_weights: tuple[float, float] = (0.5, 0.5),
    ):
        self.xgb = None
        self.multitarget = None
        self.w_xgb = float(blend_weights[0])
        self.w_mt = float(blend_weights[1])

        try:
            self.xgb = XGBTrendBrain.load(xgb_meta_path)
            logger.info("✅ SupervisedTrendHub: XGBTrendBrain 로드 완료")
        except Exception as e:
            logger.warning("⚠️ SupervisedTrendHub: XGBTrendBrain 미로드: %s", e)

        try:
            self.multitarget = MultiTargetLGBMBrain.load(multitarget_meta_path)
        except Exception as e:
            logger.warning("⚠️ SupervisedTrendHub: MultiTargetLGBMBrain 미로드: %s", e)

    @property
    def available(self) -> bool:
        return (self.xgb is not None) or (self.multitarget is not None)

    def status(self) -> dict:
        return {
            "xgb_loaded": self.xgb is not None,
            "multitarget_loaded": self.multitarget is not None,
            "available": self.available,
            "weights": [self.w_xgb, self.w_mt],
        }

    def predict_from_df(self, df: pd.DataFrame) -> dict | None:
        xgb_signal_dict = None
        mt_signal_dict = None

        if self.xgb is not None:
            xgb_signal = self.xgb.predict_from_df(df)
            xgb_signal_dict = trend_signal_to_dict(xgb_signal, default_model="XGB_TREND")
        if self.multitarget is not None:
            mt_signal = self.multitarget.predict_from_df(df)
            mt_signal_dict = trend_signal_to_dict(mt_signal, default_model="MULTITARGET_LGBM")

        if xgb_signal_dict is not None and mt_signal_dict is not None:
            return blend_trend_signals(xgb_signal_dict, mt_signal_dict, w_a=self.w_xgb, w_b=self.w_mt)
        if xgb_signal_dict is not None:
            return xgb_signal_dict
        if mt_signal_dict is not None:
            return mt_signal_dict
        return None


class EntryPriceHub:
    """Entry price recommendation model loader."""

    def __init__(self, meta_path: str = "data/ensemble/supervised/entry_price_model.json"):
        self.brain = None
        try:
            self.brain = EntryPriceBrain.load(meta_path)
        except Exception as e:
            logger.warning("⚠️ EntryPriceHub: EntryPriceBrain 미로드: %s", e)

    @property
    def available(self) -> bool:
        return self.brain is not None

    def status(self) -> dict:
        return {
            "entry_price_loaded": self.available,
            "feature_count": len(self.brain.feature_cols) if self.brain is not None else 0,
        }

    def predict_from_df(self, df: pd.DataFrame) -> dict | None:
        if self.brain is None:
            return None
        return self.brain.predict_from_df(df)
