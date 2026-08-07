from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_core_numeric
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_core_numeric)
except Exception:
    pass

from ensemble.supervised.train_trend_xgb import XGBTrendBrain
from ensemble.supervised.train_entry_price_model import EntryPriceBrain

logger = logging.getLogger(__name__)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

def _softmax2(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -50.0, 50.0))
    s = ez.sum(axis=1, keepdims=True)
    s = np.where(s <= 1e-12, 1.0, s)
    return ez / s

def _safe_prob2(arr: np.ndarray, *, source: str) -> np.ndarray:
    p = np.asarray(arr, dtype=np.float64)
    if p.ndim == 1:
        cls = np.clip(p.astype(np.int64), 0, 1)
        out = np.zeros((len(cls), 2), dtype=np.float64)
        out[np.arange(len(cls)), cls] = 1.0
        return out
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"{source} must emit 2-class direction probabilities [DOWN, UP], got shape={p.shape}")
    p = np.nan_to_num(p, nan=0.0)
    denom = p.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return p / denom

def _robust_z_series(s: pd.Series, window: int = 288, min_periods: int = 48) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    med = x.rolling(window, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median()
    z = (x - med) / (1.4826 * mad.replace(0.0, np.nan) + 1e-6)
    return z.fillna(0.0).clip(-8.0, 8.0).to_numpy(dtype=np.float64)

def _resolve_model_path(meta_path: str, model_ref: str | None) -> str:
    if not model_ref: return os.path.splitext(meta_path)[0] + ".pkl"
    if os.path.isabs(model_ref): return model_ref
    return os.path.join(os.path.dirname(meta_path), model_ref)

def _enrich_m7_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        hour, minute = ts.dt.hour.fillna(0), ts.dt.minute.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        df["minute_sin"] = np.sin(2 * np.pi * minute / 60).astype(np.float32)
        df["minute_cos"] = np.cos(2 * np.pi * minute / 60).astype(np.float32)
        df["session_europe"] = ((hour >= 8) & (hour < 16)).astype(np.float32)
        df["is_hour_open"] = (minute < 5).astype(np.float32)
    if "chop_index" not in df.columns and all(c in df.columns for c in ["high", "low", "close"]):
        h, l, c = pd.to_numeric(df["high"]), pd.to_numeric(df["low"]), pd.to_numeric(df["close"])
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr_sum = tr.rolling(14).sum()
        hmax, lmin = h.rolling(14).max(), l.rolling(14).min()
        df["chop_index"] = (100 * np.log10((atr_sum+1e-8)/(hmax-lmin+1e-8))/np.log10(14)).fillna(50.0).astype(np.float32)
    return df

def _add_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = pd.to_numeric(df["close"], errors="coerce")
    df["ret_12"] = np.tanh(c.pct_change(12) * 10)
    df["ret_48"] = np.tanh(c.pct_change(48) * 10)
    df["trend_accel"] = np.tanh((c.pct_change(12) - c.pct_change(48)/4)*20)
    return df

def _to_numeric_frame(df: pd.DataFrame, cols: list[str], fill_mode: str = "median") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols: out[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    if fill_mode == "median": out = out.fillna(out.median(numeric_only=True).fillna(0))
    return out.astype(np.float32)

@dataclass
class _ModelState:
    available: bool
    model: Any = None
    feature_cols: list[str] | None = None
    extra: dict[str, Any] | None = None
    reason: str = ""

class SevenModelEnsemble:
    DEFAULT_META_PATHS = {
        "trend_xgb": "data/ensemble/supervised/trend_xgb.json",
        "entry_price_model": "data/ensemble/supervised/entry_price_model.json",
        "multi_target_lgbm": "data/ensemble/supervised/multi_target_lgbm.json",
        "quantile_forest": "data/ensemble/supervised/quantile_forest.json",
        "lightgbm_ensemble": "data/ensemble/supervised/lightgbm_ensemble.json",
    }

    def __init__(self, meta_paths: dict = None, weight_trend_xgb: float = 0.45, weight_multitarget: float = 0.35, weight_quantile: float = 0.20, strict: bool = True):
        self.meta_paths = {**self.DEFAULT_META_PATHS, **(meta_paths or {})}
        self.weight_trend_xgb, self.weight_multitarget, self.weight_quantile = weight_trend_xgb, weight_multitarget, weight_quantile
        self.strict = strict
        self.trend_xgb = self.entry_price = self.multi_target = self.quantile = self.lightgbm_ensemble = _ModelState(False)
        self._load_all()

    def _load_all(self):
        self._load_trend_xgb()
        self._load_entry_price()
        self._load_multitarget()
        self._load_quantile()
        self._load_lightgbm_ensemble()

    def _load_meta(self, key):
        path = self.meta_paths.get(key)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        return None

    def _load_pickle_from_meta(self, key):
        meta = self._load_meta(key)
        if not meta: return None, None
        model_path = _resolve_model_path(self.meta_paths[key], meta.get("model_path"))
        if not os.path.exists(model_path): return None, meta
        try:
            with open(model_path, "rb") as f: return pickle.load(f), meta
        except Exception: return None, meta

    def _load_trend_xgb(self):
        try:
            brain = XGBTrendBrain.load(self.meta_paths["trend_xgb"])
            self.trend_xgb = _ModelState(True, model=brain.model, feature_cols=list(brain.feature_cols))
        except Exception as e: logger.warning(f"trend_xgb load failed: {e}")

    def _load_entry_price(self):
        try:
            brain = EntryPriceBrain.load(self.meta_paths["entry_price_model"])
            self.entry_price = _ModelState(True, model=brain, feature_cols=list(brain.feature_cols))
        except Exception: pass

    def _load_multitarget(self):
        payload, meta = self._load_pickle_from_meta("multi_target_lgbm")
        if payload: self.multi_target = _ModelState(True, model=payload, feature_cols=list(meta.get("feature_cols", [])))

    def _load_quantile(self):
        payload, meta = self._load_pickle_from_meta("quantile_forest")
        if payload: self.quantile = _ModelState(True, model=payload.get("models", {}), feature_cols=list(meta.get("feature_cols", [])))

    def _load_lightgbm_ensemble(self):
        payload, meta = self._load_pickle_from_meta("lightgbm_ensemble")
        if payload:
            self.lightgbm_ensemble = _ModelState(True, model=payload.get("models", {}), feature_cols=list(meta.get("feature_cols", [])), extra=payload)

    def _predict_trend_xgb(self, df):
        if not self.trend_xgb.available: return np.full((len(df), 2), np.nan)
        x = _to_numeric_frame(_add_trend_structure_features(df), self.trend_xgb.feature_cols)
        return _safe_prob2(self.trend_xgb.model.predict_proba(x), source="trend_xgb")

    def _predict_multitarget(self, df):
        n = len(df)
        res = {"probs": np.full((n, 2), np.nan), "quality": np.full(n, np.nan), "hold": np.full(n, np.nan)}
        if not self.multi_target.available: return res
        x = _to_numeric_frame(df, self.multi_target.feature_cols)
        m = self.multi_target.model
        if "direction_model" in m: res["probs"] = _safe_prob2(m["direction_model"].predict_proba(x), source="multi_target_lgbm")
        if "quality_model" in m: res["quality"] = m["quality_model"].predict(x)
        if "hold_model" in m: res["hold"] = m["hold_model"].predict(x)
        return res

    def _predict_quantile(self, df):
        n = len(df)
        res = {"q10": np.full(n, np.nan), "q50": np.full(n, np.nan), "q90": np.full(n, np.nan), "probs": np.full((n, 2), np.nan)}
        if not self.quantile.available: return res
        x = _to_numeric_frame(df, self.quantile.feature_cols)
        m = self.quantile.model
        res["q10"], res["q50"], res["q90"] = m["q10"].predict(x), m["q50"].predict(x), m["q90"].predict(x)
        width = np.maximum(res["q90"] - res["q10"], 1e-6)
        z = res["q50"] / width
        res["probs"] = _softmax2(np.column_stack([-z, z]))
        return res

    def _predict_entry_price(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy()
        res = {
            "entry_long_price": close.copy(),
            "entry_short_price": close.copy(),
            "entry_long_offset": np.zeros(n, dtype=np.float64),
            "entry_short_offset": np.zeros(n, dtype=np.float64),
        }
        if not self.entry_price.available: return res
        try:
            pred = self.entry_price.model.predict_batch_from_df(df)
            if not pred.empty:
                res["entry_long_price"] = pred["entry_long_price"].to_numpy()
                res["entry_short_price"] = pred["entry_short_price"].to_numpy()
                if "entry_long_offset" in pred.columns:
                    res["entry_long_offset"] = pred["entry_long_offset"].to_numpy()
                if "entry_short_offset" in pred.columns:
                    res["entry_short_offset"] = pred["entry_short_offset"].to_numpy()
        except Exception: pass
        return res

    def _predict_lightgbm_ensemble(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        keys = [
            "long_edge",
            "short_edge",
            "tradeability_score",
            "long_mae_q90",
            "short_mae_q90",
            "long_adverse_prob",
            "short_adverse_prob",
            "path_best_side",
        ]
        res = {k: np.zeros(n, dtype=np.float64) for k in keys}
        if not self.lightgbm_ensemble.available:
            return res
        x = _to_numeric_frame(df, self.lightgbm_ensemble.feature_cols)
        models = self.lightgbm_ensemble.model
        res["long_edge"] = models["long_edge"].predict(x)
        res["short_edge"] = models["short_edge"].predict(x)
        res["tradeability_score"] = models["tradeability"].predict(x)
        res["long_mae_q90"] = np.maximum(0.0, models["long_mae_q90"].predict(x))
        res["short_mae_q90"] = np.maximum(0.0, models["short_mae_q90"].predict(x))
        res["long_adverse_prob"] = models["long_adverse"].predict_proba(x)[:, 1]
        res["short_adverse_prob"] = models["short_adverse"].predict_proba(x)[:, 1]
        res["path_best_side"] = np.where(res["long_edge"] >= res["short_edge"], 1.0, -1.0)
        return res

    def _blend_supervised(self, df, p_xgb, p_mtl, p_q, q50, quality):
        w = np.array([self.weight_trend_xgb, self.weight_multitarget, self.weight_quantile])
        w /= w.sum()
        base = (p_xgb * w[0] + p_mtl * w[1] + p_q * w[2])
        bias = 0.25 * np.sign(np.nan_to_num(q50)) + 0.15 * np.tanh(np.nan_to_num(quality))
        logits = np.log(np.clip(base, 1e-7, 1.0)) + np.column_stack([-bias, bias])
        return _safe_prob2(_softmax2(logits), source="m7_blend")

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _enrich_m7_features(df)
        p_xgb = self._predict_trend_xgb(df)
        mtl = self._predict_multitarget(df)
        q = self._predict_quantile(df)
        path = self._predict_lightgbm_ensemble(df)
        
        probs = self._blend_supervised(df, p_xgb, mtl["probs"], q["probs"], q["q50"], mtl["quality"])
        conf = np.max(probs, axis=1)
        action = np.where(probs[:, 1] >= probs[:, 0], 1.0, -1.0)
        
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy()
        
        out = pd.DataFrame(index=df.index)
        # Individual directional model outputs: [DOWN, UP]. Tradeability is separate.
        out["m7_trend_xgb_dn"], out["m7_trend_xgb_up"] = p_xgb[:, 0], p_xgb[:, 1]
        out["m7_mtl_dn"], out["m7_mtl_up"] = mtl["probs"][:, 0], mtl["probs"][:, 1]
        out["m7_quant_dn"], out["m7_quant_up"] = q["probs"][:, 0], q["probs"][:, 1]
        
        # Core Blended Outputs
        out["m7_prob_dn"], out["m7_prob_up"] = probs[:, 0], probs[:, 1]
        out["m7_target_quality"], out["m7_target_hold"] = mtl["quality"], mtl["hold"]
        out["m7_confidence"] = conf
        out["m7_action"] = action
        
        # Supervised Details
        out["m7_q10"], out["m7_q50"], out["m7_q90"] = q["q10"], q["q50"], q["q90"]
        out["m7_qwidth"] = np.abs(q["q90"] - q["q10"])
        out["m7_quality_pred"], out["m7_hold_pred"] = mtl["quality"], mtl["hold"]
        out["m7_long_edge"], out["m7_short_edge"] = path["long_edge"], path["short_edge"]
        out["m7_tradeability_score"] = path["tradeability_score"]
        out["m7_long_mae_q90"], out["m7_short_mae_q90"] = path["long_mae_q90"], path["short_mae_q90"]
        out["m7_long_adverse_prob"], out["m7_short_adverse_prob"] = path["long_adverse_prob"], path["short_adverse_prob"]
        out["m7_path_best_side"] = path["path_best_side"]
        
        # Entry/Exit Targets
        ep = self._predict_entry_price(df)
        out["m7_entry_long_price"], out["m7_entry_short_price"] = ep["entry_long_price"], ep["entry_short_price"]
        out["m7_entry_long_offset"], out["m7_entry_short_offset"] = ep.get("entry_long_offset", 0.0), ep.get("entry_short_offset", 0.0)
        
        tp_offset = np.maximum(np.abs(q["q90"]), 0.0008)
        sl_offset = 0.0006
        out["m7_tp_offset"], out["m7_sl_offset"] = tp_offset, sl_offset
        out["m7_tp_price"] = close * (1.0 + np.where(out["m7_action"]==1, tp_offset, -tp_offset))
        out["m7_sl_price"] = close * (1.0 + np.where(out["m7_action"]==1, -sl_offset, sl_offset))
        
        # Additional metrics for compatibility
        out["m7_expected_ret"] = q["q50"]
        out["m7_tail_risk"] = np.maximum(0.0, np.abs(q["q10"]) - 0.01)
        out["m7_composite_score"] = out["m7_action"] * conf
        
        return out.fillna(0.0).astype(np.float32)

    def predict_last(self, df: pd.DataFrame) -> dict:
        res = self.predict_batch(df.tail(200)).tail(1)
        return res.iloc[-1].to_dict() if not res.empty else {}
