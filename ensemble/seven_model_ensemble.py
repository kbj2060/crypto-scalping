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

# Some persisted artifacts were created with NumPy 2.x, which pickles objects
# under the private ``numpy._core`` package path. NumPy 1.26 still exposes the
# equivalent implementation under ``numpy.core`` only, so we alias the legacy
# module path before loading any pickled models.
try:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_core_numeric

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.numeric", _np_core_numeric)
except Exception:
    pass

from ensemble.supervised.train_trend_xgb import XGBTrendBrain
from ensemble.supervised.train_entry_price_model import EntryPriceBrain

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax3(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -50.0, 50.0))
    s = ez.sum(axis=1, keepdims=True)
    s = np.where(s <= 1e-12, 1.0, s)
    return ez / s


def _safe_prob3(arr: np.ndarray) -> np.ndarray:
    p = np.asarray(arr, dtype=np.float64)
    if p.ndim == 1:
        cls = np.clip(p.astype(np.int64), 0, 2)
        out = np.zeros((len(cls), 3), dtype=np.float64)
        out[np.arange(len(cls)), cls] = 1.0
        return out
    if p.ndim == 2 and p.shape[1] >= 3:
        p = p[:, :3]
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        denom = p.sum(axis=1, keepdims=True)
        bad = denom[:, 0] <= 1e-12
        denom = np.where(denom <= 1e-12, 1.0, denom)
        p = p / denom
        if np.any(bad):
            p[bad] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return p
    n = len(p) if p.ndim > 0 else 1
    return np.full((n, 3), 1.0 / 3.0, dtype=np.float64)


def _robust_z_series(s: pd.Series, window: int = 288, min_periods: int = 48) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    med = x.rolling(window=window, min_periods=min_periods).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=min_periods).median()
    z = (x - med) / (1.4826 * mad.replace(0.0, np.nan) + 1e-6)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0).to_numpy(dtype=np.float64)


def _resolve_model_path(meta_path: str, model_ref: str | None) -> str:
    if not model_ref:
        return os.path.splitext(meta_path)[0] + ".pkl"
    if os.path.isabs(model_ref):
        return model_ref
    return os.path.join(os.path.dirname(meta_path), model_ref)


def _enrich_m7_features(df: pd.DataFrame) -> pd.DataFrame:
    """M7 모델이 요구하는 피처 중 processed_df에 누락된 것들을 추론 직전에 보완한다.

    이미 컬럼이 존재하면 덮어쓰지 않는다(if 조건 guard).
    계산에 필요한 원시 컬럼이 없으면 해당 파생 컬럼은 생성하지 않는다.
    (필수 컬럼 검증은 _to_numeric_frame에서 fail-fast 처리)
    """
    df = df.copy()

    # ── 1. signal_* = pred_* × conf_* 합성 ─────────────────────────
    _pred_conf_pairs = [
        ("pred_patchtst", "conf_patchtst"),
        ("pred_chronos",  "conf_chronos"),
        ("pred_tide",     "conf_tide"),
    ]
    for pred_col, conf_col in _pred_conf_pairs:
        sig_col = pred_col.replace("pred_", "signal_")
        if sig_col in df.columns:
            continue
        if pred_col in df.columns and conf_col in df.columns:
            df[sig_col] = pd.to_numeric(df[pred_col], errors="coerce") * pd.to_numeric(df[conf_col], errors="coerce")
        elif pred_col in df.columns:
            df[sig_col] = pd.to_numeric(df[pred_col], errors="coerce")

    # ── 2. 시간 파생 피처 ─────────────────────────────────────────
    if "timestamp" in df.columns and any(
        c not in df.columns for c in ["hour_sin", "minute_sin", "minute_cos", "session_europe", "is_hour_open"]
    ):
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            hour = ts.dt.hour.fillna(0).astype(float)
            minute = ts.dt.minute.fillna(0).astype(float)
            if "hour_sin" not in df.columns:
                df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
            if "minute_sin" not in df.columns:
                df["minute_sin"] = np.sin(2 * np.pi * minute / 60).astype(np.float32)
            if "minute_cos" not in df.columns:
                df["minute_cos"] = np.cos(2 * np.pi * minute / 60).astype(np.float32)
            if "session_europe" not in df.columns:
                df["session_europe"] = ((hour >= 8) & (hour < 16)).astype(np.float32)
            if "is_hour_open" not in df.columns:
                df["is_hour_open"] = (minute < 5).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive time features for M7 inference: {e}") from e

    # ── 3. chop_index ────────────────────────────────────────────
    if "chop_index" not in df.columns and all(c in df.columns for c in ["high", "low", "close"]):
        try:
            _length = 14
            _h = pd.to_numeric(df["high"], errors="coerce")
            _l = pd.to_numeric(df["low"], errors="coerce")
            _c = pd.to_numeric(df["close"], errors="coerce")
            _tr1 = _h - _l
            _tr2 = (_h - _c.shift(1)).abs()
            _tr3 = (_l - _c.shift(1)).abs()
            _tr = pd.concat([_tr1, _tr2, _tr3], axis=1).max(axis=1)
            _atr_sum = _tr.rolling(_length).sum()
            _hmax = _h.rolling(_length).max()
            _lmin = _l.rolling(_length).min()
            df["chop_index"] = (
                100 * np.log10((_atr_sum + 1e-8) / (_hmax - _lmin + 1e-8)) / np.log10(_length)
            ).fillna(50.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive chop_index for M7 inference: {e}") from e

    # ── 4. realized_skewness ─────────────────────────────────────
    if "realized_skewness" not in df.columns and "close" in df.columns:
        try:
            _window = 96
            _rets = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0)
            def _skew_fn(x: np.ndarray) -> float:
                if len(x) < 8:
                    return 0.0
                mu = x.mean()
                sig = x.std()
                if sig < 1e-10:
                    return 0.0
                return float(((x - mu) ** 3).mean() / (sig ** 3 + 1e-10))
            df["realized_skewness"] = (
                _rets.rolling(_window, min_periods=_window // 2)
                .apply(_skew_fn, raw=True)
                .clip(-3, 3)
                .fillna(0)
                .astype(np.float32)
            )
        except Exception as e:
            raise RuntimeError(f"failed to derive realized_skewness for M7 inference: {e}") from e

    # ── 5. regime_trending (hurst_48 > 0.5) ─────────────────────
    if "regime_trending" not in df.columns:
        if "hurst_48" in df.columns:
            df["regime_trending"] = (
                pd.to_numeric(df["hurst_48"], errors="coerce").fillna(0.5) > 0.5
            ).astype(np.float32)

    # ── 5.1 regime one-hot (cvp_regime 기반) ───────────────────
    regime_cols = ["regime_chop", "regime_whipsaw", "regime_bull", "regime_bear", "regime_normal"]
    if any(c not in df.columns for c in regime_cols):
        if "cvp_regime" not in df.columns:
            raise KeyError("cvp_regime is required to derive regime one-hot features for M7 inference")
        _reg = pd.to_numeric(df["cvp_regime"], errors="coerce").fillna(0.0)
        _abs = _reg.abs()
        _bull = (_reg >= 0.35)
        _bear = (_reg <= -0.35)
        _chop = (_abs < 0.15)
        _whipsaw = (_abs >= 0.15) & (_abs < 0.35)
        _normal = ~(_bull | _bear | _chop | _whipsaw)
        if "regime_bull" not in df.columns:
            df["regime_bull"] = _bull.astype(np.float32)
        if "regime_bear" not in df.columns:
            df["regime_bear"] = _bear.astype(np.float32)
        if "regime_chop" not in df.columns:
            df["regime_chop"] = _chop.astype(np.float32)
        if "regime_whipsaw" not in df.columns:
            df["regime_whipsaw"] = _whipsaw.astype(np.float32)
        if "regime_normal" not in df.columns:
            df["regime_normal"] = _normal.astype(np.float32)

    # ── 6. funding_roc_48 ────────────────────────────────────────
    if "funding_roc_48" not in df.columns and "last_funding_rate" in df.columns:
        try:
            _fr = pd.to_numeric(df["last_funding_rate"], errors="coerce").fillna(0)
            _shifted = _fr.shift(48)
            df["funding_roc_48"] = (
                (_fr - _shifted) / (_shifted.abs().clip(lower=1e-4) + 1e-8)
            ).clip(-10, 10).fillna(0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive funding_roc_48 for M7 inference: {e}") from e

    # ── 7. mta_funding ───────────────────────────────────────────
    if "mta_funding" not in df.columns and all(
        c in df.columns for c in ["funding_roc_12", "squeeze_power", "last_funding_rate"]
    ):
        try:
            _roc12 = pd.to_numeric(df["funding_roc_12"], errors="coerce").fillna(0)
            _roc48 = pd.to_numeric(df.get("funding_roc_48", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
            _roc288 = pd.to_numeric(df["funding_roc_288"], errors="coerce").fillna(0) if "funding_roc_288" in df.columns else 0.0
            _fr = pd.to_numeric(df["last_funding_rate"], errors="coerce").abs().clip(lower=1e-5)
            _sq = pd.to_numeric(df["squeeze_power"], errors="coerce").fillna(0)
            _roll = 288
            _sq_mean = _sq.rolling(_roll, min_periods=1).mean()
            _sq_std = _sq.rolling(_roll, min_periods=1).std().replace(0, 1e-8).fillna(1e-8)
            _sq_z = (_sq - _sq_mean) / _sq_std
            _weighted_roc = 0.5 * _roc12 + 0.3 * _roc48 + 0.2 * _roc288
            df["mta_funding"] = (
                (_weighted_roc / _fr) * np.tanh(_sq_z)
            ).clip(-3, 3).fillna(0).div(3).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive mta_funding for M7 inference: {e}") from e

    # ── 9. svps ──────────────────────────────────────────────────
    if "svps" not in df.columns and all(
        c in df.columns for c in ["cvp_poc_dist", "cvp_volume_imbalance"]
    ):
        try:
            _poc = pd.to_numeric(df["cvp_poc_dist"], errors="coerce").fillna(0)
            _vim = pd.to_numeric(df["cvp_volume_imbalance"], errors="coerce").fillna(0)
            _reg = pd.to_numeric(df["cvp_regime"], errors="coerce").fillna(0)
            df["svps"] = (
                np.tanh(2.0 * _poc * _vim * np.exp(-np.abs(_reg).clip(0, 5)))
            ).fillna(0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive svps for M7 inference: {e}") from e

    # ── 10. kalman_velocity ──────────────────────────────────────
    if "kalman_velocity" not in df.columns and "close" in df.columns:
        try:
            _vals = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0).to_numpy(dtype=np.float64)
            _n = len(_vals)
            _F = np.array([[1.0, 1.0], [0.0, 1.0]])
            _H = np.array([[1.0, 0.0]])
            _Q = np.eye(2) * 1e-5
            _R = np.array([[1e-3]])
            _x = np.array([_vals[0], 0.0])
            _P = np.eye(2)
            _vels = np.empty(_n, dtype=np.float64)
            for _i in range(_n):
                _x = _F @ _x
                _P = _F @ _P @ _F.T + _Q
                _S = float((_H @ _P @ _H.T + _R)[0, 0])
                _K = (_P @ _H.T).flatten() / _S
                _inn = _vals[_i] - float((_H @ _x)[0])
                _x = _x + _K * _inn
                _P = (np.eye(2) - np.outer(_K, _H)) @ _P
                _vels[_i] = _x[1]
            df["kalman_velocity"] = np.clip(_vels / (_vals + 1e-8), -0.05, 0.05).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to derive kalman_velocity for M7 inference: {e}") from e

    return df


def _add_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """trend_xgb 학습 시 사용된 OHLC 파생 피처를 추론 전에 보장한다."""
    df = df.copy()
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce") if "high" in df.columns else c
    lo = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else c
    if "ret_12" not in df.columns:
        df["ret_12"] = np.tanh(c.pct_change(12) * 10)
    if "ret_24" not in df.columns:
        df["ret_24"] = np.tanh(c.pct_change(24) * 10)
    if "ret_48" not in df.columns:
        df["ret_48"] = np.tanh(c.pct_change(48) * 10)
    if "hh_count_24" not in df.columns:
        df["hh_count_24"] = (h > h.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
    if "hl_count_24" not in df.columns:
        df["hl_count_24"] = (lo > lo.shift(1)).astype(float).rolling(24, min_periods=1).sum() / 24.0
    if "trend_accel" not in df.columns:
        df["trend_accel"] = np.tanh((c.pct_change(12) - c.pct_change(48) / 4) * 20)
    return df


def _to_numeric_frame(df: pd.DataFrame, cols: list[str], fill_mode: str = "median") -> pd.DataFrame:
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        preview = ", ".join(missing_cols[:10])
        suffix = " ..." if len(missing_cols) > 10 else ""
        raise KeyError(f"M7 required feature(s) missing: {preview}{suffix}")

    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [c for c in cols if out[c].isna().all()]
    if all_nan_cols:
        preview = ", ".join(all_nan_cols[:10])
        suffix = " ..." if len(all_nan_cols) > 10 else ""
        raise ValueError(f"M7 required feature(s) are all-NaN: {preview}{suffix}")

    if fill_mode == "median":
        med = out.median(numeric_only=True)
        out = out.fillna(med)
    elif fill_mode == "zero":
        pass

    if out.isna().any().any():
        bad_cols = [c for c in cols if out[c].isna().any()]
        preview = ", ".join(bad_cols[:10])
        suffix = " ..." if len(bad_cols) > 10 else ""
        raise ValueError(f"M7 required feature(s) contain NaN after preprocessing: {preview}{suffix}")

    return out.astype(np.float32)


if _TORCH_AVAILABLE:
    class _VAE(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            h = self.encoder(x)
            mu = self.mu(h)
            logvar = self.logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon = self.decoder(z)
            return recon, mu, logvar


@dataclass
class _ModelState:
    available: bool
    model: Any = None
    feature_cols: list[str] | None = None
    extra: dict[str, Any] | None = None
    reason: str = ""


class SevenModelEnsemble:
    """7개 지도/비지도 모델 통합 추론기.

    목표:
    1) 방향(UP/DOWN/FLAT) 확률 통합
    2) quantile 기반 포지션 사이징
    3) 변동성/클러스터 기반 라우팅
    4) 이상탐지 기반 진입 차단(게이트)
    5) 품질/홀딩 예측 기반 청산 보조
    """

    DEFAULT_META_PATHS = {
        "trend_xgb": "data/ensemble/supervised/trend_xgb.json",
        "manifold_hgb": "data/ensemble/supervised/manifold_hgb.json",
        "entry_price_model": "data/ensemble/supervised/entry_price_model.json",
        "multi_target_lgbm": "data/ensemble/supervised/multi_target_lgbm.json",
        "quantile_forest": "data/ensemble/supervised/quantile_forest.json",
        "gmm_volatility": "data/ensemble/unsupervised/gmm_volatility.json",
        "isolation_forest": "data/ensemble/unsupervised/isolation_forest.json",
        "vae_anomaly": "data/ensemble/unsupervised/vae_anomaly.json",
    }

    def __init__(
        self,
        meta_paths: dict[str, str] | None = None,
        weight_trend_xgb: float = 0.45,
        weight_multitarget: float = 0.35,
        weight_quantile: float = 0.20,
        strict: bool = True,
    ):
        self.meta_paths = dict(self.DEFAULT_META_PATHS)
        if meta_paths:
            self.meta_paths.update(meta_paths)

        self.weight_trend_xgb = float(weight_trend_xgb)
        self.weight_multitarget = float(weight_multitarget)
        self.weight_quantile = float(weight_quantile)
        self.strict = bool(strict)

        self.trend_xgb = _ModelState(False)
        self.manifold = _ModelState(False)
        self.entry_price = _ModelState(False)
        self.multi_target = _ModelState(False)
        self.quantile = _ModelState(False)
        self.gmm = _ModelState(False)
        self.isolation = _ModelState(False)
        self.vae = _ModelState(False)

        self._load_all()
        self._assert_ready()

    def _missing_models(self) -> list[str]:
        checks = [
            ("trend_xgb", self.trend_xgb),
            ("manifold_hgb", self.manifold),
            ("entry_price_model", self.entry_price),
            ("multi_target_lgbm", self.multi_target),
            ("quantile_forest", self.quantile),
            ("gmm_volatility", self.gmm),
            ("isolation_forest", self.isolation),
            ("vae_anomaly", self.vae),
        ]
        missing: list[str] = []
        for name, state in checks:
            if not state.available:
                reason = state.reason or "unavailable"
                missing.append(f"{name}({reason})")
        return missing

    def _assert_ready(self) -> None:
        if not self.strict:
            return
        missing = self._missing_models()
        if missing:
            raise RuntimeError("SevenModelEnsemble strict mode: missing required model(s): " + ", ".join(missing))

    def _load_meta(self, key: str) -> dict[str, Any] | None:
        path = self.meta_paths[key]
        if not os.path.exists(path):
            logger.warning("[%s] meta not found: %s", key, path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[%s] meta load failed: %s", key, e)
            return None

    def _load_pickle_from_meta(self, key: str) -> tuple[Any | None, dict[str, Any] | None]:
        meta = self._load_meta(key)
        if not meta:
            return None, None
        model_path = _resolve_model_path(self.meta_paths[key], meta.get("model_path"))
        if not os.path.exists(model_path):
            logger.warning("[%s] model file missing: %s", key, model_path)
            return None, meta
        try:
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            return payload, meta
        except Exception as e:
            if key == "vae_anomaly" and _TORCH_AVAILABLE:
                try:
                    payload = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
                    return payload, meta
                except Exception as e2:
                    logger.warning("[%s] model load failed: %s / torch_fallback=%s", key, e, e2)
                    return None, meta
            logger.warning("[%s] model load failed: %s", key, e)
            return None, meta

    def _load_all(self) -> None:
        self._load_trend_xgb()
        self._load_manifold_hgb()
        self._load_entry_price()
        self._load_multitarget()
        self._load_quantile()
        self._load_gmm()
        self._load_isolation()
        self._load_vae()

    def _load_trend_xgb(self) -> None:
        path = self.meta_paths["trend_xgb"]
        try:
            brain = XGBTrendBrain.load(path)
            self.trend_xgb = _ModelState(True, model=brain.model, feature_cols=list(brain.feature_cols), extra={})
            logger.info("✅ trend_xgb loaded (%d features)", len(brain.feature_cols))
        except Exception as e:
            self.trend_xgb = _ModelState(False, reason=str(e))
            logger.warning("⚠️ trend_xgb unavailable: %s", e)

    def _load_manifold_hgb(self) -> None:
        payload, meta = self._load_pickle_from_meta("manifold_hgb")
        if payload is None:
            self.manifold = _ModelState(False, reason="missing_payload")
            logger.warning("⚠️ manifold_hgb unavailable")
            return
        fcols = list(meta.get("feature_cols", payload.get("feature_cols", []))) if meta else list(payload.get("feature_cols", []))
        self.manifold = _ModelState(True, model=payload.get("model"), feature_cols=fcols, extra={})
        logger.info("✅ manifold_hgb loaded (%d features)", len(fcols))

    def _load_entry_price(self) -> None:
        path = self.meta_paths["entry_price_model"]
        try:
            brain = EntryPriceBrain.load(path)
            self.entry_price = _ModelState(True, model=brain, feature_cols=list(brain.feature_cols), extra={})
            logger.info("✅ entry_price_model loaded (%d features)", len(brain.feature_cols))
        except Exception as e:
            self.entry_price = _ModelState(False, reason=str(e))
            logger.warning("⚠️ entry_price_model unavailable: %s", e)

    def _load_multitarget(self) -> None:
        payload, meta = self._load_pickle_from_meta("multi_target_lgbm")
        if payload is None:
            self.multi_target = _ModelState(False, reason="missing_payload")
            return
        fcols = list(meta.get("feature_cols", payload.get("feature_cols", []))) if meta else list(payload.get("feature_cols", []))
        self.multi_target = _ModelState(
            True,
            model=payload,
            feature_cols=fcols,
            extra={},
        )
        logger.info("✅ multi_target_lgbm loaded (%d features)", len(fcols))

    def _load_quantile(self) -> None:
        payload, meta = self._load_pickle_from_meta("quantile_forest")
        if payload is None:
            self.quantile = _ModelState(False, reason="missing_payload")
            return
        fcols = list(meta.get("feature_cols", payload.get("feature_cols", []))) if meta else list(payload.get("feature_cols", []))
        flat_th = float(meta.get("flat_threshold", payload.get("flat_threshold", 5e-4))) if meta else float(payload.get("flat_threshold", 5e-4))
        self.quantile = _ModelState(
            True,
            model=payload.get("models", {}),
            feature_cols=fcols,
            extra={"flat_threshold": flat_th},
        )
        logger.info("✅ quantile_forest loaded (%d features)", len(fcols))

    def _load_gmm(self) -> None:
        payload, meta = self._load_pickle_from_meta("gmm_volatility")
        if payload is None:
            self.gmm = _ModelState(False, reason="missing_payload")
            return
        fcols = list(payload.get("feature_cols", []))
        rank_map = payload.get("cluster_rank_map", {})
        rank_map = {int(k): int(v) for k, v in rank_map.items()}
        self.gmm = _ModelState(
            True,
            model=payload.get("model"),
            feature_cols=fcols,
            extra={
                "mean": np.asarray(payload.get("mean"), dtype=np.float32),
                "std": np.asarray(payload.get("std"), dtype=np.float32),
                "cluster_rank_map": rank_map,
            },
        )
        logger.info("✅ gmm_volatility loaded (%d features)", len(fcols))

    def _load_hdbscan(self) -> None:
        payload, _ = self._load_pickle_from_meta("hdbscan_regime")
        if payload is None:
            self.hdbscan = _ModelState(False, reason="missing_payload")
            return
        self.hdbscan = _ModelState(
            True,
            model=payload.get("model"),
            feature_cols=list(payload.get("feature_cols", [])),
            extra={
                "mean": np.asarray(payload.get("mean"), dtype=np.float32),
                "std": np.asarray(payload.get("std"), dtype=np.float32),
            },
        )
        logger.info("✅ hdbscan_regime loaded (%d features)", len(self.hdbscan.feature_cols or []))

    def _load_isolation(self) -> None:
        payload, _ = self._load_pickle_from_meta("isolation_forest")
        if payload is None:
            self.isolation = _ModelState(False, reason="missing_payload")
            return
        self.isolation = _ModelState(
            True,
            model=payload.get("model"),
            feature_cols=list(payload.get("feature_cols", [])),
            extra={
                "mean": np.asarray(payload.get("mean"), dtype=np.float32),
                "std": np.asarray(payload.get("std"), dtype=np.float32),
            },
        )
        logger.info("✅ isolation_forest loaded (%d features)", len(self.isolation.feature_cols or []))

    def _load_vae(self) -> None:
        payload, _ = self._load_pickle_from_meta("vae_anomaly")
        if payload is None:
            self.vae = _ModelState(False, reason="missing_payload")
            return
        if not _TORCH_AVAILABLE:
            self.vae = _ModelState(False, reason="torch_unavailable")
            logger.warning("⚠️ vae_anomaly unavailable: torch not installed")
            return

        fcols = list(payload.get("feature_cols", []))
        meta = payload.get("meta", {})
        latent_dim = int(meta.get("latent_dim", 8))
        hidden_dim = int(meta.get("hidden_dim", 128))
        try:
            net = _VAE(input_dim=len(fcols), latent_dim=latent_dim, hidden_dim=hidden_dim)
            net.load_state_dict(payload.get("state_dict", {}), strict=False)
            net.eval()
            self.vae = _ModelState(
                True,
                model=net,
                feature_cols=fcols,
                extra={
                    "mean": np.asarray(payload.get("mean"), dtype=np.float32),
                    "std": np.asarray(payload.get("std"), dtype=np.float32),
                    "threshold": float(payload.get("threshold", 0.0)),
                },
            )
            logger.info("✅ vae_anomaly loaded (%d features)", len(fcols))
        except Exception as e:
            self.vae = _ModelState(False, reason=str(e))
            logger.warning("⚠️ vae_anomaly unavailable: %s", e)

    def _predict_trend_xgb(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if not self.trend_xgb.available or self.trend_xgb.model is None:
            return np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
        df = _add_trend_structure_features(df)
        x = _to_numeric_frame(df, self.trend_xgb.feature_cols or [], fill_mode="median")
        model = self.trend_xgb.model
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(x), dtype=np.float64)
        else:
            out = np.asarray(model.predict(x.values), dtype=np.float64)
        return _safe_prob3(out)

    def _predict_manifold_hgb(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if not self.manifold.available or self.manifold.model is None:
            return np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
        x = _to_numeric_frame(df, self.manifold.feature_cols or [], fill_mode="median")
        model = self.manifold.model
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(x), dtype=np.float64)
        else:
            out = np.asarray(model.predict(x.values), dtype=np.float64)
        return _safe_prob3(out)

    def _predict_multitarget(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
        quality = np.full(n, np.nan, dtype=np.float64)
        hold = np.full(n, np.nan, dtype=np.float64)
        if not self.multi_target.available or self.multi_target.model is None:
            return {"probs": probs, "quality": quality, "hold": hold}
        x = _to_numeric_frame(df, self.multi_target.feature_cols or [], fill_mode="median")
        m = self.multi_target.model
        d_model = m.get("direction_model")
        q_model = m.get("quality_model")
        h_model = m.get("hold_model")
        if d_model is not None:
            if hasattr(d_model, "predict_proba"):
                probs = _safe_prob3(np.asarray(d_model.predict_proba(x), dtype=np.float64))
            else:
                probs = _safe_prob3(np.asarray(d_model.predict(x), dtype=np.float64))
        if q_model is not None:
            quality = np.asarray(q_model.predict(x), dtype=np.float64).reshape(-1)
        if h_model is not None:
            hold = np.asarray(h_model.predict(x), dtype=np.float64).reshape(-1)
        return {"probs": probs, "quality": quality, "hold": hold}

    def _predict_entry_price(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        long_offset = np.zeros(n, dtype=np.float64)
        short_offset = np.zeros(n, dtype=np.float64)
        long_price = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        short_price = long_price.copy()
        if not self.entry_price.available or self.entry_price.model is None:
            return {
                "entry_long_offset": long_offset,
                "entry_short_offset": short_offset,
                "entry_long_price": long_price,
                "entry_short_price": short_price,
            }
        pred = self.entry_price.model.predict_batch_from_df(df)
        if pred.empty:
            return {
                "entry_long_offset": long_offset,
                "entry_short_offset": short_offset,
                "entry_long_price": long_price,
                "entry_short_price": short_price,
            }
        return {
            "entry_long_offset": np.asarray(pred["entry_long_offset"], dtype=np.float64),
            "entry_short_offset": np.asarray(pred["entry_short_offset"], dtype=np.float64),
            "entry_long_price": np.asarray(pred["entry_long_price"], dtype=np.float64),
            "entry_short_price": np.asarray(pred["entry_short_price"], dtype=np.float64),
        }

    def _predict_quantile(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        q10 = np.full(n, np.nan, dtype=np.float64)
        q50 = np.full(n, np.nan, dtype=np.float64)
        q90 = np.full(n, np.nan, dtype=np.float64)
        pseudo_probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
        flat_threshold = 5e-4
        if not self.quantile.available or self.quantile.model is None:
            return {"q10": q10, "q50": q50, "q90": q90, "probs": pseudo_probs}
        x = _to_numeric_frame(df, self.quantile.feature_cols or [], fill_mode="median")
        models = self.quantile.model
        flat_threshold = float((self.quantile.extra or {}).get("flat_threshold", flat_threshold))
        if "q10" in models:
            q10 = np.asarray(models["q10"].predict(x), dtype=np.float64).reshape(-1)
        if "q50" in models:
            q50 = np.asarray(models["q50"].predict(x), dtype=np.float64).reshape(-1)
        if "q90" in models:
            q90 = np.asarray(models["q90"].predict(x), dtype=np.float64).reshape(-1)
        width = np.maximum(np.nan_to_num(q90 - q10, nan=2e-3), 1e-6)
        z = np.nan_to_num(q50 / width, nan=0.0)
        up_raw = _sigmoid(4.0 * (z - flat_threshold))
        dn_raw = _sigmoid(4.0 * (-z - flat_threshold))
        fl_raw = np.exp(-np.abs(z) * 2.8)
        pseudo_probs = _softmax3(np.column_stack([dn_raw, fl_raw, up_raw]))
        return {"q10": q10, "q50": q50, "q90": q90, "probs": pseudo_probs}

    def _predict_gmm(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        cluster = np.full(n, -1, dtype=np.int64)
        conf = np.zeros(n, dtype=np.float64)
        vol_rank = np.full(n, 0.5, dtype=np.float64)
        if not self.gmm.available or self.gmm.model is None:
            return {"cluster": cluster, "confidence": conf, "vol_rank": vol_rank}
        cols = self.gmm.feature_cols or []
        x_raw = _to_numeric_frame(df, cols, fill_mode="median").values.astype(np.float32)
        mean = (self.gmm.extra or {}).get("mean")
        std = (self.gmm.extra or {}).get("std")
        std = np.where(np.asarray(std) < 1e-8, 1.0, std)
        x = (x_raw - mean) / std
        model = self.gmm.model
        cluster = np.asarray(model.predict(x), dtype=np.int64)
        probs = np.asarray(model.predict_proba(x), dtype=np.float64)
        conf = np.max(probs, axis=1)
        rank_map = (self.gmm.extra or {}).get("cluster_rank_map", {})
        if rank_map:
            max_rank = max(rank_map.values()) if rank_map else 0
            denom = max(max_rank, 1)
            vol_rank = np.array([rank_map.get(int(c), 0) / denom for c in cluster], dtype=np.float64)
        return {"cluster": cluster, "confidence": conf, "vol_rank": vol_rank}

    def _predict_hdbscan(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)
        probs = np.zeros(n, dtype=np.float64)
        return {"label": labels, "prob": probs}

    def _predict_isolation(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        pred = np.ones(n, dtype=np.int64)
        score = np.zeros(n, dtype=np.float64)
        if not self.isolation.available or self.isolation.model is None:
            return {"pred": pred, "score": score}
        cols = self.isolation.feature_cols or []
        x_raw = _to_numeric_frame(df, cols, fill_mode="median").values.astype(np.float32)
        mean = (self.isolation.extra or {}).get("mean")
        std = (self.isolation.extra or {}).get("std")
        std = np.where(np.asarray(std) < 1e-8, 1.0, std)
        x = (x_raw - mean) / std
        model = self.isolation.model
        pred = np.asarray(model.predict(x), dtype=np.int64)
        score = -np.asarray(model.decision_function(x), dtype=np.float64)
        return {"pred": pred, "score": score}

    def _predict_vae(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(df)
        err = np.full(n, np.nan, dtype=np.float64)
        is_anom = np.zeros(n, dtype=np.int64)
        threshold = np.nan
        if not self.vae.available or self.vae.model is None or not _TORCH_AVAILABLE:
            return {"error": err, "is_anomaly": is_anom, "threshold": threshold}

        cols = self.vae.feature_cols or []
        x_raw = _to_numeric_frame(df, cols, fill_mode="median").values.astype(np.float32)
        mean = (self.vae.extra or {}).get("mean")
        std = (self.vae.extra or {}).get("std")
        threshold = float((self.vae.extra or {}).get("threshold", 0.0))
        std = np.where(np.asarray(std) < 1e-8, 1.0, std)
        x = (x_raw - mean) / std

        model = self.vae.model
        xt = torch.from_numpy(x)
        bs = 4096
        errs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(xt), bs):
                xb = xt[i : i + bs]
                recon, _, _ = model(xb)
                e = torch.mean((recon - xb) ** 2, dim=1).cpu().numpy()
                errs.append(e)
        err = np.concatenate(errs) if errs else np.full(n, np.nan, dtype=np.float64)
        is_anom = (err > threshold).astype(np.int64)
        return {"error": err, "is_anomaly": is_anom, "threshold": threshold}

    def _blend_supervised_manifold(
        self,
        df: pd.DataFrame,
        p_xgb: np.ndarray,
        p_meta: np.ndarray,
        p_q: np.ndarray,
        q50: np.ndarray,
        q_width: np.ndarray,
        quality: np.ndarray,
    ) -> np.ndarray:
        # Convex supervised fusion: reward agreement, downweight high-entropy disagreement,
        # and use quantile edge as a directional geometric bias.
        probs_stack = np.stack([_safe_prob3(p_xgb), _safe_prob3(p_meta), _safe_prob3(p_q)], axis=1)
        weights = np.array(
            [self.weight_trend_xgb, self.weight_multitarget, self.weight_quantile],
            dtype=np.float64,
        )
        weights = weights / max(float(np.sum(weights)), 1e-12)

        base = np.sum(probs_stack * weights[None, :, None], axis=1)
        pair_gap = (
            np.abs(probs_stack[:, 0, :] - probs_stack[:, 1, :]).sum(axis=1)
            + np.abs(probs_stack[:, 0, :] - probs_stack[:, 2, :]).sum(axis=1)
            + np.abs(probs_stack[:, 1, :] - probs_stack[:, 2, :]).sum(axis=1)
        ) / 3.0
        agreement = np.clip(1.0 - 0.9 * pair_gap, 0.25, 1.15)

        rr = np.abs(np.nan_to_num(q50, nan=0.0)) / np.maximum(np.nan_to_num(q_width, nan=1e-6), 1e-6)
        edge = np.clip(np.tanh(rr * 0.9), 0.0, 1.0)
        quality_norm = np.clip(np.nan_to_num(quality, nan=0.0) * 80.0, -1.0, 1.0)

        mtf1 = pd.to_numeric(df["mtf_trend_1h"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) if "mtf_trend_1h" in df.columns else np.zeros(len(df), dtype=np.float64)
        mtf4 = pd.to_numeric(df["mtf_trend_4h"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) if "mtf_trend_4h" in df.columns else np.zeros(len(df), dtype=np.float64)
        trend_bias = np.clip(0.55 * mtf1 + 0.45 * mtf4, -2.0, 2.0)
        directional_bias = 0.55 * np.sign(np.nan_to_num(q50, nan=0.0)) * edge + 0.25 * np.tanh(trend_bias) + 0.20 * quality_norm
        logits_bias = np.column_stack([
            -directional_bias,
            -0.35 * edge,
            directional_bias,
        ])

        entropy = -np.sum(base * np.log(np.clip(base, 1e-8, 1.0)), axis=1) / np.log(3.0)
        confidence_boost = np.clip(1.0 - entropy, 0.0, 1.0)
        logits = np.log(np.clip(base, 1e-8, 1.0)) * agreement[:, None] + (0.20 + 0.35 * confidence_boost)[:, None] * logits_bias

        probs = _softmax3(logits)
        prior = np.clip(np.mean(probs, axis=0), 1e-6, 1.0)
        target_prior = np.array([0.42, 0.16, 0.42], dtype=np.float64)
        prior_scale = np.clip(target_prior / prior, 0.75, 1.35)
        probs = probs * prior_scale[None, :]
        return _safe_prob3(probs)

    def _experimental_unsup_redesign(
        self,
        df: pd.DataFrame,
        probs: np.ndarray,
        confidence: np.ndarray,
    ) -> dict[str, np.ndarray]:
        side = np.where(np.argmax(probs, axis=1) == 2, 1.0, np.where(np.argmax(probs, axis=1) == 0, -1.0, 0.0))

        rs = _robust_z_series(df.get("rogers_satchell_vol", pd.Series(0.0, index=df.index)))
        gk = _robust_z_series(df.get("garman_klass_vol", pd.Series(0.0, index=df.index)))
        amihud = _robust_z_series(df.get("amihud_illiquidity_z", pd.Series(0.0, index=df.index)))
        lv = _robust_z_series(df.get("liquidity_vacuum", pd.Series(0.0, index=df.index)))
        curvature = _robust_z_series(pd.to_numeric(df.get("cross_scale_curvature", 0.0), errors="coerce").abs())
        execq = pd.to_numeric(df.get("execution_quality", 0.0), errors="coerce").fillna(0.0).clip(-1.5, 1.5).to_numpy(dtype=np.float64)
        crowd = pd.to_numeric(df.get("crowding_pressure", 0.0), errors="coerce").fillna(0.0).clip(-3.0, 3.0).to_numpy(dtype=np.float64)
        whale = _robust_z_series(pd.to_numeric(df.get("whale_conviction", 0.0), errors="coerce").abs())
        funding_div = _robust_z_series(pd.to_numeric(df.get("funding_price_divergence", 0.0), errors="coerce").abs())
        smf = _robust_z_series(pd.to_numeric(df.get("smart_money_flow", 0.0), errors="coerce").abs())
        cvp_imb = _robust_z_series(pd.to_numeric(df.get("cvp_volume_imbalance", 0.0), errors="coerce").abs())
        poc = _robust_z_series(pd.to_numeric(df.get("cvp_poc_dist", 0.0), errors="coerce").abs())
        taker = _robust_z_series(pd.to_numeric(df.get("taker_acceleration", 0.0), errors="coerce").abs())
        nti = _robust_z_series(pd.to_numeric(df.get("net_taker_ratio", 0.0), errors="coerce").abs())
        regime_persist = pd.to_numeric(df.get("regime_persistence", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.5).to_numpy(dtype=np.float64)
        trade_int = _robust_z_series(pd.to_numeric(df.get("trade_intensity", 0.0), errors="coerce").abs())

        vol_surface = 0.26 * rs + 0.22 * gk + 0.18 * amihud + 0.20 * lv + 0.14 * curvature
        vol_rank = _sigmoid(1.1 * vol_surface)
        gmm_conf = np.clip(1.0 - 0.55 * vol_rank + 0.18 * np.clip(execq, -1.0, 1.0), 0.0, 1.0)
        gmm_cluster = np.digitize(vol_rank, bins=[0.22, 0.45, 0.68]).astype(np.int64)

        flow_dislocation = 0.24 * cvp_imb + 0.20 * poc + 0.18 * nti + 0.18 * taker + 0.20 * lv
        crowd_pressure = 0.85 * (0.38 * np.abs(crowd) + 0.22 * whale + 0.20 * funding_div + 0.20 * smf)
        persistence_stress = 1.15 * (0.55 * _robust_z_series(pd.Series(1.0 - regime_persist)) + 0.45 * curvature)
        execution_fragility = 0.45 * _robust_z_series(pd.Series(-execq)) + 0.20 * lv + 0.15 * trade_int + 0.20 * cvp_imb

        iso_score = np.maximum(0.0, 0.42 * crowd_pressure + 0.33 * flow_dislocation + 0.25 * execution_fragility)
        vae_error = np.maximum(0.0, 0.36 * execution_fragility + 0.34 * persistence_stress + 0.30 * vol_surface)

        gate_energy = iso_score + 0.7 * vae_error
        soft_th = float(np.quantile(gate_energy, 0.90))
        hard_th = float(np.quantile(gate_energy, 0.972))
        iso_anom = (gate_energy >= soft_th).astype(np.int64)
        vae_anom = (gate_energy >= hard_th).astype(np.int64)
        gate_block = (gate_energy >= hard_th).astype(np.int64)

        # directional over-crowding: penalize only when current side aligns with crowding pressure
        directional_crowd = np.maximum(side * crowd, 0.0)
        route_scale = np.clip(
            1.0
            - 0.34 * vol_rank
            - 0.48 * np.clip((gate_energy - soft_th) / max(hard_th - soft_th, 1e-6), 0.0, 1.0)
            + 0.06 * np.clip(execq, -1.0, 1.0)
            - 0.06 * np.clip(directional_crowd, 0.0, 2.0),
            0.15,
            1.25,
        )
        min_conf = np.clip(0.42 + 0.08 * vol_rank + 0.10 * (gate_energy >= soft_th), 0.42, 0.78)

        return {
            "gmm_cluster": gmm_cluster,
            "gmm_conf": gmm_conf,
            "vol_rank": vol_rank,
            "iso_score": iso_score,
            "vae_error": vae_error,
            "iso_anom": iso_anom,
            "vae_anom": vae_anom,
            "gate_block": gate_block,
            "route_scale": route_scale,
            "min_conf": min_conf,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        self._assert_ready()
        n = len(df)
        if n == 0:
            return pd.DataFrame(index=df.index)

        df = _enrich_m7_features(df)

        p_xgb = self._predict_trend_xgb(df)
        p_meta = self._predict_manifold_hgb(df)
        mtl = self._predict_multitarget(df)
        p_mtl = mtl["probs"]
        q = self._predict_quantile(df)
        p_q = q["probs"]
        entry_px = self._predict_entry_price(df)

        gmm = self._predict_gmm(df)
        hdb = self._predict_hdbscan(df)
        iso = self._predict_isolation(df)
        vae = self._predict_vae(df)

        q10 = np.nan_to_num(q["q10"], nan=0.0)
        q50 = np.nan_to_num(q["q50"], nan=0.0)
        q90 = np.nan_to_num(q["q90"], nan=0.0)
        q_width = np.maximum(q90 - q10, 1e-6)
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        quality = np.nan_to_num(mtl["quality"], nan=0.0)

        probs = self._blend_supervised_manifold(df, p_xgb, p_meta, p_q, q50, q_width, quality)

        sort_p = np.sort(probs, axis=1)
        p_top = sort_p[:, 2]
        p_second = sort_p[:, 1]
        confidence = np.clip((p_top - 1.0 / 3.0) * 1.5 + (p_top - p_second) * 0.6, 0.0, 1.0)
        direction = np.argmax(probs, axis=1).astype(np.int64)  # 0=DOWN,1=FLAT,2=UP

        # 롱/숏 완전 대칭 사이징: edge=|q50|, risk=qwidth
        rr_sym = np.abs(q50) / np.maximum(q_width, 1e-6)
        dir_gap = np.abs(probs[:, 2] - probs[:, 0])  # 방향 우위
        conf_mix = np.clip(0.65 * confidence + 0.35 * dir_gap, 0.0, 1.0)
        rr = np.where(direction == 1, 0.0, rr_sym)
        base_size = np.tanh(rr * 1.1) * conf_mix

        quality_scale = np.clip(0.8 + quality * 80.0, 0.25, 1.25)
        size = np.clip(base_size * quality_scale, 0.0, 1.0)

        uns = self._experimental_unsup_redesign(df, probs, confidence)
        vol_rank = np.clip(np.nan_to_num(uns["vol_rank"], nan=0.5), 0.0, 1.0)
        route_scale = np.asarray(uns["route_scale"], dtype=np.float64)
        size = np.clip(size * route_scale, 0.0, 1.0)

        iso_anom = np.asarray(uns["iso_anom"], dtype=np.int64)
        vae_anom = np.asarray(uns["vae_anom"], dtype=np.int64)
        gate_block = np.asarray(uns["gate_block"], dtype=np.int64)
        size = np.where(gate_block == 1, 0.0, size)
        size = np.where((gate_block == 0) & ((iso_anom == 1) | (vae_anom == 1)), size * 0.5, size)

        # 방향별 평균 사이징 균형화(한쪽으로만 size가 죽는 현상 방지)
        m_long = (direction == 2) & (gate_block == 0)
        m_short = (direction == 0) & (gate_block == 0)
        long_mean = float(np.mean(size[m_long])) if np.any(m_long) else 0.0
        short_mean = float(np.mean(size[m_short])) if np.any(m_short) else 0.0
        if long_mean > 1e-6 and short_mean > 1e-6:
            bal_scale = float(np.clip(long_mean / short_mean, 0.70, 1.40))
            size = np.where(m_short, np.clip(size * bal_scale, 0.0, 1.0), size)

        min_conf = np.asarray(uns["min_conf"], dtype=np.float64)
        action = np.zeros(n, dtype=np.int64)  # -1 short, 0 hold, +1 long
        long_cond = (direction == 2) & (confidence >= min_conf) & (gate_block == 0)
        short_cond = (direction == 0) & (confidence >= min_conf) & (gate_block == 0)
        action[long_cond] = 1
        action[short_cond] = -1
        action[size < 0.05] = 0

        hold_raw = np.where(np.isfinite(mtl["hold"]), mtl["hold"], 12.0)
        target_hold = np.clip(np.round(hold_raw), 1, 48).astype(np.int64)
        target_hold = np.where(action == 0, 0, target_hold)
        target_hold = np.where(quality < 0.0, np.minimum(target_hold, 6), target_hold)
        target_hold = np.where((iso_anom == 1) | (vae_anom == 1), np.minimum(target_hold, 6), target_hold)
        target_hold = np.where(gate_block == 1, np.minimum(target_hold, 3), target_hold)

        expected_ret = np.where(action == 1, q50, np.where(action == -1, -q50, 0.0))
        tail_risk = np.where(action == 1, np.minimum(q10, 0.0), np.where(action == -1, -np.maximum(q90, 0.0), 0.0))
        composite = np.clip(expected_ret * (0.5 + confidence) * (1.0 - 0.5 * gate_block), -1.0, 1.0)
        entry_long_offset = np.asarray(entry_px["entry_long_offset"], dtype=np.float64)
        entry_short_offset = np.asarray(entry_px["entry_short_offset"], dtype=np.float64)
        entry_long_price = np.asarray(entry_px["entry_long_price"], dtype=np.float64)
        entry_short_price = np.asarray(entry_px["entry_short_price"], dtype=np.float64)

        tp_floor = 8e-4
        sl_floor = 6e-4
        ref_side = np.where(action != 0, action, np.where(direction == 2, 1, np.where(direction == 0, -1, 0)))
        tp_offset = np.where(
            ref_side > 0,
            np.maximum(q90, tp_floor),
            np.where(ref_side < 0, np.minimum(q10, -tp_floor), 0.0),
        )
        sl_offset = np.where(
            ref_side > 0,
            np.minimum(q10, -sl_floor),
            np.where(ref_side < 0, np.maximum(q90, sl_floor), 0.0),
        )
        tp_price = close * (1.0 + tp_offset)
        sl_price = close * (1.0 + sl_offset)

        out = pd.DataFrame(index=df.index)
        out["m7_trend_xgb_dn"] = p_xgb[:, 0]
        out["m7_trend_xgb_fl"] = p_xgb[:, 1]
        out["m7_trend_xgb_up"] = p_xgb[:, 2]
        out["m7_mtl_dn"] = p_mtl[:, 0]
        out["m7_mtl_fl"] = p_mtl[:, 1]
        out["m7_mtl_up"] = p_mtl[:, 2]
        out["m7_quant_dn"] = p_q[:, 0]
        out["m7_quant_fl"] = p_q[:, 1]
        out["m7_quant_up"] = p_q[:, 2]

        out["m7_prob_dn"] = probs[:, 0]
        out["m7_prob_fl"] = probs[:, 1]
        out["m7_prob_up"] = probs[:, 2]
        out["m7_direction"] = direction.astype(np.float32)
        out["m7_confidence"] = confidence
        out["m7_action"] = action.astype(np.float32)
        out["m7_size"] = size

        out["m7_q10"] = q10
        out["m7_q50"] = q50
        out["m7_q90"] = q90
        out["m7_qwidth"] = q_width
        out["m7_quality_pred"] = quality
        out["m7_hold_pred"] = np.nan_to_num(mtl["hold"], nan=0.0)
        out["m7_target_hold"] = target_hold.astype(np.float32)
        out["m7_entry_long_offset"] = entry_long_offset.astype(np.float32)
        out["m7_entry_short_offset"] = entry_short_offset.astype(np.float32)
        out["m7_entry_long_price"] = entry_long_price.astype(np.float32)
        out["m7_entry_short_price"] = entry_short_price.astype(np.float32)
        out["m7_tp_offset"] = tp_offset.astype(np.float32)
        out["m7_sl_offset"] = sl_offset.astype(np.float32)
        out["m7_tp_price"] = tp_price.astype(np.float32)
        out["m7_sl_price"] = sl_price.astype(np.float32)

        out["m7_gmm_cluster"] = np.asarray(uns["gmm_cluster"], dtype=np.float32)
        out["m7_gmm_conf"] = np.asarray(uns["gmm_conf"], dtype=np.float64)
        out["m7_gmm_vol_rank"] = vol_rank
        out["m7_hdb_label"] = np.asarray(hdb["label"], dtype=np.float32)
        out["m7_hdb_prob"] = np.asarray(hdb["prob"], dtype=np.float64)

        out["m7_iso_pred"] = np.where(iso_anom == 1, -1.0, 1.0).astype(np.float32)
        out["m7_iso_score"] = np.asarray(uns["iso_score"], dtype=np.float64)
        out["m7_vae_error"] = np.asarray(uns["vae_error"], dtype=np.float64)
        out["m7_vae_threshold"] = float(np.quantile(np.asarray(uns["vae_error"], dtype=np.float64), 0.972)) if n > 0 else 0.0
        out["m7_iso_anom"] = iso_anom.astype(np.float32)
        out["m7_vae_anom"] = vae_anom.astype(np.float32)
        out["m7_gate_block"] = gate_block.astype(np.float32)

        out["m7_expected_ret"] = expected_ret
        out["m7_tail_risk"] = tail_risk
        out["m7_composite_score"] = composite
        return out.astype(np.float32)

    def predict_last(self, df: pd.DataFrame) -> dict[str, float]:
        # Keep sufficient history so rolling/lag features for the final row are computable.
        pred = self.predict_batch(df.tail(512)).tail(1)
        if pred.empty:
            return {}
        row = pred.iloc[-1]
        return {k: float(v) for k, v in row.to_dict().items()}
