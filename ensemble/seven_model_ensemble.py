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


def _resolve_model_path(meta_path: str, model_ref: str | None) -> str:
    if not model_ref:
        return os.path.splitext(meta_path)[0] + ".pkl"
    if os.path.isabs(model_ref):
        return model_ref
    return os.path.join(os.path.dirname(meta_path), model_ref)


def _to_numeric_frame(df: pd.DataFrame, cols: list[str], fill_mode: str = "median") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    if fill_mode == "median":
        med = out.median(numeric_only=True)
        out = out.fillna(med).fillna(0.0)
    elif fill_mode == "zero":
        out = out.fillna(0.0)
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
        "entry_price_model": "data/ensemble/supervised/entry_price_model.json",
        "multi_target_lgbm": "data/ensemble/supervised/multi_target_lgbm.json",
        "quantile_forest": "data/ensemble/supervised/quantile_forest.json",
        "gmm_volatility": "data/ensemble/unsupervised/gmm_volatility.json",
        "hdbscan_regime": "data/ensemble/unsupervised/hdbscan_regime.json",
        "isolation_forest": "data/ensemble/unsupervised/isolation_forest.json",
        "vae_anomaly": "data/ensemble/unsupervised/vae_anomaly.json",
    }

    def __init__(
        self,
        meta_paths: dict[str, str] | None = None,
        weight_trend_xgb: float = 0.45,
        weight_multitarget: float = 0.35,
        weight_quantile: float = 0.20,
    ):
        self.meta_paths = dict(self.DEFAULT_META_PATHS)
        if meta_paths:
            self.meta_paths.update(meta_paths)

        self.weight_trend_xgb = float(weight_trend_xgb)
        self.weight_multitarget = float(weight_multitarget)
        self.weight_quantile = float(weight_quantile)

        self.trend_xgb = _ModelState(False)
        self.entry_price = _ModelState(False)
        self.multi_target = _ModelState(False)
        self.quantile = _ModelState(False)
        self.gmm = _ModelState(False)
        self.hdbscan = _ModelState(False)
        self.isolation = _ModelState(False)
        self.vae = _ModelState(False)

        self._load_all()

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
        self._load_entry_price()
        self._load_multitarget()
        self._load_quantile()
        self._load_gmm()
        self._load_hdbscan()
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
        x = _to_numeric_frame(df, self.trend_xgb.feature_cols or [], fill_mode="median")
        model = self.trend_xgb.model
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
        labels = np.full(n, -1, dtype=np.int64)
        probs = np.zeros(n, dtype=np.float64)
        if not self.hdbscan.available or self.hdbscan.model is None:
            return {"label": labels, "prob": probs}
        cols = self.hdbscan.feature_cols or []
        x_raw = _to_numeric_frame(df, cols, fill_mode="median").values.astype(np.float32)
        mean = (self.hdbscan.extra or {}).get("mean")
        std = (self.hdbscan.extra or {}).get("std")
        std = np.where(np.asarray(std) < 1e-8, 1.0, std)
        x = (x_raw - mean) / std
        model = self.hdbscan.model
        try:
            import hdbscan  # type: ignore

            labels, probs = hdbscan.approximate_predict(model, x)
            labels = np.asarray(labels, dtype=np.int64)
            probs = np.asarray(probs, dtype=np.float64)
        except Exception:
            if hasattr(model, "labels_") and len(getattr(model, "labels_", [])) == n:
                labels = np.asarray(model.labels_, dtype=np.int64)
                probs = np.asarray(getattr(model, "probabilities_", np.zeros(n)), dtype=np.float64)
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

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        if n == 0:
            return pd.DataFrame(index=df.index)

        p_xgb = self._predict_trend_xgb(df)
        mtl = self._predict_multitarget(df)
        p_mtl = mtl["probs"]
        q = self._predict_quantile(df)
        p_q = q["probs"]
        entry_px = self._predict_entry_price(df)

        gmm = self._predict_gmm(df)
        hdb = self._predict_hdbscan(df)
        iso = self._predict_isolation(df)
        vae = self._predict_vae(df)

        probs = (
            self.weight_trend_xgb * p_xgb
            + self.weight_multitarget * p_mtl
            + self.weight_quantile * p_q
        )
        denom = self.weight_trend_xgb + self.weight_multitarget + self.weight_quantile
        probs = probs / max(denom, 1e-12)
        probs = _safe_prob3(probs)
        # Batch-level prior rebalance: 클래스 사전확률이 한쪽으로 과도하게 쏠릴 때 완만히 보정
        # target prior는 DOWN/FLAT/UP = 0.42/0.16/0.42 로 설정
        prior = np.clip(np.mean(probs, axis=0), 1e-6, 1.0)
        target_prior = np.array([0.42, 0.16, 0.42], dtype=np.float64)
        prior_scale = np.clip(target_prior / prior, 0.75, 1.35)
        probs = probs * prior_scale[None, :]
        probs = _safe_prob3(probs)

        sort_p = np.sort(probs, axis=1)
        p_top = sort_p[:, 2]
        p_second = sort_p[:, 1]
        confidence = np.clip((p_top - 1.0 / 3.0) * 1.5 + (p_top - p_second) * 0.6, 0.0, 1.0)
        direction = np.argmax(probs, axis=1).astype(np.int64)  # 0=DOWN,1=FLAT,2=UP

        q10 = np.nan_to_num(q["q10"], nan=0.0)
        q50 = np.nan_to_num(q["q50"], nan=0.0)
        q90 = np.nan_to_num(q["q90"], nan=0.0)
        q_width = np.maximum(q90 - q10, 1e-6)
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        # 롱/숏 완전 대칭 사이징: edge=|q50|, risk=qwidth
        rr_sym = np.abs(q50) / np.maximum(q_width, 1e-6)
        dir_gap = np.abs(probs[:, 2] - probs[:, 0])  # 방향 우위
        conf_mix = np.clip(0.65 * confidence + 0.35 * dir_gap, 0.0, 1.0)
        rr = np.where(direction == 1, 0.0, rr_sym)
        base_size = np.tanh(rr * 1.1) * conf_mix

        quality = np.nan_to_num(mtl["quality"], nan=0.0)
        quality_scale = np.clip(0.8 + quality * 80.0, 0.25, 1.25)
        size = np.clip(base_size * quality_scale, 0.0, 1.0)

        vol_rank = np.clip(np.nan_to_num(gmm["vol_rank"], nan=0.5), 0.0, 1.0)
        route_scale = np.where(
            vol_rank >= 0.8,
            0.55,
            np.where(vol_rank >= 0.6, 0.75, np.where(vol_rank <= 0.2, 1.10, 1.0)),
        )
        route_scale = np.where(hdb["label"] == -1, route_scale * 0.80, route_scale)
        size = np.clip(size * route_scale, 0.0, 1.0)

        # VAE unavailable/degenerate fallback: iso score + vol rank 기반 대체 anomaly 신호 생성
        vae_err_cur = np.nan_to_num(np.asarray(vae.get("error", np.full(n, np.nan)), dtype=np.float64), nan=0.0)
        if (not self.vae.available) or float(np.std(vae_err_cur)) < 1e-12:
            iso_score_pos = np.maximum(np.nan_to_num(iso["score"], nan=0.0), 0.0)
            vae_err_proxy = iso_score_pos + 0.25 * vol_rank
            if len(vae_err_proxy) >= 200:
                th = float(np.quantile(vae_err_proxy, 0.97))
            else:
                th = float(np.mean(vae_err_proxy) + 2.5 * np.std(vae_err_proxy))
            vae = {
                "error": vae_err_proxy.astype(np.float64),
                "is_anomaly": (vae_err_proxy >= th).astype(np.int64),
                "threshold": float(th),
            }

        iso_anom = (iso["pred"] == -1).astype(np.int64)
        vae_anom = np.asarray(vae["is_anomaly"], dtype=np.int64)
        gate_block = ((iso_anom == 1) & (vae_anom == 1)).astype(np.int64)
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

        min_conf = np.where(vol_rank >= 0.8, 0.60, 0.45)
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
        target_hold = np.where((iso_anom == 1) | (vae_anom == 1), np.minimum(target_hold, 4), target_hold)

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

        out["m7_gmm_cluster"] = np.asarray(gmm["cluster"], dtype=np.float32)
        out["m7_gmm_conf"] = np.asarray(gmm["confidence"], dtype=np.float64)
        out["m7_gmm_vol_rank"] = vol_rank
        out["m7_hdb_label"] = np.asarray(hdb["label"], dtype=np.float32)
        out["m7_hdb_prob"] = np.asarray(hdb["prob"], dtype=np.float64)

        out["m7_iso_pred"] = np.asarray(iso["pred"], dtype=np.float32)
        out["m7_iso_score"] = np.asarray(iso["score"], dtype=np.float64)
        out["m7_vae_error"] = np.nan_to_num(vae["error"], nan=0.0)
        out["m7_vae_threshold"] = float(vae.get("threshold", np.nan)) if np.isfinite(float(vae.get("threshold", np.nan))) else 0.0
        out["m7_iso_anom"] = iso_anom.astype(np.float32)
        out["m7_vae_anom"] = vae_anom.astype(np.float32)
        out["m7_gate_block"] = gate_block.astype(np.float32)

        out["m7_expected_ret"] = expected_ret
        out["m7_tail_risk"] = tail_risk
        out["m7_composite_score"] = composite
        return out.astype(np.float32)

    def predict_last(self, df: pd.DataFrame) -> dict[str, float]:
        pred = self.predict_batch(df.tail(1))
        if pred.empty:
            return {}
        row = pred.iloc[-1]
        return {k: float(v) for k, v in row.to_dict().items()}
