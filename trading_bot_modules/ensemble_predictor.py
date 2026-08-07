import gc
import logging
import time

import numpy as np
import pandas as pd
import torch

from ensemble.ensemble_router import (
    DLinearOFIForecaster,
    PatchTSTForecaster,
    TiDEVolatilityForecaster,
    TimesNetCycleForecaster,
)
from trading_bot_modules.position_accounting import _safe_float
from trading_bot_modules.runtime_config import (
    CONSOLE_LOG_MODEL_TRACE,
    FINAL_GOVERNOR_AI_FEATURE_GROUPS,
    FINAL_GOVERNOR_AI_FEATURE_STALE_SEC,
    FINAL_GOVERNOR_AI_TIMING_LOG_ENABLE,
)

logger = logging.getLogger("LiveBot")

def _traj_direction(traj: np.ndarray) -> float:
    """slope+delta 합의 → {-1.0, 0.0, 1.0}  (get_direction 동일 로직)"""
    if len(traj) < 2:
        return float(np.sign(np.mean(traj)))
    slope = float(np.polyfit(np.arange(len(traj)), traj, 1)[0])
    delta = float(traj[-1] - traj[0])
    if slope > 0 and delta > 0:
        return 1.0
    if slope < 0 and delta < 0:
        return -1.0
    return 0.0

def _traj_conf(traj: np.ndarray) -> float:
    """tanh(|기울기|/표준편차) — get_conf 동일 로직"""
    if len(traj) < 2:
        return 0.5
    slope = float(np.polyfit(np.arange(len(traj), dtype=float), traj, 1)[0])
    std = float(np.std(traj)) + 1e-6
    return float(np.tanh(abs(slope) / std))

class EnsemblePredictor:
    MODEL_ORDER = []
    AI_FEATURE_COLUMNS = {
        "patchtst": [
            "ai_dir_edge",
            "ai_dir_p_up",
            "ai_dir_p_down",
            "ai_dir_p_flat",
            "ai_dir_entropy",
            "patchtst_median",
            "patchtst_regime_sim",
            "pred_patchtst",
            "conf_patchtst",
        ],
        "tide": [
            "ai_adverse_risk",
            "ai_reward_risk",
            "ai_vol_regime_pct",
            "tide_vol_raw",
            "tide_vol_zscore",
        ],
        "timesnet": [
            "ai_anchor_revert_prob",
            "ai_anchor_overheat",
            "ai_anchor_trend_escape_prob",
            "timesnet_cycle_sin",
            "timesnet_cycle_cos",
            "timesnet_cycle_delta",
        ],
        "dlinear": [
            "ai_flow_pressure",
            "ai_flow_exhaustion",
            "ai_flow_flip_prob",
            "ai_flow_slope",
            "dlinear_smf_ema",
            "dlinear_smf_slope",
        ],
    }
    AI_FEATURE_MODELS = {
        "patchtst": "PatchTST",
        "tide": "TiDE",
        "timesnet": "TimesNet",
        "dlinear": "DLinear",
    }

    def __init__(self):
        model_factories = {
            "PatchTST": PatchTSTForecaster,
            "TiDE": TiDEVolatilityForecaster,
            "TimesNet": TimesNetCycleForecaster,
            "DLinear": DLinearOFIForecaster,
        }
        active_model_names = {
            str(self.AI_FEATURE_MODELS.get(str(group).lower()) or "")
            for group in FINAL_GOVERNOR_AI_FEATURE_GROUPS
        }
        active_model_names.discard("")
        missing_factories = sorted(name for name in active_model_names if name not in model_factories)
        if missing_factories:
            raise RuntimeError(f"missing AI model factories: {missing_factories}")
        self.models = {name: model_factories[name]() for name in sorted(active_model_names)}
        self.last_trace: list[dict[str, object]] = []
        self.last_errors: list[dict[str, object]] = []
        self.last_timing: dict[str, object] = {}
        self._feature_cache: dict[str, tuple[float, float, tuple | None]] = {}
        self._active_feature_frame_key: tuple | None = None
        self._prediction_cache_key: tuple | None = None
        self._prediction_cache: dict[tuple[str, int], object] = {}

    def _frame_cache_key(self, df: pd.DataFrame) -> tuple:
        try:
            rows = int(len(df))
            if df is None or rows == 0:
                return (0, "", 0.0)
            last = df.iloc[-1]
            ts = str(last.get("timestamp", df.index[-1]) if hasattr(last, "get") else df.index[-1])
            close = _safe_float(last.get("close", 0.0) if hasattr(last, "get") else 0.0, 0.0)
            return (rows, ts, round(float(close), 8))
        except Exception:
            return (int(len(df)) if df is not None else 0, "", 0.0)

    def _begin_frame(self, df: pd.DataFrame) -> tuple:
        frame_key = self._frame_cache_key(df)
        self._active_feature_frame_key = frame_key
        if self._prediction_cache_key != frame_key:
            self._prediction_cache_key = frame_key
            self._prediction_cache = {}
        return frame_key

    @staticmethod
    def _slice_prediction(res, horizon: int):
        try:
            med = np.asarray(getattr(res, "median"), dtype=np.float32)
            conf = np.asarray(getattr(res, "confidence"), dtype=np.float32)
            if med.ndim >= 2:
                med = med[:, :horizon]
            else:
                med = med[:horizon]
            if conf.ndim >= 2:
                conf = conf[:, :horizon]
            else:
                conf = conf[:horizon]
            return type(res)(med, conf)
        except Exception:
            return res

    def _predict_cached(self, name: str, model, df: pd.DataFrame, *, horizon: int) -> tuple[object, bool, float]:
        self._begin_frame(df)
        cache_key = (str(name), int(horizon))
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key], True, 0.0
        for (cached_name, cached_horizon), cached_res in list(self._prediction_cache.items()):
            if cached_name == str(name) and int(cached_horizon) >= int(horizon):
                sliced = self._slice_prediction(cached_res, int(horizon))
                self._prediction_cache[cache_key] = sliced
                return sliced, True, 0.0
        t0 = time.perf_counter()
        res = model.predict(df, horizon=int(horizon))
        elapsed = float(time.perf_counter() - t0)
        self._prediction_cache[cache_key] = res
        return res, False, elapsed

    def _refined_features_from_prediction(self, model, pred, df: pd.DataFrame) -> dict[str, float]:
        if pred is None or getattr(pred, "median", None) is None or pred.median.size == 0 or np.isnan(pred.median).all():
            nan_df = pd.DataFrame(index=[0])
            tmp = model._apply_refined_batch_logic(nan_df, np.array([np.nan], dtype=np.float32), df)
        else:
            val = float(np.asarray(pred.median, dtype=np.float32)[0, 0])
            one = pd.DataFrame(index=[0])
            tmp = model._apply_refined_batch_logic(one, np.array([val], dtype=np.float32), df)
        out: dict[str, float] = {}
        for c in tmp.columns:
            v = tmp.iloc[-1][c]
            out[str(c)] = float(v) if np.isfinite(v) else 0.0
        return out

    def _record_ai_error(self, model: str | None, stage: str, error: Exception | str, fallback: str) -> dict[str, object]:
        item = {
            "model": str(model or "unknown"),
            "stage": str(stage),
            "error": str(error),
            "fallback": str(fallback),
            "ts": time.time(),
        }
        self.last_errors.append(item)
        return item

    def _cache_features(self, features: dict[str, float], frame_key: tuple | None = None) -> None:
        now = time.time()
        cache_frame_key = frame_key if frame_key is not None else self._active_feature_frame_key
        for col, val in dict(features or {}).items():
            try:
                fval = float(val)
            except Exception:
                continue
            if np.isfinite(fval):
                self._feature_cache[str(col)] = (fval, now, cache_frame_key)

    def _cached_features(self, cols: list[str], frame_key: tuple | None = None) -> dict[str, float]:
        now = time.time()
        target_frame_key = frame_key if frame_key is not None else self._active_feature_frame_key
        out: dict[str, float] = {}
        for col in cols:
            cached = self._feature_cache.get(str(col))
            if cached is None:
                continue
            try:
                val, ts, cached_frame_key = cached
            except ValueError:
                continue
            if cached_frame_key != target_frame_key:
                continue
            if now - float(ts) <= float(FINAL_GOVERNOR_AI_FEATURE_STALE_SEC):
                out[str(col)] = float(val)
        return out

    def best_ai_features(self, df: pd.DataFrame) -> dict[str, float]:
        frame_key = self._begin_frame(df)
        out: dict[str, float] = {}
        trace: list[dict[str, object]] = []
        timing_rows: list[dict[str, object]] = []
        stage_t0 = time.perf_counter()
        for group in FINAL_GOVERNOR_AI_FEATURE_GROUPS:
            model_name = self.AI_FEATURE_MODELS.get(str(group).lower())
            model = self.models.get(model_name or "")
            expected_cols = list(self.AI_FEATURE_COLUMNS.get(str(group).lower(), []))
            group_t0 = time.perf_counter()
            cache_hit = False
            if model is None:
                raise RuntimeError(f"missing AI model for group={group} model={model_name}")
            try:
                if not getattr(model, "available", False):
                    raise RuntimeError(f"AI model unavailable for group={group} model={model_name}")
                feats = model.get_refined_features(df)
                feats = {str(k): float(v) for k, v in dict(feats or {}).items() if np.isfinite(float(v))}
                if str(group).lower() == "patchtst":
                    edge = float(feats.get("ai_dir_edge", np.nan))
                    if not np.isfinite(edge):
                        raise RuntimeError("PatchTST missing ai_dir_edge")
                    conf = float(feats.get("patchtst_regime_sim", abs(edge)))
                    feats["pred_patchtst"] = float(np.clip(edge, -1.0, 1.0))
                    feats["conf_patchtst"] = float(np.clip(conf, 0.0, 1.0))
                if expected_cols:
                    missing_expected = [c for c in expected_cols if c not in feats]
                    if missing_expected:
                        raise RuntimeError(
                            f"AI feature group={group} model={model_name} missing expected cols={missing_expected[:8]}"
                        )
                    if all(abs(float(feats.get(c, 0.0))) <= 1e-12 for c in expected_cols):
                        raise RuntimeError(
                            f"AI feature group={group} model={model_name} produced all-zero output for expected cols"
                        )
                for col, val in dict(feats or {}).items():
                    try:
                        fval = float(val)
                    except Exception:
                        fval = 0.0
                    out[str(col)] = fval if np.isfinite(fval) else 0.0
                self._cache_features(out, frame_key)
                trace.append({"group": group, "model": model_name, "status": "live", "cols": sorted(dict(feats or {}).keys()), "cache_hit": bool(cache_hit)})
                timing_rows.append({
                    "group": str(group),
                    "model": str(model_name),
                    "sec": float(time.perf_counter() - group_t0),
                    "cache_hit": bool(cache_hit),
                    "status": "live",
                })
            except Exception as e:
                trace.append({"group": group, "model": model_name, "status": "failed", "error": str(e), "cache_hit": bool(cache_hit)})
                timing_rows.append({
                    "group": str(group),
                    "model": str(model_name),
                    "sec": float(time.perf_counter() - group_t0),
                    "cache_hit": bool(cache_hit),
                    "status": "failed",
                })
                self._record_ai_error(
                    model_name,
                    "best_ai_features",
                    e,
                    "fail_fast",
                )
                self.last_trace = trace
                self.last_timing["best_ai_features"] = {
                    "sec": float(time.perf_counter() - stage_t0),
                    "groups": timing_rows,
                    "frame_key": list(frame_key),
                }
                logger.error("%s refined AI feature generation failed: %s", model_name, e)
                raise
        self._cache_features(out, frame_key)
        self.last_trace = trace
        best_sec = float(time.perf_counter() - stage_t0)
        self.last_timing["best_ai_features"] = {
            "sec": best_sec,
            "groups": timing_rows,
            "frame_key": list(frame_key),
        }
        if FINAL_GOVERNOR_AI_TIMING_LOG_ENABLE:
            try:
                predict_sec = float(dict(self.last_timing.get("predict_all", {}) or {}).get("sec", 0.0) or 0.0)
                cache_hits = sum(1 for row in timing_rows if bool(row.get("cache_hit", False)))
                group_s = ",".join(
                    f"{row.get('model')}:{float(row.get('sec', 0.0)):.2f}s{'*' if bool(row.get('cache_hit', False)) else ''}"
                    for row in timing_rows
                )
                logger.info(
                    "TIMING ai_features predict_all=%.2fs best=%.2fs cache_hits=%d groups=%s",
                    predict_sec,
                    best_sec,
                    int(cache_hits),
                    group_s,
                )
            except Exception:
                pass
        return out

    async def predict_all_async(self, df: pd.DataFrame):
        frame_key = self._begin_frame(df)
        preds, confs = [], []
        results = []
        inference_errors: dict[str, str] = {}
        timing_rows: list[dict[str, object]] = []
        stage_t0 = time.perf_counter()
        for name in self.MODEL_ORDER:
            m = self.models[name]
            model_t0 = time.perf_counter()
            if not getattr(m, 'available', False):
                results.append(None)
                timing_rows.append({
                    "model": str(name),
                    "sec": float(time.perf_counter() - model_t0),
                    "cache_hit": False,
                    "status": "unavailable",
                })
                continue
            try:
                pred, cache_hit, _pred_sec = self._predict_cached(str(name), m, df, horizon=6)
                results.append(pred)
                timing_rows.append({
                    "model": str(name),
                    "sec": float(time.perf_counter() - model_t0),
                    "cache_hit": bool(cache_hit),
                    "status": "live",
                })
            except Exception as e:
                inference_errors[name] = str(e)
                self._record_ai_error(name, "predict_all_async", e, "model_prediction_unavailable")
                logger.error("%s 추론 실패: %s", name, e)
                results.append(None)
                timing_rows.append({
                    "model": str(name),
                    "sec": float(time.perf_counter() - model_t0),
                    "cache_hit": False,
                    "status": "failed",
                    "error": str(e),
                })

        def _extract_last_conf(res) -> float:
            try:
                c = getattr(res, "confidence", None)
                if c is None:
                    return float("nan")
                arr = np.asarray(c, dtype=np.float32)
                if arr.ndim == 0:
                    v = float(arr)
                elif arr.ndim == 1:
                    v = float(arr[-1])
                else:
                    v = float(arr[-1][-1])
                return v if np.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        traces: list[dict[str, object]] = []
        for name, res in zip(self.MODEL_ORDER, results):
            p_val, c_val = float("nan"), float("nan")
            conf_src = "none"
            traj_last = float("nan")
            traj_std = float("nan")
            traj_zero_like = False
            if res is not None and getattr(res, 'median', None) is not None:
                traj = np.array(res.median[-1], dtype=np.float32)
                if np.all(np.isfinite(traj)):
                    traj_last = float(traj[-1]) if traj.size > 0 else float("nan")
                    traj_std = float(np.std(traj)) if traj.size > 0 else float("nan")
                    traj_zero_like = bool(np.allclose(traj, 0.0, atol=1e-9))
                    p_val = _traj_direction(traj)
                    c_val = _extract_last_conf(res)
                    conf_src = "model"
                    if not np.isfinite(c_val):
                        c_val = _traj_conf(traj)
                        conf_src = "traj_fallback"
                    c_val = float(np.clip(c_val, 0.0, 1.0))
            traces.append({
                "model": name,
                "pred": float(p_val) if np.isfinite(p_val) else float("nan"),
                "conf": float(c_val) if np.isfinite(c_val) else float("nan"),
                "traj_last": traj_last,
                "traj_std": traj_std,
                "traj_zero_like": traj_zero_like,
                "conf_src": conf_src,
                "error": inference_errors.get(name, ""),
                "ok": bool(np.isfinite(p_val) and np.isfinite(c_val)),
                "is_zero": bool(np.isfinite(p_val) and np.isfinite(c_val) and abs(float(p_val)) < 1e-12 and abs(float(c_val)) < 1e-12),
            })
            preds.append(p_val)
            confs.append(c_val)
        self.last_trace = traces
        self.last_timing["predict_all"] = {
            "sec": float(time.perf_counter() - stage_t0),
            "models": timing_rows,
            "frame_key": list(frame_key),
        }

        if CONSOLE_LOG_MODEL_TRACE:
            try:
                _parts = []
                for t in traces:
                    _p = t["pred"]
                    _c = t["conf"]
                    _p_s = "nan" if not np.isfinite(_p) else f"{float(_p):+.4f}"
                    _c_s = "nan" if not np.isfinite(_c) else f"{float(_c):.4f}"
                    _ts = t.get("traj_std", float("nan"))
                    _ts_s = "nan" if not np.isfinite(_ts) else f"{float(_ts):.6f}"
                    _z = "Z0" if bool(t.get("traj_zero_like", False)) else "Z-"
                    _flag = "OK" if t["ok"] else "MISS"
                    _parts.append(f"{t['model']}:{_flag}(pred={_p_s},conf={_c_s},src={t['conf_src']},std={_ts_s},{_z})")
                logger.info("MODEL_TRACE %s", " | ".join(_parts))
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.array(preds), np.array(confs)
