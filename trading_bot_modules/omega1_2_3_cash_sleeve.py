from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from trading_bot_modules.omega1_2_1_live import Omega121Decision


OMEGA123_MODEL_ID = "omega1_2_3_ev_hgb_cash_sleeve_20260615"
OMEGA124_MODEL_ID = "omega1_2_4_ev_calibrated_cash_sleeve_20260616"
OMEGA128B_MODEL_ID = "omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260618"
OMEGA3_MODEL_ID = "omega3_full_retrain_hf7_dynamic_cash_sleeve_20260618"
OMEGA123_SLEEVE = "omega1_2_8b_numeric_cash_sleeve"
BASE_MODEL_ID = "omega1_2_1_tp_runner_clean_repair_20260613"
OMEGA128B_PARENT_MODEL_ID = "omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608"
NUMERIC_FULL_RETRAIN_MODEL_IDS = {OMEGA128B_MODEL_ID, OMEGA3_MODEL_ID}
RISK_MODEL_KEYS = {
    "long_take_profit",
    "short_take_profit",
    "long_stop_loss",
    "short_stop_loss",
    "long_notional",
    "short_notional",
    "long_leverage",
    "short_leverage",
}
ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


@dataclass(frozen=True)
class Omega123SleeveDecision:
    action: int
    side: int
    notional_exposure: float
    leverage: float
    position_fraction: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    confidence: float
    long_ev: float
    short_ev: float
    trace: dict[str, Any]


class Omega123CashSleeveAdapter:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = str(bundle_path)
        bundle = joblib.load(self.bundle_path)
        self.model_id = str(bundle.get("model_id"))
        if self.model_id not in {OMEGA123_MODEL_ID, OMEGA124_MODEL_ID, *NUMERIC_FULL_RETRAIN_MODEL_IDS}:
            raise RuntimeError(f"Omega1.2.3 bundle model_id mismatch: {bundle.get('model_id')}")
        expected_base = OMEGA128B_PARENT_MODEL_ID if self.model_id in NUMERIC_FULL_RETRAIN_MODEL_IDS else BASE_MODEL_ID
        if str(bundle.get("base_model_id")) != expected_base:
            raise RuntimeError(f"Omega1.2.3 base model mismatch: {bundle.get('base_model_id')}")
        self.long_model = bundle["long_model"]
        self.short_model = bundle["short_model"]
        self.utility_long_model = bundle.get("utility_long_model")
        self.utility_short_model = bundle.get("utility_short_model")
        self.feature_cols = list(bundle["feature_cols"])
        self.risk = dict(bundle["risk"])
        self.dynamic_risk = bool(bundle.get("dynamic_risk", False))
        self.risk_models = dict(bundle.get("risk_models") or {})
        self.risk_label_bounds = dict(bundle.get("risk_label_bounds") or {})
        if self.dynamic_risk:
            if set(self.risk_models.keys()) != RISK_MODEL_KEYS:
                raise RuntimeError(f"Omega3 dynamic risk model keys mismatch: {sorted(self.risk_models.keys())}")
            if set(self.risk_label_bounds.keys()) != RISK_MODEL_KEYS:
                raise RuntimeError(f"Omega3 dynamic risk bounds mismatch: {sorted(self.risk_label_bounds.keys())}")
        self.ev_min = float(bundle["ev_min"])
        self.utility_min = float(bundle.get("utility_min", 0.0) or 0.0)
        self.margin_min = float(bundle.get("margin_min", 0.0) or 0.0)
        self.support_profile = dict(bundle.get("support_profile") or {})
        self.conservative_gate = dict(bundle.get("conservative_gate") or {})
        if self.model_id in NUMERIC_FULL_RETRAIN_MODEL_IDS:
            if not self.support_profile:
                raise RuntimeError("Numeric cash sleeve requires support_profile in bundle")
            if not self.conservative_gate:
                raise RuntimeError("Numeric cash sleeve requires conservative_gate in bundle")
        calibration = dict(bundle.get("calibration") or {})
        self.long_ev_offset = float(calibration.get("long_abs_residual_offset", 0.0) or 0.0)
        self.short_ev_offset = float(calibration.get("short_abs_residual_offset", 0.0) or 0.0)
        self.long_utility_offset = float(calibration.get("long_utility_abs_residual_offset", 0.0) or 0.0)
        self.short_utility_offset = float(calibration.get("short_utility_abs_residual_offset", 0.0) or 0.0)
        if self.model_id in NUMERIC_FULL_RETRAIN_MODEL_IDS and (self.utility_long_model is None or self.utility_short_model is None):
            raise RuntimeError("Numeric cash sleeve requires utility models")
        self.primary_active_history: list[int] = []
        self.primary_side_history: list[int] = []
        self.primary_cash_streak = 0
        self.last_primary_active_len = 0
        self._current_active_len = 0
        self.last_primary_side = 0

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "primary_active_history": [int(x) for x in self.primary_active_history[-512:]],
            "primary_side_history": [int(x) for x in self.primary_side_history[-512:]],
            "primary_cash_streak": int(self.primary_cash_streak),
            "last_primary_active_len": int(self.last_primary_active_len),
            "current_active_len": int(self._current_active_len),
            "last_primary_side": int(self.last_primary_side),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise RuntimeError("Omega1.2.3 state restore requires object payload")
        active = [1 if int(x) else 0 for x in list(state.get("primary_active_history", []))[-512:]]
        sides = [int(x) for x in list(state.get("primary_side_history", []))[-512:]]
        if len(active) != len(sides):
            raise RuntimeError("Omega1.2.3 state restore history length mismatch")
        bad_sides = [x for x in sides if x not in (-1, 0, 1)]
        if bad_sides:
            raise RuntimeError(f"Omega1.2.3 state restore invalid side: {bad_sides[:5]}")
        self.primary_active_history = active
        self.primary_side_history = sides
        self.primary_cash_streak = int(state.get("primary_cash_streak", 0) or 0)
        self.last_primary_active_len = int(state.get("last_primary_active_len", 0) or 0)
        self._current_active_len = int(state.get("current_active_len", 0) or 0)
        self.last_primary_side = int(state.get("last_primary_side", 0) or 0)

    def observe_primary(self, primary_active: bool, primary_side: int) -> None:
        active = 1 if bool(primary_active) else 0
        side = int(primary_side) if active else 0
        if active:
            self._current_active_len += 1
            self.primary_cash_streak = 0
            self.last_primary_side = side
        else:
            if self.primary_active_history and self.primary_active_history[-1] == 1:
                self.last_primary_active_len = int(self._current_active_len)
                self._current_active_len = 0
            self.primary_cash_streak += 1
        self.primary_active_history.append(active)
        self.primary_side_history.append(side)
        if len(self.primary_active_history) > 512:
            self.primary_active_history = self.primary_active_history[-512:]
            self.primary_side_history = self.primary_side_history[-512:]

    @staticmethod
    def _market_features(frame: pd.DataFrame) -> dict[str, float]:
        required = ["open", "high", "low", "close", "timestamp"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Omega1.2.3 missing market columns: {missing}")
        close = pd.to_numeric(frame["close"], errors="raise")
        high = pd.to_numeric(frame["high"], errors="raise")
        low = pd.to_numeric(frame["low"], errors="raise")
        open_ = pd.to_numeric(frame["open"], errors="raise")
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        out = pd.DataFrame(index=frame.index)
        out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
        out["body_pct"] = ((close - open_) / close).replace([np.inf, -np.inf], np.nan)
        out["atr14_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)
        for lag in (1, 3, 6, 12, 24):
            out[f"ret_{lag}"] = close.pct_change(lag).replace([np.inf, -np.inf], np.nan)
        for win in (6, 12, 24, 48):
            out[f"ret_vol_{win}"] = ret.rolling(win, min_periods=max(3, win // 3)).std()
            out[f"range_mean_{win}"] = out["bar_range_pct"].rolling(win, min_periods=max(3, win // 3)).mean()
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        out["ema9_21_gap"] = ((ema9 - ema21) / close).replace([np.inf, -np.inf], np.nan)
        ts = pd.to_datetime(frame["timestamp"], errors="raise")
        minute = ts.dt.hour * 60 + ts.dt.minute
        out["tod_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
        out["tod_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
        return {
            str(k): float(v)
            for k, v in out.replace([np.inf, -np.inf], np.nan).fillna(0.0).iloc[-1].to_dict().items()
        }

    @staticmethod
    def _trace_features(primary: Omega121Decision) -> dict[str, float]:
        trace = dict(primary.trace or {})
        direction = dict(trace.get("direction_proba") or {})
        quality = dict(trace.get("quality_proba") or {})
        expert = str(trace.get("router_expert") or "")
        side = int(primary.side)
        take_profit = float(primary.take_profit)
        stop_loss = float(primary.stop_loss)
        features = {
            "tabm_router_confidence": float(trace.get("router_confidence", 0.0)),
            "tabm_router_margin": float(trace.get("router_margin", 0.0)),
            "tabm_dir_p_cash": float(direction.get("cash", 0.0)),
            "tabm_dir_p_long": float(direction.get("long", 0.0)),
            "tabm_dir_p_short": float(direction.get("short", 0.0)),
            "tabm_dir_confidence": float(max(direction.values()) if direction else 0.0),
            "tabm_dir_side_edge": float(direction.get("long", 0.0) - direction.get("short", 0.0)),
            "tabm_dir_trade_prob": float(direction.get("long", 0.0) + direction.get("short", 0.0)),
            "tabm_quality_p_cash": float(quality.get("cash", 0.0)),
            "tabm_quality_p_long": float(quality.get("long", 0.0)),
            "tabm_quality_p_short": float(quality.get("short", 0.0)),
            "tabm_quality_for_action": float(trace.get("quality_for_action", primary.quality_score)),
            "tabm_router_bull": 1.0 if expert == "bull" else 0.0,
            "tabm_router_bear": 1.0 if expert == "bear" else 0.0,
            "tabm_router_chop_expert": 1.0 if expert in {"chop", "chop_expert"} else 0.0,
            "dec_action": float(primary.action),
            "dec_side": float(side),
            "dec_quality_score": float(primary.quality_score),
            "dec_confidence": float(primary.confidence),
            "dec_notional_exposure": float(primary.notional_exposure),
            "dec_position_fraction": float(primary.position_fraction),
            "dec_leverage": float(primary.leverage),
            "dec_take_profit": take_profit,
            "dec_stop_loss": stop_loss,
            "dec_rr": take_profit / max(abs(stop_loss), 1e-8),
        }
        features.update(
            {
                "router_confidence": features["tabm_router_confidence"],
                "router_margin": features["tabm_router_margin"],
                "dir_p_cash": features["tabm_dir_p_cash"],
                "dir_p_long": features["tabm_dir_p_long"],
                "dir_p_short": features["tabm_dir_p_short"],
                "dir_confidence": features["tabm_dir_confidence"],
                "dir_side_edge": features["tabm_dir_side_edge"],
                "dir_trade_prob": features["tabm_dir_trade_prob"],
                "quality_p_cash": features["tabm_quality_p_cash"],
                "quality_p_long": features["tabm_quality_p_long"],
                "quality_p_short": features["tabm_quality_p_short"],
                "quality_for_action": features["tabm_quality_for_action"],
                "router_is_bull": features["tabm_router_bull"],
                "router_is_bear": features["tabm_router_bear"],
                "router_is_chop": features["tabm_router_chop_expert"],
                "side": float(side),
                "base_notional": float(trace.get("base_notional", primary.notional_exposure)),
                "base_tp": take_profit,
                "base_sl": stop_loss,
            }
        )
        return features

    def _history_features(self, frame: pd.DataFrame, primary: Omega121Decision) -> dict[str, float]:
        active = np.asarray(self.primary_active_history, dtype=np.float64)
        if len(active) == 0:
            raise RuntimeError("Omega1.2.3 primary history is empty; call observe_primary before decide_latest")
        market = self._market_features(frame)
        close = pd.to_numeric(frame["close"], errors="raise")
        high = pd.to_numeric(frame["high"], errors="raise")
        low = pd.to_numeric(frame["low"], errors="raise")
        ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rng = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cash_range_ratio = (
            rng.rolling(12, min_periods=1).mean() / rng.rolling(48, min_periods=1).mean().replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        trace_features = self._trace_features(primary)
        probs = np.clip(
            np.asarray(
                [
                    trace_features["tabm_dir_p_cash"],
                    trace_features["tabm_dir_p_long"],
                    trace_features["tabm_dir_p_short"],
                ],
                dtype=np.float64,
            ),
            1e-9,
            1.0,
        )
        out = {
            **market,
            **trace_features,
            "primary_is_cash": 1.0 if int(primary.side) == 0 else 0.0,
            "primary_active_roll_12": float(active[-12:].mean()),
            "primary_active_roll_48": float(active[-48:].mean()),
            "primary_cash_streak": float(np.tanh(float(self.primary_cash_streak) / 144.0)),
            "cash_ret_sum_12": float(ret1.rolling(12, min_periods=1).sum().iloc[-1]),
            "cash_ret_sum_48": float(ret1.rolling(48, min_periods=1).sum().iloc[-1]),
            "cash_ret_vol_12": float(ret1.rolling(12, min_periods=2).std().fillna(0.0).iloc[-1]),
            "cash_ret_vol_48": float(ret1.rolling(48, min_periods=2).std().fillna(0.0).iloc[-1]),
            "cash_range_ratio_12_48": float(cash_range_ratio.iloc[-1]),
            "tabm_dir_entropy": float((-(probs * np.log(probs)).sum() / np.log(3.0))),
            "tabm_long_short_gap": float(trace_features["tabm_dir_p_long"] - trace_features["tabm_dir_p_short"]),
            "tabm_abs_side_gap": float(abs(trace_features["tabm_dir_p_long"] - trace_features["tabm_dir_p_short"])),
            "tabm_quality_side_gap": float(trace_features["tabm_quality_p_long"] - trace_features["tabm_quality_p_short"]),
            "tabm_quality_abs_gap": float(abs(trace_features["tabm_quality_p_long"] - trace_features["tabm_quality_p_short"])),
            "time_since_primary_exit": float(np.tanh(float(self.primary_cash_streak) / 144.0)),
            "last_primary_active_len": float(np.tanh(float(self.last_primary_active_len) / 288.0)),
            "last_primary_side": float(self.last_primary_side),
        }
        missing = [c for c in self.feature_cols if c not in out]
        if missing:
            raise RuntimeError(f"Omega1.2.3 missing sleeve features: {missing}")
        bad = [c for c in self.feature_cols if not np.isfinite(float(out[c]))]
        if bad:
            raise RuntimeError(f"Omega1.2.3 non-finite sleeve features: {bad}")
        return out

    def _support_gate(self, features: dict[str, float]) -> dict[str, Any]:
        if self.model_id not in NUMERIC_FULL_RETRAIN_MODEL_IDS:
            return {"enabled": False, "pass": True}
        profile = self.support_profile
        low = dict(profile.get("low") or {})
        high = dict(profile.get("high") or {})
        median = dict(profile.get("median") or {})
        iqr = dict(profile.get("iqr") or {})
        missing = [c for c in self.feature_cols if c not in low or c not in high or c not in median or c not in iqr]
        if missing:
            raise RuntimeError(f"Omega1.2.8b support_profile missing features: {missing[:20]}")
        in_support = 0
        robust_abs_z: list[float] = []
        out_features: list[str] = []
        for col in self.feature_cols:
            val = float(features[col])
            lo = float(low[col])
            hi = float(high[col])
            med = float(median[col])
            scale = max(float(iqr[col]), 1.0e-8)
            rz = abs((val - med) / scale)
            robust_abs_z.append(float(rz))
            if lo <= val <= hi:
                in_support += 1
            else:
                out_features.append(str(col))
        support_fraction = float(in_support / max(len(self.feature_cols), 1))
        max_robust_abs_z = float(max(robust_abs_z) if robust_abs_z else 0.0)
        min_fraction = float(profile.get("min_fraction_in_support", 0.92))
        max_allowed_z = float(profile.get("max_robust_abs_z", 8.0))
        passed = bool(support_fraction >= min_fraction and max_robust_abs_z <= max_allowed_z)
        return {
            "enabled": True,
            "pass": passed,
            "support_fraction": support_fraction,
            "min_fraction_in_support": min_fraction,
            "max_robust_abs_z": max_robust_abs_z,
            "max_allowed_robust_abs_z": max_allowed_z,
            "out_of_support_count": int(len(out_features)),
            "out_of_support_features": out_features[:12],
        }

    def _risk_for_side(self, arr: np.ndarray, side: int) -> tuple[dict[str, float], dict[str, Any]]:
        if not self.dynamic_risk:
            risk = dict(self.risk)
            return risk, {"enabled": False}

        prefix = "long" if int(side) > 0 else "short"
        risk = dict(self.risk)
        trace: dict[str, Any] = {"enabled": True}
        for suffix, fallback_key in (
            ("take_profit", "take_profit"),
            ("stop_loss", "stop_loss"),
            ("notional", "notional"),
            ("leverage", "leverage"),
        ):
            key = f"{prefix}_{suffix}"
            value = float(self.risk_models[key].predict(arr)[0])
            bounds = dict(self.risk_label_bounds[key])
            lo = float(bounds["min"])
            hi = float(bounds["max"])
            clipped = float(np.clip(value, lo, hi))
            risk[fallback_key] = clipped
            trace[key] = {"raw": value, "clipped": clipped, "min": lo, "max": hi}
        risk["notional_exposure"] = float(risk["notional"])
        risk["position_fraction"] = float(risk["notional"]) / max(float(risk["leverage"]), 1.0e-8)
        return risk, trace

    def decide_latest(self, frame: pd.DataFrame, primary: Omega121Decision) -> Omega123SleeveDecision:
        features = self._history_features(frame.reset_index(drop=True), primary)
        support_gate = self._support_gate(features)
        x = pd.DataFrame([{c: float(features[c]) for c in self.feature_cols}], columns=self.feature_cols)
        arr = x.to_numpy(dtype=np.float64)
        raw_long_ev = float(self.long_model.predict(arr)[0])
        raw_short_ev = float(self.short_model.predict(arr)[0])
        long_ev = raw_long_ev - float(self.long_ev_offset)
        short_ev = raw_short_ev - float(self.short_ev_offset)
        utility_filter = {}
        if self.utility_long_model is not None and self.utility_short_model is not None:
            raw_long_utility = float(self.utility_long_model.predict(arr)[0])
            raw_short_utility = float(self.utility_short_model.predict(arr)[0])
            long_utility = raw_long_utility - float(self.long_utility_offset)
            short_utility = raw_short_utility - float(self.short_utility_offset)
            utility_filter = {
                "utility_min": float(self.utility_min),
                "margin_min": float(self.margin_min),
                "raw_long_utility": raw_long_utility,
                "raw_short_utility": raw_short_utility,
                "long_utility": long_utility,
                "short_utility": short_utility,
                "long_utility_offset": float(self.long_utility_offset),
                "short_utility_offset": float(self.short_utility_offset),
            }
        else:
            long_utility = long_ev
            short_utility = short_ev

        if max(long_ev, short_ev) <= self.ev_min:
            action = ACTION_CASH
            side = 0
        elif long_ev >= short_ev:
            action = ACTION_LONG
            side = 1
        else:
            action = ACTION_SHORT
            side = -1
        if side > 0 and not (long_utility > self.utility_min and (long_utility - short_utility) >= self.margin_min):
            action = ACTION_CASH
            side = 0
        elif side < 0 and not (short_utility > self.utility_min and (short_utility - long_utility) >= self.margin_min):
            action = ACTION_CASH
            side = 0
        if bool(self.conservative_gate.get("block_if_out_of_support", False)) and not bool(support_gate.get("pass", True)):
            action = ACTION_CASH
            side = 0
        confidence = float(np.clip((max(long_ev, short_ev) - self.ev_min) / 0.02, 0.0, 1.0))
        if not side:
            confidence = 0.0
        risk, dynamic_risk_trace = self._risk_for_side(arr, side) if side else (dict(self.risk), {"enabled": self.dynamic_risk})
        leverage = float(risk["leverage"]) if side else 1.0
        notional_exposure = float(risk.get("notional_exposure", risk["notional"])) if side else 0.0
        position_fraction = float(risk.get("position_fraction", notional_exposure / max(leverage, 1e-8))) if side else 0.0
        trace = {
            "model_id": self.model_id,
            "base_model_id": OMEGA128B_PARENT_MODEL_ID if self.model_id in NUMERIC_FULL_RETRAIN_MODEL_IDS else BASE_MODEL_ID,
            "model_sleeve": OMEGA123_SLEEVE,
            "bundle_path": self.bundle_path,
            "ev_min": float(self.ev_min),
            "long_ev": long_ev,
            "short_ev": short_ev,
            "raw_long_ev": raw_long_ev,
            "raw_short_ev": raw_short_ev,
            "long_ev_offset": float(self.long_ev_offset),
            "short_ev_offset": float(self.short_ev_offset),
            "feature_count": int(len(self.feature_cols)),
            "primary_cash_streak": int(self.primary_cash_streak),
            "last_primary_active_len": int(self.last_primary_active_len),
            "last_primary_side": int(self.last_primary_side),
            "primary_trace": dict(primary.trace or {}),
            "support_gate": dict(support_gate),
            "conservative_gate": dict(self.conservative_gate),
            "dynamic_risk": dynamic_risk_trace,
        }
        trace.update(utility_filter)
        return Omega123SleeveDecision(
            action=int(action),
            side=int(side),
            notional_exposure=notional_exposure,
            leverage=leverage,
            position_fraction=position_fraction,
            take_profit=float(risk["take_profit"]) if side else 0.0,
            stop_loss=float(risk["stop_loss"]) if side else 0.0,
            max_hold_bars=int(risk["max_hold_bars"]) if side else 0,
            confidence=confidence,
            long_ev=long_ev,
            short_ev=short_ev,
            trace=trace,
        )
