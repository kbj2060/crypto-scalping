from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


MODEL_ID = "clean_base_causal_sleeve_conformal_veto_v1_5"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _latest(frame: pd.DataFrame, name: str, default: float = 0.0) -> float:
    if frame is None or frame.empty or name not in frame.columns:
        return default
    return _safe_float(frame[name].iloc[-1], default)


def _decision_value(decision: Any, name: str, default: Any = 0.0) -> Any:
    if isinstance(decision, Mapping):
        return decision.get(name, default)
    if hasattr(decision, "get"):
        try:
            return decision.get(name, default)
        except Exception:
            pass
    return getattr(decision, name, default)


@dataclass(frozen=True)
class ConformalSleeveDecision:
    action: str
    core_notional: float
    sleeve_notional: float
    total_notional: float
    sleeve_fraction: float
    sleeve_exit_bars: int
    features: dict[str, float]
    predictions: dict[str, float]
    reason: str

    @property
    def active_sleeve(self) -> bool:
        return self.sleeve_notional > 0.0 and self.action.startswith("ADD_SAME_SIDE")


class ConformalSleeveV15Adapter:
    """Live adapter for the V1.5 clean-base conformal sleeve artifact."""

    def __init__(self, artifact: Mapping[str, Any], report: Mapping[str, Any] | None = None):
        self.artifact = dict(artifact)
        self.report = dict(report or {})
        self.model_id = str(self.artifact.get("model_id") or MODEL_ID)
        self.sleeve_model = self.artifact["sleeve_model"]
        self.conformal_model = self.artifact["conformal_model"]
        self.sleeve_features = list(self.artifact["sleeve_features"])
        self.conformal_features = list(self.artifact["conformal_features"])
        self.selected_config = dict(self.artifact.get("selected_config") or {})
        self.selected_residual_q = _safe_float(self.artifact.get("selected_residual_q"), 0.0)

        if self.model_id != MODEL_ID:
            raise ValueError(f"unexpected V1.5 model_id: {self.model_id}")
        if "same" not in self.sleeve_model:
            raise ValueError("V1.5 sleeve artifact is missing the same-side model")
        if "full" not in self.conformal_model or "adverse" not in self.conformal_model:
            raise ValueError("V1.5 conformal artifact is missing full/adverse models")

    @classmethod
    def load(cls, model_path: str | Path, report_path: str | Path | None = None) -> "ConformalSleeveV15Adapter":
        artifact = joblib.load(model_path)
        report: dict[str, Any] = {}
        if report_path:
            path = Path(report_path)
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
        return cls(artifact=artifact, report=report)

    def build_features(
        self,
        frame: pd.DataFrame,
        decision: Any,
        account_context: Mapping[str, Any] | None,
        core_notional: float,
    ) -> dict[str, float]:
        account_context = account_context or {}
        side = int(_decision_value(decision, "side", 0) or 0)
        leverage = _safe_float(_decision_value(decision, "leverage", 1.0), 1.0)
        quality = _safe_float(
            _decision_value(decision, "quality", _decision_value(decision, "quality_score", 0.0)),
            0.0,
        )
        confidence = _safe_float(_decision_value(decision, "confidence", 0.0), 0.0)

        return {
            "side": float(side),
            "quality": quality,
            "confidence": confidence,
            "core_notional": _safe_float(core_notional, 0.0),
            "leverage": leverage,
            "account_dd": _safe_float(account_context.get("account_dd"), 0.0),
            "daily_dd": _safe_float(
                account_context.get("daily_dd", account_context.get("daily_dd_proxy")),
                0.0,
            ),
            "loss_streak": _safe_float(account_context.get("loss_streak"), 0.0),
            "funding_abs": abs(_latest(frame, "funding_abs", _latest(frame, "funding_rate", 0.0))),
            "funding_pressure": _latest(frame, "funding_pressure", 0.0),
            "liquidity_vacuum": _latest(frame, "liquidity_vacuum", 0.0),
            "amihud_illiquidity_z": _latest(frame, "amihud_illiquidity_z", 0.0),
            "evt_tail_flag": _latest(frame, "evt_tail_flag", 0.0),
            "ai_adverse_risk": _latest(frame, "ai_adverse_risk", 0.0),
        }

    def _frame_for(self, features: Mapping[str, float], columns: list[str]) -> pd.DataFrame:
        row = {name: _safe_float(features.get(name), 0.0) for name in columns}
        return pd.DataFrame([row], columns=columns).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _predict(self, features: Mapping[str, float]) -> dict[str, float]:
        sleeve_x = self._frame_for(features, self.sleeve_features)
        conformal_x = self._frame_for(features, self.conformal_features)
        same_pred = _safe_float(self.sleeve_model["same"].predict(sleeve_x)[0], 0.0)
        full_pred = _safe_float(self.conformal_model["full"].predict(conformal_x)[0], 0.0)
        adverse_pred = _safe_float(self.conformal_model["adverse"].predict(conformal_x)[0], 0.0)
        pred_lcb = full_pred - self.selected_residual_q
        return {
            "same_pred": same_pred,
            "pred_full": full_pred,
            "pred_adverse": adverse_pred,
            "pred_lcb": pred_lcb,
            "selected_residual_q": self.selected_residual_q,
        }

    def decide(
        self,
        frame: pd.DataFrame,
        decision: Any,
        account_context: Mapping[str, Any] | None,
        core_notional: float,
        *,
        max_total_notional: float = 3.6,
    ) -> ConformalSleeveDecision:
        core = max(0.0, _safe_float(core_notional, 0.0))
        cap = max(0.0, min(_safe_float(max_total_notional, 3.6), 3.6))
        features = self.build_features(frame, decision, account_context, core)
        preds = self._predict(features)
        cfg = self.selected_config

        same_threshold = _safe_float(cfg.get("same_threshold"), 0.0)
        max_sleeve_frac = max(0.0, _safe_float(cfg.get("max_sleeve_frac"), 0.0))
        max_sleeve_bars = max(1, int(_safe_float(cfg.get("max_sleeve_bars"), 6.0)))
        account_dd_disable = _safe_float(cfg.get("account_dd_disable"), 1.0)
        daily_dd_disable = _safe_float(cfg.get("daily_dd_disable"), 1.0)
        lcb_veto_threshold = _safe_float(cfg.get("lcb_veto_threshold"), -1.0)
        adverse_veto_cut = _safe_float(cfg.get("adverse_veto_cut"), 1.0)

        stress = (
            features["evt_tail_flag"] > 0.0
            or abs(features["liquidity_vacuum"]) > 1.0
            or abs(features["funding_pressure"]) > 0.12
            or abs(features["ai_adverse_risk"]) > 0.75
        )

        action = "NO_SLEEVE"
        reason = "below_same_side_threshold"
        sleeve_fraction = 0.0
        sleeve_notional = 0.0

        if features["account_dd"] >= account_dd_disable:
            reason = "account_dd_disable"
        elif features["daily_dd"] >= daily_dd_disable:
            reason = "daily_dd_disable"
        elif stress:
            reason = "stress_filter"
        elif preds["same_pred"] >= same_threshold and core > 0.0:
            sleeve_fraction = min(max_sleeve_frac, 0.25)
            sleeve_notional = core * sleeve_fraction
            action = "ADD_SAME_SIDE_25" if sleeve_fraction >= 0.20 else "ADD_SAME_SIDE_15"
            reason = "predicted_same_edge"

        if action.startswith("ADD_SAME_SIDE") and (
            preds["pred_lcb"] <= lcb_veto_threshold or preds["pred_adverse"] >= adverse_veto_cut
        ):
            action = "CONFORMAL_VETO"
            reason = "conformal_veto"
            sleeve_fraction = 0.0
            sleeve_notional = 0.0

        total = core + sleeve_notional
        if sleeve_notional > 0.0 and total > cap:
            action = "NO_SLEEVE"
            reason = "gross_cap"
            sleeve_fraction = 0.0
            sleeve_notional = 0.0
            total = core

        return ConformalSleeveDecision(
            action=action,
            core_notional=core,
            sleeve_notional=sleeve_notional,
            total_notional=total,
            sleeve_fraction=sleeve_fraction,
            sleeve_exit_bars=max_sleeve_bars,
            features=dict(features),
            predictions=preds,
            reason=reason,
        )
