from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _decision_value(dec: Any, key: str, default: float) -> float:
    try:
        if isinstance(dec, pd.Series):
            return float(dec.get(key, default))
        return float(getattr(dec, key, default))
    except Exception:
        return float(default)


def _positive_probability(model: Any, x: pd.DataFrame) -> float:
    try:
        classes = list(getattr(model[-1], "classes_", []))
        if 1 not in classes:
            return 0.0
        return float(model.predict_proba(x)[0][classes.index(1)])
    except Exception:
        try:
            classes = list(getattr(model, "classes_", []))
            if 1 not in classes:
                return 0.0
            return float(model.predict_proba(x)[0][classes.index(1)])
        except Exception:
            return 0.0


class JackpotRunnerV21_2Adapter:
    model_version = "V21.2"

    def __init__(self, *, model_path: str | Path, report_path: str | Path, audit_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.report_path = str(report_path)
        self.audit_path = str(audit_path)
        self.payload = joblib.load(self.model_path)
        with open(self.report_path, "r", encoding="utf-8") as f:
            self.report = json.load(f)
        with open(self.audit_path, "r", encoding="utf-8") as f:
            self.audit = json.load(f)
        if str(self.audit.get("status", "")).lower() != "pass":
            raise ValueError(f"v21_2_audit_not_pass:{self.audit.get('blocking', [])}")
        self.model_id = str(self.payload.get("model_id") or self.report.get("model_id") or "hf_v13_jackpot_runner_v21_2_20260511")
        self.selected_config = dict(self.payload.get("selected_config") or self.report.get("selected_config") or {})
        self.runner = dict(self.payload.get("cost_runner") or {})
        if not self.runner:
            raise ValueError("v21_2_cost_runner_missing")
        self.base_model_path = str(self.payload.get("base_model") or self.report.get("base_model") or "")
        self.feature_cols = list(self.runner.get("feature_cols") or [])
        if not self.feature_cols:
            raise ValueError("v21_2_feature_cols_missing")

    @classmethod
    def load(cls, model_path: str | Path, report_path: str | Path, audit_path: str | Path) -> "JackpotRunnerV21_2Adapter":
        return cls(model_path=model_path, report_path=report_path, audit_path=audit_path)

    def max_entry_notional(self) -> float:
        return float(self.selected_config.get("max_entry_notional", 2.75) or 2.75)

    def max_total_mult(self) -> float:
        return float(self.selected_config.get("max_total_mult", 1.35) or 1.35)

    def add_fraction(self) -> float:
        return float(self.selected_config.get("full_add_frac", 0.20) or 0.20)

    def min_unrealized(self) -> float:
        return float(self.selected_config.get("min_unrealized", 0.004) or 0.004)

    def min_bars_since_entry(self) -> int:
        return int(self.selected_config.get("min_bars_since_entry", 3) or 3)

    def _feature_frame(
        self,
        frame: pd.DataFrame,
        dec: Any,
        state: dict[str, float],
        *,
        parent_bundle: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        idx = int(len(frame) - 1)
        parent_cols = list(dict(parent_bundle or {}).get("feature_cols") or [])
        if parent_cols:
            row = frame.iloc[[idx]].reindex(columns=parent_cols).replace([np.inf, -np.inf], np.nan).copy()
            if row.isna().all(axis=None):
                row = frame.iloc[[idx]].select_dtypes(include=[np.number]).copy()
        else:
            numeric = frame.iloc[[idx]].select_dtypes(include=[np.number]).copy()
            row = numeric.reindex(columns=self.feature_cols)
        extra = {
            "parent_side": _decision_value(dec, "side", state.get("side", 0.0)),
            "parent_notional": float(state["parent_notional"]),
            "current_notional": float(state["notional"]),
            "bars_since_entry": float(state["bars_since_entry"]),
            "unrealized_pct": float(state["unrealized"]),
            "mfe_so_far": float(state["mfe"]),
            "mae_so_far": float(state["mae"]),
            "drawdown_abs": float(state["drawdown_abs"]),
            "parent_take_profit": float(state["take_profit"]),
            "parent_stop_loss": float(state["stop_loss"]),
            "parent_max_hold_bars": float(state["max_hold"]),
            "parent_confidence": _decision_value(dec, "confidence", 0.0),
            "parent_quality_score": _decision_value(dec, "quality_score", 0.0),
        }
        for key, value in extra.items():
            row[key] = float(value)
        return row.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def scores(
        self,
        frame: pd.DataFrame,
        dec: Any,
        state: dict[str, float],
        *,
        parent_bundle: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        x = self._feature_frame(frame, dec, state, parent_bundle=parent_bundle).reindex(columns=self.feature_cols)
        reg = self.runner["regressor"]
        q90 = self.runner["q90_regressor"]
        jackpot = self.runner["jackpot_classifier"]
        bad = self.runner["bad_classifier"]
        cost3 = self.runner["cost3_classifier"]
        out = {
            "edge": float(reg.predict(x)[0]),
            "q90": float(q90.predict(x)[0]),
            "p_jackpot": _positive_probability(jackpot, x),
            "p_bad_addon": _positive_probability(bad, x),
            "p_cost3_survive": _positive_probability(cost3, x),
        }
        return out

    def add_on_decision(
        self,
        frame: pd.DataFrame,
        dec: Any,
        *,
        side: int,
        parent_notional: float,
        current_notional: float,
        bars_since_entry: int,
        unrealized: float,
        mfe: float,
        mae: float,
        drawdown_abs: float,
        take_profit: float,
        stop_loss: float,
        max_hold: int,
        router_cap: float,
        parent_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.selected_config
        meta: dict[str, Any] = {
            "enabled": True,
            "applied": False,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model": self.model_path,
            "report": self.report_path,
            "audit": self.audit_path,
            "selected_config": dict(cfg),
            "parent_notional": float(parent_notional),
            "current_notional": float(current_notional),
            "bars_since_entry": int(bars_since_entry),
            "unrealized": float(unrealized),
        }
        if int(side) == 0 or parent_notional <= 1e-12 or current_notional <= 1e-12:
            meta["reason"] = "no_active_parent_position"
            return meta
        if float(unrealized) < self.min_unrealized() or int(bars_since_entry) < self.min_bars_since_entry():
            meta["reason"] = "jackpot_min_state_not_met"
            return meta
        if float(drawdown_abs) > float(cfg.get("dd_block", 0.30) or 0.30):
            meta["reason"] = "jackpot_dd_block"
            return meta
        state = {
            "side": float(side),
            "parent_notional": float(parent_notional),
            "notional": float(current_notional),
            "bars_since_entry": float(bars_since_entry),
            "unrealized": float(unrealized),
            "mfe": float(mfe),
            "mae": float(mae),
            "drawdown_abs": float(drawdown_abs),
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "max_hold": float(max_hold),
        }
        scores = self.scores(frame, dec, state, parent_bundle=parent_bundle)
        cap = float(min(float(parent_notional) * self.max_total_mult(), float(router_cap)))
        delta = float(max(0.0, min(float(parent_notional) * self.add_fraction(), cap - float(current_notional))))
        pass_gate = bool(
            scores["p_jackpot"] >= float(cfg.get("jackpot_p", 0.20) or 0.20)
            and scores["q90"] >= float(cfg.get("jackpot_q90", 0.015) or 0.015)
            and scores["p_bad_addon"] <= float(cfg.get("bad_cap", 0.50) or 0.50)
            and scores["p_cost3_survive"] >= 0.40
            and delta > 1e-12
        )
        meta.update(
            {
                **scores,
                "cap": float(cap),
                "delta_notional": float(delta),
                "output_notional": float(current_notional + delta) if pass_gate else float(current_notional),
                "applied": bool(pass_gate),
                "reason": "v21_2_jackpot_add" if pass_gate else "v21_2_jackpot_reject",
            }
        )
        return meta
