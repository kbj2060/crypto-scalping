"""Runtime-native Omega4.6.2 source-parent adapter for Omega5.

This adapter intentionally does not replay validation/OOS ledgers.  It rebuilds
the promoted source parent from the live feature frame:

1. run the two promoted TabM parent bundles (h48qual q050, zig075 q075),
2. route h48qual before zig075,
3. apply the cap220 short RSI gate and sidecar sizing,
4. apply the v5 fine exposure and path-causal loss governor state.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_eval_omega1_2_tabm_3head_20260603 as omega_tabm  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402


OMEGA462_SOURCE_PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
OMEGA462_SOURCE_PARENT_VERSION = "Omega4.6.2-source-parent-live-native-20260702"
OMEGA462_PARENT_BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
OMEGA462_REFERENCE_POLICY_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"

DEFAULT_SOURCE_PARENT_REPORT = (
    ROOT / "tmp/causal_regen_20260516" / OMEGA462_SOURCE_PARENT_MODEL_ID / "report.json"
)
DEFAULT_CAP220_RUNTIME_CONTRACT = (
    ROOT / "tmp/causal_regen_20260516" / OMEGA462_PARENT_BASE_MODEL_ID / "runtime_contract.json"
)
DEFAULT_REFERENCE_POLICY_REPORT = (
    ROOT / "tmp/causal_regen_20260516" / OMEGA462_REFERENCE_POLICY_MODEL_ID / "report.json"
)
DEFAULT_CURRENT_REGIME_PATH = (
    ROOT
    / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/"
    / "regime3_current_sensitive_hmm_wide24_2024.joblib"
)
DEFAULT_CMAMBA_PATH = (
    ROOT
    / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531/"
    / "regime3_cryptomamba_pred_h6_nocurrent_20260531_2024.pt"
)
DEFAULT_RISK_PATH = (
    ROOT
    / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/"
    / "regime3_stability_risk_h6.joblib"
)

SOURCE_COMPONENT_ORDER = ("h48qual", "zig075")
COMPONENT_QUALITY_THRESHOLDS = {"h48qual": 0.50, "zig075": 0.75}
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
CMAMBA_PREFIX = "regime3_cmamba_h6_sidecar_"
RISK_COLS = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]
ROUTE_COLS = [
    f"{CURRENT_PREFIX}bull_prob",
    f"{CURRENT_PREFIX}bear_prob",
    f"{CURRENT_PREFIX}chop_prob",
]
FORBIDDEN_FEATURE_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
FORBIDDEN_FEATURE_NAMES = {"tp_sl_action_score"}
BASE_NOTIONAL = 0.45
BASE_LEVERAGE = 2.0
BASE_TAKE_PROFIT = 0.026
BASE_STOP_LOSS = 0.014
ATR_WINDOW = 192
CAP220_SHORT_RSI_THRESHOLD = 56.656189
CAP220_LONG_FACTOR = 0.90
CAP220_SHORT_FACTOR = 1.25
CAP220_NOTIONAL_CAP = 2.20
V5_LONG_FACTOR = 1.30
V5_SHORT_FACTOR = 1.955
V5_CAP_NOTIONAL = 4.106
V5_LEVERAGE_CAP = 5.0
V5_MAX_MARGIN_FRACTION = 1.0
V5_LOSS1_SCALE = 0.50
V5_LOSS2_SCALE = 1.00
V5_LOSS_WINDOW_HOURS = 12.0
V5_MAX_HOLD_BARS = int(90 * 12)
EPS = 1.0e-12


@dataclass(frozen=True)
class Omega121Decision:
    action: int
    side: int
    notional_exposure: float
    leverage: float
    position_fraction: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    cooldown_bars: int
    quality_score: float
    confidence: float
    router_expert: str
    trace: dict[str, Any]


@dataclass(frozen=True)
class Omega462SourceParentConfig:
    source_parent_report_path: str | Path = DEFAULT_SOURCE_PARENT_REPORT
    cap220_runtime_contract_path: str | Path = DEFAULT_CAP220_RUNTIME_CONTRACT
    reference_policy_report_path: str | Path = DEFAULT_REFERENCE_POLICY_REPORT
    current_regime_path: str | Path = DEFAULT_CURRENT_REGIME_PATH
    cmamba_path: str | Path = DEFAULT_CMAMBA_PATH
    risk_path: str | Path = DEFAULT_RISK_PATH
    device: Any = "cuda"


@dataclass
class _Component:
    alias: str
    report_path: Path
    bundle_path: Path
    sidecar_path: Path
    quality_threshold: float
    bundle: dict[str, Any]
    models: dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]]
    sidecar: dict[str, Any]


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    proba = state_prob @ state_class
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1.0e-300, None)
    return proba


class Regime3CurrentLiveFeatures:
    def __init__(self, *, current_path: str | Path) -> None:
        self.current_payload = joblib.load(Path(current_path))

    @staticmethod
    def _reject_forbidden(cols: list[str], tag: str) -> None:
        bad = [
            c
            for c in cols
            if c in FORBIDDEN_FEATURE_NAMES
            or any(str(c).startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
        ]
        if bad:
            raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")

    @staticmethod
    def _require_finite_frame(raw: pd.DataFrame, tag: str) -> None:
        bad = [str(c) for c in raw.columns if bool(raw[c].isna().any())]
        if bad:
            raise RuntimeError(f"{tag} non-finite model inputs: {bad[:40]}")

    @staticmethod
    def _impute_training_medians(raw: pd.DataFrame, payload: dict[str, Any], tag: str) -> pd.DataFrame:
        medians = payload.get("feature_medians")
        if medians is None:
            raise RuntimeError(f"{tag} payload missing feature_medians")
        fill = pd.Series({str(k): float(v) for k, v in dict(medians).items()})
        missing = [str(c) for c in raw.columns if str(c) not in fill.index]
        if missing:
            raise RuntimeError(f"{tag} feature_medians missing columns: {missing[:40]}")
        return raw.fillna(fill.reindex(raw.columns)).fillna(0.0)

    @staticmethod
    def _finite_latest(frame: pd.DataFrame, cols: list[str], tag: str) -> None:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{tag} missing columns: {missing[:40]}")
        if not len(frame):
            raise RuntimeError(f"{tag} empty frame")
        latest = frame.iloc[-1]
        bad = []
        for col in cols:
            try:
                val = float(latest[col])
            except Exception:
                bad.append(col)
                continue
            if not np.isfinite(val):
                bad.append(col)
        if bad:
            raise RuntimeError(f"{tag} non-finite latest columns: {bad[:40]}")

    @staticmethod
    def _with_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = _with_raw_state12(frame.copy())
        for col in cols:
            if col not in out.columns:
                raise RuntimeError(f"missing current HMM feature column: {col}")
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _append_current(self, frame: pd.DataFrame) -> pd.DataFrame:
        payload = self.current_payload
        cols = list(payload["feature_cols"])
        self._reject_forbidden(cols, "Regime3 current")
        work = self._with_features(frame, cols)
        raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = self._impute_training_medians(raw, payload, "Regime3 current")
        self._require_finite_frame(raw, "Regime3 current")
        xz = payload["scaler"].transform(raw)
        state = payload["model"].filter_proba(xz)
        proba = _class_proba(state, np.asarray(payload["state_class_matrix"], dtype=np.float64))
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)

        out = frame.copy()
        for i, name in enumerate(payload["classes"]):
            out[f"{CURRENT_PREFIX}{name}_prob"] = proba[:, i]
        sorted_p = np.sort(proba, axis=1)
        out[f"{CURRENT_PREFIX}confidence"] = proba.max(axis=1)
        out[f"{CURRENT_PREFIX}margin"] = sorted_p[:, -1] - sorted_p[:, -2]
        out[f"{CURRENT_PREFIX}entropy"] = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1) / np.log(3.0)
        return out

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._append_current(frame)
        self._finite_latest(
            out,
            ROUTE_COLS + [f"{CURRENT_PREFIX}confidence", f"{CURRENT_PREFIX}entropy", f"{CURRENT_PREFIX}margin"],
            "Regime3 current",
        )
        return out


def _read_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        raise RuntimeError(f"Omega4.6.2 source parent required artifact is missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"Omega4.6.2 non-finite {name}: {value!r}")
    return out


def _atr_pct(frame: pd.DataFrame, window: int) -> float:
    required = {"high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Omega4.6.2 ATR missing columns: {missing}")
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    out = atr / np.maximum(close, EPS)
    latest = float(out[-1])
    if not np.isfinite(latest):
        raise RuntimeError("Omega4.6.2 non-finite ATR percent")
    return latest


class Omega462SourceParentLiveAdapter:
    def __init__(self, config: Omega462SourceParentConfig | None = None) -> None:
        self.config = config or Omega462SourceParentConfig()
        self.source_parent_report = _read_json(self.config.source_parent_report_path)
        self.cap220_contract = _read_json(self.config.cap220_runtime_contract_path)
        self.reference_policy_report = _read_json(self.config.reference_policy_report_path)
        self._validate_contracts()
        if not torch.cuda.is_available() and str(self.config.device) == "cuda":
            raise RuntimeError("Omega4.6.2 source parent live adapter requires CUDA")
        self.device = torch.device(
            self.config.device if str(self.config.device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type != "cuda":
            raise RuntimeError("Omega4.6.2 source parent live path must run on CUDA")
        self.regime3 = Regime3CurrentLiveFeatures(current_path=self.config.current_regime_path)
        self.components = self._load_components()
        self._validate_current_only_runtime()
        self.loss_streak = 0
        self.last_loss_exit_ts: pd.Timestamp | None = None

    def _validate_current_only_runtime(self) -> None:
        bad: dict[str, list[str]] = {}
        for alias, component in self.components.items():
            cols: set[str] = set()
            for _, _, input_cols in component.models.values():
                cols.update(input_cols)
            found = sorted(
                c
                for c in cols
                if c in RISK_COLS or str(c).startswith("m7_") or str(c).startswith(CMAMBA_PREFIX)
            )
            if found:
                bad[alias] = found
        if bad:
            raise RuntimeError(
                "Omega4.6.2 current-only live parent cannot run because component inputs "
                f"require full live append columns: {bad}"
            )

    def _validate_contracts(self) -> None:
        if self.source_parent_report.get("model_id") != OMEGA462_SOURCE_PARENT_MODEL_ID:
            raise RuntimeError("Omega4.6.2 source parent report model_id mismatch")
        if self.source_parent_report.get("base_model_id") != OMEGA462_PARENT_BASE_MODEL_ID:
            raise RuntimeError("Omega4.6.2 source parent base_model_id mismatch")
        selected = self.source_parent_report.get("selected_variant")
        if not isinstance(selected, dict) or not bool(selected.get("validation_upgrade_gate_pass")):
            raise RuntimeError("Omega4.6.2 source parent selected variant did not pass validation gate")
        if self.cap220_contract.get("model_id") != OMEGA462_PARENT_BASE_MODEL_ID:
            raise RuntimeError("Omega4.6.2 cap220 runtime contract model_id mismatch")
        if self.reference_policy_report.get("model_id") != OMEGA462_REFERENCE_POLICY_MODEL_ID:
            raise RuntimeError("Omega4.6.2 reference policy report model_id mismatch")
        if self.reference_policy_report.get("parent_model_id") != OMEGA462_SOURCE_PARENT_MODEL_ID:
            raise RuntimeError("Omega4.6.2 reference policy parent_model_id mismatch")
        runtime_req = dict(self.cap220_contract.get("promotion_requirements_for_successors") or {})
        if bool(runtime_req.get("historical_trade_ledger_fallback_allowed", True)):
            raise RuntimeError("Omega4.6.2 cap220 contract unexpectedly allows historical ledger fallback")

    def _load_components(self) -> dict[str, _Component]:
        raw_components = self.cap220_contract.get("components")
        if not isinstance(raw_components, dict):
            raise RuntimeError("Omega4.6.2 cap220 runtime contract missing components")
        out: dict[str, _Component] = {}
        for alias in SOURCE_COMPONENT_ORDER:
            raw = raw_components.get(alias)
            if not isinstance(raw, dict):
                raise RuntimeError(f"Omega4.6.2 cap220 runtime missing component: {alias}")
            report_path = _resolve(raw["report"])
            report = _read_json(report_path)
            quality_threshold = _finite_float(raw.get("quality_threshold"), f"{alias}.quality_threshold")
            expected_threshold = COMPONENT_QUALITY_THRESHOLDS[alias]
            if abs(quality_threshold - expected_threshold) > 1.0e-12:
                raise RuntimeError(f"Omega4.6.2 {alias} quality threshold mismatch")
            risk_model = dict(report.get("risk_model") or {})
            pred_dir = _resolve(risk_model.get("precomputed_prediction_dir", ""))
            bundle_path = pred_dir / "true_3head_tabm_bundle.pt"
            if not bundle_path.exists():
                raise RuntimeError(f"Omega4.6.2 {alias} TabM bundle missing: {bundle_path}")
            sidecar_path = _resolve(dict(report.get("artifacts") or {}).get("risk_sidecar", report_path.parent / "risk_sidecar.pkl"))
            if not sidecar_path.exists():
                raise RuntimeError(f"Omega4.6.2 {alias} risk sidecar missing: {sidecar_path}")
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            models = self._load_tabm_models(bundle, alias)
            with sidecar_path.open("rb") as f:
                sidecar = pickle.load(f)
            self._validate_sidecar(sidecar, alias)
            out[alias] = _Component(
                alias=alias,
                report_path=report_path,
                bundle_path=bundle_path,
                sidecar_path=sidecar_path,
                quality_threshold=quality_threshold,
                bundle=bundle,
                models=models,
                sidecar=sidecar,
            )
        return out

    def _load_tabm_models(
        self,
        bundle: dict[str, Any],
        alias: str,
    ) -> dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]]:
        models: dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]] = {}
        for expert, payload_raw in dict(bundle["models"]).items():
            payload = dict(payload_raw)
            input_cols = list(payload["input_columns"])
            Regime3CurrentLiveFeatures._reject_forbidden(input_cols, f"Omega4.6.2 {alias} {expert} TabM")
            cfg = omega_tabm.ThreeHeadConfig(**dict(payload["config"]))
            model = omega_tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            models[str(expert)] = (model, dict(payload["scaler"]), input_cols)
        missing = sorted(set(("bull", "bear", "chop")) - set(models))
        if missing:
            raise RuntimeError(f"Omega4.6.2 {alias} bundle missing experts: {missing}")
        return models

    @staticmethod
    def _validate_sidecar(sidecar: dict[str, Any], alias: str) -> None:
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar payload is not a dict")
        if sidecar.get("risk_feature_mode") != "parent_outputs":
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar risk_feature_mode mismatch")
        if not bool(sidecar.get("side_split_model")):
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar must be side-split")
        if not bool(sidecar.get("dynamic_leverage")):
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar must use dynamic leverage")
        model = sidecar.get("model")
        if not isinstance(model, dict) or not {-1, 1}.issubset(set(model)):
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar missing side models")
        columns = sidecar.get("feature_columns")
        if not isinstance(columns, list) or not columns:
            raise RuntimeError(f"Omega4.6.2 {alias} sidecar missing feature columns")

    @staticmethod
    def _latest_timestamp(frame: pd.DataFrame) -> pd.Timestamp:
        if not len(frame):
            raise RuntimeError("Omega4.6.2 source parent received empty frame")
        row = frame.iloc[-1]
        for col in ("timestamp", "ts", "datetime", "date", "open_time"):
            if col in frame.columns:
                ts = pd.Timestamp(row[col])
                if pd.isna(ts):
                    raise RuntimeError(f"Omega4.6.2 invalid latest timestamp column: {col}")
                return ts.tz_localize(None) if ts.tzinfo is not None else ts
        if isinstance(frame.index, pd.DatetimeIndex):
            ts = pd.Timestamp(frame.index[-1])
            if pd.isna(ts):
                raise RuntimeError("Omega4.6.2 invalid latest DatetimeIndex timestamp")
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        raise RuntimeError("Omega4.6.2 source parent requires a timestamp column")

    @staticmethod
    def _cash(*, reason: str, trace: dict[str, Any]) -> Omega121Decision:
        out = dict(trace)
        out["omega462_reason"] = reason
        return Omega121Decision(
            action=0,
            side=0,
            notional_exposure=0.0,
            leverage=1.0,
            position_fraction=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            max_hold_bars=0,
            cooldown_bars=0,
            quality_score=float(out.get("quality_score", 0.0) or 0.0),
            confidence=float(out.get("confidence", 0.0) or 0.0),
            router_expert=str(out.get("router_expert", "") or ""),
            trace=out,
        )

    @staticmethod
    def _route_expert(row: pd.Series) -> tuple[str, float, float]:
        probs = np.asarray([float(row[c]) for c in ROUTE_COLS], dtype=np.float64)
        if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            raise RuntimeError("Omega4.6.2 invalid Regime3 route probabilities")
        probs = probs / np.clip(probs.sum(), EPS, None)
        idx = int(np.argmax(probs))
        sorted_p = np.sort(probs)
        return ("bull", "bear", "chop")[idx], float(probs[idx]), float(sorted_p[-1] - sorted_p[-2])

    def _latest_input(self, frame: pd.DataFrame, input_cols: list[str], pos_cols: list[str]) -> pd.DataFrame:
        Regime3CurrentLiveFeatures._reject_forbidden(input_cols, "Omega4.6.2 TabM")
        row = frame.iloc[-1]
        data: dict[str, float] = {}
        missing: list[str] = []
        bad: list[str] = []
        for col in input_cols:
            if col in pos_cols:
                data[col] = 0.0
                continue
            if col not in frame.columns:
                missing.append(col)
                continue
            try:
                val = float(row[col])
            except Exception:
                bad.append(col)
                continue
            if not np.isfinite(val):
                bad.append(col)
                continue
            data[col] = val
        if missing:
            raise RuntimeError(f"Omega4.6.2 missing input features: {missing[:60]}")
        if bad:
            raise RuntimeError(f"Omega4.6.2 non-finite input features: {bad[:60]}")
        return pd.DataFrame([data], columns=input_cols)

    @staticmethod
    def _standardize(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
        cols = list(scaler["columns"])
        if list(x.columns) != cols:
            raise RuntimeError("Omega4.6.2 TabM feature column contract mismatch")
        arr = x.to_numpy(dtype=np.float32)
        z = (arr - scaler["mean"]) / scaler["std"]
        if not np.isfinite(z).all():
            raise RuntimeError("Omega4.6.2 standardized feature matrix has non-finite values")
        return z.astype(np.float32)

    @torch.no_grad()
    def _predict_component(self, component: _Component, frame: pd.DataFrame, atr_pct: float) -> dict[str, Any]:
        row = frame.iloc[-1]
        expert, route_conf, route_margin = self._route_expert(row)
        model, scaler, input_cols = component.models[expert]
        pos_cols = list(component.bundle["pos_cols"])
        x = self._latest_input(frame, input_cols, pos_cols)
        z = self._standardize(x, scaler)
        out = model(torch.from_numpy(z).to(self.device))
        direction = torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        quality = torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        dir_action = int(np.argmax(direction))
        quality_for_action = float(quality[dir_action] if dir_action > 0 else quality[0])
        final_action = dir_action if dir_action != 0 and quality_for_action >= component.quality_threshold else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)
        dec = {
            "action": int(final_action),
            "side": int(side),
            "quality_score": float(quality_for_action if side else 0.0),
            "confidence": float(np.max(direction)),
            "notional_exposure": float(BASE_NOTIONAL if side else 0.0),
            "leverage": float(BASE_LEVERAGE if side else 1.0),
            "position_fraction": float(BASE_NOTIONAL if side else 0.0),
            "take_profit": float(BASE_TAKE_PROFIT if side else 0.0),
            "stop_loss": float(BASE_STOP_LOSS if side else 0.0),
        }
        features = self._risk_features(
            component=component,
            expert=expert,
            route_conf=route_conf,
            route_margin=route_margin,
            direction=direction,
            quality=quality,
            dir_action=dir_action,
            quality_for_action=quality_for_action,
            final_action=final_action,
            dec=dec,
            atr_pct=atr_pct,
        )
        margin, leverage, sidecar_score = self._sidecar_sizing(component, features, dec)
        notional = margin * leverage
        return {
            "alias": component.alias,
            "expert": expert,
            "route_confidence": route_conf,
            "route_margin": route_margin,
            "direction": direction,
            "quality": quality,
            "dir_action": dir_action,
            "final_action": final_action,
            "side": side,
            "quality_for_action": quality_for_action,
            "confidence": float(np.max(direction)),
            "base_margin_fraction": margin,
            "base_leverage": leverage,
            "base_notional": notional,
            "sidecar_score": sidecar_score,
            "component_report": str(component.report_path),
            "component_bundle": str(component.bundle_path),
            "component_sidecar": str(component.sidecar_path),
        }

    def _risk_features(
        self,
        *,
        component: _Component,
        expert: str,
        route_conf: float,
        route_margin: float,
        direction: np.ndarray,
        quality: np.ndarray,
        dir_action: int,
        quality_for_action: float,
        final_action: int,
        dec: dict[str, float | int],
        atr_pct: float,
    ) -> pd.DataFrame:
        row = {
            "parent_router_confidence": float(route_conf),
            "parent_router_margin": float(route_margin),
            "parent_dir_p_cash": float(direction[0]),
            "parent_dir_p_long": float(direction[1]),
            "parent_dir_p_short": float(direction[2]),
            "parent_dir_confidence": float(np.max(direction)),
            "parent_dir_side_edge": float(abs(direction[1] - direction[2])),
            "parent_dir_trade_prob": float(direction[1] + direction[2]),
            "parent_dir_action": int(dir_action),
            "parent_quality_p_cash": float(quality[0]),
            "parent_quality_p_long": float(quality[1]),
            "parent_quality_p_short": float(quality[2]),
            "parent_quality_for_action": float(quality_for_action),
            "parent_quality_threshold": float(component.quality_threshold),
            "parent_final_action": int(final_action),
            "parent_router_expert_bear": 1.0 if expert == "bear" else 0.0,
            "parent_router_expert_bull": 1.0 if expert == "bull" else 0.0,
            "parent_router_expert_chop": 1.0 if expert == "chop" else 0.0,
            "decision_action": int(dec["action"]),
            "decision_side": int(dec["side"]),
            "decision_quality_score": float(dec["quality_score"]),
            "decision_confidence": float(dec["confidence"]),
            "decision_notional_exposure": float(dec["notional_exposure"]),
            "decision_leverage": float(dec["leverage"]),
            "decision_position_fraction": float(dec["position_fraction"]),
            "decision_take_profit": float(dec["take_profit"]),
            "decision_stop_loss": float(dec["stop_loss"]),
            "decision_rr": float(dec["take_profit"]) / max(abs(float(dec["stop_loss"])), 1.0e-8),
            "atr_pct_runtime": float(atr_pct),
        }
        cols = list(component.sidecar["feature_columns"])
        missing = sorted(set(cols) - set(row))
        if missing:
            raise RuntimeError(f"Omega4.6.2 {component.alias} sidecar feature builder missing: {missing}")
        return pd.DataFrame([{col: row[col] for col in cols}], columns=cols, dtype=np.float32)

    @staticmethod
    def _sidecar_sizing(component: _Component, features: pd.DataFrame, dec: dict[str, float | int]) -> tuple[float, float, float]:
        side = int(dec["side"])
        if side == 0:
            return 0.0, 0.0, 0.0
        model = component.sidecar["model"][side]
        score = float(model.predict(features)[0])
        mapping = dict(component.sidecar["selected_mapping"])
        q50 = float(component.sidecar["train_score_q50"])
        iqr = max(float(component.sidecar["train_score_iqr"]), 1.0e-8)
        z_margin = float(np.clip((score - q50) / iqr, -8.0, 8.0))
        unit_margin = 1.0 / (1.0 + np.exp(-float(mapping["temp"]) * z_margin))
        scale = float(mapping["min_scale"]) + (float(mapping["max_scale"]) - float(mapping["min_scale"])) * unit_margin
        base_margin = float(dec["notional_exposure"]) / max(float(dec["leverage"]), EPS)
        margin = float(np.clip(base_margin * scale, float(mapping["floor"]), float(mapping["cap"])))
        if side > 0:
            margin *= float(mapping.get("long_scale", 1.0))
        else:
            margin *= float(mapping.get("short_scale", 1.0))
        margin = float(np.clip(margin, float(mapping["floor"]), float(mapping["cap"])))

        z_lev = float(np.clip((score - q50) / iqr, -8.0, 8.0))
        unit_lev = 1.0 / (1.0 + np.exp(-float(mapping["leverage_temp"]) * z_lev))
        leverage = float(mapping["leverage_min"]) + (
            float(mapping["leverage_max"]) - float(mapping["leverage_min"])
        ) * unit_lev
        if side > 0:
            leverage *= float(mapping.get("long_leverage_scale", 1.0))
        else:
            leverage *= float(mapping.get("short_leverage_scale", 1.0))
        leverage = float(np.clip(leverage, float(mapping["leverage_floor"]), float(mapping["leverage_cap"])))
        return margin, leverage, score

    @staticmethod
    def _apply_cap220(candidate: dict[str, Any], latest: pd.Series) -> dict[str, Any]:
        out = dict(candidate)
        side = int(out["side"])
        if side == 0:
            return out
        rsi = _finite_float(latest.get("rsi"), "rsi")
        if side < 0 and rsi >= CAP220_SHORT_RSI_THRESHOLD:
            out["cap220_skipped"] = True
            out["notional"] = 0.0
            out["margin_fraction"] = 0.0
            out["leverage"] = 0.0
            return out
        factor = CAP220_LONG_FACTOR if side > 0 else CAP220_SHORT_FACTOR
        notional = min(float(out["base_notional"]) * factor, CAP220_NOTIONAL_CAP)
        leverage = float(out["base_leverage"])
        margin = notional / max(leverage, EPS)
        out.update(
            {
                "cap220_skipped": False,
                "cap220_factor": factor,
                "notional": float(notional),
                "margin_fraction": float(margin),
                "leverage": float(leverage),
            }
        )
        return out

    def _loss_governor_scale(self, now: pd.Timestamp) -> float:
        effective_loss_streak = int(self.loss_streak)
        if self.last_loss_exit_ts is not None:
            hours_from_loss = (now - self.last_loss_exit_ts).total_seconds() / 3600.0
            if hours_from_loss > V5_LOSS_WINDOW_HOURS:
                effective_loss_streak = 0
        if effective_loss_streak >= 2:
            return float(V5_LOSS2_SCALE)
        if effective_loss_streak == 1:
            return float(V5_LOSS1_SCALE)
        return 1.0

    def _apply_v5_exposure(self, candidate: dict[str, Any], now: pd.Timestamp) -> dict[str, Any]:
        out = dict(candidate)
        side = int(out["side"])
        notional = float(out.get("notional", 0.0) or 0.0)
        if side == 0 or notional <= EPS:
            out.update({"loss_governor_scale": 0.0, "source_parent_side_factor": 0.0})
            return out
        side_factor = V5_LONG_FACTOR if side > 0 else V5_SHORT_FACTOR
        pre_governor = min(notional * side_factor, V5_CAP_NOTIONAL, V5_LEVERAGE_CAP * V5_MAX_MARGIN_FRACTION)
        loss_scale = self._loss_governor_scale(now)
        final_notional = pre_governor * loss_scale
        leverage = V5_LEVERAGE_CAP
        margin = final_notional / max(leverage, EPS)
        if margin > V5_MAX_MARGIN_FRACTION + 1.0e-12:
            raise RuntimeError(f"Omega4.6.2 margin exceeds v5 cap: {margin}")
        out.update(
            {
                "source_parent_side_factor": float(side_factor),
                "pre_governor_notional": float(pre_governor),
                "loss_governor_scale": float(loss_scale),
                "notional": float(final_notional),
                "leverage": float(leverage),
                "margin_fraction": float(margin),
            }
        )
        return out

    def record_closed_trade(self, *, exit_timestamp: Any, net_per_notional: float) -> None:
        ts = pd.Timestamp(exit_timestamp)
        if pd.isna(ts):
            raise RuntimeError("Omega4.6.2 loss governor received invalid close timestamp")
        ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
        net = _finite_float(net_per_notional, "net_per_notional")
        if net < 0.0:
            self.loss_streak += 1
            self.last_loss_exit_ts = ts
        else:
            self.loss_streak = 0
            self.last_loss_exit_ts = None

    def decide_latest(self, frame: pd.DataFrame) -> Omega121Decision:
        now = self._latest_timestamp(frame)
        work = self.regime3.append(frame.copy().reset_index(drop=True))
        latest = work.iloc[-1]
        atr = _atr_pct(work, ATR_WINDOW)
        trace: dict[str, Any] = {
            "model_id": OMEGA462_SOURCE_PARENT_MODEL_ID,
            "model_version": OMEGA462_SOURCE_PARENT_VERSION,
            "base_model_id": OMEGA462_PARENT_BASE_MODEL_ID,
            "decision_timestamp": str(now),
            "notional_contract": "notional=margin_fraction*leverage",
            "ledger_replay_used": False,
            "source_policy_interval_adapter": False,
            "reference_policy_entry_event_adapter": False,
            "source_parent_live_native_adapter": True,
            "source_parent_predictive_artifact": "tabm_bundle+risk_sidecar_runtime_forward",
            "reference_policy_model_id": OMEGA462_REFERENCE_POLICY_MODEL_ID,
            "reference_policy_roundtrip_cost": 0.000612,
            "reference_policy_raw_exit_price_move": 0.0,
            "reference_policy_net_per_notional": 0.0,
            "source_parent_policy_row": -1,
            "reference_policy_row": -1,
            "source_policy_signal_offset_minutes": 0,
            "loss_governor_live_state": {
                "loss_streak": int(self.loss_streak),
                "last_loss_exit_ts": str(self.last_loss_exit_ts or ""),
            },
        }
        candidates = []
        for alias in SOURCE_COMPONENT_ORDER:
            pred = self._predict_component(self.components[alias], work, atr)
            pred = self._apply_cap220(pred, latest)
            pred = self._apply_v5_exposure(pred, now)
            candidates.append(pred)
        trace["component_predictions"] = [
            {
                "alias": c["alias"],
                "expert": c["expert"],
                "final_action": int(c["final_action"]),
                "side": int(c["side"]),
                "quality_for_action": float(c["quality_for_action"]),
                "confidence": float(c["confidence"]),
                "base_notional": float(c["base_notional"]),
                "base_margin_fraction": float(c["base_margin_fraction"]),
                "base_leverage": float(c["base_leverage"]),
                "cap220_skipped": bool(c.get("cap220_skipped", False)),
                "notional": float(c.get("notional", 0.0) or 0.0),
                "margin_fraction": float(c.get("margin_fraction", 0.0) or 0.0),
                "leverage": float(c.get("leverage", 0.0) or 0.0),
                "sidecar_score": float(c["sidecar_score"]),
                "loss_governor_scale": float(c.get("loss_governor_scale", 0.0) or 0.0),
            }
            for c in candidates
        ]
        selected = next((c for c in candidates if int(c["side"]) != 0 and float(c.get("notional", 0.0) or 0.0) > EPS), None)
        if selected is None:
            return self._cash(reason="source_parent_live_native_cash", trace=trace)
        side = int(selected["side"])
        notional = _finite_float(selected["notional"], "source_parent_notional")
        leverage = _finite_float(selected["leverage"], "source_parent_leverage")
        margin = _finite_float(selected["margin_fraction"], "source_parent_margin_fraction")
        expected = margin * leverage
        if abs(expected - notional) > 1.0e-8:
            raise RuntimeError(
                "Omega4.6.2 source parent violates notional=margin_fraction*leverage: "
                f"notional={notional} margin={margin} leverage={leverage}"
            )
        action = 1 if side > 0 else 2
        trace.update(
            {
                "omega462_reason": "entry",
                "side": int(side),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "position_fraction": float(margin),
                "max_hold_bars": int(V5_MAX_HOLD_BARS),
                "quality_score": float(selected["quality_for_action"]),
                "confidence": float(selected["confidence"]),
                "router_expert": f"{selected['alias']}:{selected['expert']}",
                "source_alias": str(selected["alias"]),
                "source_parent_component_report": str(selected["component_report"]),
                "source_parent_component_bundle": str(selected["component_bundle"]),
                "source_parent_component_sidecar": str(selected["component_sidecar"]),
                "source_parent_policy_artifact": str(selected["component_bundle"]),
                "source_parent_policy_entry_timestamp": str(now),
                "source_parent_policy_exit_timestamp": "",
                "source_parent_side_factor": float(selected["source_parent_side_factor"]),
                "pre_governor_notional": float(selected["pre_governor_notional"]),
                "loss_governor_scale": float(selected["loss_governor_scale"]),
                "cap220_short_rsi_gate": {
                    "feature": "rsi",
                    "op": ">=",
                    "threshold": float(CAP220_SHORT_RSI_THRESHOLD),
                    "latest": float(latest["rsi"]),
                },
            }
        )
        return Omega121Decision(
            action=int(action),
            side=int(side),
            notional_exposure=float(notional),
            leverage=float(leverage),
            position_fraction=float(margin),
            take_profit=0.0,
            stop_loss=0.0,
            max_hold_bars=int(V5_MAX_HOLD_BARS),
            cooldown_bars=0,
            quality_score=float(selected["quality_for_action"]),
            confidence=float(selected["confidence"]),
            router_expert=f"{selected['alias']}:{selected['expert']}",
            trace=trace,
        )
