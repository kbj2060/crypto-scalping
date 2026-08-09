from __future__ import annotations

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

import experiment_regime3_current_hmm_wide24_20260529 as regime3_current  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as omega_tabm  # noqa: E402
import train_regime3_cryptomamba_pred_20260531 as regime3_cmamba  # noqa: E402
import train_regime3_stability_risk_20260530 as regime3_risk  # noqa: E402


OMEGA121_MODEL_ID = "omega3_aggressive_compensated_scale200_cap090_20260618"
OMEGA121_OWNER = "omega1_2_1"
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
THR_MAP = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
EXPERT_SCALES = {"bull": 0.75, "bear": 0.90, "chop": 0.90}
OVERLAY_SCALES = {"bull": 0.65, "bear": 0.90, "chop": 0.90}
BASE_NOTIONAL = 0.45
BASE_LEVERAGE = 2.0
BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
NOTIONAL_CAP = 0.90
TRUE_LEVERAGE_EXPOSURE = True
PRESERVE_PRICE_BARRIER = True
FORBIDDEN_FEATURE_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
FORBIDDEN_FEATURE_NAMES = {"tp_sl_action_score"}


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


class Regime3LiveFeatures:
    def __init__(
        self,
        *,
        current_path: str | Path,
        cmamba_path: str | Path,
        risk_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        self.current_payload = joblib.load(Path(current_path))
        self.risk_payload = joblib.load(Path(risk_path))
        if not torch.cuda.is_available() and str(device) == "cuda":
            raise RuntimeError("Omega1.2.1 Regime3 CMamba requires CUDA")
        self.device = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError("Omega1.2.1 Regime3 CMamba live path must run on CUDA")
        self.cmamba_payload = torch.load(Path(cmamba_path), map_location="cpu", weights_only=False)
        self.cmamba_model = regime3_cmamba.CryptoMambaRegimePred(
            len(self.cmamba_payload["feature_cols"]),
            int(self.cmamba_payload["seq_len"]),
            int(self.cmamba_payload["d_model"]),
            int(self.cmamba_payload["cblocks"]),
            int(self.cmamba_payload["cmblocks"]),
            int(self.cmamba_payload["d_state"]),
            0.0,
        ).to(self.device)
        self.cmamba_model.load_state_dict(self.cmamba_payload["state_dict"])
        self.cmamba_model.eval()

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

    def _append_current(self, frame: pd.DataFrame) -> pd.DataFrame:
        payload = self.current_payload
        cols = list(payload["feature_cols"])
        self._reject_forbidden(cols, "Regime3 current")
        work = regime3_current._with_features(frame, cols)
        raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = self._impute_training_medians(raw, payload, "Regime3 current")
        self._require_finite_frame(raw, "Regime3 current")
        x = raw
        xz = payload["scaler"].transform(x)
        state = payload["model"].filter_proba(xz)
        proba = regime3_current._class_proba(state, np.asarray(payload["state_class_matrix"], dtype=np.float64))
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)

        out = frame.copy()
        for i, name in enumerate(payload["classes"]):
            out[f"{CURRENT_PREFIX}{name}_prob"] = proba[:, i]
        sorted_p = np.sort(proba, axis=1)
        out[f"{CURRENT_PREFIX}confidence"] = proba.max(axis=1)
        out[f"{CURRENT_PREFIX}margin"] = sorted_p[:, -1] - sorted_p[:, -2]
        out[f"{CURRENT_PREFIX}entropy"] = -(proba * np.log(np.clip(proba, 1e-12, None))).sum(axis=1) / np.log(3.0)
        return out

    def _append_cmamba(self, frame: pd.DataFrame) -> pd.DataFrame:
        payload = self.cmamba_payload
        seq_len = int(payload["seq_len"])
        if len(frame) < seq_len:
            raise RuntimeError(f"Omega1.2.1 CMamba requires at least {seq_len} bars, got {len(frame)}")
        work = regime3_cmamba._add_volume_features(regime3_cmamba._add_rolling_stable_features(frame.copy()))
        cols = list(payload["feature_cols"])
        self._reject_forbidden(cols, "Regime3 CMamba")
        missing = [c for c in cols if c not in work.columns]
        if missing:
            raise RuntimeError(f"Omega1.2.1 CMamba missing feature columns: {missing[:40]}")
        raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = self._impute_training_medians(raw, payload, "Regime3 CMamba")
        self._require_finite_frame(raw.tail(seq_len), "Regime3 CMamba tail window")
        x = raw.to_numpy(dtype=np.float32)
        x = (x - np.asarray(payload["scaler_mean"], dtype=np.float32)) / np.asarray(payload["scaler_scale"], dtype=np.float32)
        x = x.astype(np.float32)
        idx = np.asarray([len(x) - 1], dtype=np.int64)
        probs = regime3_cmamba._predict(self.cmamba_model, x, idx, seq_len, 512, self.device)
        current = frame[ROUTE_COLS].to_numpy(dtype=np.float64)
        current = current / np.clip(current.sum(axis=1, keepdims=True), 1e-12, None)
        current_id = np.argmax(current, axis=1).astype(np.int64)
        pred = np.argmax(probs, axis=1).astype(np.int64)

        out = frame.copy()
        for name in ("bull", "bear", "chop"):
            out[f"{CMAMBA_PREFIX}{name}_prob"] = np.nan
        out[f"{CMAMBA_PREFIX}class_id"] = np.nan
        out[f"{CMAMBA_PREFIX}confidence"] = np.nan
        out[f"{CMAMBA_PREFIX}transition_prob"] = np.nan
        out[f"{CMAMBA_PREFIX}stability_score"] = np.nan
        for i, name in enumerate(payload["classes"]):
            out.loc[idx, f"{CMAMBA_PREFIX}{name}_prob"] = probs[:, i]
        out.loc[idx, f"{CMAMBA_PREFIX}class_id"] = pred
        out.loc[idx, f"{CMAMBA_PREFIX}confidence"] = probs.max(axis=1)
        stay_p = probs[np.arange(len(idx)), current_id[idx]]
        out.loc[idx, f"{CMAMBA_PREFIX}transition_prob"] = 1.0 - stay_p
        out.loc[idx, f"{CMAMBA_PREFIX}stability_score"] = stay_p
        return out

    def _append_risk(self, frame: pd.DataFrame) -> pd.DataFrame:
        payload = self.risk_payload
        work = regime3_risk._add_stability_features(regime3_risk._add_rolling_stable_features(frame.copy()))
        cols = list(payload["feature_cols"])
        self._reject_forbidden(cols, "Regime3 stability/risk")
        missing = [c for c in cols if c not in work.columns]
        if missing:
            raise RuntimeError(f"Omega1.2.1 Regime3 risk missing feature columns: {missing[:40]}")
        raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = self._impute_training_medians(raw, payload, "Regime3 stability/risk")
        x = raw.tail(1)
        self._require_finite_frame(x, "Regime3 stability/risk latest")
        xz = payload["scaler"].transform(x).astype(np.float32)
        transition_proba = payload["transition_model"].predict_proba(xz)
        transition_p = np.zeros(len(xz), dtype=np.float64)
        for i, cls in enumerate(payload["transition_model"].classes_):
            if int(cls) == 1:
                transition_p = transition_proba[:, i].astype(np.float64)
                break
        risk_score = np.asarray(payload["risk_model"].predict(xz), dtype=np.float64)
        out = frame.copy()
        last_idx = out.index[-1]
        out["regime3_stability_h6_score"] = np.nan
        out["regime3_transition_h6_risk_prob"] = np.nan
        out["regime3_transition_h6_risk_pred"] = np.nan
        out["regime3_churn_h6_risk_score"] = np.nan
        out.loc[last_idx, "regime3_stability_h6_score"] = float(1.0 - transition_p[-1])
        out.loc[last_idx, "regime3_transition_h6_risk_prob"] = float(transition_p[-1])
        out.loc[last_idx, "regime3_transition_h6_risk_pred"] = int(transition_p[-1] >= float(payload["threshold"]))
        out.loc[last_idx, "regime3_churn_h6_risk_score"] = float(np.clip(risk_score[-1], 0.0, 1.0))
        return out

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._append_current(frame)
        self._finite_latest(out, ROUTE_COLS + [f"{CURRENT_PREFIX}confidence", f"{CURRENT_PREFIX}entropy", f"{CURRENT_PREFIX}margin"], "Regime3 current")
        out = self._append_cmamba(out)
        cmamba_cols = [
            f"{CMAMBA_PREFIX}bull_prob",
            f"{CMAMBA_PREFIX}bear_prob",
            f"{CMAMBA_PREFIX}chop_prob",
            f"{CMAMBA_PREFIX}class_id",
            f"{CMAMBA_PREFIX}confidence",
            f"{CMAMBA_PREFIX}transition_prob",
            f"{CMAMBA_PREFIX}stability_score",
        ]
        self._finite_latest(out, cmamba_cols, "Regime3 CMamba")
        out = self._append_risk(out)
        self._finite_latest(out, RISK_COLS, "Regime3 stability/risk")
        return out


class Omega121LiveAdapter:
    def __init__(
        self,
        *,
        bundle_path: str | Path,
        current_path: str | Path,
        cmamba_path: str | Path,
        risk_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        if not torch.cuda.is_available() and str(device) == "cuda":
            raise RuntimeError("Omega1.2.1 live adapter requires CUDA")
        self.device = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            raise RuntimeError("Omega1.2.1 live adapter must run on CUDA")
        self.regime3 = Regime3LiveFeatures(
            current_path=current_path,
            cmamba_path=cmamba_path,
            risk_path=risk_path,
            device=self.device,
        )
        self.bundle = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
        self.base_cols = list(self.bundle["base_cols"])
        self.pos_cols = list(self.bundle["pos_cols"])
        self.models: dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]] = {}
        for expert, payload in dict(self.bundle["models"]).items():
            Regime3LiveFeatures._reject_forbidden(list(payload["input_columns"]), f"Omega1.2.1 {expert} TabM")
            cfg = omega_tabm.ThreeHeadConfig(**dict(payload["config"]))
            model = omega_tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.models[str(expert)] = (model, payload["scaler"], list(payload["input_columns"]))
        missing_experts = sorted(set(("bull", "bear", "chop")) - set(self.models))
        if missing_experts:
            raise RuntimeError(f"Omega1.2.1 bundle missing experts: {missing_experts}")

    @staticmethod
    def _quality_for_action(quality: np.ndarray, action: int) -> float:
        if int(action) <= 0:
            return float(quality[0])
        return float(quality[int(action)])

    @staticmethod
    def _route_expert(row: pd.Series) -> tuple[str, float, float]:
        probs = np.asarray([float(row[c]) for c in ROUTE_COLS], dtype=np.float64)
        if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            raise RuntimeError("Omega1.2.1 invalid Regime3 route probabilities")
        probs = probs / np.clip(probs.sum(), 1e-12, None)
        idx = int(np.argmax(probs))
        sorted_p = np.sort(probs)
        return ("bull", "bear", "chop")[idx], float(probs[idx]), float(sorted_p[-1] - sorted_p[-2])

    def _latest_input(self, frame: pd.DataFrame, input_cols: list[str]) -> pd.DataFrame:
        Regime3LiveFeatures._reject_forbidden(input_cols, "Omega1.2.1 TabM")
        row = frame.iloc[-1]
        data: dict[str, float] = {}
        missing = []
        bad = []
        for col in input_cols:
            if col in self.pos_cols:
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
            raise RuntimeError(f"Omega1.2.1 missing input features: {missing[:60]}")
        if bad:
            raise RuntimeError(f"Omega1.2.1 non-finite input features: {bad[:60]}")
        return pd.DataFrame([data], columns=input_cols)

    @staticmethod
    def _standardize(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
        cols = list(scaler["columns"])
        if list(x.columns) != cols:
            raise RuntimeError("Omega1.2.1 TabM feature column contract mismatch")
        arr = x.to_numpy(dtype=np.float32)
        z = (arr - scaler["mean"]) / scaler["std"]
        if not np.isfinite(z).all():
            raise RuntimeError("Omega1.2.1 standardized feature matrix has non-finite values")
        return z.astype(np.float32)

    @torch.no_grad()
    def _predict_expert(self, expert: str, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model, scaler, _ = self.models[expert]
        z = self._standardize(x, scaler)
        out = model(torch.from_numpy(z).to(self.device))
        direction = torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        quality = torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        exit_p = torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        return direction, quality, exit_p

    @staticmethod
    def _base_notional(expert: str) -> float:
        parent = BASE_NOTIONAL * float(EXPERT_SCALES[expert])
        return parent * (float(OVERLAY_SCALES[expert]) / float(EXPERT_SCALES[expert]))

    @staticmethod
    def _apply_aggressive_risk(base_notional: float) -> tuple[float, float, float, float]:
        leverage = float(BASE_LEVERAGE)
        effective_exposure = min(float(base_notional) * COMPENSATED_SCALE, NOTIONAL_CAP)
        margin_notional = effective_exposure / max(leverage, 1e-12) if TRUE_LEVERAGE_EXPOSURE else effective_exposure
        barrier_scale = leverage if PRESERVE_PRICE_BARRIER and TRUE_LEVERAGE_EXPOSURE else 1.0
        return effective_exposure, margin_notional, BASE_TP * barrier_scale, BASE_SL * barrier_scale

    def decide_latest(self, frame: pd.DataFrame) -> Omega121Decision:
        enriched = self.regime3.append(frame.copy().reset_index(drop=True))
        row = enriched.iloc[-1]
        expert, route_conf, route_margin = self._route_expert(row)
        _, _, input_cols = self.models[expert]
        x = self._latest_input(enriched, input_cols)
        direction, quality, exit_p = self._predict_expert(expert, x)
        dir_action = int(np.argmax(direction))
        q_action = self._quality_for_action(quality, dir_action)
        threshold = float(THR_MAP[expert])
        final_action = dir_action if dir_action != 0 and q_action >= threshold else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)
        base_notional = self._base_notional(expert) if side else 0.0
        notional, margin_notional, take_profit, stop_loss = self._apply_aggressive_risk(base_notional) if side else (0.0, 0.0, 0.0, 0.0)
        leverage = BASE_LEVERAGE if side else 1.0
        trace = {
            "model_id": OMEGA121_MODEL_ID,
            "router_expert": expert,
            "router_confidence": route_conf,
            "router_margin": route_margin,
            "direction_proba": {
                "cash": float(direction[0]),
                "long": float(direction[1]),
                "short": float(direction[2]),
            },
            "quality_proba": {
                "cash": float(quality[0]),
                "long": float(quality[1]),
                "short": float(quality[2]),
            },
            "exit_proba": {
                "hold": float(exit_p[0]),
                "exit": float(exit_p[1]),
            },
            "dir_action": dir_action,
            "quality_for_action": float(q_action),
            "quality_threshold": threshold,
            "base_notional": float(base_notional),
            "compensated_scale": COMPENSATED_SCALE,
            "notional_cap": NOTIONAL_CAP,
            "margin_notional": float(margin_notional),
            "effective_exposure": float(notional),
            "true_leverage_exposure": bool(TRUE_LEVERAGE_EXPOSURE),
            "preserve_price_barrier": bool(PRESERVE_PRICE_BARRIER),
            "regime3": {
                "current_confidence": float(row[f"{CURRENT_PREFIX}confidence"]),
                "cmamba_transition_prob": float(row[f"{CMAMBA_PREFIX}transition_prob"]),
                "transition_risk_prob": float(row["regime3_transition_h6_risk_prob"]),
                "churn_risk_score": float(row["regime3_churn_h6_risk_score"]),
            },
        }
        return Omega121Decision(
            action=final_action,
            side=side,
            notional_exposure=float(notional),
            leverage=float(leverage),
            position_fraction=float(margin_notional) if side else 0.0,
            take_profit=float(take_profit),
            stop_loss=float(stop_loss),
            max_hold_bars=0,
            cooldown_bars=0,
            quality_score=float(q_action),
            confidence=float(np.max(direction)),
            router_expert=expert,
            trace=trace,
        )
