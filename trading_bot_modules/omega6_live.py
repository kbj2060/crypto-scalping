"""Omega6 synthesis research adapter.

Composes validated pieces from prior Alpha/Omega generations per
docs/model_contracts/omega6_synthesis_design_20260703.md:

  L2 primary/fallback TabM 3-head parent (CASH-triggered fallback, Alpha7-style)
  L3 TCN sequence entry gate (short-only, CASH-only trigger, Omega462 pattern)
  L4 risk sizing sidecar (margin_fraction/leverage, Omega4.6.2 pattern)
  L5 true-leverage price barrier + fixed time-stop (Omega1.2.1 pattern)
  L6 event-risk governor (macro veto + shock haircut, Omega5 pattern; reduce-only)

Status: draft_research_not_live_wired. This module is NOT imported by trading_bot.py
and defines no FINAL_GOVERNOR_OMEGA6_* flags. It is driven only by
scripts/backtest_omega6_synthesis_fresh_forward_20260703.py for offline research.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_eval_omega6_tabm_3head_20260703 as omega6_tabm  # noqa: E402


class SequenceEntryTCN(nn.Module):
    """Exact architecture copy of scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py
    ::SequenceEntryTCN, defined locally to avoid importing that script's heavy transitive
    dependency chain (it pulls in the live Omega4.6.2 adapter and tmp/ OOS-replay modules)."""

    def __init__(self, seq_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.net(seq.transpose(1, 2))
        pooled = x.mean(dim=-1)
        last = x[:, :, -1]
        return self.head(torch.cat([pooled, last], dim=1)).squeeze(-1)

EPS = 1.0e-8

# ---- L2 parent contract ----
MODEL_ID = "omega6_synthesis_v1_20260703"
ROUTE_PREFIX = "regime3_current_sensitive_wide24_"
ROUTE_COLS = [f"{ROUTE_PREFIX}bull_prob", f"{ROUTE_PREFIX}bear_prob", f"{ROUTE_PREFIX}chop_prob"]
EXPERTS = ("bull", "bear", "chop")
FORBIDDEN_FEATURE_PREFIXES = (
    "teacher_",
    "teacher_oof_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
DEFAULT_QUALITY_THRESHOLD = 0.45  # matches train_eval_omega6_tabm_3head_20260703.py --quality-threshold default

# ---- L3 TCN sequence gate contract (ported from
# scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py) ----

# ---- L5 true-leverage barrier contract (ported from
# trading_bot_modules/omega1_2_1_live.py::_apply_aggressive_risk) ----
L5_BASE_TP_PRICE_MOVE = 0.026
L5_BASE_SL_PRICE_MOVE = 0.014
L5_MAX_HOLD_HOURS = 24.0  # design principle P8: cap hold budget, unlike Omega4.6's unresolved 222h
L5_BARS_PER_HOUR = 12
L5_MAX_HOLD_BARS = int(L5_MAX_HOLD_HOURS * L5_BARS_PER_HOUR)

# Baseline sizing fed to the L4 sidecar as "decision_*" context features, matching the
# convention the reused omega4_4_v18-lineage sidecar was trained against
# (trading_bot_modules/omega4_6_2_source_parent_live.py BASE_NOTIONAL/BASE_LEVERAGE).
# The sidecar's own output (margin_fraction/leverage) is what actually sizes the trade.
L4_BASELINE_NOTIONAL = 0.45
L4_BASELINE_LEVERAGE = 2.0

# ---- L6 event-risk governor contract (ported from trading_bot_modules/omega5_live.py;
# pure calendar/feature function only, no ledger-derived selection state) ----
L6_MACRO_PRE_MINUTES = 30
L6_MACRO_POST_MINUTES = 120
L6_SHOCK_NOTIONAL_SCALE = 0.50
L6_SHOCK_JUMP_Z_THRESHOLD = 3.0
L6_SHOCK_RET_1H_THRESHOLD = 0.030
L6_SHOCK_RET_4H_THRESHOLD = 0.040
L6_FOMC_DECISION_DATES = {
    # 2026 source: trading_bot_modules/omega5_live.py OMEGA5_EVENT_RISK_FOMC_DECISION_DATES.
    # 2025 source: federalreserve.gov/monetarypolicy/fomccalendars.htm (fetched 2026-07-03),
    # eight 2025 meetings: Jan 28-29, Mar 18-19, May 6-7, Jun 17-18, Jul 29-30, Sep 16-17,
    # Oct 28-29, Dec 9-10 -- date used below is each meeting's second/announcement day
    # (2pm ET statement release), matching the 2026 list's day-of-decision convention.
    2025: ("2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"),
    2026: ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"),
}


@dataclass(frozen=True)
class Omega6Decision:
    action: int
    side: int
    notional_exposure: float
    margin_notional: float
    leverage: float
    take_profit: float
    stop_loss: float
    max_hold_bars: int
    quality_score: float
    confidence: float
    gate_source: str
    trace: dict[str, Any]


@dataclass
class _ParentComponent:
    alias: str
    quality_threshold: float
    models: dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]]
    pos_cols: list[str]


def _reject_forbidden(cols: list[str], tag: str) -> None:
    bad = sorted(c for c in cols if any(str(c).startswith(p) for p in FORBIDDEN_FEATURE_PREFIXES))
    if bad:
        raise RuntimeError(f"{tag} contains forbidden feature prefixes: {bad}")


class Omega6LiveAdapter:
    def __init__(
        self,
        *,
        primary_bundle_path: str | Path,
        fallback_bundle_path: str | Path,
        tcn_gate_path: str | Path,
        risk_sidecar_path: str | Path,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        tcn_short_threshold: float | None = None,
        atr_window: int = 192,
        device: str = "cpu",
        enable_l3_gate: bool = False,
    ) -> None:
        # L3: originally disabled because scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py's
        # feature_cols are the Omega4.6.2 dual-parent's own decision trace (h48qual_*/zig075_*),
        # not generic market features -- feeding Omega6's own parent outputs under those column
        # names would have silently miscalibrated the frozen TCN weights. Fixed by retraining a
        # dedicated gate against Omega6's own decision trace: scripts/train_omega6_sequence_gate_20260703.py.
        # That trainer's threshold convention is score >= threshold (predicts counterfactual SHORT
        # trade_return; keep high-scoring/high-predicted-return bars), read from the artifact's own
        # "threshold" field unless explicitly overridden here.
        self.device = torch.device(device)
        self.quality_threshold = float(quality_threshold)
        self.atr_window = int(atr_window)
        self.enable_l3_gate = bool(enable_l3_gate)
        self.primary = self._load_component("primary", primary_bundle_path)
        self.fallback = self._load_component("fallback", fallback_bundle_path)
        self.tcn = self._load_tcn_gate(tcn_gate_path)
        self.tcn_short_threshold = float(tcn_short_threshold) if tcn_short_threshold is not None else float(self.tcn["threshold"])
        self.sidecar_path = Path(risk_sidecar_path)
        if not self.sidecar_path.exists():
            raise RuntimeError(f"Omega6 risk sidecar missing: {self.sidecar_path}")
        with self.sidecar_path.open("rb") as f:
            self.sidecar = pickle.load(f)
        self._validate_sidecar(self.sidecar)

    # ---- loading / fail-fast validation ----

    def _load_component(self, alias: str, bundle_path: str | Path) -> _ParentComponent:
        path = Path(bundle_path)
        if not path.exists():
            raise RuntimeError(f"Omega6 {alias} bundle missing: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        pos_cols = list(bundle["pos_cols"])
        models: dict[str, tuple[torch.nn.Module, dict[str, Any], list[str]]] = {}
        for expert, payload_raw in dict(bundle["models"]).items():
            payload = dict(payload_raw)
            input_cols = list(payload["input_columns"])
            _reject_forbidden(input_cols, f"Omega6 {alias} {expert} TabM")
            cfg = omega6_tabm.ThreeHeadConfig(**dict(payload["config"]))
            model = omega6_tabm.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            models[str(expert)] = (model, dict(payload["scaler"]), input_cols)
        missing = sorted(set(EXPERTS) - set(models))
        if missing:
            raise RuntimeError(f"Omega6 {alias} bundle missing experts: {missing}")
        return _ParentComponent(alias=alias, quality_threshold=self.quality_threshold, models=models, pos_cols=pos_cols)

    @staticmethod
    def _load_tcn_gate(path: str | Path) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Omega6 TCN sequence gate artifact missing: {p}")
        payload = torch.load(p, map_location="cpu", weights_only=False)
        feature_cols = list(payload["feature_cols"])
        model = SequenceEntryTCN(seq_dim=len(feature_cols))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return {
            "model": model,
            "lookback": int(payload["lookback"]),
            "feature_cols": feature_cols,
            "mean": np.asarray(payload["mean"], dtype=np.float32),
            "std": np.asarray(payload["std"], dtype=np.float32),
            "threshold": float(payload["threshold"]),
        }

    @staticmethod
    def _validate_sidecar(sidecar: dict[str, Any]) -> None:
        if not isinstance(sidecar, dict):
            raise RuntimeError("Omega6 sidecar payload is not a dict")
        if sidecar.get("risk_feature_mode") != "parent_outputs":
            raise RuntimeError("Omega6 sidecar risk_feature_mode mismatch")
        if not bool(sidecar.get("side_split_model")):
            raise RuntimeError("Omega6 sidecar must be side-split")
        if not bool(sidecar.get("dynamic_leverage")):
            raise RuntimeError("Omega6 sidecar must use dynamic leverage")
        model = sidecar.get("model")
        if not isinstance(model, dict) or not {-1, 1}.issubset(set(model)):
            raise RuntimeError("Omega6 sidecar missing side models")
        columns = sidecar.get("feature_columns")
        if not isinstance(columns, list) or not columns:
            raise RuntimeError("Omega6 sidecar missing feature columns")

    # ---- L2 parent inference ----

    @staticmethod
    def _route_expert(row: pd.Series) -> tuple[str, float, float]:
        probs = np.asarray([float(row[c]) for c in ROUTE_COLS], dtype=np.float64)
        if not np.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            raise RuntimeError("Omega6 invalid Regime3 route probabilities")
        probs = probs / np.clip(probs.sum(), EPS, None)
        idx = int(np.argmax(probs))
        sorted_p = np.sort(probs)
        return EXPERTS[idx], float(probs[idx]), float(sorted_p[-1] - sorted_p[-2])

    @staticmethod
    def _latest_input(frame: pd.DataFrame, input_cols: list[str], pos_cols: list[str]) -> pd.DataFrame:
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
            raise RuntimeError(f"Omega6 missing input features: {missing[:60]}")
        if bad:
            raise RuntimeError(f"Omega6 non-finite input features: {bad[:60]}")
        return pd.DataFrame([data], columns=input_cols)

    @staticmethod
    def _standardize(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
        cols = list(scaler["columns"])
        if list(x.columns) != cols:
            raise RuntimeError("Omega6 TabM feature column contract mismatch")
        arr = x.to_numpy(dtype=np.float32)
        z = (arr - scaler["mean"]) / scaler["std"]
        if not np.isfinite(z).all():
            raise RuntimeError("Omega6 standardized feature matrix has non-finite values")
        return z.astype(np.float32)

    @torch.no_grad()
    def _predict_parent(self, component: _ParentComponent, frame: pd.DataFrame) -> dict[str, Any]:
        row = frame.iloc[-1]
        expert, route_conf, route_margin = self._route_expert(row)
        model, scaler, input_cols = component.models[expert]
        x = self._latest_input(frame, input_cols, component.pos_cols)
        z = self._standardize(x, scaler)
        out = model(torch.from_numpy(z).to(self.device))
        direction = torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        quality = torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy()[0].astype(np.float64)
        dir_action = int(np.argmax(direction))
        quality_for_action = float(quality[dir_action] if dir_action > 0 else quality[0])
        final_action = dir_action if dir_action != 0 and quality_for_action >= component.quality_threshold else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)
        return {
            "component": component.alias,
            "expert": expert,
            "route_confidence": route_conf,
            "route_margin": route_margin,
            "direction": direction,
            "quality": quality,
            "dir_action": dir_action,
            "quality_for_action": quality_for_action,
            "action": int(final_action),
            "side": int(side),
            "quality_score": float(quality_for_action if side else 0.0),
            "confidence": float(np.max(direction)),
        }

    # ---- L3 TCN sequence gate ----

    @staticmethod
    def l3_tape_row(primary_out: dict[str, Any], fallback_out: dict[str, Any], atr_pct: float, ts: pd.Timestamp) -> dict[str, float]:
        """Feature-row construction shared byte-for-byte between L3 training
        (scripts/train_omega6_sequence_gate_20260703.py) and inference (_tcn_gate_score below)
        to avoid a train/inference feature-construction mismatch."""
        p_dir, p_qual = primary_out["direction"], primary_out["quality"]
        f_dir, f_qual = fallback_out["direction"], fallback_out["quality"]
        dow = ts.dayofweek + ts.hour / 24.0
        return {
            "primary_dir_p_cash": float(p_dir[0]),
            "primary_dir_p_long": float(p_dir[1]),
            "primary_dir_p_short": float(p_dir[2]),
            "primary_dir_confidence": float(np.max(p_dir)),
            "primary_quality_p_cash": float(p_qual[0]),
            "primary_expert_bull": 1.0 if primary_out["expert"] == "bull" else 0.0,
            "primary_expert_bear": 1.0 if primary_out["expert"] == "bear" else 0.0,
            "primary_expert_chop": 1.0 if primary_out["expert"] == "chop" else 0.0,
            "primary_route_confidence": float(primary_out["route_confidence"]),
            "fallback_dir_p_cash": float(f_dir[0]),
            "fallback_dir_p_long": float(f_dir[1]),
            "fallback_dir_p_short": float(f_dir[2]),
            "fallback_dir_confidence": float(np.max(f_dir)),
            "fallback_quality_p_cash": float(f_qual[0]),
            "fallback_expert_bull": 1.0 if fallback_out["expert"] == "bull" else 0.0,
            "fallback_expert_bear": 1.0 if fallback_out["expert"] == "bear" else 0.0,
            "fallback_expert_chop": 1.0 if fallback_out["expert"] == "chop" else 0.0,
            "fallback_route_confidence": float(fallback_out["route_confidence"]),
            "atr_pct": float(atr_pct),
            "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
            "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        }

    def _tcn_gate_score(self, frame: pd.DataFrame) -> float | None:
        lookback = int(self.tcn["lookback"])
        feature_cols = self.tcn["feature_cols"]
        if len(frame) < lookback + self.atr_window:
            return None
        # L3's feature_cols are derived from primary/fallback decision-trace outputs, not raw
        # frame columns -- rebuild each of the last `lookback` bars' tape row causally (each
        # row only sees data up to and including that bar), matching how the trainer built the
        # tape via scripts/train_omega6_sequence_gate_20260703.py::_build_tape_and_labels.
        rows: list[dict[str, float]] = []
        for offset in range(lookback, 0, -1):
            sub = frame.iloc[: len(frame) - offset + 1]
            p_out = self._predict_parent(self.primary, sub)
            f_out = self._predict_parent(self.fallback, sub)
            atr = self._atr_pct(sub, self.atr_window)
            ts = pd.Timestamp(sub.iloc[-1]["timestamp"])
            rows.append(self.l3_tape_row(p_out, f_out, atr, ts))
        tape = pd.DataFrame(rows)
        missing = sorted(set(feature_cols) - set(tape.columns))
        if missing:
            raise RuntimeError(f"Omega6 TCN gate missing input features: {missing[:60]}")
        tail = tape[feature_cols].apply(pd.to_numeric, errors="coerce")
        seq = tail.to_numpy(dtype=np.float32)
        if not np.isfinite(seq).all():
            raise RuntimeError("Omega6 TCN gate received non-finite sequence input")
        x = (seq - self.tcn["mean"][None, :]) / self.tcn["std"][None, :]
        with torch.no_grad():
            score = float(self.tcn["model"](torch.from_numpy(x[None, :, :].astype(np.float32))).numpy()[0])
        return score

    # ---- L4 risk sizing sidecar ----

    @staticmethod
    def _atr_pct(frame: pd.DataFrame, window: int) -> float:
        required = {"high", "low", "close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Omega6 ATR missing columns: {missing}")
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
            raise RuntimeError("Omega6 non-finite ATR percent")
        return latest

    def _risk_features(self, parent_out: dict[str, Any], atr_pct: float) -> pd.DataFrame:
        direction = parent_out["direction"]
        quality = parent_out["quality"]
        expert = parent_out["expert"]
        side = int(parent_out["side"])
        dec_notional = float(L4_BASELINE_NOTIONAL) if side else 0.0
        dec_leverage = float(L4_BASELINE_LEVERAGE) if side else 1.0
        dec_tp = float(L5_BASE_TP_PRICE_MOVE) if side else 0.0
        dec_sl = float(L5_BASE_SL_PRICE_MOVE) if side else 0.0
        row = {
            "parent_router_confidence": float(parent_out["route_confidence"]),
            "parent_router_margin": float(parent_out["route_margin"]),
            "parent_dir_p_cash": float(direction[0]),
            "parent_dir_p_long": float(direction[1]),
            "parent_dir_p_short": float(direction[2]),
            "parent_dir_confidence": float(np.max(direction)),
            "parent_dir_side_edge": float(abs(direction[1] - direction[2])),
            "parent_dir_trade_prob": float(direction[1] + direction[2]),
            "parent_dir_action": int(parent_out["dir_action"]),
            "parent_quality_p_cash": float(quality[0]),
            "parent_quality_p_long": float(quality[1]),
            "parent_quality_p_short": float(quality[2]),
            "parent_quality_for_action": float(parent_out["quality_for_action"]),
            "parent_quality_threshold": float(self.quality_threshold),
            "parent_final_action": int(parent_out["action"]),
            "parent_router_expert_bear": 1.0 if expert == "bear" else 0.0,
            "parent_router_expert_bull": 1.0 if expert == "bull" else 0.0,
            "parent_router_expert_chop": 1.0 if expert == "chop" else 0.0,
            "decision_action": int(parent_out["action"]),
            "decision_side": int(parent_out["side"]),
            "decision_quality_score": float(parent_out["quality_score"]),
            "decision_confidence": float(parent_out["confidence"]),
            "decision_notional_exposure": float(dec_notional),
            "decision_leverage": float(dec_leverage),
            "decision_position_fraction": float(dec_notional),
            "decision_take_profit": float(dec_tp),
            "decision_stop_loss": float(dec_sl),
            "decision_rr": float(dec_tp) / max(float(dec_sl), EPS),
            "atr_pct_runtime": float(atr_pct),
        }
        cols = list(self.sidecar["feature_columns"])
        missing = sorted(set(cols) - set(row))
        if missing:
            raise RuntimeError(f"Omega6 sidecar feature builder missing: {missing}")
        return pd.DataFrame([{col: row[col] for col in cols}], columns=cols, dtype=np.float32)

    def _sidecar_sizing(self, features: pd.DataFrame, side: int) -> tuple[float, float, float]:
        if side == 0:
            return 0.0, 0.0, 0.0
        model = self.sidecar["model"][side]
        score = float(model.predict(features)[0])
        mapping = dict(self.sidecar["selected_mapping"])
        q50 = float(self.sidecar["train_score_q50"])
        iqr = max(float(self.sidecar["train_score_iqr"]), 1.0e-8)
        z_margin = float(np.clip((score - q50) / iqr, -8.0, 8.0))
        unit_margin = 1.0 / (1.0 + np.exp(-float(mapping["temp"]) * z_margin))
        scale = float(mapping["min_scale"]) + (float(mapping["max_scale"]) - float(mapping["min_scale"])) * unit_margin
        margin = float(np.clip(scale, float(mapping["floor"]), float(mapping["cap"])))
        margin *= float(mapping.get("long_scale", 1.0)) if side > 0 else float(mapping.get("short_scale", 1.0))
        margin = float(np.clip(margin, float(mapping["floor"]), float(mapping["cap"])))

        z_lev = z_margin
        unit_lev = 1.0 / (1.0 + np.exp(-float(mapping["leverage_temp"]) * z_lev))
        leverage = float(mapping["leverage_min"]) + (float(mapping["leverage_max"]) - float(mapping["leverage_min"])) * unit_lev
        leverage *= float(mapping.get("long_leverage_scale", 1.0)) if side > 0 else float(mapping.get("short_leverage_scale", 1.0))
        leverage = float(np.clip(leverage, float(mapping["leverage_floor"]), float(mapping["leverage_cap"])))
        return margin, leverage, score

    # ---- L5 true-leverage price barrier + time-stop ----

    @staticmethod
    def _apply_true_leverage_barrier(margin_fraction: float, leverage: float) -> tuple[float, float, float]:
        # AGENTS.md Futures Risk Sizing Contract: notional = margin_fraction * leverage.
        # TP/SL price-move barriers are scaled by leverage (omega1_2_1_live.py::_apply_aggressive_risk
        # pattern) so the effective price distance is preserved under true-leverage accounting.
        notional_exposure = float(margin_fraction) * float(leverage)
        take_profit = float(L5_BASE_TP_PRICE_MOVE) * float(leverage)
        stop_loss = float(L5_BASE_SL_PRICE_MOVE) * float(leverage)
        return notional_exposure, take_profit, stop_loss

    # ---- L6 event-risk governor (pure calendar/feature function; no ledger dependence) ----

    @staticmethod
    def _weekday_on_or_after(year: int, month: int, day: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=day)
        while out.weekday() >= 5:
            out += pd.Timedelta(days=1)
        return out

    @staticmethod
    def _nth_weekday(year: int, month: int, n: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=1)
        count = 0
        while True:
            if out.weekday() < 5:
                count += 1
                if count == int(n):
                    return out
            out += pd.Timedelta(days=1)

    @staticmethod
    def _first_friday(year: int, month: int) -> pd.Timestamp:
        out = pd.Timestamp(year=year, month=month, day=1)
        while out.weekday() != 4:
            out += pd.Timedelta(days=1)
        return out

    @staticmethod
    def _et_to_utc_naive(day: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
        ny = ZoneInfo("America/New_York")
        dt = datetime(int(day.year), int(day.month), int(day.day), int(hour), int(minute), tzinfo=ny)
        return pd.Timestamp(dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))

    @classmethod
    def _macro_events_for_year(cls, year: int) -> list[tuple[str, pd.Timestamp]]:
        events: list[tuple[str, pd.Timestamp]] = []
        for month in range(1, 13):
            nfp = cls._first_friday(year, month)
            events.append(("NFP_8h30_ET_rule_based", cls._et_to_utc_naive(nfp, 8, 30)))
            manufacturing = cls._nth_weekday(year, month, 1)
            events.append(("ISM_MANUFACTURING_10h_ET_rule_based", cls._et_to_utc_naive(manufacturing, 10, 0)))
            services = cls._nth_weekday(year, month, 3)
            events.append(("ISM_SERVICES_10h_ET_rule_based", cls._et_to_utc_naive(services, 10, 0)))
            flash = cls._weekday_on_or_after(year, month, 23)
            events.append(("SPGLOBAL_FLASH_PMI_9h45_ET_rule_based", cls._et_to_utc_naive(flash, 9, 45)))
        for raw in L6_FOMC_DECISION_DATES.get(int(year), ()):
            day = pd.Timestamp(raw)
            events.append(("FOMC_14h_ET_static_calendar", cls._et_to_utc_naive(day, 14, 0)))
        return events

    @staticmethod
    def _latest_optional_float(frame: pd.DataFrame, feature: str) -> float:
        if feature not in frame.columns:
            return 0.0
        value = float(frame.iloc[-1][feature])
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _latest_return(frame: pd.DataFrame, bars: int) -> float:
        if "close" not in frame.columns:
            raise RuntimeError("Omega6 event-risk governor missing required feature: close")
        if len(frame) <= int(bars):
            return 0.0
        latest = float(frame.iloc[-1]["close"])
        previous = float(frame.iloc[-1 - int(bars)]["close"])
        if not np.isfinite(latest) or not np.isfinite(previous) or previous <= 0.0:
            raise RuntimeError("Omega6 event-risk governor received invalid close history")
        return float(latest / previous - 1.0)

    @classmethod
    def _event_risk_latest(cls, frame: pd.DataFrame) -> dict[str, Any]:
        if not len(frame):
            raise RuntimeError("Omega6 event-risk governor received empty feature frame")
        if "timestamp" not in frame.columns:
            raise RuntimeError("Omega6 event-risk governor missing required feature: timestamp")
        ts = pd.Timestamp(frame.iloc[-1]["timestamp"])
        if pd.isna(ts):
            raise RuntimeError("Omega6 event-risk governor received NaT timestamp")
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        event_hits: list[str] = []
        for year in (ts.year - 1, ts.year, ts.year + 1):
            for name, event_ts in cls._macro_events_for_year(int(year)):
                start = event_ts - pd.Timedelta(minutes=L6_MACRO_PRE_MINUTES)
                end = event_ts + pd.Timedelta(minutes=L6_MACRO_POST_MINUTES)
                if start <= ts <= end:
                    event_hits.append(name)
        jump_flag = cls._latest_optional_float(frame, "jump_flag")
        evt_tail_flag = cls._latest_optional_float(frame, "evt_tail_flag")
        jump_z = cls._latest_optional_float(frame, "jump_z")
        ret_1h = cls._latest_return(frame, 12)
        ret_4h = cls._latest_return(frame, 48)
        shock_hit = (
            jump_flag > 0.0
            or evt_tail_flag > 0.0
            or abs(jump_z) >= L6_SHOCK_JUMP_Z_THRESHOLD
            or abs(ret_1h) >= L6_SHOCK_RET_1H_THRESHOLD
            or abs(ret_4h) >= L6_SHOCK_RET_4H_THRESHOLD
        )
        return {
            "timestamp": str(ts),
            "macro_entry_veto": bool(event_hits),
            "macro_event_names": event_hits,
            "shock_haircut": bool(shock_hit),
            "shock_notional_scale": float(L6_SHOCK_NOTIONAL_SCALE if shock_hit else 1.0),
            "jump_flag": float(jump_flag),
            "evt_tail_flag": float(evt_tail_flag),
            "jump_z": float(jump_z),
            "ret_1h_past": float(ret_1h),
            "ret_4h_past": float(ret_4h),
        }

    # ---- top-level entry decision ----

    def _cash_decision(self, gate_source: str, trace: dict[str, Any]) -> Omega6Decision:
        return Omega6Decision(
            action=0,
            side=0,
            notional_exposure=0.0,
            margin_notional=0.0,
            leverage=1.0,
            take_profit=0.0,
            stop_loss=0.0,
            max_hold_bars=0,
            quality_score=0.0,
            confidence=0.0,
            gate_source=gate_source,
            trace=trace,
        )

    def decide_latest(self, frame: pd.DataFrame) -> Omega6Decision:
        if not len(frame):
            raise RuntimeError("Omega6 received empty feature frame")
        trace: dict[str, Any] = {"model_id": MODEL_ID}

        primary_out = self._predict_parent(self.primary, frame)
        trace["primary"] = {k: v for k, v in primary_out.items() if k not in ("direction", "quality")}
        if primary_out["side"] != 0:
            parent_out, gate_source = primary_out, "primary"
        else:
            fallback_out = self._predict_parent(self.fallback, frame)
            trace["fallback"] = {k: v for k, v in fallback_out.items() if k not in ("direction", "quality")}
            if fallback_out["side"] != 0:
                parent_out, gate_source = fallback_out, "fallback"
            else:
                tcn_score = self._tcn_gate_score(frame) if self.enable_l3_gate else None
                trace["tcn_gate_score"] = tcn_score
                trace["l3_gate_enabled"] = self.enable_l3_gate
                # score >= threshold: scripts/train_omega6_sequence_gate_20260703.py trains the
                # gate to predict the counterfactual SHORT trade's net_per_notional return, so a
                # HIGH score means "this bar looks like a good short entry" (confirmed against
                # scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py's own
                # select_threshold(), which keeps `scores >= threshold` and maximizes summed
                # trade_return -- not the reversed `<=` this module used before retraining).
                if tcn_score is not None and tcn_score >= self.tcn_short_threshold:
                    parent_out = {
                        "component": "l3_tcn_gate",
                        "expert": fallback_out["expert"],
                        "route_confidence": fallback_out["route_confidence"],
                        "route_margin": fallback_out["route_margin"],
                        "direction": fallback_out["direction"],
                        "quality": fallback_out["quality"],
                        "dir_action": 2,
                        "quality_for_action": float(fallback_out["quality"][0]),
                        "action": 2,
                        "side": -1,
                        "quality_score": float(fallback_out["quality"][0]),
                        "confidence": float(abs(tcn_score)),
                    }
                    gate_source = "l3_tcn_gate"
                else:
                    parent_out, gate_source = fallback_out, "cash"

        if parent_out["side"] == 0:
            return self._cash_decision(gate_source, trace)

        atr_pct = self._atr_pct(frame, self.atr_window)
        features = self._risk_features(parent_out, atr_pct)
        margin_fraction, leverage, sidecar_score = self._sidecar_sizing(features, int(parent_out["side"]))
        trace["sidecar_score"] = sidecar_score
        trace["margin_fraction_raw"] = margin_fraction
        trace["leverage_raw"] = leverage
        if margin_fraction <= 0.0 or leverage <= 0.0:
            return self._cash_decision("l4_sidecar_zero_size", trace)

        notional_exposure, take_profit, stop_loss = self._apply_true_leverage_barrier(margin_fraction, leverage)

        event_risk = self._event_risk_latest(frame)
        trace["event_risk"] = event_risk
        if event_risk["macro_entry_veto"]:
            return self._cash_decision("l6_macro_veto", trace)
        if event_risk["shock_haircut"]:
            margin_fraction *= L6_SHOCK_NOTIONAL_SCALE
            notional_exposure *= L6_SHOCK_NOTIONAL_SCALE

        return Omega6Decision(
            action=int(parent_out["action"]),
            side=int(parent_out["side"]),
            notional_exposure=float(notional_exposure),
            margin_notional=float(margin_fraction),
            leverage=float(leverage),
            take_profit=float(take_profit),
            stop_loss=float(stop_loss),
            max_hold_bars=int(L5_MAX_HOLD_BARS),
            quality_score=float(parent_out["quality_score"]),
            confidence=float(parent_out["confidence"]),
            gate_source=gate_source,
            trace=trace,
        )
