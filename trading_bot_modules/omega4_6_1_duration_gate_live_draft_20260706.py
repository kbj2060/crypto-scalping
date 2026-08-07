"""DRAFT, NOT WIRED live adapter for omega4_6_1_duration_ou_halflife_risk_gate_20260630 (base
form, event-flat overlay EXCLUDED -- that idea failed under genuine fresh-forward testing, see
docs/model_contracts/omega4_6_1_event_flat_fresh_forward_correction_20260706.md).

Scope note: this is a bounded draft for the promotion checklist item, not a full production-grade
adapter matching the rigor of trading_bot_modules/omega4_6_2_source_parent_live.py (846 lines,
extensive fail-fast contract checks for every sub-artifact). It implements the essential
decide_latest() flow -- two TabM parent components, priority router (h48qual > zig075), risk
sidecar sizing, and the VAL-reselected duration gate -- with fail-fast validation on the artifacts
this module directly touches. It is NOT imported by trading_bot.py and has no FINAL_GOVERNOR_*
flag; wiring it live requires the same runtime-native parity work every other Omega candidate in
this repo needs before promotion (see docs/model_contracts/omega4_6_1_event_flat_live_promotion_audit_20260706.md
for the full outstanding gate list -- this file only closes the "live adapter code exists" gap).
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as _atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as _parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as _omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as _hard  # noqa: E402
from omega4_6_2_source_parent_live import (  # noqa: E402
    DEFAULT_CURRENT_REGIME_PATH as _DEFAULT_CURRENT_REGIME_PATH,
    Regime3CurrentLiveFeatures as _Regime3CurrentLiveFeatures,
)

MODEL_ID = "omega4_6_1_duration_ou_halflife_risk_gate_20260630"
DURATION_FEATURE = "ou_halflife"
DURATION_THRESHOLD = 0.005417  # VAL-reselected 2026-07-06; original frozen value was 0.005415348
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ("h48qual", "zig075")


@dataclass(frozen=True)
class Omega461Decision:
    action: int  # 0=CASH, 1=LONG, 2=SHORT
    side: int  # 0/1/-1
    source_component: str
    margin_fraction: float
    leverage: float
    notional_exposure: float
    take_profit: float
    stop_loss: float
    duration_gate_hit: bool
    trace: dict[str, Any]


@dataclass(frozen=True)
class _ComponentConfig:
    alias: str
    bundle_path: Path
    sidecar_path: Path
    quality_threshold: float
    atr_window: int = 192
    tp_mult: float = 12.0
    sl_mult: float = 6.0
    min_tp: float = 0.075
    min_sl: float = 0.040
    max_tp: float = 0.22
    max_sl: float = 0.12
    exit_threshold: float = 0.95


class _Component:
    def __init__(self, cfg: _ComponentConfig, *, device: torch.device) -> None:
        if not cfg.bundle_path.exists():
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: missing parent bundle {cfg.bundle_path}")
        if not cfg.sidecar_path.exists():
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: missing risk sidecar {cfg.sidecar_path}")
        self.cfg = cfg
        self.device = device
        bundle = torch.load(cfg.bundle_path, map_location="cpu", weights_only=False)
        self.base_cols: list[str] = list(bundle["base_cols"])
        if any(c.startswith(("m7_", "ai_", "patchtst", "tide_", "dlinear")) for c in self.base_cols):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: unexpected m7/NF feature dependency (contract drift)")
        models = dict(bundle["models"])
        missing_experts = sorted(set(_hard.EXPERT_NAMES) - set(models))
        if missing_experts:
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: bundle missing experts {missing_experts}")
        self.models = models
        with open(cfg.sidecar_path, "rb") as f:
            pkl = pickle.load(f)
        if pkl.get("risk_feature_mode") != "parent_outputs":
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar risk_feature_mode contract mismatch")
        if not pkl.get("side_split_model"):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar must be side-split")
        if not pkl.get("dynamic_leverage"):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar must use dynamic leverage")
        self.sidecar = pkl

    def decide(self, frame: pd.DataFrame) -> dict[str, Any]:
        """frame: causal feature history up to and including the current bar (last row = latest)."""
        missing = [c for c in self.base_cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Omega4.6.1 {self.cfg.alias}: missing input features {missing[:20]}")
        x_all = frame.reindex(columns=self.base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x_all.isna().any().any():
            raise RuntimeError(f"Omega4.6.1 {self.cfg.alias}: non-finite input features in history window")
        x_row = x_all.iloc[[-1]].reset_index(drop=True)
        for c in _parent.POS_COLS:
            x_row[c] = 0.0
        x_np = x_row.astype(np.float32)

        route_val = _hard._route_id(frame.iloc[[-1]].reset_index(drop=True))[0]
        expert = _hard.EXPERT_NAMES[int(route_val)]
        pred = _parent._predict_payload(self.models[expert], x_np, device=self.device)
        direction = pred["direction"][0]
        quality = pred["quality"][0]
        action = int(np.argmax(direction))
        qual_for_action = float(quality[action]) if action > 0 else float(quality[0])
        final_action = action if (action != 0 and qual_for_action >= self.cfg.quality_threshold) else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)

        atr_pct = float(_atr_eval._atr_pct(frame, self.cfg.atr_window)[-1])
        if side != 0 and not np.isfinite(atr_pct):
            raise RuntimeError(f"Omega4.6.1 {self.cfg.alias}: non-finite ATR for active decision")
        take_profit = float(np.clip(max(self.cfg.min_tp, atr_pct * self.cfg.tp_mult), 0.0, self.cfg.max_tp)) if side != 0 else 0.0
        stop_loss = float(np.clip(max(self.cfg.min_sl, atr_pct * self.cfg.sl_mult), 0.0, self.cfg.max_sl)) if side != 0 else 0.0

        margin_fraction = leverage = 0.0
        if side != 0:
            feat_cols = self.sidecar["feature_columns"]
            router_probs = frame[_hard.ROUTE_COLS].iloc[-1].to_numpy(dtype=np.float64)
            router_expert_onehot = {f"parent_router_expert_{e}": float(expert == e) for e in _hard.EXPERT_NAMES}
            # baseline (pre-risk-sizing) fixed-template decision, matching omega._to_fixed_decisions
            # + _apply_expert_scale exactly -- the risk sidecar's score model was trained on THESE
            # baseline decision_* features as input, not zeros.
            expert_scale_key = "chop_expert" if expert == "chop" else expert
            expert_scale = float(_omega.EXPERT_SCALES.get(expert_scale_key, 1.0))
            base_notional = float(_omega.BASE_TEMPLATE["notional"]) * expert_scale
            base_leverage = float(_omega.BASE_TEMPLATE["leverage"])
            risk_row = {
                "parent_router_confidence": float(router_probs.max()),
                "parent_router_margin": float(frame["regime3_current_sensitive_wide24_margin"].iloc[-1]),
                "parent_dir_p_cash": float(direction[0]), "parent_dir_p_long": float(direction[1]), "parent_dir_p_short": float(direction[2]),
                "parent_dir_confidence": float(direction.max()), "parent_dir_side_edge": float(direction[1] - direction[2]),
                "parent_dir_trade_prob": float(direction[1] + direction[2]), "parent_dir_action": float(action),
                "parent_quality_p_cash": float(quality[0]), "parent_quality_p_long": float(quality[1]), "parent_quality_p_short": float(quality[2]),
                "parent_quality_for_action": qual_for_action, "parent_quality_threshold": float(self.cfg.quality_threshold),
                "parent_final_action": float(final_action),
                **router_expert_onehot,
                "decision_action": float(final_action), "decision_side": float(side), "decision_quality_score": qual_for_action,
                "decision_confidence": float(direction.max()), "decision_notional_exposure": base_notional, "decision_leverage": base_leverage,
                "decision_position_fraction": base_notional, "decision_take_profit": take_profit, "decision_stop_loss": stop_loss,
                "decision_rr": take_profit / max(stop_loss, 1e-8), "atr_pct_runtime": atr_pct,
            }
            x_risk = pd.DataFrame([{c: risk_row.get(c, 0.0) for c in feat_cols}], columns=feat_cols).astype(np.float32)
            model = self.sidecar["model"][side]
            score = float(model.predict(x_risk)[0])
            z = float(np.clip((score - self.sidecar["train_score_q50"]) / max(self.sidecar["train_score_iqr"], 1e-8), -8.0, 8.0))
            m = self.sidecar["selected_mapping"]
            unit_m = 1.0 / (1.0 + np.exp(-m["temp"] * z))
            base_margin = base_notional / max(base_leverage, 1e-12)
            margin_fraction = float(np.clip(base_margin * (m["min_scale"] + (m["max_scale"] - m["min_scale"]) * unit_m), m["floor"], m["cap"]))
            margin_fraction *= float(m["long_scale"] if side > 0 else m["short_scale"])
            margin_fraction = float(np.clip(margin_fraction, m["floor"], m["cap"]))
            unit_l = 1.0 / (1.0 + np.exp(-m["leverage_temp"] * z))
            leverage = float(m["leverage_min"] + (m["leverage_max"] - m["leverage_min"]) * unit_l)
            leverage *= float(m["long_leverage_scale"] if side > 0 else m["short_leverage_scale"])
            leverage = float(np.clip(leverage, m["leverage_floor"], m["leverage_cap"]))

        return {
            "action": final_action, "side": side, "margin_fraction": margin_fraction, "leverage": leverage,
            "notional_exposure": margin_fraction * leverage, "take_profit": take_profit, "stop_loss": stop_loss,
        }


class Omega461DurationGateLiveAdapterDraft:
    """DRAFT. Not imported by trading_bot.py. See module docstring for scope."""

    def __init__(self, *, h48qual_bundle: str | Path, h48qual_sidecar: str | Path,
                 zig075_bundle: str | Path, zig075_sidecar: str | Path,
                 current_regime_path: str | Path = _DEFAULT_CURRENT_REGIME_PATH, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.components = {
            "h48qual": _Component(_ComponentConfig("h48qual", Path(h48qual_bundle), Path(h48qual_sidecar), quality_threshold=0.50), device=self.device),
            "zig075": _Component(_ComponentConfig("zig075", Path(zig075_bundle), Path(zig075_sidecar), quality_threshold=0.75), device=self.device),
        }
        # NOTE (2026-07-06 live-path parity finding): trading_bot.py's shared processed_df/
        # FeatureEngineer pipeline does NOT populate regime3_current_sensitive_wide24_* columns --
        # every Omega-family adapter computes this itself (see
        # omega4_6_2_source_parent_live.py::Regime3CurrentLiveFeatures). The initial draft of this
        # adapter incorrectly assumed these columns were already present in `frame`; fixed here to
        # compute them live via the same causal HMM filter_proba() call the sibling adapter uses.
        self.regime3_current = _Regime3CurrentLiveFeatures(current_path=current_regime_path)

    def decide_latest(self, frame: pd.DataFrame) -> Omega461Decision:
        if frame.empty:
            raise RuntimeError("Omega4.6.1 draft adapter received empty frame")
        frame = self.regime3_current.append(frame)
        if DURATION_FEATURE not in frame.columns:
            raise RuntimeError(f"Omega4.6.1 draft adapter missing required feature: {DURATION_FEATURE}")
        halflife = float(frame[DURATION_FEATURE].iloc[-1])
        if not np.isfinite(halflife):
            raise RuntimeError("Omega4.6.1 draft adapter received non-finite ou_halflife")

        chosen = None
        source = ""
        for alias in PRIORITY:
            out = self.components[alias].decide(frame)
            if out["side"] != 0:
                chosen, source = out, alias
                break
        if chosen is None:
            chosen, source = {"action": 0, "side": 0, "margin_fraction": 0.0, "leverage": 0.0,
                               "notional_exposure": 0.0, "take_profit": 0.0, "stop_loss": 0.0}, "none"

        duration_hit = halflife <= DURATION_THRESHOLD
        notional = chosen["notional_exposure"]
        leverage = chosen["leverage"]
        margin = chosen["margin_fraction"]
        if chosen["side"] != 0 and source != "none":
            scale = SCALE_MAP.get(f"{source}_{'L' if chosen['side'] > 0 else 'S'}", 1.0)
            leverage = min(leverage * scale, LEVERAGE_CAP)
            notional = min(margin * leverage, NOTIONAL_CAP)
            leverage = notional / max(margin, 1e-12)
        if duration_hit:
            notional, leverage, margin = 0.0, 0.0, 0.0

        return Omega461Decision(
            action=chosen["action"] if not duration_hit else 0,
            side=chosen["side"] if not duration_hit else 0,
            source_component=source,
            margin_fraction=margin, leverage=leverage, notional_exposure=notional,
            take_profit=chosen["take_profit"], stop_loss=chosen["stop_loss"],
            duration_gate_hit=duration_hit,
            trace={"ou_halflife": halflife, "duration_threshold": DURATION_THRESHOLD, "pre_gate_side": chosen["side"]},
        )
