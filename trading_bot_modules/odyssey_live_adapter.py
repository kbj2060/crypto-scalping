"""Odyssey ETH live decision adapter -- entry/exit decisions for the h48qual + zig075 3-Head TabM
components, rebuilt as a standalone module for the Odyssey shadow lineage.

This is a cleanroom reimplementation of `trading_bot_modules/omega4_6_1_live.py`'s
`Omega461LiveAdapter`/`_Component`/`_ComponentConfig`, scoped to exactly what Odyssey's ETH shadow
uses. It intentionally does NOT import `omega4_6_1_live.py` (shared with real ETH live order
execution in `trading_bot.py` -- must never be modified for a shadow-only rewrite),
`omega4_6_2_source_parent_live.py` (an Omega5-oriented adapter, only 2 symbols of which Odyssey
needs -- see `odyssey_regime3_live.py`), or `trading_bot_modules/runtime_config.py` (pulls in
`omega5_live.py` and raises at import time on unrelated Omega5 env-var mismatches -- a real crash
risk for a module that has nothing to do with Omega5). See docs/experiments/eth_odyssey_live_
cleanroom_dependency_rewrite_20260816.md for the traced dependency graph that motivated this file.

The one piece of the original that IS reused rather than duplicated is `trading_bot_modules.
omega4_6_1_runtime_contract` (`strict_feature_values`/`validate_sidecar_lineage`): it's a
dependency-free, general-purpose safety utility (the Omega Artifact Integrity Gate check), and
duplicating safety-critical validation logic risks the two copies silently drifting apart.

Two behavior fixes vs. the original (both already verified as behavioral no-ops against the two
currently-deployed bundles -- see the parity checks in the dependency-rewrite doc):
  1. `_Component` builds and caches each expert's model ONCE at construction (via
     `odyssey_tabm_core.build_model`, which always reconstructs `ThreeHeadConfig` from the
     bundle's own recorded config) and reuses it for both entry decisions and exit probability.
     The original rebuilt a fresh, uncached model from a module-global config singleton on every
     single entry decision.
  2. The `"chop"` -> `"chop_expert"` EXPERT_SCALES key remap is centralized in
     `odyssey_tabm_core.resolve_expert_scale_key` instead of being re-derived ad hoc at each call
     site.

Duration-gate (L4.5 OU-halflife) and SCALE_MAP/LEVERAGE_CAP/NOTIONAL_CAP values are carried over
unchanged -- this is a shadow bot whose entire purpose is a valid comparison against the real live
decision path, so those must stay byte-identical, not "cleaned up".
"""
from __future__ import annotations

import os
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

from trading_bot_modules import odyssey_tabm_core as core
from trading_bot_modules.odyssey_regime3_live import DEFAULT_CURRENT_REGIME_PATH, Regime3CurrentLiveFeatures
from trading_bot_modules.omega4_6_1_runtime_contract import strict_feature_values, validate_sidecar_lineage

MODEL_ID = "odyssey_live_cleanroom_20260816"
DURATION_FEATURE = "ou_halflife"
DURATION_THRESHOLD = 0.005417
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ("h48qual", "zig075")
EXIT_THRESHOLD = 0.95

# Odyssey-owned artifact path constants -- same literal defaults as trading_bot_modules/
# runtime_config.py's FINAL_GOVERNOR_OMEGA4_6_1_* constants, but under Odyssey-scoped env var
# names (deliberately NOT reusing the FINAL_GOVERNOR_* names) so configuring the real live system
# can never accidentally reconfigure this shadow, or vice versa.
ODYSSEY_H48QUAL_BUNDLE_PATH = os.getenv(
    "ODYSSEY_H48QUAL_BUNDLE_PATH",
    "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
)
ODYSSEY_H48QUAL_SIDECAR_PATH = os.getenv(
    "ODYSSEY_H48QUAL_SIDECAR_PATH",
    "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl",
)
ODYSSEY_ZIG075_BUNDLE_PATH = os.getenv(
    "ODYSSEY_ZIG075_BUNDLE_PATH",
    "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
)
ODYSSEY_ZIG075_SIDECAR_PATH = os.getenv(
    "ODYSSEY_ZIG075_SIDECAR_PATH",
    "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl",
)


@dataclass(frozen=True)
class OdysseyEntryDecision:
    side: int  # 1=LONG, -1=SHORT
    source_component: str
    margin_fraction: float
    leverage: float
    notional_exposure: float
    take_profit: float
    stop_loss: float
    quality_score: float
    confidence: float
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
    exit_threshold: float = EXIT_THRESHOLD
    allowed_selection_scopes: frozenset[str] = frozenset({"validation_only"})


class _Component:
    def __init__(self, cfg: _ComponentConfig, *, device: torch.device,
                 base_template: dict[str, float] | None = None,
                 expert_scales: dict[str, float] | None = None) -> None:
        if not cfg.bundle_path.exists():
            raise RuntimeError(f"Odyssey {cfg.alias}: missing parent bundle {cfg.bundle_path}")
        if not cfg.sidecar_path.exists():
            raise RuntimeError(f"Odyssey {cfg.alias}: missing risk sidecar {cfg.sidecar_path}")
        try:
            validate_sidecar_lineage(
                repo_root=ROOT,
                bundle_path=cfg.bundle_path,
                sidecar_path=cfg.sidecar_path,
                quality_threshold=cfg.quality_threshold,
                allowed_selection_scopes=cfg.allowed_selection_scopes,
            )
        except ValueError as exc:
            raise RuntimeError(f"Odyssey {cfg.alias}: invalid artifact lineage: {exc}") from exc
        self.cfg = cfg
        self.device = device
        self.base_template = dict(base_template) if base_template is not None else dict(core.BASE_TEMPLATE)
        self.expert_scales = dict(expert_scales) if expert_scales is not None else dict(core.EXPERT_SCALES)
        bundle = torch.load(cfg.bundle_path, map_location="cpu", weights_only=False)
        self.base_cols: list[str] = list(bundle["base_cols"])
        if any(c.startswith(("m7_", "ai_", "patchtst", "tide_", "dlinear")) for c in self.base_cols):
            raise RuntimeError(f"Odyssey {cfg.alias}: unexpected m7/NF feature dependency (contract drift)")
        models = dict(bundle["models"])
        missing_experts = sorted(set(core.EXPERT_NAMES) - set(models))
        if missing_experts:
            raise RuntimeError(f"Odyssey {cfg.alias}: bundle missing experts {missing_experts}")
        # Built + cached once here (fix #1, see module docstring) -- used for BOTH entry decisions
        # and exit-probability evaluation, unlike the original which only cached for exit and
        # rebuilt a fresh model on every entry decision.
        self.loaded: dict[str, tuple[Any, dict[str, Any]]] = {
            expert: (core.build_model(payload, device=device), payload["scaler"]) for expert, payload in models.items()
        }
        with open(cfg.sidecar_path, "rb") as f:
            pkl = pickle.load(f)
        if pkl.get("risk_feature_mode") != "parent_outputs":
            raise RuntimeError(f"Odyssey {cfg.alias}: sidecar risk_feature_mode contract mismatch")
        if not pkl.get("side_split_model"):
            raise RuntimeError(f"Odyssey {cfg.alias}: sidecar must be side-split")
        if not pkl.get("dynamic_leverage"):
            raise RuntimeError(f"Odyssey {cfg.alias}: sidecar must use dynamic leverage")
        self.sidecar = pkl

    def entry_decision(self, frame: pd.DataFrame) -> dict[str, Any]:
        missing = [c for c in self.base_cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Odyssey {self.cfg.alias}: missing input features {missing[:20]}")
        x_all = frame.reindex(columns=self.base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x_all.isna().any().any():
            raise RuntimeError(f"Odyssey {self.cfg.alias}: non-finite input features in history window")
        x_row = x_all.iloc[[-1]].reset_index(drop=True)
        for c in core.POS_COLS:
            x_row[c] = 0.0

        route_val = core._route_id(frame.iloc[[-1]].reset_index(drop=True))[0]
        expert = core.EXPERT_NAMES[int(route_val)]
        model, scaler = self.loaded[expert]
        pred = core.predict_proba(model, x_row, scaler, device=self.device)
        direction = pred["direction"][0]
        quality = pred["quality"][0]
        action = int(np.argmax(direction))
        qual_for_action = float(quality[action]) if action > 0 else float(quality[0])
        final_action = action if (action != 0 and qual_for_action >= self.cfg.quality_threshold) else 0
        side = 1 if final_action == 1 else (-1 if final_action == 2 else 0)

        atr_pct = float(core._atr_pct(frame, self.cfg.atr_window)[-1])
        if side != 0 and not np.isfinite(atr_pct):
            raise RuntimeError(f"Odyssey {self.cfg.alias}: non-finite ATR for active decision")
        take_profit = float(np.clip(max(self.cfg.min_tp, atr_pct * self.cfg.tp_mult), 0.0, self.cfg.max_tp)) if side != 0 else 0.0
        stop_loss = float(np.clip(max(self.cfg.min_sl, atr_pct * self.cfg.sl_mult), 0.0, self.cfg.max_sl)) if side != 0 else 0.0

        margin_fraction = leverage = 0.0
        if side != 0:
            expert_scale_key = core.resolve_expert_scale_key(expert)
            expert_scale = float(self.expert_scales.get(expert_scale_key, 1.0))
            base_notional = float(self.base_template["notional"]) * expert_scale
            base_leverage = float(self.base_template["leverage"])
            router_probs = frame[core.ROUTE_COLS].iloc[-1].to_numpy(dtype=np.float64)
            router_expert_onehot = {f"parent_router_expert_{e}": float(expert == e) for e in core.EXPERT_NAMES}
            feat_cols = self.sidecar["feature_columns"]
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
            risk_values = strict_feature_values(feat_cols, risk_row)
            x_risk = pd.DataFrame([risk_values], columns=feat_cols, dtype=np.float32)
            risk_model = self.sidecar["model"][side]
            score = float(risk_model.predict(x_risk)[0])
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
            "side": side, "margin_fraction": margin_fraction, "leverage": leverage,
            "notional_exposure": margin_fraction * leverage, "take_profit": take_profit, "stop_loss": stop_loss,
            "quality_score": qual_for_action, "confidence": float(direction.max()), "expert": expert,
        }

    def exit_probability(self, frame: pd.DataFrame, *, side: int, hold_bars: int, unrealized_move: float,
                          mfe: float, mae: float, notional: float, leverage: float,
                          take_profit: float, stop_loss: float) -> float:
        missing = [c for c in self.base_cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Odyssey {self.cfg.alias}: missing input features for exit eval {missing[:20]}")
        x_all = frame.reindex(columns=self.base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x_all.isna().any().any():
            raise RuntimeError(f"Odyssey {self.cfg.alias}: non-finite input features for exit eval")
        route_val = core._route_id(frame.iloc[[-1]].reset_index(drop=True))[0]
        expert = core.EXPERT_NAMES[int(route_val)]
        model, scaler = self.loaded[expert]
        giveback = (mfe - unrealized_move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
        pos_values = {
            "pos_side": float(side), "pos_hold_bars": float(hold_bars), "pos_unrealized": float(unrealized_move),
            "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
            "pos_dist_to_tp": float(take_profit - unrealized_move), "pos_dist_to_sl": float(unrealized_move + abs(stop_loss)),
            "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
            "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
        }
        row = x_all.iloc[[-1]].reset_index(drop=True).copy()
        for c in core.POS_COLS:
            row[c] = pos_values.get(c, 0.0)
        x_np = row[list(scaler["columns"])].to_numpy(dtype=np.float32)
        xz = (x_np - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(xz).to(self.device))["exit"], dim=-1).mean(dim=1)
        return float(probs.detach().cpu().numpy()[0, 1])


class OdysseyLiveAdapter:
    """Cleanroom Odyssey ETH live adapter -- see module docstring."""

    def __init__(self, *, h48qual_bundle: str | Path = "", h48qual_sidecar: str | Path = "",
                 zig075_bundle: str | Path = "", zig075_sidecar: str | Path = "",
                 current_regime_path: str | Path = DEFAULT_CURRENT_REGIME_PATH, device: str = "cpu",
                 components_override: dict[str, dict[str, Any]] | None = None,
                 priority: tuple[str, ...] | None = None,
                 duration_threshold: float = DURATION_THRESHOLD,
                 scale_map: dict[str, float] | None = None,
                 base_template: dict[str, float] | None = None,
                 expert_scales: dict[str, float] | None = None) -> None:
        self.device = torch.device(device)
        self.priority = tuple(priority or PRIORITY)
        self.duration_threshold = float(duration_threshold)
        self.scale_map = dict(scale_map or SCALE_MAP)
        self.base_template = dict(base_template) if base_template is not None else dict(core.BASE_TEMPLATE)
        self.expert_scales = dict(expert_scales) if expert_scales is not None else dict(core.EXPERT_SCALES)
        if components_override is None:
            self.components = {
                "h48qual": _Component(_ComponentConfig("h48qual", Path(h48qual_bundle), Path(h48qual_sidecar), quality_threshold=0.50), device=self.device, base_template=self.base_template, expert_scales=self.expert_scales),
                "zig075": _Component(_ComponentConfig("zig075", Path(zig075_bundle), Path(zig075_sidecar), quality_threshold=0.75), device=self.device, base_template=self.base_template, expert_scales=self.expert_scales),
            }
        else:
            self.components = {}
            for alias, cfg in components_override.items():
                _scopes = cfg.get("allowed_selection_scopes")
                self.components[str(alias)] = _Component(
                    _ComponentConfig(
                        str(alias),
                        Path(cfg["bundle"]),
                        Path(cfg["sidecar"]),
                        quality_threshold=float(cfg["quality_threshold"]),
                        **({"allowed_selection_scopes": frozenset(_scopes)} if _scopes else {}),
                    ),
                    device=self.device,
                    base_template=self.base_template,
                    expert_scales=self.expert_scales,
                )
            missing = [alias for alias in self.priority if alias not in self.components]
            if missing:
                raise RuntimeError(f"Odyssey adapter priority references missing components: {missing}")
        self.regime3_current = Regime3CurrentLiveFeatures(current_path=current_regime_path)

    def _with_regime3(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise RuntimeError("Odyssey adapter received empty frame")
        return self.regime3_current.append(frame)

    def decide_entry(self, frame: pd.DataFrame) -> OdysseyEntryDecision | None:
        frame = self._with_regime3(frame)
        if DURATION_FEATURE not in frame.columns:
            raise RuntimeError(f"Odyssey adapter missing required feature: {DURATION_FEATURE}")
        halflife = float(frame[DURATION_FEATURE].iloc[-1])
        if not np.isfinite(halflife):
            raise RuntimeError("Odyssey adapter received non-finite ou_halflife")
        duration_hit = halflife <= self.duration_threshold

        for alias in self.priority:
            out = self.components[alias].entry_decision(frame)
            if out["side"] == 0:
                continue
            if duration_hit:
                return None
            side = out["side"]
            scale = self.scale_map.get(f"{alias}_{'L' if side > 0 else 'S'}", 1.0)
            leverage = min(out["leverage"] * scale, LEVERAGE_CAP)
            notional = min(out["margin_fraction"] * leverage, NOTIONAL_CAP)
            leverage = notional / max(out["margin_fraction"], 1e-12)
            if notional <= 0.0:
                continue
            return OdysseyEntryDecision(
                side=side, source_component=alias, margin_fraction=out["margin_fraction"], leverage=leverage,
                notional_exposure=notional, take_profit=out["take_profit"], stop_loss=out["stop_loss"],
                quality_score=out["quality_score"], confidence=out["confidence"],
                trace={"ou_halflife": halflife, "duration_threshold": self.duration_threshold, "expert": out["expert"]},
            )
        return None

    def evaluate_exit(self, frame: pd.DataFrame, *, source_component: str, side: int, hold_bars: int,
                       unrealized_move: float, mfe: float, mae: float, notional: float, leverage: float,
                       take_profit: float, stop_loss: float,
                       bar_high_move: float | None = None, bar_low_move: float | None = None) -> tuple[bool, str, float]:
        frame = self._with_regime3(frame)
        tp_move = unrealized_move if bar_high_move is None else bar_high_move
        sl_move = unrealized_move if bar_low_move is None else bar_low_move
        if stop_loss > 0.0 and sl_move <= -abs(stop_loss):
            return True, "stop_loss", 0.0
        if take_profit > 0.0 and tp_move >= take_profit:
            return True, "take_profit", 0.0
        if source_component not in self.components:
            raise RuntimeError(f"Odyssey evaluate_exit: unknown source_component {source_component!r}")
        prob = self.components[source_component].exit_probability(
            frame, side=side, hold_bars=hold_bars, unrealized_move=unrealized_move, mfe=mfe, mae=mae,
            notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss,
        )
        if prob >= EXIT_THRESHOLD:
            return True, "exit_head", prob
        return False, "hold", prob
