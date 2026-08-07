"""Production live adapter for omega4_6_1_duration_ou_halflife_risk_gate_20260630 (base form,
event-flat overlay excluded -- that idea failed genuine fresh-forward testing, see
docs/model_contracts/omega4_6_1_event_flat_fresh_forward_correction_20260706.md).

Verification trail (2026-07-06): extended-OOS retest, Gate 2 artifact-integrity fix
(promotion_pass=true), runtime-native parity testing (found+fixed a sizing bug and a
greedy-vs-offline-reconciled routing gap, found+fixed a missing live regime3 feature
computation), and a lookahead/contamination/lag audit (clean). See
docs/model_contracts/omega4_6_1_live_path_parity_and_lookahead_audit_20260706.md and
docs/model_contracts/omega4_6_1_full_architecture_blueprint_20260706.md for full detail.

This module separates ENTRY decisions (decide_entry) from EXIT evaluation (evaluate_exit) because
trading_bot.py owns persistent position state (entry price, hold count, MFE/MAE) across the
position's lifetime, unlike the offline backtest replay which tracked this locally in one loop.
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
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as _omega_sol  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_20260708 as _omega_btc  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as _sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as _hard  # noqa: E402

# Exposed so trading_bot.py can pass each asset's OWN copy into Omega461LiveAdapter instead of
# silently defaulting to the ETH module above (found 2026-07-20, same failure class as the
# regime3 bug -- these per-asset files already existed but nothing referenced them live).
SOL_BASE_TEMPLATE = dict(_omega_sol.BASE_TEMPLATE)
SOL_EXPERT_SCALES = dict(_omega_sol.EXPERT_SCALES)
BTC_BASE_TEMPLATE = dict(_omega_btc.BASE_TEMPLATE)
BTC_EXPERT_SCALES = dict(_omega_btc.EXPERT_SCALES)
from trading_bot_modules.omega4_6_2_source_parent_live import (  # noqa: E402
    DEFAULT_CURRENT_REGIME_PATH as _DEFAULT_CURRENT_REGIME_PATH,
    Regime3CurrentLiveFeatures as _Regime3CurrentLiveFeatures,
)
from trading_bot_modules.omega4_6_1_runtime_contract import (
    strict_feature_values,
    validate_sidecar_lineage,
)

OMEGA4_6_1_MODEL_ID = "omega4_6_1_duration_ou_halflife_risk_gate_20260630"
OMEGA4_6_1_MODEL_VERSION = "Omega4.6.1-live-20260706"
OMEGA4_6_1_OWNER = "omega4_6_1"
DURATION_FEATURE = "ou_halflife"
DURATION_THRESHOLD = 0.005417  # VAL-reselected 2026-07-06; original frozen value was 0.005415348
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ("h48qual", "zig075")
EXIT_THRESHOLD = 0.95


@dataclass(frozen=True)
class Omega461EntryDecision:
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
    # SOL's live zig075 sidecar (sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720)
    # predates the validation_only-only Omega Artifact Integrity policy (2026-06-30) and was
    # selected+promoted live under 'validation_oos_guard' on 2026-07-20 (see
    # project-sol-adaptive-squeeze-v2-live-20260720 memory) -- allow it explicitly rather than
    # loosening the default for ETH/BTC's stricter, policy-compliant sidecars.
    allowed_selection_scopes: frozenset[str] = frozenset({"validation_only"})


class _Component:
    def __init__(self, cfg: _ComponentConfig, *, device: torch.device,
                 base_template: dict[str, float] | None = None,
                 expert_scales: dict[str, float] | None = None) -> None:
        if not cfg.bundle_path.exists():
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: missing parent bundle {cfg.bundle_path}")
        if not cfg.sidecar_path.exists():
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: missing risk sidecar {cfg.sidecar_path}")
        try:
            validate_sidecar_lineage(
                repo_root=ROOT,
                bundle_path=cfg.bundle_path,
                sidecar_path=cfg.sidecar_path,
                quality_threshold=cfg.quality_threshold,
                allowed_selection_scopes=cfg.allowed_selection_scopes,
            )
        except ValueError as exc:
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: invalid artifact lineage: {exc}") from exc
        self.cfg = cfg
        self.device = device
        # Defaults preserve the prior hardcoded-ETH-module behavior for any caller that doesn't
        # pass these explicitly (found 2026-07-20: same failure class as the regime3 bug).
        self.base_template = dict(base_template) if base_template is not None else dict(_omega.BASE_TEMPLATE)
        self.expert_scales = dict(expert_scales) if expert_scales is not None else dict(_omega.EXPERT_SCALES)
        bundle = torch.load(cfg.bundle_path, map_location="cpu", weights_only=False)
        self.base_cols: list[str] = list(bundle["base_cols"])
        if any(c.startswith(("m7_", "ai_", "patchtst", "tide_", "dlinear")) for c in self.base_cols):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: unexpected m7/NF feature dependency (contract drift)")
        models = dict(bundle["models"])
        missing_experts = sorted(set(_hard.EXPERT_NAMES) - set(models))
        if missing_experts:
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: bundle missing experts {missing_experts}")
        self.models = models
        # keep loaded (model, scaler) pairs for exit-head evaluation
        self.loaded = {expert: (self._build_model(payload), payload["scaler"]) for expert, payload in models.items()}
        with open(cfg.sidecar_path, "rb") as f:
            pkl = pickle.load(f)
        if pkl.get("risk_feature_mode") != "parent_outputs":
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar risk_feature_mode contract mismatch")
        if not pkl.get("side_split_model"):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar must be side-split")
        if not pkl.get("dynamic_leverage"):
            raise RuntimeError(f"Omega4.6.1 {cfg.alias}: sidecar must use dynamic leverage")
        self.sidecar = pkl

    def _build_model(self, payload: dict[str, Any]) -> _parent.ThreeHeadTabM:
        cfg = _parent.ThreeHeadConfig(**dict(payload["config"]))
        model = _parent.ThreeHeadTabM(int(payload["n_features"]), cfg=cfg).to(self.device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def entry_decision(self, frame: pd.DataFrame) -> dict[str, Any]:
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
            expert_scale_key = "chop_expert" if expert == "chop" else expert
            expert_scale = float(self.expert_scales.get(expert_scale_key, 1.0))
            base_notional = float(self.base_template["notional"]) * expert_scale
            base_leverage = float(self.base_template["leverage"])
            router_probs = frame[_hard.ROUTE_COLS].iloc[-1].to_numpy(dtype=np.float64)
            router_expert_onehot = {f"parent_router_expert_{e}": float(expert == e) for e in _hard.EXPERT_NAMES}
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
            "side": side, "margin_fraction": margin_fraction, "leverage": leverage,
            "notional_exposure": margin_fraction * leverage, "take_profit": take_profit, "stop_loss": stop_loss,
            "quality_score": qual_for_action, "confidence": float(direction.max()), "expert": expert,
        }

    @torch.no_grad()
    def exit_probability(self, frame: pd.DataFrame, *, side: int, hold_bars: int, unrealized_move: float,
                          mfe: float, mae: float, notional: float, leverage: float,
                          take_profit: float, stop_loss: float) -> float:
        missing = [c for c in self.base_cols if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Omega4.6.1 {self.cfg.alias}: missing input features for exit eval {missing[:20]}")
        x_all = frame.reindex(columns=self.base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x_all.isna().any().any():
            raise RuntimeError(f"Omega4.6.1 {self.cfg.alias}: non-finite input features for exit eval")
        route_val = _hard._route_id(frame.iloc[[-1]].reset_index(drop=True))[0]
        expert = _hard.EXPERT_NAMES[int(route_val)]
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
        for c in _parent.POS_COLS:
            row[c] = pos_values.get(c, 0.0)
        x_np = row[list(scaler["columns"])].to_numpy(dtype=np.float32)
        xz = (x_np - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
        probs = torch.softmax(model(torch.from_numpy(xz).to(self.device))["exit"], dim=-1).mean(dim=1)
        return float(probs.detach().cpu().numpy()[0, 1])


class Omega461LiveAdapter:
    """Production Omega4.6.1 adapter. See module docstring for verification trail."""

    def __init__(self, *, h48qual_bundle: str | Path, h48qual_sidecar: str | Path,
                 zig075_bundle: str | Path, zig075_sidecar: str | Path,
                 current_regime_path: str | Path = _DEFAULT_CURRENT_REGIME_PATH, device: str = "cpu",
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
        # Defaults preserve the prior hardcoded-ETH-module behavior exactly for any caller that
        # doesn't pass these explicitly. SOL/BTC now pass their own copies (found 2026-07-20: same
        # failure class as the regime3 bug -- per-asset files already existed, live path didn't
        # reference them; today's values are identical across ETH/SOL/BTC so this was latent, not
        # yet manifesting, but would silently mis-size entries the moment one asset's copy is
        # retuned independently).
        self.base_template = dict(base_template) if base_template is not None else dict(_omega.BASE_TEMPLATE)
        self.expert_scales = dict(expert_scales) if expert_scales is not None else dict(_omega.EXPERT_SCALES)
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
                raise RuntimeError(f"Omega4.6.1 adapter priority references missing components: {missing}")
        # trading_bot.py's shared FeatureEngineer pipeline does NOT compute
        # regime3_current_sensitive_wide24_* -- every Omega-family adapter must compute it itself
        # (found 2026-07-06; see omega4_6_1_live_path_parity_and_lookahead_audit_20260706.md).
        self.regime3_current = _Regime3CurrentLiveFeatures(current_path=current_regime_path)

    def _with_regime3(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise RuntimeError("Omega4.6.1 adapter received empty frame")
        return self.regime3_current.append(frame)

    def decide_entry(self, frame: pd.DataFrame) -> Omega461EntryDecision | None:
        frame = self._with_regime3(frame)
        if DURATION_FEATURE not in frame.columns:
            raise RuntimeError(f"Omega4.6.1 adapter missing required feature: {DURATION_FEATURE}")
        halflife = float(frame[DURATION_FEATURE].iloc[-1])
        if not np.isfinite(halflife):
            raise RuntimeError("Omega4.6.1 adapter received non-finite ou_halflife")
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
            return Omega461EntryDecision(
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
        """Bar-by-bar exit check for an already-open position. TP/SL first (raw price-move
        barriers, independent of notional per the Futures Risk Sizing Contract), then the
        ORIGINATING component's learned exit head. Returns (should_exit, reason, exit_prob).

        bar_high_move/bar_low_move are the best/worst price-move the *already-completed* bar
        touched intraday (e.g. move-to-high for a LONG). They're optional and default to the
        close-only unrealized_move for backward compatibility. Passing them lets TP/SL barriers
        trigger on an intrabar touch of the completed bar rather than requiring its close to also
        clear the threshold -- this mirrors a real resting TP/SL order, which fills the instant
        price touches it, not only if the candle happens to close beyond it. This does not add
        lookahead: both bars are already fully closed/confirmed by the time this runs, and the
        fill itself still executes at the next bar's open per the existing execution-delay model.
        Stop-loss is checked before take-profit since intrabar ordering within the bar is unknown;
        assuming the adverse touch first is the conservative choice.
        """
        frame = self._with_regime3(frame)
        tp_move = unrealized_move if bar_high_move is None else bar_high_move
        sl_move = unrealized_move if bar_low_move is None else bar_low_move
        if stop_loss > 0.0 and sl_move <= -abs(stop_loss):
            return True, "stop_loss", 0.0
        if take_profit > 0.0 and tp_move >= take_profit:
            return True, "take_profit", 0.0
        if source_component not in self.components:
            raise RuntimeError(f"Omega4.6.1 evaluate_exit: unknown source_component {source_component!r}")
        prob = self.components[source_component].exit_probability(
            frame, side=side, hold_bars=hold_bars, unrealized_move=unrealized_move, mfe=mfe, mae=mae,
            notional=notional, leverage=leverage, take_profit=take_profit, stop_loss=stop_loss,
        )
        if prob >= EXIT_THRESHOLD:
            return True, "exit_head", prob
        return False, "hold", prob
