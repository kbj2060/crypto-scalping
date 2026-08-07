from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PureRLDecision:
    final_action: int
    kelly: float
    source: str


@dataclass(frozen=True)
class PureRLConfig:
    pos_th: float
    close_th: float
    flip_th: float
    flip_kelly_mult: float
    max_kelly: float
    force_close: bool

    @classmethod
    def from_env(cls) -> "PureRLConfig":
        pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.155"))
        return cls(
            pos_th=pos_th,
            close_th=float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.00")),
            flip_th=float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(pos_th))),
            flip_kelly_mult=float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "1.0")),
            max_kelly=float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0")),
            force_close=str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"},
        )


def decide_pure_rl_action(
    *,
    action_val: float,
    current_pos: str | None,
    live_unrealized_pnl: float,
    alpha_focus_enabled: bool,
    alpha_focus_row: dict[str, Any] | None,
    alpha_focus_regime: str,
    alpha_focus_sizing_fn: Callable[[int, float, dict[str, Any], str, float], dict[str, Any]] | None,
    alpha_focus_exposure_cap: float = 1.0,
    oos_parity_mode: bool = False,
    dsac_action: int = 0,
    dsac_lev: float = 0.0,
    source_pure: str = "DSAC_PURE_RL",
    source_parity: str = "DSAC_OOS_PARITY",
) -> PureRLDecision:
    cfg = PureRLConfig.from_env()
    abs_action = abs(float(action_val))
    final_action = 0
    kelly = 0.0
    source = source_pure

    if oos_parity_mode:
        return PureRLDecision(
            final_action=int(dsac_action),
            kelly=float(np.clip(dsac_lev, 0.0, 1.0)),
            source=source_parity,
        )

    if current_pos is None:
        if action_val > cfg.pos_th:
            final_action, kelly = 1, min(abs_action, cfg.max_kelly)
        elif action_val < -cfg.pos_th:
            final_action, kelly = 2, min(abs_action, cfg.max_kelly)
    elif current_pos == "LONG":
        if cfg.force_close and live_unrealized_pnl <= -0.025:
            final_action, kelly, source = 0, 0.0, f"{source_pure}_FORCE_CLOSE"
        elif abs_action < cfg.close_th:
            final_action, kelly = 0, 0.0
        elif action_val < -cfg.flip_th:
            final_action, kelly = 2, min(abs_action, cfg.max_kelly) * cfg.flip_kelly_mult
        else:
            final_action, kelly = 1, min(abs_action, cfg.max_kelly)
    else:
        if cfg.force_close and live_unrealized_pnl <= -0.025:
            final_action, kelly, source = 0, 0.0, f"{source_pure}_FORCE_CLOSE"
        elif abs_action < cfg.close_th:
            final_action, kelly = 0, 0.0
        elif action_val > cfg.flip_th:
            final_action, kelly = 1, min(abs_action, cfg.max_kelly) * cfg.flip_kelly_mult
        else:
            final_action, kelly = 2, min(abs_action, cfg.max_kelly)

    kelly = float(np.clip(kelly, 0.0, 1.0))

    if (
        alpha_focus_enabled
        and final_action in (1, 2)
        and alpha_focus_row is not None
        and alpha_focus_sizing_fn is not None
    ):
        af_profile = alpha_focus_sizing_fn(
            final_action,
            kelly,
            alpha_focus_row,
            alpha_focus_regime,
            alpha_focus_exposure_cap,
        )
        new_exposure = float(af_profile.get("target_exposure", kelly))
        if abs(new_exposure - kelly) > 1e-9:
            kelly = float(np.clip(new_exposure, 0.0, max(alpha_focus_exposure_cap, 1.0)))
            source = f"{source}|ALPHA_FOCUS_STRICT(tag={af_profile.get('tag', 'base')},lev=1.000)"

    return PureRLDecision(final_action=int(final_action), kelly=float(kelly), source=source)
