from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from features.integrated_overlay import (
    _align_score,
    _clip,
    _safe_float,
    build_micro_overlay_features,
    build_polymarket_overlay_features,
    build_tail_overlay_features,
)


@dataclass(frozen=True)
class NewsShockGuardConfig:
    shock_trigger_th: float = 0.86
    aftershock_trigger_th: float = 0.78
    toxicity_trigger_th: float = 0.72
    queue_trigger_th: float = 0.78
    poly_momentum_trigger: float = 0.010
    poly_gap_trigger: float = 0.018
    reduce_mult: float = 0.35
    severe_reduce_mult: float = 0.0
    cooldown_bars: int = 6
    severe_cooldown_bars: int = 10

    @property
    def name(self) -> str:
        return (
            f"st{self.shock_trigger_th:.2f}_at{self.aftershock_trigger_th:.2f}"
            f"_tx{self.toxicity_trigger_th:.2f}_rm{self.reduce_mult:.2f}"
        )


def compute_news_shock_guard(side: str, row, cfg: NewsShockGuardConfig | None = None) -> dict:
    cfg = cfg or NewsShockGuardConfig()
    s = str(side or "").upper()

    micro = build_micro_overlay_features(row)
    tail = build_tail_overlay_features(row)
    poly = build_polymarket_overlay_features(row)

    poly_align = _align_score(s, poly["direction_pressure"])
    gap = _safe_float(poly["gap"], 0.0)
    adverse_gap = (gap <= -cfg.poly_gap_trigger) if s == "LONG" else (gap >= cfg.poly_gap_trigger)
    adverse_momentum = _safe_float(poly["momentum_1m"], 0.0)
    adverse_momentum = (
        adverse_momentum <= -cfg.poly_momentum_trigger
        if s == "LONG"
        else adverse_momentum >= cfg.poly_momentum_trigger
    )

    shock_score = _clip(
        (0.34 * tail["aftershock"])
        + (0.20 * micro["toxicity"])
        + (0.14 * micro["queue_risk"])
        + (0.16 * min(abs(_safe_float(poly["momentum_1m"], 0.0)) / max(cfg.poly_momentum_trigger, 1e-6), 1.5))
        + (0.16 * min(abs(gap) / max(cfg.poly_gap_trigger, 1e-6), 1.5)),
        0.0,
        1.5,
    )

    reasons: list[str] = []
    if tail["aftershock"] >= cfg.aftershock_trigger_th:
        reasons.append("TAIL_AFTERSHOCK_SPIKE")
    if micro["toxicity"] >= cfg.toxicity_trigger_th:
        reasons.append("MS_TOXICITY_SPIKE")
    if micro["queue_risk"] >= cfg.queue_trigger_th:
        reasons.append("MS_QUEUE_COLLAPSE")
    if adverse_gap:
        reasons.append("POLY_GAP_SHOCK")
    if adverse_momentum:
        reasons.append("POLY_MOM_SHOCK")
    if poly_align < -0.45:
        reasons.append("POLY_DIRECTION_FLIP")

    severe = (
        shock_score >= (cfg.shock_trigger_th + 0.12)
        and tail["aftershock"] >= (cfg.aftershock_trigger_th + 0.08)
        and adverse_gap
        and adverse_momentum
    )
    trigger = (
        shock_score >= cfg.shock_trigger_th
        and tail["aftershock"] >= cfg.aftershock_trigger_th
        and (
            (micro["toxicity"] >= cfg.toxicity_trigger_th and micro["queue_risk"] >= cfg.queue_trigger_th)
            or (adverse_gap and adverse_momentum)
            or (adverse_gap and poly_align < -0.45)
        )
    )

    reduce_mult = 1.0
    if trigger:
        reduce_mult = float(cfg.severe_reduce_mult if severe else cfg.reduce_mult)

    cooldown_bars = 0
    if trigger:
        cooldown_bars = int(cfg.severe_cooldown_bars if severe else cfg.cooldown_bars)

    return {
        "config": asdict(cfg),
        "trigger": bool(trigger),
        "severe": bool(severe),
        "reduce_mult": float(reduce_mult),
        "cooldown_bars": int(cooldown_bars),
        "shock_score": float(shock_score),
        "micro": micro,
        "tail": tail,
        "poly": poly,
        "reasons": reasons,
    }
