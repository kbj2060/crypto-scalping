from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _align_score(side: str, value: float) -> float:
    s = str(side or "").upper()
    x = _safe_float(value, 0.0)
    if s == "LONG":
        return x
    if s == "SHORT":
        return -x
    return 0.0


@dataclass(frozen=True)
class IntegratedOverlayConfig:
    entry_score_th: float = -0.35
    risk_block_th: float = 0.92
    risk_exit_th: float = 0.88
    micro_toxicity_block_th: float = 1.10
    tail_aftershock_block_th: float = 0.90
    tail_aftershock_exit_th: float = 0.82
    poly_adverse_gap_th: float = 0.0035
    poly_severe_gap_th: float = 0.0060
    poly_conf_low_th: float = 0.08
    max_size_mult: float = 1.15
    min_size_mult: float = 0.50
    cooldown_med_bars: int = 3
    cooldown_high_bars: int = 6

    @property
    def name(self) -> str:
        return (
            f"es{self.entry_score_th:+.2f}_rb{self.risk_block_th:.2f}"
            f"_re{self.risk_exit_th:.2f}_pg{self.poly_adverse_gap_th:.4f}"
            f"_mx{self.max_size_mult:.2f}_mn{self.min_size_mult:.2f}"
        )


def build_micro_overlay_features(row) -> dict[str, float]:
    signal_bias = _clip(_safe_float(getattr(row, "signal_bias", row.get("signal_bias", 0.0) if hasattr(row, "get") else 0.0), 0.0), -1.0, 1.0)
    nif_whale = _clip(_safe_float(getattr(row, "nif_whale", row.get("nif_whale", 0.0) if hasattr(row, "get") else 0.0), 0.0), -1.0, 1.0)
    taker_buy_ratio = _clip(_safe_float(getattr(row, "taker_buy_ratio", row.get("taker_buy_ratio", 0.5) if hasattr(row, "get") else 0.5), 0.5), 0.0, 1.0)
    toxicity = _clip(_safe_float(getattr(row, "shadow_toxicity_score", row.get("shadow_toxicity_score", 0.0) if hasattr(row, "get") else 0.0), 0.0) / 1.2, 0.0, 1.2)
    queue_collapse = _clip(_safe_float(getattr(row, "shadow_queue_collapse", row.get("shadow_queue_collapse", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    absorption = _clip(_safe_float(getattr(row, "shadow_absorption_score", row.get("shadow_absorption_score", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    regime_conf = _clip(_safe_float(getattr(row, "shadow_regime_conf", row.get("shadow_regime_conf", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    flow_align = _clip(
        0.45 * signal_bias + 0.35 * nif_whale + 0.20 * ((2.0 * taker_buy_ratio) - 1.0),
        -1.0,
        1.0,
    )
    queue_risk = _clip((0.6 * queue_collapse) + (0.4 * (1.0 - absorption)), 0.0, 1.0)
    return {
        "flow_align": float(flow_align),
        "toxicity": float(toxicity),
        "queue_risk": float(queue_risk),
        "regime_conf": float(regime_conf),
    }


def build_tail_overlay_features(row) -> dict[str, float]:
    aftershock = _clip(_safe_float(getattr(row, "shadow_aftershock_prob", row.get("shadow_aftershock_prob", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    half_life = _clip(_safe_float(getattr(row, "shadow_decay_half_life", row.get("shadow_decay_half_life", 0.0) if hasattr(row, "get") else 0.0), 0.0) / 10.0, 0.0, 1.0)
    bucket = str(getattr(row, "shadow_risk_bucket", row.get("shadow_risk_bucket", "normal") if hasattr(row, "get") else "normal") or "normal").lower()
    bucket_risk = 0.0 if bucket == "normal" else 0.5 if bucket in {"watch", "elevated"} else 1.0
    risk_score = _clip((0.65 * aftershock) + (0.20 * half_life) + (0.15 * bucket_risk), 0.0, 1.0)
    return {
        "aftershock": float(aftershock),
        "decay_pressure": float(half_life),
        "bucket_risk": float(bucket_risk),
        "risk_score": float(risk_score),
    }


def build_polymarket_overlay_features(row) -> dict[str, float]:
    gap = _safe_float(getattr(row, "target_gap", row.get("target_gap", 0.0) if hasattr(row, "get") else 0.0), 0.0)
    mom_1m = _safe_float(getattr(row, "target_gap_delta_1m", row.get("target_gap_delta_1m", 0.0) if hasattr(row, "get") else 0.0), 0.0)
    mom_mode = _safe_float(getattr(row, "prob_mom_1m", row.get("prob_mom_1m", 0.0) if hasattr(row, "get") else 0.0), 0.0)
    mode_prob = _clip(_safe_float(getattr(row, "mode_prob", row.get("mode_prob", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    spread = _clip(_safe_float(getattr(row, "mode_spread", row.get("mode_spread", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    entropy = _clip(_safe_float(getattr(row, "entropy", row.get("entropy", 1.0) if hasattr(row, "get") else 1.0), 1.0), 0.0, 1.0)
    tail_up = _clip(_safe_float(getattr(row, "tail_up_prob", row.get("tail_up_prob", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    tail_down = _clip(_safe_float(getattr(row, "tail_down_prob", row.get("tail_down_prob", 0.0) if hasattr(row, "get") else 0.0), 0.0), 0.0, 1.0)
    confidence = _clip(spread * (1.0 - entropy), 0.0, 1.0)
    uncertainty = _clip((0.65 * entropy) + (0.35 * (1.0 - spread)), 0.0, 1.0)
    tail_bias = _clip(tail_up - tail_down, -1.0, 1.0)
    direction_pressure = _clip(np.tanh(gap / 0.01) + 0.5 * np.tanh(mom_1m / 0.003) + 0.25 * np.tanh(mom_mode / 0.03), -1.0, 1.0)
    return {
        "gap": float(gap),
        "momentum_1m": float(mom_1m),
        "mode_momentum_1m": float(mom_mode),
        "mode_prob": float(mode_prob),
        "spread": float(spread),
        "entropy": float(entropy),
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "tail_bias": float(tail_bias),
        "direction_pressure": float(direction_pressure),
    }


def compute_integrated_overlay(
    side: str,
    row,
    cfg: IntegratedOverlayConfig | None = None,
    dsac_strength: float = 1.0,
) -> dict:
    cfg = cfg or IntegratedOverlayConfig()
    s = str(side or "").upper()
    dsac_strength = _clip(_safe_float(dsac_strength, 1.0), 0.0, 1.0)

    micro = build_micro_overlay_features(row)
    tail = build_tail_overlay_features(row)
    poly = build_polymarket_overlay_features(row)

    micro_align = _align_score(s, micro["flow_align"])
    poly_align = _align_score(s, poly["direction_pressure"])
    tail_bias_align = _align_score(s, poly["tail_bias"])
    adverse_poly = (_safe_float(poly["gap"], 0.0) <= -cfg.poly_adverse_gap_th) if s == "LONG" else (_safe_float(poly["gap"], 0.0) >= cfg.poly_adverse_gap_th)
    severe_poly = (_safe_float(poly["gap"], 0.0) <= -cfg.poly_severe_gap_th) if s == "LONG" else (_safe_float(poly["gap"], 0.0) >= cfg.poly_severe_gap_th)

    entry_score = (
        0.50 * micro_align
        + 0.35 * poly_align
        + 0.10 * tail_bias_align
        - 0.15 * micro["toxicity"]
    ) * (0.75 + 0.25 * dsac_strength)
    confidence_score = _clip(
        (0.55 * poly["confidence"]) + (0.25 * micro["regime_conf"]) + (0.20 * abs(micro["flow_align"])),
        0.0,
        1.0,
    )
    risk_score = _clip(
        (0.50 * tail["risk_score"])
        + (0.25 * micro["toxicity"])
        + (0.15 * micro["queue_risk"])
        + (0.10 * poly["uncertainty"]),
        0.0,
        1.0,
    )

    reasons: list[str] = []
    if micro_align > 0.10:
        reasons.append("MS_ALIGN")
    elif micro_align < -0.10:
        reasons.append("MS_ADVERSE")
    if poly_align > 0.10:
        reasons.append("POLY_SUPPORT")
    elif poly_align < -0.10:
        reasons.append("POLY_ADVERSE")
    if tail["aftershock"] >= cfg.tail_aftershock_block_th:
        reasons.append("TAIL_AFTERSHOCK")
    if micro["toxicity"] >= cfg.micro_toxicity_block_th:
        reasons.append("MS_TOXIC")
    if poly["confidence"] < cfg.poly_conf_low_th:
        reasons.append("POLY_LOW_CONF")

    # First rollout is intentionally size-first. Entry veto is reserved for
    # extreme risk states so the overlay does not erase the base strategy.
    allow_entry = True
    if (
        risk_score >= cfg.risk_block_th
        and tail["aftershock"] >= cfg.tail_aftershock_block_th
        and severe_poly
    ):
        allow_entry = False
    if (
        entry_score <= cfg.entry_score_th
        and micro["toxicity"] >= cfg.micro_toxicity_block_th
        and adverse_poly
    ):
        allow_entry = False

    raw_mult = (
        1.00
        + 0.07 * max(-entry_score, 0.0)
        + 0.03 * max(poly["confidence"] - cfg.poly_conf_low_th, 0.0) * (1.0 if entry_score < -0.05 else 0.0)
        - 0.03 * max(risk_score - 0.60, 0.0)
    )
    if micro_align < -0.15 and poly_align < -0.05 and risk_score < 0.55:
        raw_mult += 0.04
    elif micro_align > 0.20 and poly_align > 0.10 and risk_score < 0.35:
        raw_mult += 0.02
    if adverse_poly and risk_score >= 0.70:
        raw_mult -= 0.02
    size_mult = _clip(raw_mult, cfg.min_size_mult, cfg.max_size_mult)

    exit_now = False
    if tail["aftershock"] >= cfg.tail_aftershock_exit_th:
        exit_now = True
    if risk_score >= cfg.risk_exit_th and severe_poly and micro_align < -0.05:
        exit_now = True
    if severe_poly and micro["toxicity"] >= 0.70 and micro["queue_risk"] >= 0.70:
        exit_now = True

    cooldown_bars = 0
    if tail["aftershock"] >= cfg.tail_aftershock_exit_th or risk_score >= cfg.risk_exit_th:
        cooldown_bars = int(cfg.cooldown_high_bars)
    elif tail["aftershock"] >= cfg.tail_aftershock_block_th or risk_score >= cfg.risk_block_th:
        cooldown_bars = int(cfg.cooldown_med_bars)

    return {
        "config": asdict(cfg),
        "micro": micro,
        "tail": tail,
        "poly": poly,
        "allow_entry": bool(allow_entry),
        "size_mult": float(size_mult),
        "exit_now": bool(exit_now),
        "cooldown_bars": int(cooldown_bars),
        "entry_score": float(entry_score),
        "risk_score": float(risk_score),
        "confidence_score": float(confidence_score),
        "reasons": reasons,
    }
