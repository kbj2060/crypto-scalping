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


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -40.0, 40.0))))


def _align_score(side: str, value: float) -> float:
    s = str(side or "").upper()
    x = _safe_float(value, 0.0)
    if s == "LONG":
        return x
    if s == "SHORT":
        return -x
    return 0.0


@dataclass(frozen=True)
class PlaybookMetaConfig:
    event_k: float = 1.10
    hazard_k: float = 1.20
    continuation_k: float = 1.00
    pullback_k: float = 0.95
    size_boost: float = 0.12
    size_floor: float = 0.65
    size_cap: float = 1.20
    delay_scale: float = 2.0
    hold_base_bars: int = 180
    hold_scale: float = 0.30
    exit_aggr: float = 1.00
    skip_hazard_th: float = 0.86
    sparse_event_th: float = 0.78
    sparse_hazard_th: float = 0.72
    severe_exit_th: float = 0.90
    mild_reduce_th: float = 0.82

    @property
    def name(self) -> str:
        return (
            f"ek{self.event_k:.2f}_hk{self.hazard_k:.2f}_ck{self.continuation_k:.2f}"
            f"_pk{self.pullback_k:.2f}_sb{self.size_boost:.2f}_ds{self.delay_scale:.1f}"
            f"_xa{self.exit_aggr:.2f}"
        )


def build_playbook_features(row, side: str) -> dict[str, float]:
    signal_bias = _clip(_safe_float(getattr(row, "signal_bias", row.get("signal_bias", 0.0) if hasattr(row, "get") else 0.0)), -1.0, 1.0)
    nif_whale = _clip(_safe_float(getattr(row, "nif_whale", row.get("nif_whale", 0.0) if hasattr(row, "get") else 0.0)), -1.0, 1.0)
    taker_buy_ratio = _clip(_safe_float(getattr(row, "taker_buy_ratio", row.get("taker_buy_ratio", 0.5) if hasattr(row, "get") else 0.5)), 0.0, 1.0)
    toxicity = _clip(_safe_float(getattr(row, "shadow_toxicity_score", row.get("shadow_toxicity_score", 0.0) if hasattr(row, "get") else 0.0)) / 1.2, 0.0, 1.5)
    queue_collapse = _clip(_safe_float(getattr(row, "shadow_queue_collapse", row.get("shadow_queue_collapse", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    absorption = _clip(_safe_float(getattr(row, "shadow_absorption_score", row.get("shadow_absorption_score", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    regime_conf = _clip(_safe_float(getattr(row, "shadow_regime_conf", row.get("shadow_regime_conf", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    aftershock = _clip(_safe_float(getattr(row, "shadow_aftershock_prob", row.get("shadow_aftershock_prob", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    half_life = _clip(_safe_float(getattr(row, "shadow_decay_half_life", row.get("shadow_decay_half_life", 0.0) if hasattr(row, "get") else 0.0)) / 10.0, 0.0, 1.2)
    bucket = str(getattr(row, "shadow_risk_bucket", row.get("shadow_risk_bucket", "normal") if hasattr(row, "get") else "normal") or "normal").lower()
    bucket_risk = 0.0 if bucket == "normal" else 0.45 if bucket in {"watch", "elevated"} else 0.9

    gap = _safe_float(getattr(row, "target_gap", row.get("target_gap", 0.0) if hasattr(row, "get") else 0.0))
    mom_1m = _safe_float(getattr(row, "target_gap_delta_1m", row.get("target_gap_delta_1m", 0.0) if hasattr(row, "get") else 0.0))
    prob_mom = _safe_float(getattr(row, "prob_mom_1m", row.get("prob_mom_1m", 0.0) if hasattr(row, "get") else 0.0))
    mode_prob = _clip(_safe_float(getattr(row, "mode_prob", row.get("mode_prob", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    mode_spread = _clip(_safe_float(getattr(row, "mode_spread", row.get("mode_spread", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    entropy = _clip(_safe_float(getattr(row, "entropy", row.get("entropy", 1.0) if hasattr(row, "get") else 1.0)), 0.0, 1.0)
    tail_up = _clip(_safe_float(getattr(row, "tail_up_prob", row.get("tail_up_prob", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    tail_down = _clip(_safe_float(getattr(row, "tail_down_prob", row.get("tail_down_prob", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)
    qwidth = _clip(_safe_float(getattr(row, "m7_qwidth", row.get("m7_qwidth", 0.0) if hasattr(row, "get") else 0.0)) / 0.02, 0.0, 1.5)
    m7_conf = _clip(_safe_float(getattr(row, "m7_confidence", row.get("m7_confidence", 0.0) if hasattr(row, "get") else 0.0)), 0.0, 1.0)

    flow_align = _clip(0.45 * signal_bias + 0.35 * nif_whale + 0.20 * ((2.0 * taker_buy_ratio) - 1.0), -1.0, 1.0)
    direction_pressure = _clip(np.tanh(gap / 0.01) + 0.55 * np.tanh(mom_1m / 0.003) + 0.20 * np.tanh(prob_mom / 0.03), -1.5, 1.5)
    poly_conf = _clip(mode_spread * (1.0 - entropy), 0.0, 1.0)
    uncertainty = _clip((0.60 * entropy) + (0.25 * (1.0 - mode_spread)) + (0.15 * qwidth), 0.0, 1.2)
    tail_bias = _clip(tail_up - tail_down, -1.0, 1.0)
    micro_quality = _clip(np.exp(-(0.85 * toxicity + 0.75 * queue_collapse)) * (0.55 + 0.45 * absorption), 0.0, 1.2)

    supportive_gap = gap if str(side).upper() == "LONG" else -gap
    adverse_gap = -supportive_gap

    return {
        "flow_align": float(flow_align),
        "toxicity": float(toxicity),
        "queue_collapse": float(queue_collapse),
        "absorption": float(absorption),
        "regime_conf": float(regime_conf),
        "aftershock": float(aftershock),
        "half_life": float(half_life),
        "bucket_risk": float(bucket_risk),
        "gap": float(gap),
        "supportive_gap": float(supportive_gap),
        "adverse_gap": float(adverse_gap),
        "momentum_1m": float(mom_1m),
        "prob_mom_1m": float(prob_mom),
        "mode_prob": float(mode_prob),
        "mode_spread": float(mode_spread),
        "entropy": float(entropy),
        "tail_bias": float(tail_bias),
        "poly_conf": float(poly_conf),
        "uncertainty": float(uncertainty),
        "direction_pressure": float(direction_pressure),
        "micro_quality": float(micro_quality),
        "m7_confidence": float(m7_conf),
        "qwidth": float(qwidth),
    }


def compute_playbook_meta_controller(side: str, row, cfg: PlaybookMetaConfig | None = None) -> dict:
    cfg = cfg or PlaybookMetaConfig()
    s = str(side or "").upper()
    f = build_playbook_features(row, side=s)

    poly_align = _align_score(s, f["direction_pressure"])
    micro_align = _align_score(s, f["flow_align"])
    tail_align = _align_score(s, f["tail_bias"])
    align = _clip((0.50 * poly_align) + (0.35 * micro_align) + (0.15 * tail_align), -1.5, 1.5)

    event_energy = _sigmoid(
        cfg.event_k
        * (
            1.20 * abs(np.tanh(f["momentum_1m"] / 0.003))
            + 0.85 * abs(np.tanh(f["supportive_gap"] / 0.008))
            + 0.55 * f["poly_conf"]
            - 0.45 * f["uncertainty"]
        )
    )
    hazard = _sigmoid(
        cfg.hazard_k
        * (
            1.05 * f["aftershock"]
            + 0.65 * f["toxicity"]
            + 0.55 * f["queue_collapse"]
            + 0.35 * f["uncertainty"]
            + 0.25 * f["bucket_risk"]
            - 0.35 * f["micro_quality"]
        )
    )
    continuation_score = _sigmoid(
        3.2
        * cfg.continuation_k
        * (
            0.75 * align
            + 0.35 * event_energy
            + 0.25 * f["micro_quality"]
            + 0.20 * f["m7_confidence"]
            - 0.55 * hazard
        )
    )
    pullback_score = _sigmoid(
        3.0
        * cfg.pullback_k
        * (
            0.75 * np.tanh(f["adverse_gap"] / 0.008)
            + 0.45 * f["absorption"]
            + 0.15 * (1.0 - f["queue_collapse"])
            - 0.45 * hazard
        )
    )
    decay_score = _sigmoid(
        3.0
        * (
            0.70 * f["aftershock"]
            + 0.40 * f["uncertainty"]
            + 0.20 * f["entropy"]
            - 0.20 * f["micro_quality"]
        )
    )
    reversal_score = _sigmoid(
        3.1
        * (
            0.70 * hazard
            + 0.30 * np.tanh(f["adverse_gap"] / 0.008)
            + 0.20 * event_energy
            - 0.25 * align
        )
    )
    calm_drift_score = _sigmoid(
        2.9
        * (
            0.60 * align
            + 0.30 * f["micro_quality"]
            + 0.20 * f["poly_conf"]
            - 0.55 * hazard
            - 0.15 * abs(np.tanh(f["momentum_1m"] / 0.003))
        )
    )

    playbook_scores = {
        "EVENT_BREAKOUT_FOLLOW": continuation_score,
        "MEAN_REVERT_PULLBACK": pullback_score,
        "POST_NEWS_DECAY": decay_score,
        "TOXIC_EVENT_REVERSAL": reversal_score,
        "CALM_DRIFT_ALIGN": calm_drift_score,
    }
    playbook = max(playbook_scores.items(), key=lambda kv: kv[1])[0]

    sparse_active = bool(event_energy >= cfg.sparse_event_th and hazard >= cfg.sparse_hazard_th)
    severe_sparse = bool(event_energy >= max(cfg.sparse_event_th + 0.06, 0.84) and hazard >= cfg.severe_exit_th)
    skip_entry = bool(
        severe_sparse
        and hazard >= cfg.skip_hazard_th
        and playbook in {"POST_NEWS_DECAY", "TOXIC_EVENT_REVERSAL"}
    )
    delay_bars = 0
    if sparse_active and playbook == "MEAN_REVERT_PULLBACK":
        delay_bars = int(np.clip(round(cfg.delay_scale * (0.40 + 0.90 * pullback_score) * (1.0 - min(hazard, 0.85))), 1, 4))
    elif sparse_active and playbook == "POST_NEWS_DECAY":
        delay_bars = int(np.clip(round(cfg.delay_scale * (0.55 + 0.45 * decay_score)), 1, 3))
    elif sparse_active and playbook == "TOXIC_EVENT_REVERSAL":
        delay_bars = int(np.clip(round(cfg.delay_scale * (0.65 + 0.35 * reversal_score)), 1, 3))

    raw_size = 1.0
    if sparse_active:
        raw_size = (
            0.98
            + 0.55 * cfg.size_boost * continuation_score
            + 0.30 * cfg.size_boost * calm_drift_score
            - 0.70 * cfg.size_boost * hazard
            - 0.25 * cfg.size_boost * decay_score
            + 0.15 * cfg.size_boost * max(align, 0.0)
        )
        if playbook == "EVENT_BREAKOUT_FOLLOW":
            raw_size += 0.18 * cfg.size_boost
        elif playbook in {"POST_NEWS_DECAY", "TOXIC_EVENT_REVERSAL"}:
            raw_size -= 0.55 * cfg.size_boost
    size_mult = _clip(raw_size, cfg.size_floor, cfg.size_cap)
    if skip_entry:
        size_mult = 0.0

    # Passive hold caps are intentionally disabled. The controller should only
    # intervene when there is an active event/risk reason, not by shortening
    # every trade through a baked-in time limit.
    max_hold_bars = None

    exit_danger = _clip(
        0.42 * hazard
        + 0.24 * _clip(-align, 0.0, 1.5)
        + 0.14 * abs(np.tanh(f["momentum_1m"] / 0.003))
        + 0.12 * f["uncertainty"]
        + 0.08 * (1.0 - f["micro_quality"]),
        0.0,
        1.5,
    )
    exit_trigger = _clip(0.94 - 0.06 * cfg.exit_aggr + 0.02 * continuation_score, 0.78, 0.98)

    mode = "free_trade"
    if skip_entry:
        mode = "cooldown"
    elif sparse_active and playbook in {"POST_NEWS_DECAY", "TOXIC_EVENT_REVERSAL"} and hazard >= cfg.mild_reduce_th:
        mode = "reduce_only"
    elif sparse_active and delay_bars > 0:
        mode = "delayed_entry"
    elif sparse_active and size_mult > 1.01:
        mode = "conviction_boost"

    return {
        "config": asdict(cfg),
        "playbook": playbook,
        "playbook_scores": {k: float(v) for k, v in playbook_scores.items()},
        "features": f,
        "align": float(align),
        "event_energy": float(event_energy),
        "hazard": float(hazard),
        "sparse_active": bool(sparse_active),
        "severe_sparse": bool(severe_sparse),
        "size_mult": float(size_mult),
        "delay_bars": int(delay_bars),
        "max_hold_bars": max_hold_bars,
        "exit_danger": float(exit_danger),
        "exit_trigger": float(exit_trigger),
        "skip_entry": bool(skip_entry),
        "mode": mode,
    }
