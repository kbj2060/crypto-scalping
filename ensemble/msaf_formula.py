from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def _sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@dataclass
class MSAFConfig:
    eps: float = 1e-8

    # Layer 1: FAS
    w_flow: float = 0.30
    w_obi: float = 0.25
    w_nif: float = 0.20

    # Layer 2: SIS
    w_abs: float = 0.20
    w_tox: float = 0.15
    w_vpin: float = 0.10
    w_tasd: float = 0.08
    use_tasd: bool = True

    # Layer 3: LCS
    w_liq: float = 0.15
    w_aft: float = 0.40

    # Layer 4: ROA
    gamma: float = 0.15
    beta: float = 1.50

    # Layer 5: EAI amplification
    w_eai: float = 0.25
    eai_apply_min: float = 0.50

    # Layer 6: staleness
    k_stale: float = 0.80
    stale_half_life_sec: float = 60.0
    stale_force_close_sec: float = 120.0

    # Regime-conditional multipliers
    regime_hot_th: float = 1.5
    regime_cold_th: float = -1.5
    hot_fas_mult: float = 0.85
    hot_sis_mult: float = 1.15
    hot_lcs_mult: float = 1.00
    hot_roa_mult: float = 1.20
    cold_fas_mult: float = 0.95
    cold_sis_mult: float = 0.90
    cold_lcs_mult: float = 1.20
    cold_roa_mult: float = 1.05

    # Layer 7: alignment booster
    k_align: float = 0.20

    # Layer 8: Kelly scaling
    c_kelly: float = 2.0
    base_size: float = 1.0
    psi_sigma_window: int = 200

    # Overheat fallback
    overheat_alpha: float = 0.60
    overheat_window: int = 96
    overheat_min_samples: int = 20

    # Optional execution threshold
    min_abs_hat_for_trade: float = 0.05


class MSAFEngine:
    """
    MSAF v1.0
    - 입력: microstructure snapshot(dict)
    - 출력: layer attribution + final side/size
    """

    def __init__(self, cfg: MSAFConfig | None = None):
        self.cfg = cfg or MSAFConfig()
        self._oi_hist: deque[float] = deque(maxlen=max(self.cfg.overheat_window, 8))
        self._fund_hist: deque[float] = deque(maxlen=max(self.cfg.overheat_window, 8))
        self._psi_hist: deque[float] = deque(maxlen=max(self.cfg.psi_sigma_window, 8))

    def _zscore(self, arr: list[float], x: float) -> float:
        if len(arr) < self.cfg.overheat_min_samples:
            return 0.0
        mu = float(sum(arr) / len(arr))
        var = float(sum((v - mu) * (v - mu) for v in arr) / max(1, len(arr) - 1))
        sd = math.sqrt(max(var, self.cfg.eps))
        return float((x - mu) / sd)

    def _rolling_sigma(self) -> float:
        # Use only historical values to avoid look-ahead bias.
        vals = list(self._psi_hist)
        if len(vals) < 10:
            return 1.0
        mu = float(sum(vals) / len(vals))
        var = float(sum((v - mu) * (v - mu) for v in vals) / max(1, len(vals) - 1))
        return float(max(math.sqrt(var), 1e-6))

    def compute(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg

        # Layer 0
        taker_buy_ratio = _clip(_to_float(snapshot.get("taker_buy_ratio", 0.5), 0.5), 0.0, 1.0)
        obi = _clip(_to_float(snapshot.get("obi", 0.0), 0.0), -1.0, 1.0)
        nif_whale = _clip(_to_float(snapshot.get("nif_whale", 0.0), 0.0), -1.0, 1.0)
        s_abs = _clip(_to_float(snapshot.get("shadow_absorption_score", 0.0), 0.0), 0.0, 1.0)
        s_tox = _clip(_to_float(snapshot.get("shadow_toxicity_score", 0.0), 0.0), 0.0, 1.0)
        s_qc = _clip(_to_float(snapshot.get("shadow_queue_collapse", 0.0), 0.0), 0.0, 1.0)
        eai = _clip(_to_float(snapshot.get("eai", 0.0), 0.0), 0.0, 1.0)

        short_usd_1m = _to_float(snapshot.get("short_usd_1m", 0.0), 0.0)
        long_usd_1m = _to_float(snapshot.get("long_usd_1m", 0.0), 0.0)
        oi_delta_pct = _to_float(snapshot.get("oi_delta_pct", 0.0), 0.0)
        funding_rate = _to_float(snapshot.get("funding_rate", 0.0), 0.0)
        aftershock_prob = snapshot.get("aftershock_prob", None)
        shadow_aftershock_prob = _to_float(snapshot.get("shadow_aftershock_prob", 0.0), 0.0)
        data_stale = _clip(_to_float(snapshot.get("data_stale", 0.0), 0.0), 0.0, 1.0)
        stale_seconds = _to_float(snapshot.get("stale_seconds", 0.0), 0.0)
        overheat_input = snapshot.get("overheat_score", None)

        flow = _clip(2.0 * taker_buy_ratio - 1.0, -1.0, 1.0)
        liq = (short_usd_1m - long_usd_1m) / (abs(short_usd_1m) + abs(long_usd_1m) + c.eps)
        liq = _clip(liq, -1.0, 1.0)

        # Keep histories warm regardless of overheat input source.
        self._oi_hist.append(float(oi_delta_pct))
        self._fund_hist.append(float(funding_rate))
        if overheat_input is None:
            z_oi = self._zscore(list(self._oi_hist), oi_delta_pct)
            z_fund = self._zscore(list(self._fund_hist), funding_rate)
            overheat_score = c.overheat_alpha * z_oi + (1.0 - c.overheat_alpha) * z_fund
        else:
            overheat_score = _to_float(overheat_input, 0.0)

        p_aft = _to_float(aftershock_prob, shadow_aftershock_prob) if aftershock_prob is not None else shadow_aftershock_prob
        p_aft = _clip(p_aft, 0.0, 1.0)

        # Layer 1: FAS
        fas = (
            c.w_flow * math.tanh(2.0 * flow)
            + c.w_obi * math.tanh(3.0 * obi)
            + c.w_nif * nif_whale
        )
        fas_pre = float(fas)

        # Layer 2: SIS
        fas_sign = _sign(fas)
        sis = c.w_abs * s_abs + c.w_tox * s_tox * fas_sign - c.w_vpin * s_qc
        tasd = 0.0
        if c.use_tasd:
            # Keep queue collapse as a risk term only; avoid directional boost from collapse.
            tasd = c.w_tasd * s_abs * s_tox * fas_sign
            sis += tasd

        # Layer 3: LCS
        lcs = c.w_liq * liq * (1.0 + c.w_aft * p_aft)

        # Layer 4: ROA
        roa = -c.gamma * math.tanh(c.beta * overheat_score) * fas_sign

        # Regime-conditioned weighting
        regime = "NEUTRAL"
        if overheat_score >= c.regime_hot_th:
            regime = "HOT"
            fas *= c.hot_fas_mult
            sis *= c.hot_sis_mult
            lcs *= c.hot_lcs_mult
            roa *= c.hot_roa_mult
        elif overheat_score <= c.regime_cold_th:
            regime = "COLD"
            fas *= c.cold_fas_mult
            sis *= c.cold_sis_mult
            lcs *= c.cold_lcs_mult
            roa *= c.cold_roa_mult

        # Layer 5: EAI amp
        psi_pre = fas + sis + lcs + roa
        eai_amp = 1.0 + c.w_eai * eai if eai >= c.eai_apply_min else 1.0
        psi_raw = psi_pre * eai_amp

        # Layer 6: staleness
        if stale_seconds > 0.0:
            # half-life decay
            stale_penalty = 0.5 ** (stale_seconds / max(c.stale_half_life_sec, 1e-6))
        else:
            stale_penalty = _clip(1.0 - c.k_stale * data_stale, 0.0, 1.0)
        psi = psi_raw * stale_penalty

        # Layer 7: alignment
        align = _sign(fas) * _sign(sis) * _sign(lcs)
        align_mult = 1.0 + c.k_align if abs(align) == 1.0 else 1.0
        psi_final = psi * align_mult

        # Layer 8: Kelly-scaled sizing
        sigma_psi = self._rolling_sigma()
        hat_psi = psi_final / sigma_psi
        kelly_f = _clip(abs(hat_psi) / c.c_kelly, 0.0, 1.0)
        directional = _clip(hat_psi, -1.0, 1.0)
        size_signed = directional * kelly_f * c.base_size
        size_signed = _clip(size_signed, -c.base_size, c.base_size)

        stale_forced_exit = bool(stale_seconds > c.stale_force_close_sec)
        if stale_forced_exit:
            self._psi_hist.append(0.0)
            psi_final = 0.0
            action = 0
            size_signed = 0.0
            kelly_f = 0.0
            hat_psi = 0.0
        elif abs(hat_psi) < c.min_abs_hat_for_trade:
            self._psi_hist.append(float(psi_final))
            action = 0
        else:
            self._psi_hist.append(float(psi_final))
            action = 1 if size_signed > 0.0 else (2 if size_signed < 0.0 else 0)

        return {
            "action": int(action),  # 0 HOLD, 1 LONG, 2 SHORT
            "score_signed": float(size_signed),
            "kelly_fraction": float(kelly_f),
            "size_signed": float(size_signed),
            "stale_forced_exit": bool(stale_forced_exit),
            "hat_psi": float(hat_psi),
            "psi_sigma": float(sigma_psi),
            "layers": {
                "flow": float(flow),
                "liq": float(liq),
                "overheat_score": float(overheat_score),
                "p_aft": float(p_aft),
                "fas_pre_regime": float(fas_pre),
                "fas_sign_pre_regime": float(fas_sign),
                "fas": float(fas),
                "sis": float(sis),
                "tasd": float(tasd),
                "lcs": float(lcs),
                "roa": float(roa),
                "psi_pre": float(psi_pre),
                "eai_amp": float(eai_amp),
                "psi_raw": float(psi_raw),
                "stale_penalty": float(stale_penalty),
                "psi": float(psi),
                "align": float(align),
                "align_mult": float(align_mult),
                "psi_final": float(psi_final),
                "regime": regime,
            },
            "inputs": {
                "taker_buy_ratio": float(taker_buy_ratio),
                "obi": float(obi),
                "nif_whale": float(nif_whale),
                "shadow_absorption_score": float(s_abs),
                "shadow_toxicity_score": float(s_tox),
                "shadow_queue_collapse": float(s_qc),
                "eai": float(eai),
                "oi_delta_pct": float(oi_delta_pct),
                "funding_rate": float(funding_rate),
                "short_usd_1m": float(short_usd_1m),
                "long_usd_1m": float(long_usd_1m),
                "data_stale": float(data_stale),
                "stale_seconds": float(stale_seconds),
            },
        }


__all__ = ["MSAFConfig", "MSAFEngine"]
