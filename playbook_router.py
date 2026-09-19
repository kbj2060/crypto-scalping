from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlaybookDecision:
    matched: bool
    playbook: str = "NONE"
    priority: int = 0
    action: int = 0
    kelly: float = 0.0
    reason: str = ""
    emergency_exit: bool = False
    widen_trailing_stop: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class PlaybookRouter:
    # 통합 6개 체계
    HFT_PLAYBOOKS = {
        "PB_VETO_SHIELD",
        "PB_CRISIS_SNIPER",
        "PB_SQUEEZE_SNIPER",
    }
    MFT_PLAYBOOKS = {
        "PB_TREND_SIGNAL",
        "PB_WHALE_SIGNAL",
        "PB_MEAN_REVERT_SIGNAL",
    }

    def __init__(self) -> None:
        self.fail_fast = os.getenv("PB_FAIL_FAST", "false").strip().lower() in ("1", "true", "yes", "on")
        self._fallback_warned: set[str] = set()
        self.tr_lai_threshold = float(os.getenv("PB_TR_LAI_TH", os.getenv("TR_LAI_THRESHOLD", "300000000")))
        self.pb2_eai_th = float(os.getenv("PB2_EAI_TH", "2.0"))
        self.pb2_funding_th = float(os.getenv("PB2_FUNDING_TH", "-0.001"))
        self.pb2_absorption_th = float(os.getenv("PB2_ABSORPTION_TH", "0.60"))
        self.pb2_oi_delta_min = float(os.getenv("PB2_OI_DELTA_MIN", "0.003"))
        self.pb5_lai_relax_mult = float(os.getenv("PB5_LAI_RELAX_MULT", "0.75"))
        self.pb5_nif_whale_th = float(os.getenv("PB5_NIF_WHALE_TH", "0.35"))
        self.pb5_absorption_th = float(os.getenv("PB5_ABSORPTION_TH", "0.62"))
        self.pb5_aftershock_max = float(os.getenv("PB5_AFTERSHOCK_MAX", "0.65"))
        self.pb7_obi_th = float(os.getenv("PB7_OBI_TH", "0.25"))
        self.pb7_nif_whale_th = float(os.getenv("PB7_NIF_WHALE_TH", "0.25"))
        self.pb7_nif_z_30m_th = float(os.getenv("PB7_NIF_Z_30M_TH", "1.0"))
        self.pb7_toxicity_max = float(os.getenv("PB7_TOXICITY_MAX", "0.38"))
        self.pb7_confirm_tox_avg_max = float(os.getenv("PB7_CONFIRM_TOX_AVG_MAX", "0.30"))
        self.pb7_confirm_nif_avg_min = float(os.getenv("PB7_CONFIRM_NIF_AVG_MIN", "0.10"))
        self.pb13_toxicity_th = float(os.getenv("PB13_TOXICITY_TH", "0.70"))
        self.pb13_nif_whale_th = float(os.getenv("PB13_NIF_WHALE_TH", "0.25"))
        self.pb13_breakout_soft_th = float(os.getenv("PB13_BREAKOUT_SOFT_TH", "0.015"))
        self.pb13_nif_soft_th = float(os.getenv("PB13_NIF_SOFT_TH", "0.10"))
        self.pb10_z_th = float(os.getenv("PB10_Z_TH", "2.0"))
        self.pb11_bias_abs_min = float(os.getenv("PB11_BIAS_ABS_MIN", "0.18"))
        self.pb_funding_extreme_th = float(os.getenv("PB_FUNDING_EXTREME_TH", "0.002"))
        self.pb_liq_magnet_strength_th = float(os.getenv("PB_LIQ_MAGNET_STRENGTH_TH", "0.22"))
        self.pb_liq_magnet_dist_max = float(os.getenv("PB_LIQ_MAGNET_DIST_MAX", "0.005"))
        self.pb_liq_magnet_tox_max = float(os.getenv("PB_LIQ_MAGNET_TOX_MAX", "0.40"))
        self.pb15_vwap_gap_th = float(os.getenv("PB15_VWAP_GAP_TH", "0.004"))
        self.pb15_vol_max = float(os.getenv("PB15_VOL_MAX", "0.006"))
        self.pb15_absorption_min = float(os.getenv("PB15_ABSORPTION_MIN", "0.60"))
        self.pb15_whale_neutral_max = float(os.getenv("PB15_WHALE_NEUTRAL_MAX", "0.20"))
        self.pb_conflict_kelly_penalty = float(os.getenv("PB_CONFLICT_KELLY_PENALTY", "0.60"))
        # Hysteresis + Soft Signal controls
        self.pb_on_score = float(os.getenv("PB_ON_SCORE", "0.68"))
        self.pb_off_score = float(os.getenv("PB_OFF_SCORE", "0.45"))
        self.pb_soft_band_ratio = float(os.getenv("PB_SOFT_BAND_RATIO", "0.15"))
        self.pb_vol_band_k = float(os.getenv("PB_VOL_BAND_K", "50.0"))
        self.pb_vol_band_max_mult = float(os.getenv("PB_VOL_BAND_MAX_MULT", "4.0"))
        self._pb_active: dict[str, bool] = {}

    def _fallback(self, source: str, key: str, default: float, reason: str, raw: Any = None) -> float:
        msg = f"PlaybookRouter fallback: {source}.{key} -> {default} ({reason}, raw={raw!r})"
        if self.fail_fast:
            raise ValueError(msg)
        token = f"{source}.{key}:{reason}"
        if token not in self._fallback_warned:
            self._fallback_warned.add(token)
            logger.warning(msg)
        return float(default)

    def _num(self, source: str, d: dict[str, Any], key: str, default: float = 0.0) -> float:
        if key not in d:
            return self._fallback(source, key, default, "missing_key")
        raw = d.get(key)
        try:
            v = float(raw)
        except Exception:
            return self._fallback(source, key, default, "non_numeric", raw=raw)
        if not math.isfinite(v):
            return self._fallback(source, key, default, "non_finite", raw=raw)
        return v

    @staticmethod
    def _kelly_up(kelly: float, mult: float) -> float:
        return min(float(kelly) * float(mult), 1.0)

    @staticmethod
    def _clamp01(v: float) -> float:
        return float(max(0.0, min(1.0, v)))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 40.0:
            return 1.0
        if x <= -40.0:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-x)))

    def _dynamic_band(self, th: float, vol: float | None = None, band: float | None = None) -> float:
        if band is not None:
            return max(float(band), 1e-6)
        base_ratio = self.pb_soft_band_ratio
        if vol is None or not math.isfinite(float(vol)) or float(vol) <= 0.0:
            dynamic_ratio = base_ratio
        else:
            vm = 1.0 + (float(vol) * self.pb_vol_band_k)
            vm = min(max(vm, 1.0), self.pb_vol_band_max_mult)
            dynamic_ratio = base_ratio * vm
        return max(abs(float(th)) * dynamic_ratio, 1e-6)

    def _soft_ge(self, val: float, th: float, band: float | None = None, vol: float | None = None) -> float:
        b = self._dynamic_band(th, vol=vol, band=band)
        return self._sigmoid((float(val) - float(th)) / b)

    def _soft_le(self, val: float, th: float, band: float | None = None, vol: float | None = None) -> float:
        b = self._dynamic_band(th, vol=vol, band=band)
        return self._sigmoid((float(th) - float(val)) / b)

    def _soft_abs_ge(self, val: float, th: float, band: float | None = None, vol: float | None = None) -> float:
        return self._soft_ge(abs(float(val)), abs(float(th)), band=band, vol=vol)

    def _soft_abs_le(self, val: float, th: float, band: float | None = None, vol: float | None = None) -> float:
        return self._soft_le(abs(float(val)), abs(float(th)), band=band, vol=vol)

    def _pb_hit(self, name: str, score: float, on: float | None = None, off: float | None = None) -> bool:
        on_th = float(self.pb_on_score if on is None else on)
        off_th = float(self.pb_off_score if off is None else off)
        active = bool(self._pb_active.get(name, False))
        s = self._clamp01(float(score))
        if not active and s >= on_th:
            active = True
        elif active and s <= off_th:
            active = False
        self._pb_active[name] = active
        return active

    @staticmethod
    def _decision_to_dict(d: PlaybookDecision) -> dict[str, Any]:
        return {
            "matched": bool(d.matched),
            "name": str(d.playbook),
            "priority": int(d.priority),
            "action": int(d.action),
            "kelly": float(d.kelly),
            "reason": str(d.reason),
            "emergency_exit": bool(d.emergency_exit),
            "widen_trailing_stop": bool(d.widen_trailing_stop),
            "meta": dict(d.meta or {}),
        }

    def _norm_score(self, c: PlaybookDecision) -> float:
        m = c.meta or {}
        us = m.get("unified_score")
        if us is not None:
            try:
                return self._clamp01(float(us))
            except Exception:
                pass
        n = c.playbook
        try:
            if n == "PB10_CVD_DIVERGENCE":
                s = min(abs(float(m.get("nif_z_30m", 0.0))) / max(self.pb10_z_th, 1e-6), 2.0)
                return self._clamp01(s / 2.0)
            if n == "PB7_HOLY_TRINITY_TREND":
                s = min((abs(float(m.get("obi", 0.0))) + abs(float(m.get("nif_whale", 0.0)))) / 0.5, 2.0)
                return self._clamp01(s / 2.0)
            if n == "PB8_HOLY_TRINITY_TRAP":
                s = min((abs(float(m.get("obi", 0.0))) + abs(float(m.get("nif_whale", 0.0)))) / 0.6, 2.0)
                return self._clamp01(s / 2.0)
            if n == "PB_OI_DIVERGENCE":
                s = min((abs(float(m.get("price_change", 0.0))) / 0.003) + (abs(float(m.get("oi_delta", 0.0))) / 0.005), 2.0)
                return self._clamp01(s / 2.0)
            if n == "PB_LIQUIDATION_MAGNET":
                s = min(
                    (abs(float(m.get("strength", 0.0))) / max(self.pb_liq_magnet_strength_th, 1e-6))
                    + (1.0 - min(abs(float(m.get("distance", 1.0))) / max(self.pb_liq_magnet_dist_max, 1e-6), 1.0)),
                    2.0,
                )
                return self._clamp01(s / 2.0)
        except Exception:
            return 0.0
        return 0.0

    def _best_of(self, candidates: list[PlaybookDecision], names: set[str]) -> PlaybookDecision | None:
        pool = [c for c in candidates if c.playbook in names]
        if not pool:
            return None
        return max(pool, key=lambda x: (self._norm_score(x), x.priority))

    def _evaluate_leaf_candidates(
        self,
        action: int,
        pos: str | None,
        kelly: float,
        ms: dict[str, Any] | None,
        tr: dict[str, Any] | None,
    ) -> list[PlaybookDecision]:
        ms, tr = ms or {}, tr or {}

        lai = self._num("tr", tr, "lai", 0.0)
        tr_aftershock = self._num("tr", tr, "shadow_aftershock_prob", 0.0)
        nif_whale = self._num("ms", ms, "nif_whale", 0.0)
        obi = self._num("ms", ms, "obi", 0.0)
        eai = self._num("ms", ms, "eai", 0.0)
        funding = self._num("ms", ms, "funding_rate", 0.0)
        oi_delta_pct = self._num("ms", ms, "oi_delta_pct", 0.0)
        shadow_absorption = self._num("ms", ms, "shadow_absorption_score", 0.0)
        shadow_collapse = self._num("ms", ms, "shadow_queue_collapse", 0.0)
        shadow_tox = self._num("ms", ms, "shadow_toxicity_score", 0.0)
        liq_cluster_direction = int(self._num("tr", tr, "liq_cluster_direction", 0.0))
        liq_cluster_strength = self._num("tr", tr, "liq_cluster_strength", 0.0)
        liq_distance_pct = self._num("tr", tr, "distance_to_cluster_pct", 1.0)
        liq_cluster_price = self._num("tr", tr, "liq_cluster_price", 0.0)

        price_change_30m = self._num("ms", ms, "price_change_30m", 0.0)
        price_volatility_30m = self._num("ms", ms, "price_volatility_30m", 0.0)
        vwap_gap_15m = self._num("ms", ms, "vwap_gap_15m", 0.0)
        price_breakout_60m = bool(ms.get("price_breakout_60m", False))
        price_breakdown_60m = bool(ms.get("price_breakdown_60m", False))
        nif_whale_sum_30m = self._num("ms", ms, "nif_whale_sum_30m", 0.0)
        nif_whale_avg_30m = self._num("ms", ms, "nif_whale_avg_30m", 0.0)
        nif_whale_std_30m = self._num("ms", ms, "nif_whale_std_30m", 0.0)
        absorption_avg_30m = self._num("ms", ms, "absorption_avg_30m", 0.0)
        bias_avg_30m = self._num("ms", ms, "bias_avg_30m", 0.0)
        toxicity_avg_30m = self._num("ms", ms, "toxicity_avg_30m", 0.0)
        eai_delta_15m = self._num("ms", ms, "eai_delta_15m", 0.0)

        out: list[PlaybookDecision] = []

        def add_pb_soft(
            *,
            score: float,
            name: str,
            prio: int,
            act: int,
            k_mult: float,
            rsn: str,
            widen: bool = False,
            meta: dict | None = None,
            on: float | None = None,
            off: float | None = None,
        ):
            s = self._clamp01(score)
            hit = self._pb_hit(name, s, on=on, off=off)
            if not hit:
                return
            # Soft intensity를 Kelly 증폭에 반영 (True/False 계단식 제거)
            dyn_mult = 1.0 + max(0.0, float(k_mult) - 1.0) * s if int(act) in (1, 2) else 0.0
            merged_meta = dict(meta or {})
            merged_meta.setdefault("unified_score", float(s))
            out.append(
                PlaybookDecision(
                    matched=True,
                    playbook=name,
                    priority=prio,
                    action=int(act),
                    kelly=self._kelly_up(kelly, dyn_mult) if int(act) in (1, 2) else 0.0,
                    reason=rsn,
                    widen_trailing_stop=bool(widen),
                    meta=merged_meta,
                )
            )

        # leaf PB들: hysteresis + soft signal
        adverse_dir = 1.0 if ((funding > 0 and int(action) == 1) or (funding < 0 and int(action) == 2)) else 0.0
        pb_fund_score = self._soft_abs_ge(funding, self.pb_funding_extreme_th, vol=price_volatility_30m) * adverse_dir
        add_pb_soft(
            score=pb_fund_score,
            name="PB_FUNDING_EXTREME_HOLD",
            prio=96,
            act=0,
            k_mult=0.0,
            rsn="FUNDING_EXTREME_VETO_HOLD",
            meta={"funding": funding, "base_action": int(action)},
            on=0.70,
            off=0.45,
        )

        pb5_score = min(
            self._soft_ge(lai, self.tr_lai_threshold * self.pb5_lai_relax_mult, vol=price_volatility_30m),
            self._soft_ge(nif_whale, self.pb5_nif_whale_th, vol=price_volatility_30m),
            self._soft_ge(shadow_absorption, self.pb5_absorption_th, vol=price_volatility_30m),
            self._soft_le(tr_aftershock, self.pb5_aftershock_max, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=pb5_score,
            name="PB5_MAMMOTH_SNIPER",
            prio=95,
            act=1,
            k_mult=1.5,
            rsn="SNIPER_LONG_MAMMOTH_BOTTOM",
            meta={"lai": lai, "nif_whale": nif_whale, "absorption": shadow_absorption, "aftershock": tr_aftershock},
        )

        pb13_base = min(
            self._soft_ge(shadow_tox, self.pb13_toxicity_th, vol=price_volatility_30m),
            self._soft_abs_ge(nif_whale, self.pb13_nif_whale_th, vol=price_volatility_30m),
        )
        # 불리언 breakout/breakdown 대신 연속형 수익률 기반 soft breakout 사용
        pb13_breakout = self._soft_ge(price_change_30m, self.pb13_breakout_soft_th, vol=price_volatility_30m)
        pb13_breakdown = self._soft_ge(-price_change_30m, self.pb13_breakout_soft_th, vol=price_volatility_30m)
        pb13_s = min(pb13_base, pb13_breakout, self._soft_ge(-nif_whale, self.pb13_nif_soft_th, vol=price_volatility_30m))
        pb13_l = min(pb13_base, pb13_breakdown, self._soft_ge(nif_whale, self.pb13_nif_soft_th, vol=price_volatility_30m))
        add_pb_soft(
            score=max(pb13_s, pb13_l),
            name="PB13_BREAKOUT_TRAP",
            prio=94,
            act=2 if pb13_s >= pb13_l else 1,
            k_mult=1.5,
            rsn="HFT_BREAKOUT_TRAP_FADE",
            meta={
                "toxicity": shadow_tox,
                "nif_whale": nif_whale,
                "score_short": pb13_s,
                "score_long": pb13_l,
                "soft_breakout": pb13_breakout,
                "soft_breakdown": pb13_breakdown,
                "legacy_breakout_60m": bool(price_breakout_60m),
                "legacy_breakdown_60m": bool(price_breakdown_60m),
            },
        )

        pb9_score = min(
            self._soft_ge(shadow_collapse, 0.75, vol=price_volatility_30m),
            self._soft_ge(shadow_tox, 0.85, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=pb9_score,
            name="PB9_VACUUM_WHIPSAW",
            prio=93,
            act=0,
            k_mult=0.0,
            rsn="VACUUM_WHIPSAW_HOLD",
            meta={"toxicity": shadow_tox, "collapse": shadow_collapse},
            on=0.70,
            off=0.45,
        )

        pb8_s = min(
            self._soft_ge(obi, 0.35, vol=price_volatility_30m),
            self._soft_ge(-nif_whale, 0.25, vol=price_volatility_30m),
            self._soft_ge(shadow_tox, 0.75, vol=price_volatility_30m),
        )
        pb8_l = min(
            self._soft_ge(-obi, 0.35, vol=price_volatility_30m),
            self._soft_ge(nif_whale, 0.25, vol=price_volatility_30m),
            self._soft_ge(shadow_tox, 0.75, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb8_s, pb8_l),
            name="PB8_HOLY_TRINITY_TRAP",
            prio=92,
            act=2 if pb8_s >= pb8_l else 1,
            k_mult=1.5,
            rsn="SNIPER_TOXIC_TRAP",
            meta={"obi": obi, "nif_whale": nif_whale, "toxicity": shadow_tox, "score_short": pb8_s, "score_long": pb8_l},
        )

        nif_z_30m = nif_whale_sum_30m / (max(nif_whale_std_30m, 1e-8) * math.sqrt(30.0) + 1e-8)
        pb10_s = min(
            self._soft_ge(price_change_30m, -0.002, vol=price_volatility_30m),
            self._soft_ge(-nif_z_30m, self.pb10_z_th, vol=price_volatility_30m),
        )
        pb10_l = min(
            self._soft_le(price_change_30m, 0.002, vol=price_volatility_30m),
            self._soft_ge(nif_z_30m, self.pb10_z_th, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb10_s, pb10_l),
            name="PB10_CVD_DIVERGENCE",
            prio=91,
            act=2 if pb10_s >= pb10_l else 1,
            k_mult=1.3,
            rsn="MFT_MACRO_DISTRIBUTION",
            widen=True,
            meta={"sum": nif_whale_sum_30m, "nif_z_30m": nif_z_30m, "price_change": price_change_30m, "score_short": pb10_s, "score_long": pb10_l},
        )

        liq_dir_score = 1.0 if abs(liq_cluster_direction) > 0 else 0.0
        pb_liq_score = min(
            liq_dir_score,
            self._soft_ge(liq_cluster_strength, self.pb_liq_magnet_strength_th, vol=price_volatility_30m),
            self._soft_le(liq_distance_pct, self.pb_liq_magnet_dist_max, vol=price_volatility_30m),
            self._soft_le(shadow_tox, self.pb_liq_magnet_tox_max, vol=price_volatility_30m),
        )
        pb_liq_action = 1 if liq_cluster_direction > 0 else 2
        add_pb_soft(
            score=pb_liq_score,
            name="PB_LIQUIDATION_MAGNET",
            prio=91,
            act=pb_liq_action,
            k_mult=1.30,
            rsn="MFT_LIQUIDATION_MAGNET",
            meta={"direction": liq_cluster_direction, "strength": liq_cluster_strength, "distance": liq_distance_pct, "cluster_price": liq_cluster_price, "toxicity": shadow_tox},
        )

        pb_oi_s = min(
            self._soft_ge(price_change_30m, 0.003, vol=price_volatility_30m),
            self._soft_ge(-oi_delta_pct, 0.005, vol=price_volatility_30m),
            self._soft_le(shadow_tox, 0.50, vol=price_volatility_30m),
        )
        pb_oi_l = min(
            self._soft_ge(-price_change_30m, 0.003, vol=price_volatility_30m),
            self._soft_ge(-oi_delta_pct, 0.005, vol=price_volatility_30m),
            self._soft_le(shadow_tox, 0.50, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb_oi_s, pb_oi_l),
            name="PB_OI_DIVERGENCE",
            prio=90,
            act=2 if pb_oi_s >= pb_oi_l else 1,
            k_mult=1.25,
            rsn="MFT_OI_DIVERGENCE_REVERT",
            meta={"price_change": price_change_30m, "oi_delta": oi_delta_pct, "toxicity": shadow_tox, "score_short": pb_oi_s, "score_long": pb_oi_l},
        )

        pb12_l = min(
            self._soft_ge(-funding, 0.001, vol=price_volatility_30m),
            self._soft_ge(-eai_delta_15m, 0.02, vol=price_volatility_30m),
            self._soft_ge(nif_whale, 0.2, vol=price_volatility_30m),
        )
        pb12_s = min(
            self._soft_ge(funding, 0.001, vol=price_volatility_30m),
            self._soft_ge(-eai_delta_15m, 0.02, vol=price_volatility_30m),
            self._soft_ge(-nif_whale, 0.2, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb12_l, pb12_s),
            name="PB12_FUNDING_SNAPBACK",
            prio=90,
            act=1 if pb12_l >= pb12_s else 2,
            k_mult=1.4,
            rsn="MFT_FUNDING_SNAPBACK",
            widen=True,
            meta={"funding": funding, "eai_delta": eai_delta_15m, "nif_whale": nif_whale, "score_long": pb12_l, "score_short": pb12_s},
        )

        pb2_l = min(
            self._soft_ge(eai, self.pb2_eai_th, vol=price_volatility_30m),
            self._soft_ge(-funding, abs(self.pb2_funding_th), vol=price_volatility_30m),
            self._soft_ge(shadow_absorption, self.pb2_absorption_th, vol=price_volatility_30m),
            self._soft_ge(oi_delta_pct, self.pb2_oi_delta_min, vol=price_volatility_30m),
        )
        pb2_s = min(
            self._soft_ge(eai, self.pb2_eai_th, vol=price_volatility_30m),
            self._soft_ge(funding, abs(self.pb2_funding_th), vol=price_volatility_30m),
            self._soft_ge(shadow_absorption, self.pb2_absorption_th, vol=price_volatility_30m),
            self._soft_ge(oi_delta_pct, self.pb2_oi_delta_min, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb2_l, pb2_s),
            name="PB2_SQUEEZE_IGNITION",
            prio=89,
            act=1 if pb2_l >= pb2_s else 2,
            k_mult=1.4,
            rsn="SNIPER_SQUEEZE_BREAKOUT",
            meta={"eai": eai, "funding": funding, "absorption": shadow_absorption, "oi_delta_pct": oi_delta_pct, "score_long": pb2_l, "score_short": pb2_s},
        )

        pb7_l = min(
            self._soft_ge(obi, self.pb7_obi_th, vol=price_volatility_30m),
            self._soft_ge(nif_whale, self.pb7_nif_whale_th, vol=price_volatility_30m),
            self._soft_ge(nif_z_30m, self.pb7_nif_z_30m_th, vol=price_volatility_30m),
            self._soft_le(shadow_tox, self.pb7_toxicity_max, vol=price_volatility_30m),
            self._soft_le(toxicity_avg_30m, self.pb7_confirm_tox_avg_max, vol=price_volatility_30m),
            self._soft_ge(nif_whale_avg_30m, self.pb7_confirm_nif_avg_min, vol=price_volatility_30m),
        )
        pb7_s = min(
            self._soft_ge(-obi, self.pb7_obi_th, vol=price_volatility_30m),
            self._soft_ge(-nif_whale, self.pb7_nif_whale_th, vol=price_volatility_30m),
            self._soft_ge(-nif_z_30m, self.pb7_nif_z_30m_th, vol=price_volatility_30m),
            self._soft_le(shadow_tox, self.pb7_toxicity_max, vol=price_volatility_30m),
            self._soft_le(toxicity_avg_30m, self.pb7_confirm_tox_avg_max, vol=price_volatility_30m),
            self._soft_ge(-nif_whale_avg_30m, self.pb7_confirm_nif_avg_min, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb7_l, pb7_s),
            name="PB7_HOLY_TRINITY_TREND",
            prio=88,
            act=1 if pb7_l >= pb7_s else 2,
            k_mult=1.25,
            rsn="HONEST_MOMENTUM_RIDING",
            widen=True,
            meta={"obi": obi, "nif_whale": nif_whale, "toxicity": shadow_tox, "nif_z_30m": nif_z_30m, "score_long": pb7_l, "score_short": pb7_s},
        )

        pb11_score = min(
            self._soft_ge(price_volatility_30m, 0.0, band=0.0005, vol=price_volatility_30m),
            self._soft_le(price_volatility_30m, 0.005, vol=price_volatility_30m),
            self._soft_ge(absorption_avg_30m, 0.75, vol=price_volatility_30m),
            self._soft_abs_ge(bias_avg_30m, self.pb11_bias_abs_min, vol=price_volatility_30m),
        )
        # bias<0 (taker sell 우세) + absorption이면 maker bid(고래 매집)로 해석 -> LONG
        pb11_action = 1 if bias_avg_30m < 0 else 2
        add_pb_soft(
            score=pb11_score,
            name="PB11_TWAP_ABSORPTION",
            prio=87,
            act=pb11_action,
            k_mult=1.2,
            rsn="MFT_SHADOW_ACCUMULATION",
            widen=True,
            meta={"vol": price_volatility_30m, "absorption": absorption_avg_30m, "bias": bias_avg_30m},
        )

        pb15_l = min(
            self._soft_ge(-vwap_gap_15m, self.pb15_vwap_gap_th, vol=price_volatility_30m),
            self._soft_le(price_volatility_30m, self.pb15_vol_max, vol=price_volatility_30m),
            self._soft_ge(shadow_absorption, self.pb15_absorption_min, vol=price_volatility_30m),
            self._soft_abs_le(nif_whale, self.pb15_whale_neutral_max, vol=price_volatility_30m),
        )
        pb15_s = min(
            self._soft_ge(vwap_gap_15m, self.pb15_vwap_gap_th, vol=price_volatility_30m),
            self._soft_le(price_volatility_30m, self.pb15_vol_max, vol=price_volatility_30m),
            self._soft_ge(shadow_absorption, self.pb15_absorption_min, vol=price_volatility_30m),
            self._soft_abs_le(nif_whale, self.pb15_whale_neutral_max, vol=price_volatility_30m),
        )
        add_pb_soft(
            score=max(pb15_l, pb15_s),
            name="PB15_VWAP_MEAN_REVERSION",
            prio=85,
            act=1 if pb15_l >= pb15_s else 2,
            k_mult=1.15,
            rsn="MFT_VWAP_REVERSION",
            meta={"vwap_gap": vwap_gap_15m, "vol": price_volatility_30m, "absorption": shadow_absorption, "nif_whale": nif_whale, "score_long": pb15_l, "score_short": pb15_s},
        )

        return out

    def _build_unified_candidates(
        self,
        leaves: list[PlaybookDecision],
        base_action: int,
        base_kelly: float,
        ms: dict[str, Any] | None = None,
        tr: dict[str, Any] | None = None,
    ) -> list[PlaybookDecision]:
        ms, tr = ms or {}, tr or {}
        by_name = {c.playbook: c for c in leaves}

        def best(names: set[str]) -> PlaybookDecision | None:
            return self._best_of(leaves, names)

        out: list[PlaybookDecision] = []

        # soft metrics for non-matched proximity score
        funding = self._num("ms", ms, "funding_rate", 0.0)
        shadow_collapse = self._num("ms", ms, "shadow_queue_collapse", 0.0)
        shadow_tox = self._num("ms", ms, "shadow_toxicity_score", 0.0)
        lai = self._num("tr", tr, "lai", 0.0)
        tr_aftershock = self._num("tr", tr, "shadow_aftershock_prob", 0.0)
        nif_whale = self._num("ms", ms, "nif_whale", 0.0)
        obi = self._num("ms", ms, "obi", 0.0)
        eai = self._num("ms", ms, "eai", 0.0)
        oi_delta_pct = self._num("ms", ms, "oi_delta_pct", 0.0)
        shadow_absorption = self._num("ms", ms, "shadow_absorption_score", 0.0)
        price_change_30m = self._num("ms", ms, "price_change_30m", 0.0)
        price_volatility_30m = self._num("ms", ms, "price_volatility_30m", 0.0)
        vwap_gap_15m = self._num("ms", ms, "vwap_gap_15m", 0.0)
        nif_whale_sum_30m = self._num("ms", ms, "nif_whale_sum_30m", 0.0)
        nif_whale_std_30m = self._num("ms", ms, "nif_whale_std_30m", 0.0)
        absorption_avg_30m = self._num("ms", ms, "absorption_avg_30m", 0.0)
        bias_avg_30m = self._num("ms", ms, "bias_avg_30m", 0.0)
        toxicity_avg_30m = self._num("ms", ms, "toxicity_avg_30m", 0.0)
        eai_delta_15m = self._num("ms", ms, "eai_delta_15m", 0.0)
        liq_cluster_direction = int(self._num("tr", tr, "liq_cluster_direction", 0.0))
        liq_cluster_strength = self._num("tr", tr, "liq_cluster_strength", 0.0)
        liq_distance_pct = self._num("tr", tr, "distance_to_cluster_pct", 1.0)

        # Layer 0
        veto_sources: list[str] = []
        # funding veto는 leaf 단계가 아니라 최종 winner action 기준으로 사후 판정한다.
        if "PB9_VACUUM_WHIPSAW" in by_name:
            veto_sources.append("VACUUM_WHIPSAW")
        fund_soft = self._clamp01(abs(funding) / max(self.pb_funding_extreme_th, 1e-6))
        vacuum_soft = self._clamp01(min(shadow_collapse / 0.75, shadow_tox / 0.85))
        veto_soft = max(fund_soft, vacuum_soft)
        out.append(
            PlaybookDecision(
                matched=bool(veto_sources),
                playbook="PB_VETO_SHIELD",
                priority=100,
                action=0,
                kelly=0.0,
                reason="+".join(veto_sources),
                meta={"unified_score": 1.0 if veto_sources else float(veto_soft), "sources": veto_sources},
            )
        )

        # Layer 1
        crisis = best({"PB5_MAMMOTH_SNIPER", "PB13_BREAKOUT_TRAP", "PB8_HOLY_TRINITY_TRAP"})
        crisis_score = self._norm_score(crisis) if crisis else 0.0
        crisis_soft_mammoth = self._clamp01(min(
            lai / max(self.tr_lai_threshold * self.pb5_lai_relax_mult, 1e-6),
            (nif_whale / max(self.pb5_nif_whale_th, 1e-6)) if self.pb5_nif_whale_th > 0 else 0.0,
            shadow_absorption / max(self.pb5_absorption_th, 1e-6),
            (self.pb5_aftershock_max / max(tr_aftershock, 1e-6)) if tr_aftershock > 0 else 1.0,
        ))
        crisis_soft_trap = self._clamp01(min(
            shadow_tox / max(self.pb13_toxicity_th, 1e-6),
            abs(nif_whale) / max(self.pb13_nif_whale_th, 1e-6),
        ))
        crisis_soft_pb8 = self._clamp01(min(
            abs(obi) / 0.35,
            abs(nif_whale) / 0.25,
            shadow_tox / 0.75,
        ))
        crisis_soft = max(crisis_soft_mammoth, crisis_soft_trap, crisis_soft_pb8)
        out.append(
            PlaybookDecision(
                matched=crisis is not None,
                playbook="PB_CRISIS_SNIPER",
                priority=95,
                action=int(crisis.action) if crisis else 0,
                kelly=min(float(base_kelly) * (0.9 + 0.6 * float(crisis_score)), 1.0) if crisis else float(base_kelly),
                reason=str(crisis.reason) if crisis else "",
                widen_trailing_stop=bool(crisis.widen_trailing_stop) if crisis else False,
                meta={"unified_score": float(crisis_score if crisis else crisis_soft), "source": (crisis.playbook if crisis else "")},
            )
        )

        squeeze = best({"PB2_SQUEEZE_IGNITION", "PB12_FUNDING_SNAPBACK"})
        squeeze_score = self._norm_score(squeeze) if squeeze else 0.0
        squeeze_soft_pb2 = self._clamp01(min(
            eai / max(self.pb2_eai_th, 1e-6),
            abs(funding) / max(abs(self.pb2_funding_th), 1e-6),
            shadow_absorption / max(self.pb2_absorption_th, 1e-6),
            oi_delta_pct / max(self.pb2_oi_delta_min, 1e-6),
        ))
        squeeze_soft_pb12 = self._clamp01(min(
            abs(funding) / 0.001,
            max(0.0, -eai_delta_15m) / 0.05,
            abs(nif_whale) / 0.2,
        ))
        squeeze_soft = max(squeeze_soft_pb2, squeeze_soft_pb12)
        out.append(
            PlaybookDecision(
                matched=squeeze is not None,
                playbook="PB_SQUEEZE_SNIPER",
                priority=92,
                action=int(squeeze.action) if squeeze else 0,
                kelly=min(float(base_kelly) * (0.9 + 0.6 * float(squeeze_score)), 1.0) if squeeze else float(base_kelly),
                reason=str(squeeze.reason) if squeeze else "",
                widen_trailing_stop=bool(squeeze.widen_trailing_stop) if squeeze else False,
                meta={"unified_score": float(squeeze_score if squeeze else squeeze_soft), "source": (squeeze.playbook if squeeze else "")},
            )
        )

        # Layer 2
        trend = best({"PB7_HOLY_TRINITY_TREND"})
        trend_score = self._norm_score(trend) if trend else 0.0
        trend_soft_pb7 = self._clamp01(min(
            abs(obi) / max(self.pb7_obi_th, 1e-6),
            abs(nif_whale) / max(self.pb7_nif_whale_th, 1e-6),
            self.pb7_toxicity_max / max(shadow_tox, 1e-6) if shadow_tox > 0 else 1.0,
            self.pb7_confirm_tox_avg_max / max(toxicity_avg_30m, 1e-6) if toxicity_avg_30m > 0 else 1.0,
        ))
        trend_soft = trend_soft_pb7
        out.append(
            PlaybookDecision(
                matched=trend is not None,
                playbook="PB_TREND_SIGNAL",
                priority=86,
                action=int(trend.action) if trend else int(base_action),
                kelly=min(float(base_kelly) * (0.8 + 0.6 * float(trend_score)), 1.0) if trend else float(base_kelly),
                reason=str(trend.reason) if trend else "",
                widen_trailing_stop=bool(trend.widen_trailing_stop) if trend else False,
                meta={"unified_score": float(trend_score if trend else trend_soft), "source": (trend.playbook if trend else "")},
            )
        )

        whale = best({"PB10_CVD_DIVERGENCE", "PB11_TWAP_ABSORPTION"})
        whale_score = self._norm_score(whale) if whale else 0.0
        nif_z_30m = nif_whale_sum_30m / (max(nif_whale_std_30m, 1e-8) * math.sqrt(30.0) + 1e-8)
        whale_soft_pb10 = self._clamp01(min(abs(nif_z_30m) / max(self.pb10_z_th, 1e-6), 0.002 / max(abs(price_change_30m), 1e-6)))
        whale_soft_pb11 = self._clamp01(min(
            0.005 / max(price_volatility_30m, 1e-6),
            absorption_avg_30m / 0.75,
            abs(bias_avg_30m) / max(self.pb11_bias_abs_min, 1e-6),
        ))
        whale_soft = max(whale_soft_pb10, whale_soft_pb11)
        out.append(
            PlaybookDecision(
                matched=whale is not None,
                playbook="PB_WHALE_SIGNAL",
                priority=85,
                action=int(whale.action) if whale else int(base_action),
                kelly=min(float(base_kelly) * (0.8 + 0.6 * float(whale_score)), 1.0) if whale else float(base_kelly),
                reason=str(whale.reason) if whale else "",
                widen_trailing_stop=bool(whale.widen_trailing_stop) if whale else False,
                meta={"unified_score": float(whale_score if whale else whale_soft), "source": (whale.playbook if whale else "")},
            )
        )

        revert = best({"PB15_VWAP_MEAN_REVERSION", "PB_LIQUIDATION_MAGNET", "PB_OI_DIVERGENCE"})
        revert_score = self._norm_score(revert) if revert else 0.0
        revert_soft_pb15 = self._clamp01(min(
            abs(vwap_gap_15m) / max(self.pb15_vwap_gap_th, 1e-6),
            self.pb15_vol_max / max(price_volatility_30m, 1e-6),
            shadow_absorption / max(self.pb15_absorption_min, 1e-6),
            self.pb15_whale_neutral_max / max(abs(nif_whale), 1e-6) if abs(nif_whale) > 0 else 1.0,
        ))
        revert_soft_liq = self._clamp01(min(
            liq_cluster_strength / max(self.pb_liq_magnet_strength_th, 1e-6),
            self.pb_liq_magnet_dist_max / max(liq_distance_pct, 1e-6),
            self.pb_liq_magnet_tox_max / max(shadow_tox, 1e-6) if shadow_tox > 0 else 1.0,
        ))
        revert_soft_oi = self._clamp01(min(
            abs(price_change_30m) / 0.003,
            max(0.0, -oi_delta_pct) / 0.005,
            0.50 / max(shadow_tox, 1e-6) if shadow_tox > 0 else 1.0,
        ))
        revert_soft = max(revert_soft_pb15, revert_soft_liq, revert_soft_oi)
        inferred_revert_action = int(base_action)
        if abs(vwap_gap_15m) >= abs(liq_cluster_direction) * 0.001:
            inferred_revert_action = 1 if vwap_gap_15m < 0 else 2 if vwap_gap_15m > 0 else int(base_action)
        elif liq_cluster_direction != 0:
            inferred_revert_action = 1 if liq_cluster_direction > 0 else 2
        out.append(
            PlaybookDecision(
                matched=revert is not None,
                playbook="PB_MEAN_REVERT_SIGNAL",
                priority=84,
                action=int(revert.action) if revert else int(inferred_revert_action),
                kelly=min(float(base_kelly) * (0.8 + 0.6 * float(revert_score)), 1.0) if revert else float(base_kelly),
                reason=str(revert.reason) if revert else "",
                widen_trailing_stop=bool(revert.widen_trailing_stop) if revert else False,
                meta={"unified_score": float(revert_score if revert else revert_soft), "source": (revert.playbook if revert else "")},
            )
        )

        return out

    def _resolve_winner(
        self,
        unified: list[PlaybookDecision],
        base_action: int,
        base_kelly: float,
        ms: dict[str, Any] | None = None,
    ) -> PlaybookDecision:
        ms = ms or {}
        by = {c.playbook: c for c in unified}

        veto = by.get("PB_VETO_SHIELD")
        if veto and veto.matched:
            return veto

        crisis = by.get("PB_CRISIS_SNIPER")
        squeeze = by.get("PB_SQUEEZE_SNIPER")
        sniper_pool = [x for x in (crisis, squeeze) if x and x.matched]
        if sniper_pool:
            return max(sniper_pool, key=lambda x: float((x.meta or {}).get("unified_score", 0.0)))

        # Layer 2 특별 규칙: 고래-추세 시너지/충돌 판결
        trend_sig = by.get("PB_TREND_SIGNAL")
        whale_sig = by.get("PB_WHALE_SIGNAL")
        if trend_sig and whale_sig and trend_sig.matched and whale_sig.matched:
            trend_score = float((trend_sig.meta or {}).get("unified_score", 0.0))
            whale_score = float((whale_sig.meta or {}).get("unified_score", 0.0))
            trend_action = int(trend_sig.action)
            whale_action = int(whale_sig.action)

            # 시너지: 방향 일치 + 양쪽 신호 품질이 충분하면 비중 강화
            if trend_action in (1, 2) and whale_action == trend_action and trend_score >= 0.50 and whale_score >= 0.50:
                k_synergy = min(max(float(trend_sig.kelly), float(whale_sig.kelly)) * 1.15, 1.0)
                return PlaybookDecision(
                    matched=True,
                    playbook="PB_TREND_SIGNAL",
                    priority=int(trend_sig.priority),
                    action=trend_action,
                    kelly=float(k_synergy),
                    reason="SYNERGY_TREND_WHALE_ALIGN",
                    widen_trailing_stop=bool(trend_sig.widen_trailing_stop or whale_sig.widen_trailing_stop),
                    meta={
                        "unified_score": float(max(trend_score, whale_score)),
                        "source": "PB7/PB10|PB11",
                        "synergy": True,
                        "trend_score": trend_score,
                        "whale_score": whale_score,
                    },
                )

            # 충돌: 고래/추세가 반대면 추세 단독 진입 억제
            if trend_action in (1, 2) and whale_action in (1, 2) and trend_action != whale_action:
                # 고래 우세가 뚜렷하면 저비중으로 고래 방향만 허용
                if whale_score >= trend_score + 0.15 and whale_score >= 0.55:
                    return PlaybookDecision(
                        matched=True,
                        playbook="PB_WHALE_SIGNAL",
                        priority=int(whale_sig.priority),
                        action=whale_action,
                        kelly=float(max(0.0, min(float(whale_sig.kelly) * self.pb_conflict_kelly_penalty, 1.0))),
                        reason="WHALE_OVERRIDE_LOW_KELLY",
                        widen_trailing_stop=bool(whale_sig.widen_trailing_stop),
                        meta={
                            "unified_score": float(whale_score),
                            "source": (whale_sig.meta or {}).get("source", ""),
                            "conflict_with": "PB_TREND_SIGNAL",
                            "trend_score": trend_score,
                            "whale_score": whale_score,
                        },
                    )
                # 우열 불명확하면 관망
                return PlaybookDecision(
                    matched=True,
                    playbook="PB_TREND_SIGNAL",
                    priority=int(max(trend_sig.priority, whale_sig.priority)),
                    action=0,
                    kelly=0.0,
                    reason="HOLD_TREND_WHALE_CLASH",
                    meta={
                        "unified_score": float(max(trend_score, whale_score)),
                        "source": "TREND_vs_WHALE",
                        "trend_score": trend_score,
                        "whale_score": whale_score,
                    },
                )

        signals = [x for x in (by.get("PB_TREND_SIGNAL"), by.get("PB_WHALE_SIGNAL"), by.get("PB_MEAN_REVERT_SIGNAL")) if x and x.matched]
        if not signals:
            return PlaybookDecision(matched=False, playbook="NONE", action=int(base_action), kelly=float(base_kelly), reason="NO_LAYER_MATCH")

        scored = []
        for s in signals:
            score = float((s.meta or {}).get("unified_score", 0.0))
            if score >= 0.30:
                scored.append((score, s))
        if not scored:
            return PlaybookDecision(matched=False, playbook="NONE", action=int(base_action), kelly=float(base_kelly), reason="SIGNAL_WEAK")

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sig = scored[0]
        if len(scored) >= 2:
            second_score, second_sig = scored[1]
            if best_sig.action in (1, 2) and second_sig.action in (1, 2) and best_sig.action != second_sig.action and (best_score - second_score) < 0.15:
                return PlaybookDecision(
                    matched=True,
                    playbook=str(best_sig.playbook),
                    priority=int(best_sig.priority),
                    action=0,
                    kelly=0.0,
                    reason="HOLD_SIGNAL_CONFLICT",
                    meta={
                        "unified_score": float(best_score),
                        "source": (best_sig.meta or {}).get("source", ""),
                        "conflict_with": str(second_sig.playbook),
                        "conflict_score": float(second_score),
                    },
                )
        winner = best_sig
        # 최종 우승 action 기준 funding veto (post-filter)
        funding = self._num("ms", ms, "funding_rate", 0.0)
        if winner.action in (1, 2):
            if (funding > self.pb_funding_extreme_th and winner.action == 1) or (
                funding < -self.pb_funding_extreme_th and winner.action == 2
            ):
                return PlaybookDecision(
                    matched=True,
                    playbook="PB_VETO_SHIELD",
                    priority=100,
                    action=0,
                    kelly=0.0,
                    reason="FINAL_ACTION_VETO_FUNDING_EXTREME",
                    meta={"funding": funding, "blocked_action": int(winner.action), "blocked_playbook": str(winner.playbook)},
                )
        return winner

    def _resolve_group_winner(self, unified: list[PlaybookDecision], action: int, kelly: float, allowed: set[str]) -> PlaybookDecision:
        grouped = [c for c in unified if c.playbook in allowed and c.matched]
        if not grouped:
            return PlaybookDecision(matched=False, action=int(action), kelly=float(kelly))
        return max(grouped, key=lambda x: (x.priority, float((x.meta or {}).get("unified_score", 0.0))))

    def decide(self, action: int, pos: str | None, kelly: float, ms: dict[str, Any] | None, tr: dict[str, Any] | None) -> PlaybookDecision:
        leaves = self._evaluate_leaf_candidates(action=action, pos=pos, kelly=kelly, ms=ms, tr=tr)
        unified = self._build_unified_candidates(leaves, base_action=action, base_kelly=kelly, ms=ms, tr=tr)
        return self._resolve_winner(unified, base_action=action, base_kelly=kelly, ms=ms)

    def evaluate_all(self, action: int, pos: str | None, kelly: float, ms: dict[str, Any] | None, tr: dict[str, Any] | None) -> dict[str, Any]:
        leaves = self._evaluate_leaf_candidates(action=action, pos=pos, kelly=kelly, ms=ms, tr=tr)
        unified = self._build_unified_candidates(leaves, base_action=action, base_kelly=kelly, ms=ms, tr=tr)
        winner = self._resolve_winner(unified, base_action=action, base_kelly=kelly, ms=ms)
        winner_hft = self._resolve_group_winner(unified, action, kelly, self.HFT_PLAYBOOKS)
        winner_mft = self._resolve_group_winner(unified, action, kelly, self.MFT_PLAYBOOKS)
        evals = [self._decision_to_dict(x) for x in unified]
        return {
            "winner": self._decision_to_dict(winner),
            "winner_hft": self._decision_to_dict(winner_hft),
            "winner_mft": self._decision_to_dict(winner_mft),
            "evaluations": evals,
        }


__all__ = ["PlaybookRouter", "PlaybookDecision"]
