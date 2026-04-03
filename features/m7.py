from __future__ import annotations

import numpy as np


def trend_signal_from_m7(m7_last: dict | None) -> dict | None:
    """SevenModelEnsemble 출력(dict)을 DSACTrendRouter 입력 포맷으로 변환."""
    if not isinstance(m7_last, dict) or not m7_last:
        return None

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(m7_last.get(key, default))
        except Exception:
            return float(default)

    agg_dn = float(np.clip(_f("m7_prob_dn", 0.0), 0.0, 1.0))
    agg_fl = float(np.clip(_f("m7_prob_fl", 0.0), 0.0, 1.0))
    agg_up = float(np.clip(_f("m7_prob_up", 0.0), 0.0, 1.0))
    s = agg_dn + agg_fl + agg_up
    if s <= 1e-12:
        agg_dn = agg_fl = agg_up = 1.0 / 3.0
    else:
        agg_dn, agg_fl, agg_up = agg_dn / s, agg_fl / s, agg_up / s

    p_dn = float(np.clip(_f("m7_trend_xgb_dn", agg_dn), 0.0, 1.0))
    p_fl = float(np.clip(_f("m7_trend_xgb_fl", agg_fl), 0.0, 1.0))
    p_up = float(np.clip(_f("m7_trend_xgb_up", agg_up), 0.0, 1.0))
    s_xgb = p_dn + p_fl + p_up
    if s_xgb <= 1e-12:
        p_dn = p_fl = p_up = 1.0 / 3.0
    else:
        p_dn, p_fl, p_up = p_dn / s_xgb, p_fl / s_xgb, p_up / s_xgb

    t_dir = int(np.argmax([p_dn, p_fl, p_up]))
    m7_action = int(np.clip(round(_f("m7_action", 0.0)), -1, 1))
    m7_conf = float(np.clip(_f("m7_confidence", 0.0), 0.0, 1.0))
    m7_gate_block = 1 if _f("m7_gate_block", 0.0) >= 0.5 else 0
    xgb_top = max(p_dn, p_fl, p_up)
    xgb_second = sorted([p_dn, p_fl, p_up])[1]
    strength = float(np.clip((xgb_top - 1.0 / 3.0) * 1.5 + (xgb_top - xgb_second) * 0.6, 0.0, 1.0))
    rev_prob = float(np.clip((1.0 - strength) * 0.70 + (0.30 if m7_gate_block else 0.0), 0.0, 1.0))

    return {
        "trend_dir": t_dir,
        "strength": strength,
        "rev_prob": rev_prob,
        "prob_dn": p_dn,
        "prob_flat": p_fl,
        "prob_up": p_up,
        "probs": [p_dn, p_fl, p_up],
        "trend_model": "TREND_XGB",
        "m7_confidence": m7_conf,
        "m7_action": m7_action,
        "m7_prob_dn": agg_dn,
        "m7_prob_fl": agg_fl,
        "m7_prob_up": agg_up,
        "m7_size": float(np.clip(_f("m7_size", 0.0), 0.0, 1.0)),
        "m7_gate_block": m7_gate_block,
        "m7_quality_pred": _f("m7_quality_pred", 0.0),
        "m7_hold_pred": _f("m7_hold_pred", 0.0),
        "m7_target_hold": float(max(0.0, _f("m7_target_hold", 0.0))),
        "m7_q10": _f("m7_q10", 0.0),
        "m7_q50": _f("m7_q50", 0.0),
        "m7_q90": _f("m7_q90", 0.0),
        "m7_qwidth": float(max(0.0, _f("m7_qwidth", 0.0))),
        "m7_entry_long_offset": _f("m7_entry_long_offset", 0.0),
        "m7_entry_short_offset": _f("m7_entry_short_offset", 0.0),
        "m7_entry_long_price": _f("m7_entry_long_price", 0.0),
        "m7_entry_short_price": _f("m7_entry_short_price", 0.0),
        "m7_tp_offset": _f("m7_tp_offset", 0.0),
        "m7_sl_offset": _f("m7_sl_offset", 0.0),
        "m7_tp_price": _f("m7_tp_price", 0.0),
        "m7_sl_price": _f("m7_sl_price", 0.0),
        "m7_trend_xgb_dn": float(np.clip(_f("m7_trend_xgb_dn", 0.0), 0.0, 1.0)),
        "m7_trend_xgb_fl": float(np.clip(_f("m7_trend_xgb_fl", 0.0), 0.0, 1.0)),
        "m7_trend_xgb_up": float(np.clip(_f("m7_trend_xgb_up", 0.0), 0.0, 1.0)),
        "m7_gmm_cluster": _f("m7_gmm_cluster", -1.0),
        "m7_gmm_conf": float(np.clip(_f("m7_gmm_conf", 0.0), 0.0, 1.0)),
        "m7_gmm_vol_rank": float(np.clip(_f("m7_gmm_vol_rank", 0.5), 0.0, 1.0)),
        "m7_hdb_label": _f("m7_hdb_label", -1.0),
        "m7_hdb_prob": float(np.clip(_f("m7_hdb_prob", 0.0), 0.0, 1.0)),
        "m7_iso_pred": _f("m7_iso_pred", 1.0),
        "m7_iso_score": _f("m7_iso_score", 0.0),
        "m7_iso_anom": 1.0 if _f("m7_iso_anom", 0.0) >= 0.5 else 0.0,
        "m7_vae_error": _f("m7_vae_error", 0.0),
        "m7_vae_threshold": _f("m7_vae_threshold", 0.0),
        "m7_vae_anom": 1.0 if _f("m7_vae_anom", 0.0) >= 0.5 else 0.0,
        "m7_expected_ret": _f("m7_expected_ret", 0.0),
        "m7_tail_risk": _f("m7_tail_risk", 0.0),
        "m7_composite_score": _f("m7_composite_score", 0.0),
    }
