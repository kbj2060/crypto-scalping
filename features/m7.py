from __future__ import annotations

import numpy as np


def trend_signal_from_m7(m7_last: dict | None) -> dict | None:
    """Convert allowed M7 risk/quality outputs without consuming M7 direction heads."""
    if not isinstance(m7_last, dict) or not m7_last:
        return None

    def _f(key: str, default: float = 0.0) -> float:  # noqa: ARG001
        return float(m7_last[key]) if key in m7_last else float(default)

    quality = float(np.clip(_f("m7_quality_pred", 0.0), 0.0, 1.0))

    return {
        "trend_dir": 1,
        "strength": 0.0,
        "rev_prob": 1.0,
        "prob_dn": 0.5,
        "prob_up": 0.5,
        "probs": [0.5, 0.5],
        "trend_model": "M7_DIRECTION_DISABLED",
        "m7_quality_pred": quality,
        "m7_hold_pred": _f("m7_hold_pred", 0.0),
        "m7_target_hold": float(max(0.0, _f("m7_target_hold", 0.0))),
        "m7_q10": _f("m7_q10", 0.0),
        "m7_q90": _f("m7_q90", 0.0),
        "m7_qwidth": float(max(0.0, _f("m7_qwidth", 0.0))),
        "m7_entry_long_offset": _f("m7_entry_long_offset", 0.0),
        "m7_entry_short_offset": _f("m7_entry_short_offset", 0.0),
        "m7_tp_offset": _f("m7_tp_offset", 0.0),
        "m7_sl_offset": _f("m7_sl_offset", 0.0),
        "m7_tail_risk": _f("m7_tail_risk", 0.0),
    }
