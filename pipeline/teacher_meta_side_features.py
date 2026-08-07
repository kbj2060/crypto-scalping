from __future__ import annotations

import numpy as np
import pandas as pd


TEACHER_FEATURE_COLS = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]

REQUIRED_TEACHER_INPUTS = [
    "m7_q10",
    "m7_q50",
    "m7_q90",
    "m7_qwidth",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_expected_ret",
    "m7_tail_risk",
    "ai_adverse_risk",
    "ai_reward_risk",
]


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _require_inputs(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_TEACHER_INPUTS if col not in frame.columns]
    if missing:
        raise ValueError(f"teacher_formula_missing_inputs:{missing}")


def append_side_teacher_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic, no-fit teacher meta features from OOS model outputs.

    This layer is intentionally a compressor of M7/AI risk context. It must not
    consume labels, target columns, realized PnL, or downstream action scores.
    """
    _require_inputs(frame)
    out = frame.copy()
    q10 = _num(out, "m7_q10", 0.0)
    q50 = _num(out, "m7_q50", 0.0)
    q90 = _num(out, "m7_q90", 0.0)
    qwidth = _num(out, "m7_qwidth", (q90 - q10).abs()).abs()
    q_skew = (q90 + q10 - 2.0 * q50) / np.clip(qwidth, 1e-9, None)
    m7_quality = _num(out, "m7_quality_pred", 0.0).clip(0.0, 1.0)
    expected_ret = _num(out, "m7_expected_ret", 0.0).clip(-0.05, 0.05)
    adverse = _num(out, "ai_adverse_risk", 0.0).clip(0.0, 3.0)
    reward_risk = _num(out, "ai_reward_risk", 0.0).clip(-3.0, 3.0)
    tail_risk = _num(out, "m7_tail_risk", 0.0).abs().clip(0.0, 3.0)
    transition_risk = _num(out, "regime3_transition_h6_risk_prob", 0.0).clip(0.0, 1.0)
    churn_risk = _num(out, "regime3_churn_h6_risk_score", 0.0).clip(0.0, 1.0)
    chronos_width = (
        _num(out, "chronos_atr14_width", 0.0).abs()
        + _num(out, "chronos_realized_vol24_width", 0.0).abs()
    )
    uncertainty = np.clip(
        qwidth / 0.02
        + 0.35 * chronos_width / 0.02
        + 0.70 * transition_risk
        + 0.45 * churn_risk,
        0.0,
        3.0,
    )
    long_adverse = _num(out, "m7_long_adverse_prob", adverse).clip(0.0, 3.0)
    short_adverse = _num(out, "m7_short_adverse_prob", adverse).clip(0.0, 3.0)
    ret_long = np.maximum(expected_ret, 0.0) / 0.01
    ret_short = np.maximum(-expected_ret, 0.0) / 0.01
    skew_long = np.maximum(q_skew, 0.0)
    skew_short = np.maximum(-q_skew, 0.0)

    out["teacher_long_edge"] = np.clip(
        0.45 * m7_quality + 0.30 * ret_long + 0.15 * skew_long + 0.10 * np.maximum(reward_risk, 0.0)
        - 0.25 * long_adverse - 0.20 * uncertainty,
        -3.0,
        3.0,
    )
    out["teacher_short_edge"] = np.clip(
        0.45 * m7_quality + 0.30 * ret_short + 0.15 * skew_short + 0.10 * np.maximum(reward_risk, 0.0)
        - 0.25 * short_adverse - 0.20 * uncertainty,
        -3.0,
        3.0,
    )
    out["teacher_side_margin"] = (out["teacher_long_edge"] - out["teacher_short_edge"]).abs()
    out["teacher_side_disagreement"] = np.clip(
        (np.sign(expected_ret) != np.sign(q_skew)).astype(float) * (0.5 + 0.5 * m7_quality)
        + 0.25 * transition_risk,
        0.0,
        1.0,
    )
    out["teacher_quantile_skew"] = np.clip(q_skew, -3.0, 3.0)
    out["teacher_uncertainty"] = uncertainty
    out["teacher_tail_warning"] = np.clip(
        0.45 * tail_risk + 0.35 * adverse + 0.20 * uncertainty + 0.20 * _num(out, "liquidity_vacuum", 0.0).clip(0.0, 3.0),
        0.0,
        3.0,
    )
    return out
