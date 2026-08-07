#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_alpha5_13_hgb_high_quality_training_data_20260518 import (  # noqa: E402
    ATR_WINDOW,
    DEFAULT_2025,
    DEFAULT_2026,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    REGIME_BARRIER,
    REGIME_TRADE_RATIO,
    VOL_Z_COL,
    _align_union,
    _apply_split_policy,
    _atr_pct,
    _json_default,
    _read,
    _regime_name,
    _resolve_barrier_vote,
    _resolve_score_vote,
    _scan_event,
    _uniqueness_weights,
    _verify_state24_sticky090_inputs,
    _vol_override,
)


MODEL_ID = "alpha5_27_label_factory_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_27_label_factory_20260519"
ENTRY_STATE_MAP = {0: "clean_wait", 1: "ambiguous_wait", 2: "trade"}
DIRECTION_MAP = {0: "unknown", 1: "long", 2: "short"}
PATH_TYPE_MAP = {
    0: "clean_wait",
    1: "tp_first",
    2: "sl_first",
    3: "timeout",
    4: "ambiguous_conflict",
}
REGIME_EDGE_MIN = {
    "bull": 2.0,
    "bear": 2.0,
    "chop": 1.5,
    "whipsaw": 9e9,
}
REGIME_CONS_MIN = {
    "bull": 2.0 / 3.0,
    "bear": 2.0 / 3.0,
    "chop": 2.0 / 3.0,
    "whipsaw": 9e9,
}
REGIME_INSTABILITY_PROB_MAX = 0.55
REGIME_CONFIDENCE_MIN = 0.45
REGIME_MARGIN_MIN = 0.08
REGIME_ENTROPY_MAX = 1.10


def _safe_sharpe(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    std = float(np.std(rets))
    if std < 1e-12:
        return 0.0
    return float(np.mean(rets) / std * math.sqrt(len(rets)))


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def _class_weights(labels: np.ndarray, mask: np.ndarray) -> dict[int, float]:
    y = np.asarray(labels, dtype=np.int64)
    m = np.asarray(mask, dtype=bool)
    y = y[m]
    if len(y) == 0:
        return {}
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    return {int(cls): float(total / (len(classes) * max(float(cnt), 1.0))) for cls, cnt in zip(classes, counts)}


def _scan_path_metrics(entry: float, future_close: np.ndarray, future_high: np.ndarray, future_low: np.ndarray) -> dict[str, float]:
    if len(future_close) == 0:
        return {
            "long_ret_6": 0.0, "short_ret_6": 0.0, "long_ret_12": 0.0, "short_ret_12": 0.0,
            "long_adv_6": 0.0, "short_adv_6": 0.0, "long_adv_12": 0.0, "short_adv_12": 0.0,
            "long_mfe_raw": 0.0, "long_mae_raw": 0.0, "short_mfe_raw": 0.0, "short_mae_raw": 0.0,
            "long_path_sharpe": 0.0, "short_path_sharpe": 0.0,
        }

    def _ret_at(k: int) -> float:
        idx = min(max(k - 1, 0), len(future_close) - 1)
        return float(future_close[idx] / entry - 1.0)

    n6 = min(6, len(future_close))
    n12 = min(12, len(future_close))
    long_adv_6 = float(abs(min(np.min(future_low[:n6] / entry - 1.0), 0.0)))
    short_adv_6 = float(max(np.max(future_high[:n6] / entry - 1.0), 0.0))
    long_adv_12 = float(abs(min(np.min(future_low[:n12] / entry - 1.0), 0.0)))
    short_adv_12 = float(max(np.max(future_high[:n12] / entry - 1.0), 0.0))
    long_mfe_raw = float(max(np.max(future_high / entry - 1.0), 0.0))
    long_mae_raw = float(abs(min(np.min(future_low / entry - 1.0), 0.0)))
    short_mfe_raw = float(max(np.max(1.0 - future_low / entry), 0.0))
    short_mae_raw = float(max(np.max(future_high / entry - 1.0), 0.0))
    path_rets = np.diff(np.r_[entry, future_close]) / np.r_[entry, future_close[:-1]]
    return {
        "long_ret_6": _ret_at(6),
        "short_ret_6": -_ret_at(6),
        "long_ret_12": _ret_at(12),
        "short_ret_12": -_ret_at(12),
        "long_adv_6": long_adv_6,
        "short_adv_6": short_adv_6,
        "long_adv_12": long_adv_12,
        "short_adv_12": short_adv_12,
        "long_mfe_raw": long_mfe_raw,
        "long_mae_raw": long_mae_raw,
        "short_mfe_raw": short_mfe_raw,
        "short_mae_raw": short_mae_raw,
        "long_path_sharpe": _safe_sharpe(path_rets),
        "short_path_sharpe": _safe_sharpe(-path_rets),
    }


def _regime_instability(frame: pd.DataFrame) -> np.ndarray:
    instability = _num(frame, "clean_regime4_2024_unsup_v1_instability_prob", 0.0)
    entropy = _num(frame, "clean_regime4_2024_unsup_v1_entropy", 0.0)
    confidence = _num(frame, "clean_regime4_2024_unsup_v1_confidence", 0.0)
    margin = _num(frame, "clean_regime4_2024_unsup_v1_margin", 0.0)
    return (
        (instability > REGIME_INSTABILITY_PROB_MAX)
        | (entropy > REGIME_ENTROPY_MAX)
        | (confidence < REGIME_CONFIDENCE_MIN)
        | (margin < REGIME_MARGIN_MIN)
    ).astype(np.int8)


def _label_frame(
    frame: pd.DataFrame,
    *,
    max_hold: int,
    atr_window: int,
    fixed_tp_pct: float,
    fixed_sl_pct: float,
    entry_event_ret_min: float,
) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(np.float64)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(np.float64)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(np.float64)
    vol_z = pd.to_numeric(out.get(VOL_Z_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    atr_pct = _atr_pct(out, window=atr_window)
    regime_name = _regime_name(out)
    regime_instability_flag = _regime_instability(out)
    n = len(out)

    label_action = np.full(n, -1, dtype=np.int16)
    label_vote_action = np.full(n, -1, dtype=np.int16)
    label_fixed_tp05_action = np.full(n, -1, dtype=np.int16)
    label_sharpe_action = np.full(n, -1, dtype=np.int16)
    label_mfe_action = np.full(n, -1, dtype=np.int16)
    label_consensus = np.zeros(n, dtype=np.float32)
    label_confidence = np.zeros(n, dtype=np.float32)
    label_event_end_idx = np.full(n, -1, dtype=np.int32)
    label_event_bars = np.full(n, -1, dtype=np.int16)
    label_tp_pct = np.full(n, np.nan, dtype=np.float32)
    label_sl_pct = np.full(n, np.nan, dtype=np.float32)
    label_valid = np.zeros(n, dtype=np.int8)
    regime_trade_selected = np.zeros(n, dtype=np.int8)

    meta_event_return = np.full(n, np.nan, dtype=np.float32)
    meta_tp_first = np.zeros(n, dtype=np.int8)
    meta_adverse_first = np.zeros(n, dtype=np.int8)
    meta_timeout = np.zeros(n, dtype=np.int8)
    meta_primary_reason = np.full(n, "", dtype=object)
    meta_raw_terminal_return = np.full(n, np.nan, dtype=np.float32)
    meta_long_score = np.full(n, np.nan, dtype=np.float32)
    meta_short_score = np.full(n, np.nan, dtype=np.float32)
    meta_edge_gap = np.full(n, np.nan, dtype=np.float32)
    meta_is_profitable = np.zeros(n, dtype=np.int8)

    long_ret_6 = np.full(n, np.nan, dtype=np.float32)
    short_ret_6 = np.full(n, np.nan, dtype=np.float32)
    long_ret_12 = np.full(n, np.nan, dtype=np.float32)
    short_ret_12 = np.full(n, np.nan, dtype=np.float32)
    long_adv_6 = np.full(n, np.nan, dtype=np.float32)
    short_adv_6 = np.full(n, np.nan, dtype=np.float32)
    long_adv_12 = np.full(n, np.nan, dtype=np.float32)
    short_adv_12 = np.full(n, np.nan, dtype=np.float32)
    long_mfe_raw = np.full(n, np.nan, dtype=np.float32)
    long_mae_raw = np.full(n, np.nan, dtype=np.float32)
    short_mfe_raw = np.full(n, np.nan, dtype=np.float32)
    short_mae_raw = np.full(n, np.nan, dtype=np.float32)
    direction_conflict_flag = np.zeros(n, dtype=np.int8)

    for i in range(atr_window, n - max_hold - 1):
        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        future_close = close[i + 1 : i + 1 + max_hold]
        future_high = high[i + 1 : i + 1 + max_hold]
        future_low = low[i + 1 : i + 1 + max_hold]
        if len(future_close) < max_hold:
            continue

        base_tp_mult, base_sl_mult = REGIME_BARRIER[str(regime_name[i])]
        tp_mult, sl_mult = _vol_override(base_tp_mult, base_sl_mult, float(vol_z[i]))
        tp_pct = float(max(atr_pct[i] * tp_mult, 1e-4))
        sl_pct = float(max(atr_pct[i] * sl_mult, 1e-4))
        label_tp_pct[i] = tp_pct
        label_sl_pct[i] = sl_pct

        path = _scan_event(entry, future_close, future_high, future_low, tp_pct, sl_pct, fixed_tp_pct, fixed_sl_pct)
        extra = _scan_path_metrics(entry, future_close, future_high, future_low)

        pri_vote, pri_reason, pri_bar = _resolve_barrier_vote(
            path["long_primary_tp_bar"], path["long_primary_sl_bar"], path["short_primary_tp_bar"], path["short_primary_sl_bar"]
        )
        fx_vote, _, _ = _resolve_barrier_vote(
            path["long_fixed_tp_bar"], path["long_fixed_sl_bar"], path["short_fixed_tp_bar"], path["short_fixed_sl_bar"]
        )
        sh_vote = _resolve_score_vote(path["long_sharpe"], path["short_sharpe"], threshold=0.20, margin=0.05)
        mf_vote = _resolve_score_vote(path["long_mfe_mae"], path["short_mfe_mae"], threshold=0.0005, margin=0.0002)

        votes = np.asarray([pri_vote, sh_vote, mf_vote], dtype=np.int16)
        counts = np.bincount(np.clip(votes, 0, 2), minlength=3)
        top_count = int(np.max(counts))
        ensemble_vote = int(np.argmax(counts))
        if top_count < 2:
            ensemble_vote = int(pri_vote)

        label_vote_action[i] = int(ensemble_vote)
        label_action[i] = int(ensemble_vote)
        label_fixed_tp05_action[i] = int(fx_vote)
        label_sharpe_action[i] = int(sh_vote)
        label_mfe_action[i] = int(mf_vote)
        label_consensus[i] = float(top_count / 3.0)
        label_confidence[i] = float(top_count / 3.0)

        long_success = path["long_primary_tp_bar"] < path["long_primary_sl_bar"]
        short_success = path["short_primary_tp_bar"] < path["short_primary_sl_bar"]
        direction_conflict_flag[i] = int(long_success and short_success)

        if pri_vote == 1:
            ret = tp_pct if path["long_primary_tp_bar"] < path["long_primary_sl_bar"] else -sl_pct
            if path["long_primary_tp_bar"] >= 10**9 and path["long_primary_sl_bar"] >= 10**9:
                ret = float(path["long_timeout_ret"])
        elif pri_vote == 2:
            ret = tp_pct if path["short_primary_tp_bar"] < path["short_primary_sl_bar"] else -sl_pct
            if path["short_primary_tp_bar"] >= 10**9 and path["short_primary_sl_bar"] >= 10**9:
                ret = float(path["short_timeout_ret"])
        else:
            ret = 0.0

        if pri_bar <= 0 or pri_bar >= 10**9:
            pri_bar = max_hold
        label_event_bars[i] = int(pri_bar)
        label_event_end_idx[i] = int(i + pri_bar)
        meta_event_return[i] = float(ret)
        meta_raw_terminal_return[i] = float(path["raw_terminal_ret"])
        meta_primary_reason[i] = str(pri_reason)
        if "tp_first" in pri_reason:
            meta_tp_first[i] = 1
        elif pri_reason == "adverse_or_timeout":
            if (path["long_primary_sl_bar"] < 10**9) or (path["short_primary_sl_bar"] < 10**9):
                meta_adverse_first[i] = 1
            else:
                meta_timeout[i] = 1
        elif pri_reason == "tie_conflict":
            meta_timeout[i] = 1

        meta_long_score[i] = float(path["long_sharpe"] + path["long_mfe_mae"])
        meta_short_score[i] = float(path["short_sharpe"] + path["short_mfe_mae"])
        meta_edge_gap[i] = float(abs(meta_long_score[i] - meta_short_score[i]))
        meta_is_profitable[i] = int((ensemble_vote in (1, 2)) and (ret > 0.0))

        long_ret_6[i] = float(extra["long_ret_6"])
        short_ret_6[i] = float(extra["short_ret_6"])
        long_ret_12[i] = float(extra["long_ret_12"])
        short_ret_12[i] = float(extra["short_ret_12"])
        long_adv_6[i] = float(extra["long_adv_6"])
        short_adv_6[i] = float(extra["short_adv_6"])
        long_adv_12[i] = float(extra["long_adv_12"])
        short_adv_12[i] = float(extra["short_adv_12"])
        long_mfe_raw[i] = float(extra["long_mfe_raw"])
        long_mae_raw[i] = float(extra["long_mae_raw"])
        short_mfe_raw[i] = float(extra["short_mfe_raw"])
        short_mae_raw[i] = float(extra["short_mae_raw"])
        label_valid[i] = 1

        if (i % 10000) == 0 and i > 0:
            print(json.dumps({"stage": "label_progress", "done": int(i), "total": int(n)}, ensure_ascii=False), flush=True)

    raw_action = label_action.copy()
    for regime, target_trade_ratio in REGIME_TRADE_RATIO.items():
        reg_mask = (regime_name == regime) & (raw_action >= 0)
        reg_idx = np.flatnonzero(reg_mask)
        if len(reg_idx) == 0:
            continue
        cand_idx = reg_idx[raw_action[reg_idx] != 0]
        target_n = int(round(len(reg_idx) * float(target_trade_ratio)))
        target_n = max(0, min(target_n, len(cand_idx)))
        if target_n == 0:
            label_action[cand_idx] = 0
            continue
        if len(cand_idx) > target_n:
            order = sorted(
                cand_idx.tolist(),
                key=lambda j: (
                    float(label_consensus[j]),
                    float(meta_edge_gap[j]) if np.isfinite(meta_edge_gap[j]) else -1.0,
                    float(label_confidence[j]),
                ),
                reverse=True,
            )
            selected = np.asarray(order[:target_n], dtype=np.int32)
            dropped = np.asarray(order[target_n:], dtype=np.int32)
            label_action[dropped] = 0
            regime_trade_selected[selected] = 1
        else:
            regime_trade_selected[cand_idx] = 1

    out["regime4_state"] = regime_name
    out["regime_instability_flag"] = regime_instability_flag
    out["atr14_pct"] = atr_pct.astype(np.float32)
    out["label_action"] = label_action
    out["label_vote_action"] = raw_action
    out["label_primary_action"] = label_action
    out["label_fixed_tp05_action"] = label_fixed_tp05_action
    out["label_sharpe_action"] = label_sharpe_action
    out["label_mfe_action"] = label_mfe_action
    out["label_consensus"] = label_consensus
    out["label_confidence"] = label_confidence
    out["label_event_end_idx"] = label_event_end_idx
    out["label_event_bars"] = label_event_bars
    out["label_tp_pct"] = label_tp_pct
    out["label_sl_pct"] = label_sl_pct
    out["label_valid"] = label_valid
    out["regime_trade_selected"] = regime_trade_selected
    out["meta_event_return"] = meta_event_return
    out["meta_tp_first"] = meta_tp_first
    out["meta_adverse_first"] = meta_adverse_first
    out["meta_timeout"] = meta_timeout
    out["meta_primary_reason"] = meta_primary_reason
    out["meta_raw_terminal_return"] = meta_raw_terminal_return
    out["meta_long_score"] = meta_long_score
    out["meta_short_score"] = meta_short_score
    out["meta_edge_gap"] = meta_edge_gap
    out["meta_is_profitable"] = meta_is_profitable
    out["meta_long_ret_6"] = long_ret_6
    out["meta_short_ret_6"] = short_ret_6
    out["meta_long_ret_12"] = long_ret_12
    out["meta_short_ret_12"] = short_ret_12
    out["meta_long_adv_6"] = long_adv_6
    out["meta_short_adv_6"] = short_adv_6
    out["meta_long_adv_12"] = long_adv_12
    out["meta_short_adv_12"] = short_adv_12
    out["meta_long_mfe_raw"] = long_mfe_raw
    out["meta_long_mae_raw"] = long_mae_raw
    out["meta_short_mfe_raw"] = short_mfe_raw
    out["meta_short_mae_raw"] = short_mae_raw
    out["direction_conflict_flag"] = direction_conflict_flag

    return out


def _derive_contract(frame: pd.DataFrame, *, entry_event_ret_min: float) -> pd.DataFrame:
    out = frame.copy()
    action = _num(out, "label_action", 0.0).astype(np.int64)
    consensus = _num(out, "label_consensus", 0.0)
    confidence = _num(out, "label_confidence", 0.0)
    uniq = _num(out, "sample_uniqueness_weight", 0.0)
    edge_gap = _num(out, "meta_edge_gap", 0.0)
    event_ret = _num(out, "meta_event_return", 0.0)
    raw_ret = _num(out, "meta_raw_terminal_return", 0.0)
    best_score = np.maximum(_num(out, "meta_long_score", 0.0), _num(out, "meta_short_score", 0.0))
    tp_first = _num(out, "meta_tp_first", 0.0).astype(np.int8)
    profitable = _num(out, "meta_is_profitable", 0.0).astype(np.int8)
    timeout = _num(out, "meta_timeout", 0.0).astype(np.int8)
    selected = _num(out, "regime_trade_selected", 0.0).astype(np.int8)
    regime = out["regime4_state"].astype(str).to_numpy()
    instability = _num(out, "regime_instability_flag", 0.0).astype(np.int8)
    conflict = _num(out, "direction_conflict_flag", 0.0).astype(np.int8)
    sl_pct = np.clip(_num(out, "label_sl_pct", 0.0), 1e-6, None)
    long_adv6 = _num(out, "meta_long_adv_6", 0.0)
    short_adv6 = _num(out, "meta_short_adv_6", 0.0)
    long_ret6 = _num(out, "meta_long_ret_6", 0.0)
    short_ret6 = _num(out, "meta_short_ret_6", 0.0)

    ambiguity_flag = np.zeros(len(out), dtype=np.int8)
    for reg in ("bull", "bear", "chop", "whipsaw"):
        reg_mask = regime == reg
        ambiguity_flag[reg_mask & ((edge_gap < REGIME_EDGE_MIN[reg]) | (consensus < REGIME_CONS_MIN[reg]))] = 1
    ambiguity_flag |= conflict.astype(np.int8)
    ambiguity_flag |= instability.astype(np.int8)

    signed_ret6 = np.where(action == 1, long_ret6, np.where(action == 2, short_ret6, 0.0))
    signed_adv6 = np.where(action == 1, long_adv6, np.where(action == 2, short_adv6, 0.0))
    early_adverse_ok = signed_adv6 <= (0.85 * sl_pct)
    early_drift_ok = signed_ret6 >= -0.0015

    trade_mask = (
        (action != 0)
        & (tp_first == 1)
        & (profitable == 1)
        & (event_ret >= float(entry_event_ret_min))
        & (selected == 1)
        & (regime != "whipsaw")
        & (ambiguity_flag == 0)
        & early_adverse_ok
        & early_drift_ok
    )

    clean_wait_mask = (
        (action == 0)
        & (best_score < 0.75)
        & (event_ret < float(entry_event_ret_min))
        & (np.abs(raw_ret) < float(entry_event_ret_min))
        & (conflict == 0)
    )

    entry_state = np.where(trade_mask, 2, np.where(clean_wait_mask, 0, 1)).astype(np.int8)
    direction_label = np.where(trade_mask, action, 0).astype(np.int8)

    path_type = np.full(len(out), 0, dtype=np.int8)
    path_type[conflict == 1] = 4
    path_type[(conflict == 0) & (tp_first == 1) & (action != 0)] = 1
    path_type[(conflict == 0) & (_num(out, "meta_adverse_first", 0.0).astype(np.int8) == 1) & (action != 0)] = 2
    path_type[(conflict == 0) & (timeout == 1) & (action != 0)] = 3

    signed_event = np.where(action == 0, 0.0, event_ret)
    signed_adv = np.where(action == 1, long_adv6, np.where(action == 2, short_adv6, 0.0))
    quality_raw = (
        1.50 * np.clip(signed_event / 0.01, -2.0, 2.0)
        + 0.70 * np.clip(edge_gap / 4.0, 0.0, 2.0)
        + 0.40 * np.clip(signed_ret6 / 0.005, -2.0, 2.0)
        - 0.60 * np.clip(signed_adv / sl_pct, 0.0, 3.0)
        - 0.70 * ambiguity_flag.astype(np.float64)
        - 0.50 * timeout.astype(np.float64)
        - 0.80 * conflict.astype(np.float64)
    )
    quality_score = np.tanh(quality_raw / 2.0).astype(np.float32)

    entry_keep = (pd.to_numeric(out["split_keep"], errors="coerce").fillna(0).to_numpy(np.int8) == 1).astype(np.int8)
    direction_keep = (direction_label != 0).astype(np.int8)
    entry_weight = np.clip(np.abs(quality_score).astype(np.float64) + 0.15, 1e-4, None)
    entry_weight *= (0.85 + 0.20 * confidence)
    entry_weight *= (0.85 + 0.20 * np.clip(uniq, 0.0, 1.0))
    entry_weight *= np.where(entry_state == 1, 0.75, 1.0)
    direction_weight = np.clip(np.abs(quality_score).astype(np.float64) + 0.20, 1e-4, None)
    direction_weight *= (1.0 + 0.10 * np.clip(edge_gap, 0.0, 8.0))
    direction_weight *= np.where(direction_label != 0, 1.0, 0.0)

    entry_cw = _class_weights(entry_state, entry_keep == 1)
    if entry_cw:
        entry_weight *= np.asarray([entry_cw.get(int(y), 0.0) for y in entry_state], dtype=np.float64)
    dir_cw = _class_weights(direction_label, direction_keep == 1)
    if dir_cw:
        direction_weight *= np.asarray([dir_cw.get(int(y), 0.0) for y in direction_label], dtype=np.float64)

    wait_contam = (entry_state == 0) & ((best_score >= 0.75) | (event_ret >= float(entry_event_ret_min)) | (np.abs(raw_ret) >= float(entry_event_ret_min)))
    trade_contam = (entry_state == 2) & (
        (tp_first != 1) | (profitable != 1) | (event_ret < float(entry_event_ret_min)) | (ambiguity_flag == 1)
    )

    out["entry_state"] = entry_state
    out["entry_state_name"] = np.asarray([ENTRY_STATE_MAP[int(x)] for x in entry_state], dtype=object)
    out["entry_binary_label"] = (entry_state == 2).astype(np.int8)
    out["direction_label"] = direction_label
    out["direction_name"] = np.asarray([DIRECTION_MAP[int(x)] for x in direction_label], dtype=object)
    out["quality_score"] = quality_score
    out["path_type"] = path_type
    out["path_type_name"] = np.asarray([PATH_TYPE_MAP[int(x)] for x in path_type], dtype=object)
    out["ambiguity_flag"] = ambiguity_flag.astype(np.int8)
    out["clean_wait_contamination_flag"] = wait_contam.astype(np.int8)
    out["trade_contamination_flag"] = trade_contam.astype(np.int8)
    out["entry_train_keep"] = entry_keep.astype(np.int8)
    out["direction_train_keep"] = direction_keep.astype(np.int8)
    out["entry_sample_weight"] = entry_weight.astype(np.float32)
    out["direction_sample_weight"] = direction_weight.astype(np.float32)
    return out


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    split_keep = _num(frame, "split_keep", 0.0).astype(np.int8) == 1
    work = frame.loc[split_keep].copy()
    state = _num(work, "entry_state", 1.0).astype(np.int64)
    direction = _num(work, "direction_label", 0.0).astype(np.int64)
    by_state = {ENTRY_STATE_MAP[int(k)]: int(v) for k, v in pd.Series(state).value_counts().sort_index().to_dict().items()}
    by_direction = {DIRECTION_MAP[int(k)]: int(v) for k, v in pd.Series(direction).value_counts().sort_index().to_dict().items()}
    month = pd.to_datetime(work["timestamp"], errors="coerce").dt.to_period("M").astype(str)
    monthly = []
    for key, grp in work.groupby(month):
        monthly.append({
            "month": key,
            "rows": int(len(grp)),
            "trade_ratio": float(np.mean(_num(grp, "entry_state", 0.0) == 2)),
            "ambiguous_ratio": float(np.mean(_num(grp, "entry_state", 0.0) == 1)),
            "direction_valid_ratio": float(np.mean(_num(grp, "direction_label", 0.0) != 0)),
            "quality_mean": float(pd.to_numeric(grp["quality_score"], errors="coerce").fillna(0.0).mean()),
        })
    regime_purity = {}
    for reg, grp in work.groupby("regime4_state"):
        d = _num(grp, "direction_label", 0.0).astype(np.int64)
        raw_ret = _num(grp, "meta_raw_terminal_return", 0.0)
        long_mask = d == 1
        short_mask = d == 2
        regime_purity[str(reg)] = {
            "rows": int(len(grp)),
            "trade_rows": int(np.sum(_num(grp, "entry_state", 0.0) == 2)),
            "ambiguous_rows": int(np.sum(_num(grp, "entry_state", 0.0) == 1)),
            "direction_rows": int(np.sum(d != 0)),
            "long_purity": float(np.mean(raw_ret[long_mask] > 0.0)) if np.any(long_mask) else 0.0,
            "short_purity": float(np.mean(raw_ret[short_mask] < 0.0)) if np.any(short_mask) else 0.0,
        }
    return {
        "rows": int(len(work)),
        "entry_state_counts": by_state,
        "direction_counts": by_direction,
        "trade_ratio": float(np.mean(state == 2)),
        "ambiguous_ratio": float(np.mean(state == 1)),
        "clean_wait_ratio": float(np.mean(state == 0)),
        "direction_valid_ratio": float(np.mean(direction != 0)),
        "clean_wait_contamination_rate": float(np.mean(_num(work, "clean_wait_contamination_flag", 0.0)[state == 0])) if np.any(state == 0) else 0.0,
        "trade_contamination_rate": float(np.mean(_num(work, "trade_contamination_flag", 0.0)[state == 2])) if np.any(state == 2) else 0.0,
        "ambiguity_rate": float(np.mean(_num(work, "ambiguity_flag", 0.0))),
        "regime_instability_rate": float(np.mean(_num(work, "regime_instability_flag", 0.0))),
        "quality_mean": float(pd.to_numeric(work["quality_score"], errors="coerce").fillna(0.0).mean()),
        "quality_trade_mean": float(pd.to_numeric(work.loc[state == 2, "quality_score"], errors="coerce").fillna(0.0).mean()) if np.any(state == 2) else 0.0,
        "quality_wait_mean": float(pd.to_numeric(work.loc[state == 0, "quality_score"], errors="coerce").fillna(0.0).mean()) if np.any(state == 0) else 0.0,
        "regime_purity": regime_purity,
        "monthly": monthly,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build alpha5_27 label factory with clean/ambiguous/trade entry states and independent direction/quality targets.")
    p.add_argument("--train-2025-csv", type=Path, default=DEFAULT_2025)
    p.add_argument("--oos-2026-csv", type=Path, default=DEFAULT_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-end", default="2025-10-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="2026-01-01")
    p.add_argument("--oos-start", default="2026-01-01")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--atr-window", type=int, default=ATR_WINDOW)
    p.add_argument("--fixed-tp-pct", type=float, default=0.005)
    p.add_argument("--fixed-sl-pct", type=float, default=0.005)
    p.add_argument("--embargo-bars", type=int, default=288)
    p.add_argument("--warmup-bars", type=int, default=288)
    p.add_argument("--entry-event-ret-min", type=float, default=0.0045)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.train_2025_csv)
    raw_2026 = _read(args.oos_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)
    a, b, union_cols = _align_union(raw_2025, raw_2026)
    combined = pd.concat([a, b], axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "rows_2025": int(len(raw_2025)),
        "rows_2026": int(len(raw_2026)),
        "union_cols": int(len(union_cols)),
        "entry_event_ret_min": float(args.entry_event_ret_min),
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    labeled = _label_frame(
        combined,
        max_hold=int(args.max_hold_bars),
        atr_window=int(args.atr_window),
        fixed_tp_pct=float(args.fixed_tp_pct),
        fixed_sl_pct=float(args.fixed_sl_pct),
        entry_event_ret_min=float(args.entry_event_ret_min),
    )
    labeled["sample_uniqueness_weight"] = _uniqueness_weights(labeled)

    labeled, split_meta = _apply_split_policy(
        labeled,
        train_end=str(args.train_end),
        val_start=str(args.val_start),
        val_end=str(args.val_end),
        oos_start=str(args.oos_start),
        embargo_bars=int(args.embargo_bars),
        warmup_bars=int(args.warmup_bars),
    )
    labeled = _derive_contract(labeled, entry_event_ret_min=float(args.entry_event_ret_min))

    train_df = labeled[labeled["dataset_split"] == "train"].reset_index(drop=True)
    val_df = labeled[labeled["dataset_split"] == "validation"].reset_index(drop=True)
    oos_df = labeled[labeled["dataset_split"] == "oos"].reset_index(drop=True)

    report = {
        "model_id": MODEL_ID,
        "config": {
            "entry_event_ret_min": float(args.entry_event_ret_min),
            "regime_edge_min": REGIME_EDGE_MIN,
            "regime_consensus_min": REGIME_CONS_MIN,
            "instability_prob_max": REGIME_INSTABILITY_PROB_MAX,
            "confidence_min": REGIME_CONFIDENCE_MIN,
            "margin_min": REGIME_MARGIN_MIN,
            "entropy_max": REGIME_ENTROPY_MAX,
        },
        "audit": audit,
        "split_meta": split_meta,
        "train": _report(train_df),
        "validation": _report(val_df),
        "oos": _report(oos_df),
    }

    train_path = args.out_dir / "alpha5_27_label_factory_train.parquet"
    val_path = args.out_dir / "alpha5_27_label_factory_val.parquet"
    oos_path = args.out_dir / "alpha5_27_label_factory_oos.parquet"
    report_path = args.out_dir / "alpha5_27_label_factory_report.json"
    summary_csv = args.out_dir / "alpha5_27_label_factory_summary.csv"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    oos_df.to_parquet(oos_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {
            "split": "train",
            "rows": len(train_df),
            "trade_ratio": report["train"]["trade_ratio"],
            "ambiguous_ratio": report["train"]["ambiguous_ratio"],
            "clean_wait_ratio": report["train"]["clean_wait_ratio"],
            "direction_valid_ratio": report["train"]["direction_valid_ratio"],
            "clean_wait_contamination_rate": report["train"]["clean_wait_contamination_rate"],
            "trade_contamination_rate": report["train"]["trade_contamination_rate"],
            "quality_mean": report["train"]["quality_mean"],
        },
        {
            "split": "validation",
            "rows": len(val_df),
            "trade_ratio": report["validation"]["trade_ratio"],
            "ambiguous_ratio": report["validation"]["ambiguous_ratio"],
            "clean_wait_ratio": report["validation"]["clean_wait_ratio"],
            "direction_valid_ratio": report["validation"]["direction_valid_ratio"],
            "clean_wait_contamination_rate": report["validation"]["clean_wait_contamination_rate"],
            "trade_contamination_rate": report["validation"]["trade_contamination_rate"],
            "quality_mean": report["validation"]["quality_mean"],
        },
        {
            "split": "oos",
            "rows": len(oos_df),
            "trade_ratio": report["oos"]["trade_ratio"],
            "ambiguous_ratio": report["oos"]["ambiguous_ratio"],
            "clean_wait_ratio": report["oos"]["clean_wait_ratio"],
            "direction_valid_ratio": report["oos"]["direction_valid_ratio"],
            "clean_wait_contamination_rate": report["oos"]["clean_wait_contamination_rate"],
            "trade_contamination_rate": report["oos"]["trade_contamination_rate"],
            "quality_mean": report["oos"]["quality_mean"],
        },
    ]).to_csv(summary_csv, index=False)

    print(json.dumps({
        "stage": "complete",
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "oos_path": str(oos_path),
        "report_path": str(report_path),
        "summary_csv": str(summary_csv),
        "train_report": report["train"],
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
