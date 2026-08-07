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

from ensemble.supervised.train_trend_xgb import compute_atr  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"

REGIME_PROB_COLS = {
    "bull": "clean_regime4_2024_unsup_v1_bull_prob",
    "bear": "clean_regime4_2024_unsup_v1_bear_prob",
    "chop": "clean_regime4_2024_unsup_v1_chop_prob",
    "whipsaw": "clean_regime4_2024_unsup_v1_whipsaw_prob",
}
REGIME_BARRIER = {
    "bull": (3.0, 1.2),
    "bear": (3.0, 1.2),
    "chop": (2.0, 1.5),
    "whipsaw": (1.5, 0.8),
}
REGIME_TRADE_RATIO = {
    "bull": 0.56,
    "bear": 0.56,
    "chop": 0.44,
    "whipsaw": 0.24,
}
VOL_Z_COL = "volatility_z"
ATR_WINDOW = 14


def _align_union(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cols: list[str] = []
    for c in list(a.columns) + list(b.columns):
        if c not in cols:
            cols.append(c)
    a2 = a.copy()
    b2 = b.copy()
    for c in cols:
        if c not in a2.columns:
            a2[c] = np.nan
        if c not in b2.columns:
            b2[c] = np.nan
    return a2[cols].copy(), b2[cols].copy(), cols


def _regime_name(frame: pd.DataFrame) -> np.ndarray:
    names = list(REGIME_PROB_COLS.keys())
    arr = np.column_stack([pd.to_numeric(frame[REGIME_PROB_COLS[k]], errors="coerce").fillna(0.0).to_numpy(np.float64) for k in names])
    idx = np.argmax(arr, axis=1)
    return np.asarray([names[int(i)] for i in idx], dtype=object)


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(np.float64)
    atr = compute_atr(high, low, close, window=window)
    return atr / np.clip(close, 1e-12, None)


def _vol_override(tp_mult: float, sl_mult: float, vol_z: float) -> tuple[float, float]:
    if vol_z > 2.0:
        return tp_mult * 1.4, sl_mult * 1.3
    if vol_z < -1.0:
        return tp_mult * 0.7, sl_mult * 0.8
    return tp_mult, sl_mult


def _choose_tp_first(tp_bar: int, sl_bar: int, timeout_ret: float) -> tuple[int, float]:
    if tp_bar < sl_bar:
        return 1, float(tp_bar)
    if sl_bar < tp_bar:
        return -1, float(sl_bar)
    return 0, float(timeout_ret)


def _safe_sharpe(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    std = float(np.std(rets))
    if std < 1e-12:
        return 0.0
    return float(np.mean(rets) / std * math.sqrt(len(rets)))


def _scan_event(
    entry: float,
    future_close: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    fixed_tp_pct: float,
    fixed_sl_pct: float,
) -> dict[str, Any]:
    long_tp_px = entry * (1.0 + tp_pct)
    long_sl_px = entry * (1.0 - sl_pct)
    short_tp_px = entry * (1.0 - tp_pct)
    short_sl_px = entry * (1.0 + sl_pct)
    fixed_long_tp_px = entry * (1.0 + fixed_tp_pct)
    fixed_long_sl_px = entry * (1.0 - fixed_sl_pct)
    fixed_short_tp_px = entry * (1.0 - fixed_tp_pct)
    fixed_short_sl_px = entry * (1.0 + fixed_sl_pct)

    long_tp_bar = long_sl_bar = short_tp_bar = short_sl_bar = 10**9
    fx_long_tp_bar = fx_long_sl_bar = fx_short_tp_bar = fx_short_sl_bar = 10**9
    for k in range(len(future_close)):
        hi = float(future_high[k])
        lo = float(future_low[k])
        bar = k + 1
        if long_tp_bar == 10**9 and hi >= long_tp_px:
            long_tp_bar = bar
        if long_sl_bar == 10**9 and lo <= long_sl_px:
            long_sl_bar = bar
        if short_tp_bar == 10**9 and lo <= short_tp_px:
            short_tp_bar = bar
        if short_sl_bar == 10**9 and hi >= short_sl_px:
            short_sl_bar = bar
        if fx_long_tp_bar == 10**9 and hi >= fixed_long_tp_px:
            fx_long_tp_bar = bar
        if fx_long_sl_bar == 10**9 and lo <= fixed_long_sl_px:
            fx_long_sl_bar = bar
        if fx_short_tp_bar == 10**9 and lo <= fixed_short_tp_px:
            fx_short_tp_bar = bar
        if fx_short_sl_bar == 10**9 and hi >= fixed_short_sl_px:
            fx_short_sl_bar = bar

    final_close = float(future_close[-1])
    long_timeout_ret = final_close / entry - 1.0
    short_timeout_ret = -(final_close / entry - 1.0)
    long_rets = future_close / entry - 1.0
    short_rets = -long_rets

    long_primary_state, _ = _choose_tp_first(long_tp_bar, long_sl_bar, long_timeout_ret)
    short_primary_state, _ = _choose_tp_first(short_tp_bar, short_sl_bar, short_timeout_ret)
    long_fixed_state, _ = _choose_tp_first(fx_long_tp_bar, fx_long_sl_bar, long_timeout_ret)
    short_fixed_state, _ = _choose_tp_first(fx_short_tp_bar, fx_short_sl_bar, short_timeout_ret)

    long_mfe = float(np.max(long_rets)) if len(long_rets) else 0.0
    long_mae = float(abs(np.min(long_rets))) if len(long_rets) else 0.0
    short_mfe = float(np.max(short_rets)) if len(short_rets) else 0.0
    short_mae = float(abs(np.min(short_rets))) if len(short_rets) else 0.0

    long_mfe_mae = float((long_mfe / max(long_mae, 1e-6)) * long_timeout_ret)
    short_mfe_mae = float((short_mfe / max(short_mae, 1e-6)) * short_timeout_ret)
    long_sharpe = _safe_sharpe(np.diff(np.r_[entry, future_close]) / np.r_[entry, future_close[:-1]])
    short_sharpe = _safe_sharpe(-np.diff(np.r_[entry, future_close]) / np.r_[entry, future_close[:-1]])

    return {
        "long_primary_state": long_primary_state,
        "short_primary_state": short_primary_state,
        "long_primary_tp_bar": int(long_tp_bar),
        "long_primary_sl_bar": int(long_sl_bar),
        "short_primary_tp_bar": int(short_tp_bar),
        "short_primary_sl_bar": int(short_sl_bar),
        "long_fixed_state": long_fixed_state,
        "short_fixed_state": short_fixed_state,
        "long_fixed_tp_bar": int(fx_long_tp_bar),
        "long_fixed_sl_bar": int(fx_long_sl_bar),
        "short_fixed_tp_bar": int(fx_short_tp_bar),
        "short_fixed_sl_bar": int(fx_short_sl_bar),
        "long_timeout_ret": float(long_timeout_ret),
        "short_timeout_ret": float(short_timeout_ret),
        "raw_terminal_ret": float(final_close / entry - 1.0),
        "long_sharpe": float(long_sharpe),
        "short_sharpe": float(short_sharpe),
        "long_mfe_mae": float(long_mfe_mae),
        "short_mfe_mae": float(short_mfe_mae),
        "long_mfe": float(long_mfe),
        "long_mae": float(long_mae),
        "short_mfe": float(short_mfe),
        "short_mae": float(short_mae),
    }


def _resolve_barrier_vote(
    long_tp_bar: int,
    long_sl_bar: int,
    short_tp_bar: int,
    short_sl_bar: int,
) -> tuple[int, str, int]:
    long_success = long_tp_bar < long_sl_bar
    short_success = short_tp_bar < short_sl_bar
    if long_success and short_success:
        if long_tp_bar < short_tp_bar:
            return 1, "long_tp_first", int(long_tp_bar)
        if short_tp_bar < long_tp_bar:
            return 2, "short_tp_first", int(short_tp_bar)
        return 0, "tie_conflict", min(int(long_tp_bar), int(short_tp_bar))
    if long_success:
        return 1, "long_tp_first", int(long_tp_bar)
    if short_success:
        return 2, "short_tp_first", int(short_tp_bar)
    long_fail = min(int(long_sl_bar), int(short_sl_bar))
    return 0, "adverse_or_timeout", long_fail if long_fail < 10**9 else -1


def _resolve_score_vote(long_score: float, short_score: float, threshold: float, margin: float) -> int:
    best = max(long_score, short_score)
    if best < threshold:
        return 0
    if abs(long_score - short_score) < margin:
        return 0
    return 1 if long_score > short_score else 2


def _label_frame(frame: pd.DataFrame, *, max_hold: int, atr_window: int, fixed_tp_pct: float, fixed_sl_pct: float) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(np.float64)
    high = pd.to_numeric(out["high"], errors="coerce").ffill().to_numpy(np.float64)
    low = pd.to_numeric(out["low"], errors="coerce").ffill().to_numpy(np.float64)
    vol_z = pd.to_numeric(out.get(VOL_Z_COL, 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    atr_pct = _atr_pct(out, window=atr_window)
    regime_name = _regime_name(out)
    n = len(out)

    primary = np.full(n, -1, dtype=np.int16)
    ensemble = np.full(n, -1, dtype=np.int16)
    vote_conf = np.zeros(n, dtype=np.float32)
    consensus = np.zeros(n, dtype=np.float32)
    event_end_idx = np.full(n, -1, dtype=np.int32)
    event_bars = np.full(n, -1, dtype=np.int16)
    tp_pct_used = np.full(n, np.nan, dtype=np.float32)
    sl_pct_used = np.full(n, np.nan, dtype=np.float32)
    meta_ret = np.full(n, np.nan, dtype=np.float32)
    meta_tp_first = np.zeros(n, dtype=np.int8)
    meta_adverse_first = np.zeros(n, dtype=np.int8)
    meta_timeout = np.zeros(n, dtype=np.int8)
    meta_primary_reason = np.full(n, "", dtype=object)
    meta_raw_terminal_ret = np.full(n, np.nan, dtype=np.float32)
    sharpe_vote = np.full(n, -1, dtype=np.int16)
    mfe_vote = np.full(n, -1, dtype=np.int16)
    fixed_vote = np.full(n, -1, dtype=np.int16)
    keep = np.zeros(n, dtype=np.int8)
    long_score = np.full(n, np.nan, dtype=np.float32)
    short_score = np.full(n, np.nan, dtype=np.float32)
    edge_gap = np.full(n, np.nan, dtype=np.float32)

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
        tp_pct_used[i] = tp_pct
        sl_pct_used[i] = sl_pct

        path = _scan_event(entry, future_close, future_high, future_low, tp_pct, sl_pct, fixed_tp_pct, fixed_sl_pct)
        pri_vote, pri_reason, pri_bar = _resolve_barrier_vote(
            path["long_primary_tp_bar"],
            path["long_primary_sl_bar"],
            path["short_primary_tp_bar"],
            path["short_primary_sl_bar"],
        )
        fx_vote, _, _ = _resolve_barrier_vote(
            path["long_fixed_tp_bar"],
            path["long_fixed_sl_bar"],
            path["short_fixed_tp_bar"],
            path["short_fixed_sl_bar"],
        )
        sh_vote = _resolve_score_vote(path["long_sharpe"], path["short_sharpe"], threshold=0.20, margin=0.05)
        mf_vote = _resolve_score_vote(path["long_mfe_mae"], path["short_mfe_mae"], threshold=0.0005, margin=0.0002)

        votes = np.asarray([pri_vote, fx_vote, sh_vote, mf_vote], dtype=np.int16)
        counts = np.bincount(np.clip(votes, 0, 2), minlength=3)
        ensemble_vote = int(np.argmax(counts))
        top = int(counts[ensemble_vote])
        ties = int(np.sum(counts == top))
        if ties > 1:
            ensemble_vote = 0
        primary[i] = int(pri_vote)
        fixed_vote[i] = int(fx_vote)
        sharpe_vote[i] = int(sh_vote)
        mfe_vote[i] = int(mf_vote)
        ensemble[i] = int(ensemble_vote)
        consensus[i] = float(top / 4.0)
        vote_conf[i] = float(top / 4.0)
        keep[i] = int(consensus[i] >= 0.75)

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
        event_bars[i] = int(pri_bar)
        event_end_idx[i] = int(i + pri_bar)
        meta_ret[i] = float(ret)
        meta_raw_terminal_ret[i] = float(path["raw_terminal_ret"])
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

        long_score[i] = float(path["long_sharpe"] + path["long_mfe_mae"])
        short_score[i] = float(path["short_sharpe"] + path["short_mfe_mae"])
        edge_gap[i] = float(abs(long_score[i] - short_score[i]))

        if (i % 10000) == 0 and i > 0:
            print(json.dumps({"stage": "label_progress", "done": int(i), "total": int(n)}, ensure_ascii=False), flush=True)

    raw_ensemble = ensemble.copy()
    regime_selected = np.zeros(n, dtype=np.int8)
    for regime, target_trade_ratio in REGIME_TRADE_RATIO.items():
        reg_mask = (regime_name == regime) & (raw_ensemble >= 0)
        reg_idx = np.flatnonzero(reg_mask)
        if len(reg_idx) == 0:
            continue
        cand_idx = reg_idx[raw_ensemble[reg_idx] != 0]
        target_n = int(round(len(reg_idx) * float(target_trade_ratio)))
        target_n = max(0, min(target_n, len(cand_idx)))
        if target_n == 0:
            ensemble[cand_idx] = 0
            continue
        if len(cand_idx) > target_n:
            order = sorted(
                cand_idx.tolist(),
                key=lambda j: (
                    float(consensus[j]),
                    float(edge_gap[j]) if np.isfinite(edge_gap[j]) else -1.0,
                    float(vote_conf[j]),
                ),
                reverse=True,
            )
            selected = np.asarray(order[:target_n], dtype=np.int32)
            dropped = np.asarray(order[target_n:], dtype=np.int32)
            ensemble[dropped] = 0
            regime_selected[selected] = 1
        else:
            regime_selected[cand_idx] = 1

    out["regime4_state"] = regime_name
    out["atr14_pct"] = atr_pct.astype(np.float32)
    out["label_primary_action"] = primary
    out["label_vote_action"] = raw_ensemble
    out["label_fixed_tp05_action"] = fixed_vote
    out["label_sharpe_action"] = sharpe_vote
    out["label_mfe_action"] = mfe_vote
    out["label_action"] = ensemble
    out["label_consensus"] = consensus
    out["label_confidence"] = vote_conf
    out["label_train_keep"] = keep
    out["label_event_end_idx"] = event_end_idx
    out["label_event_bars"] = event_bars
    out["label_tp_pct"] = tp_pct_used
    out["label_sl_pct"] = sl_pct_used
    out["meta_event_return"] = meta_ret
    out["meta_tp_first"] = meta_tp_first
    out["meta_adverse_first"] = meta_adverse_first
    out["meta_timeout"] = meta_timeout
    out["meta_primary_reason"] = meta_primary_reason
    out["meta_raw_terminal_return"] = meta_raw_terminal_ret
    out["meta_long_score"] = long_score
    out["meta_short_score"] = short_score
    out["meta_edge_gap"] = edge_gap
    out["regime_trade_selected"] = regime_selected
    out["meta_is_profitable"] = ((out["label_action"].isin([1, 2])) & (pd.to_numeric(out["meta_event_return"], errors="coerce").fillna(0.0) > 0.0)).astype(np.int8)
    out["meta_tp_ge_005"] = (pd.to_numeric(out["label_tp_pct"], errors="coerce").fillna(0.0) >= 0.005).astype(np.int8)
    out["label_valid"] = ((out["label_action"] >= 0) & (out["label_event_end_idx"] >= 0)).astype(np.int8)
    return out


def _uniqueness_weights(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    start = np.arange(n, dtype=np.int32) + 1
    end = pd.to_numeric(frame["label_event_end_idx"], errors="coerce").fillna(-1).to_numpy(np.int32)
    valid = (pd.to_numeric(frame["label_valid"], errors="coerce").fillna(0).to_numpy(np.int8) == 1) & (end >= start)
    diff = np.zeros(n + 2, dtype=np.int32)
    for s, e, ok in zip(start, end, valid):
        if not ok:
            continue
        s2 = int(np.clip(s, 0, n))
        e2 = int(np.clip(e, 0, n - 1))
        diff[s2] += 1
        diff[e2 + 1] -= 1
    conc = np.cumsum(diff[:-2]).astype(np.float64)
    inv = np.zeros_like(conc)
    nz = conc > 0
    inv[nz] = 1.0 / conc[nz]
    pref = np.zeros(n + 1, dtype=np.float64)
    pref[1:] = np.cumsum(inv)
    w = np.zeros(n, dtype=np.float32)
    for i, (s, e, ok) in enumerate(zip(start, end, valid)):
        if not ok:
            continue
        e2 = int(np.clip(e, 0, n - 1))
        s2 = int(np.clip(s, 0, n))
        length = max(e2 - s2 + 1, 1)
        w[i] = float((pref[e2 + 1] - pref[s2]) / length)
    return w


def _apply_split_policy(
    frame: pd.DataFrame,
    *,
    train_end: str,
    val_start: str,
    val_end: str,
    oos_start: str,
    embargo_bars: int,
    warmup_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    n = len(frame)
    idx = np.arange(n, dtype=np.int32)
    train_boundary = int(np.searchsorted(ts.to_numpy(), np.datetime64(val_start), side="left"))
    oos_boundary = int(np.searchsorted(ts.to_numpy(), np.datetime64(oos_start), side="left"))
    end_idx = pd.to_numeric(frame["label_event_end_idx"], errors="coerce").fillna(-1).to_numpy(np.int32)
    valid = pd.to_numeric(frame["label_valid"], errors="coerce").fillna(0).to_numpy(np.int8) == 1

    split = np.full(n, "drop", dtype=object)
    train_mask = (ts < pd.Timestamp(train_end)).to_numpy()
    val_mask = ((ts >= pd.Timestamp(val_start)) & (ts < pd.Timestamp(val_end))).to_numpy()
    oos_mask = (ts >= pd.Timestamp(oos_start)).to_numpy()

    purge_train = train_mask & (end_idx >= train_boundary)
    purge_val = val_mask & (end_idx >= oos_boundary)
    embargo_val = val_mask & (idx < train_boundary + embargo_bars)
    embargo_oos = oos_mask & (idx < oos_boundary + embargo_bars)
    warmup = idx < warmup_bars

    train_mask &= ~purge_train & ~warmup & valid
    val_mask &= ~purge_val & ~embargo_val & ~warmup & valid
    oos_mask &= ~embargo_oos & ~warmup & valid

    split[train_mask] = "train"
    split[val_mask] = "validation"
    split[oos_mask] = "oos"

    out = frame.copy()
    out["dataset_split"] = split
    out["split_keep"] = np.isin(split, ["train", "validation", "oos"]).astype(np.int8)
    meta = {
        "train_boundary_idx": train_boundary,
        "oos_boundary_idx": oos_boundary,
        "purge_train_rows": int(np.sum(purge_train)),
        "purge_validation_rows": int(np.sum(purge_val)),
        "embargo_validation_rows": int(np.sum(embargo_val)),
        "embargo_oos_rows": int(np.sum(embargo_oos)),
        "warmup_rows": int(np.sum(warmup)),
        "train_rows": int(np.sum(train_mask)),
        "validation_rows": int(np.sum(val_mask)),
        "oos_rows": int(np.sum(oos_mask)),
    }
    return out, meta


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    y = labels[(labels >= 0) & (labels <= 2)]
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    total = max(float(cnt.sum()), 1.0)
    return {i: float(total / (3.0 * max(cnt[i], 1.0))) for i in range(3)}


def _label_report(frame: pd.DataFrame) -> dict[str, Any]:
    keep = frame["split_keep"] == 1
    work = frame.loc[keep].copy()
    labs = pd.to_numeric(work["label_action"], errors="coerce").fillna(-1).astype(int)
    signed = np.where(labs == 1, 1.0, np.where(labs == 2, -1.0, 0.0))
    fut = pd.to_numeric(work["meta_raw_terminal_return"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    if np.std(signed) > 0 and np.std(fut) > 0:
        ic = float(pd.Series(signed).corr(pd.Series(fut), method="spearman"))
    else:
        ic = 0.0
    month = pd.to_datetime(work["timestamp"], errors="coerce").dt.to_period("M").astype(str)
    month_rows = []
    for key, grp in work.groupby(month):
        gl = pd.to_numeric(grp["label_action"], errors="coerce").fillna(-1).astype(int).to_numpy()
        gs = np.where(gl == 1, 1.0, np.where(gl == 2, -1.0, 0.0))
        gr = pd.to_numeric(grp["meta_raw_terminal_return"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        gic = float(pd.Series(gs).corr(pd.Series(gr), method="spearman")) if (np.std(gs) > 0 and np.std(gr) > 0) else 0.0
        month_rows.append({"month": key, "ic": gic, "rows": int(len(grp))})
    ic_vals = np.asarray([x["ic"] for x in month_rows], dtype=np.float64) if month_rows else np.asarray([0.0], dtype=np.float64)
    icir = float(np.mean(ic_vals) / max(np.std(ic_vals), 1e-12))

    action_counts = {k: int(v) for k, v in work["label_action"].value_counts().sort_index().to_dict().items()}
    cash_by_regime = (
        work.assign(is_cash=(pd.to_numeric(work["label_action"], errors="coerce").fillna(-1) == 0).astype(float))
        .groupby("regime4_state")["is_cash"]
        .mean()
        .to_dict()
    )
    split_counts = work["dataset_split"].value_counts().to_dict()
    report = {
        "rows": int(len(work)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "action_counts": action_counts,
        "trade_ratio": float(np.mean(pd.to_numeric(work["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)),
        "consensus_mean": float(pd.to_numeric(work["label_consensus"], errors="coerce").fillna(0.0).mean()),
        "consensus_keep_ratio": float(pd.to_numeric(work["label_train_keep"], errors="coerce").fillna(0).mean()),
        "uniqueness_mean": float(pd.to_numeric(work["sample_uniqueness_weight"], errors="coerce").fillna(0.0).mean()),
        "tp_first_ratio": float(pd.to_numeric(work["meta_tp_first"], errors="coerce").fillna(0).mean()),
        "adverse_first_ratio": float(pd.to_numeric(work["meta_adverse_first"], errors="coerce").fillna(0).mean()),
        "timeout_ratio": float(pd.to_numeric(work["meta_timeout"], errors="coerce").fillna(0).mean()),
        "cash_ratio_by_regime": {str(k): float(v) for k, v in cash_by_regime.items()},
        "ic_raw_terminal_return": ic,
        "icir_raw_terminal_return": icir,
        "event_return_mean_by_action": {
            str(int(k)): float(v) for k, v in work.groupby("label_action")["meta_event_return"].mean().to_dict().items()
        },
        "monthly_ic": month_rows,
        "class_weight_suggestion": _class_weights(pd.to_numeric(work["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64)),
        "missing_by_split": {
            split: int(work.loc[work["dataset_split"] == split].isna().sum().sum())
            for split in ["train", "validation", "oos"]
        },
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build high-quality HGB supervised training data with regime-conditioned ATR barriers, consensus labels, uniqueness weights, and purge/embargo.")
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
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
        "audit_report_model_path": audit.get("report_model_path"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    labeled = _label_frame(
        combined,
        max_hold=int(args.max_hold_bars),
        atr_window=int(args.atr_window),
        fixed_tp_pct=float(args.fixed_tp_pct),
        fixed_sl_pct=float(args.fixed_sl_pct),
    )
    labeled["sample_uniqueness_weight"] = _uniqueness_weights(labeled)
    base_weight = pd.to_numeric(labeled["label_confidence"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    uniq = pd.to_numeric(labeled["sample_uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    class_w = _class_weights(pd.to_numeric(labeled["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64))
    cw = np.asarray([class_w.get(int(x), 0.0) if int(x) >= 0 else 0.0 for x in pd.to_numeric(labeled["label_action"], errors="coerce").fillna(-1).to_numpy(np.int64)], dtype=np.float64)
    labeled["label_sample_weight"] = (base_weight * uniq * cw).astype(np.float32)

    labeled, split_meta = _apply_split_policy(
        labeled,
        train_end=str(args.train_end),
        val_start=str(args.val_start),
        val_end=str(args.val_end),
        oos_start=str(args.oos_start),
        embargo_bars=int(args.embargo_bars),
        warmup_bars=int(args.warmup_bars),
    )

    report = {
        "model_id": MODEL_ID,
        "config": {
            "train_end": str(args.train_end),
            "val_start": str(args.val_start),
            "val_end": str(args.val_end),
            "oos_start": str(args.oos_start),
            "max_hold_bars": int(args.max_hold_bars),
            "atr_window": int(args.atr_window),
            "fixed_tp_pct": float(args.fixed_tp_pct),
            "fixed_sl_pct": float(args.fixed_sl_pct),
            "embargo_bars": int(args.embargo_bars),
            "warmup_bars": int(args.warmup_bars),
            "regime_barrier": REGIME_BARRIER,
        },
        "audit": audit,
        "split_meta": split_meta,
        "label_quality": _label_report(labeled),
    }

    train_df = labeled[labeled["dataset_split"] == "train"].reset_index(drop=True)
    val_df = labeled[labeled["dataset_split"] == "validation"].reset_index(drop=True)
    oos_df = labeled[labeled["dataset_split"] == "oos"].reset_index(drop=True)

    train_path = args.out_dir / "alpha5_13_hgb_atr_barrier_labels_train.parquet"
    val_path = args.out_dir / "alpha5_13_hgb_atr_barrier_labels_val.parquet"
    oos_path = args.out_dir / "alpha5_13_hgb_atr_barrier_labels_oos.parquet"
    report_path = args.out_dir / "alpha5_13_hgb_atr_barrier_label_report.json"
    summary_csv = args.out_dir / "alpha5_13_hgb_atr_barrier_label_summary.csv"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    oos_df.to_parquet(oos_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([
        {"split": "train", "rows": len(train_df), "trade_ratio": float(np.mean(pd.to_numeric(train_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(train_df["label_train_keep"], errors="coerce").fillna(0).mean())},
        {"split": "validation", "rows": len(val_df), "trade_ratio": float(np.mean(pd.to_numeric(val_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(val_df["label_train_keep"], errors="coerce").fillna(0).mean())},
        {"split": "oos", "rows": len(oos_df), "trade_ratio": float(np.mean(pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(-1).to_numpy() != 0)), "keep_ratio": float(pd.to_numeric(oos_df["label_train_keep"], errors="coerce").fillna(0).mean())},
    ]).to_csv(summary_csv, index=False)

    print(json.dumps({
        "stage": "complete",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "oos_rows": int(len(oos_df)),
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "oos_path": str(oos_path),
        "report_path": str(report_path),
        "summary_csv": str(summary_csv),
        "label_quality": {
            "trade_ratio": report["label_quality"]["trade_ratio"],
            "consensus_keep_ratio": report["label_quality"]["consensus_keep_ratio"],
            "tp_first_ratio": report["label_quality"]["tp_first_ratio"],
            "adverse_first_ratio": report["label_quality"]["adverse_first_ratio"],
            "ic_raw_terminal_return": report["label_quality"]["ic_raw_terminal_return"],
            "icir_raw_terminal_return": report["label_quality"]["icir_raw_terminal_return"],
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
