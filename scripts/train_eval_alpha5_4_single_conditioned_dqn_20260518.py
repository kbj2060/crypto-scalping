#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.dueling_dqn_per_full_architecture import (  # noqa: E402
    ActionSpace,
    ConditionedDQNTrainer,
    ConditionedDQNTrainerConfig,
    RewardFunction,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_EVAL,
    DEFAULT_PREPROCESS_MANIFEST,
    DEFAULT_TRAIN,
    FORBIDDEN_EXACT,
    FORBIDDEN_PREFIXES,
    REGIMES,
    ROUTER_COLS,
    ROUTER_PROB_SET,
    _feature_cols as _base_feature_cols,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _days,
    _fill_price,
    _json_default,
    _read,
)


MODEL_ID = "alpha5_4_single_conditioned_dqn_per_state24_sticky090_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_4_single_conditioned_dqn_per_state24_sticky090_20260518"


def _seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _clean_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return arr


def _fit_market_scaler(train_df: pd.DataFrame, market_cols: list[str]) -> dict[str, Any]:
    arr = _clean_numeric_matrix(train_df, market_cols)
    med = np.zeros(arr.shape[1], dtype=np.float32)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        med[j] = float(np.median(finite)) if finite.size else 0.0
    arr = np.where(np.isfinite(arr), arr, med)
    mean = arr.mean(axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.maximum(arr.std(axis=0), 1e-6).astype(np.float32)
    std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
    return {"median": med, "mean": mean, "std": std}


def _transform_market(df: pd.DataFrame, market_cols: list[str], scaler: dict[str, Any]) -> np.ndarray:
    arr = _clean_numeric_matrix(df, market_cols)
    med = np.asarray(scaler["median"], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(scaler["std"], dtype=np.float32), 1e-6)
    arr = np.where(np.isfinite(arr), arr, med)
    return ((arr - mean) / std).astype(np.float32)


def _regime_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in ROUTER_COLS if c not in df.columns]
    if missing:
        raise ValueError("missing HMM regime probability columns: " + ", ".join(missing))
    p = df[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    p = np.clip(p, 0.0, None)
    denom = np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return (p / denom).astype(np.float32)


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame, *, include_future_regime_pred: bool, feature_top_k: int, feature_select_horizon: int) -> list[str]:
    common = set(train.columns) & set(eval_df.columns)
    cols = list(_base_feature_cols(train, eval_df))
    if bool(include_future_regime_pred):
        for col in sorted(c for c in common if c.startswith("regime4_pred_")):
            if col not in cols:
                cols.append(col)
    if int(feature_top_k) <= 0 or int(feature_top_k) >= len(cols):
        return cols
    close = pd.to_numeric(train["close"], errors="coerce")
    target = close.shift(-int(feature_select_horizon)) / close - 1.0
    scores: list[tuple[float, str]] = []
    for col in cols:
        x = pd.to_numeric(train[col], errors="coerce") if col in train.columns else pd.Series(index=train.index, dtype=float)
        valid = np.isfinite(x.to_numpy(dtype=np.float64)) & np.isfinite(target.to_numpy(dtype=np.float64))
        if int(valid.sum()) < 100:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(x.to_numpy(dtype=np.float64)[valid], target.to_numpy(dtype=np.float64)[valid])[0, 1]))
            if not np.isfinite(score):
                score = 0.0
        scores.append((score, col))
    keep = [c for _, c in sorted(scores, reverse=True)[: int(feature_top_k)]]
    for col in ["tp_sl_action_score", "m7_expected_ret", "ai_dir_edge", "ai_dir_p_up", "ai_dir_p_down", "ai_reward_risk"]:
        if col in common and col in cols and col not in keep:
            keep.append(col)
    for col in sorted(c for c in cols if c.startswith("regime4_pred_") or c.startswith(CLEAN4_PREFIX)):
        if col not in keep and len(keep) < int(feature_top_k) + 24:
            keep.append(col)
    return keep


def _edge_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    up = pd.to_numeric(df.get("ai_dir_p_up", pd.Series(np.zeros(n), index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    down = pd.to_numeric(df.get("ai_dir_p_down", pd.Series(np.zeros(n), index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    edge = pd.to_numeric(df.get("ai_dir_edge", pd.Series(up - down, index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    m7 = pd.to_numeric(df.get("m7_expected_ret", pd.Series(np.zeros(n), index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    rr = pd.to_numeric(df.get("ai_reward_risk", pd.Series(np.ones(n), index=df.index)), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    rr_scale = np.clip(rr / 4.0, 0.0, 2.0)
    long_edge = np.maximum(up - down, 0.0) + np.maximum(edge, 0.0) * 0.25 + np.maximum(m7, 0.0) * 50.0
    short_edge = np.maximum(down - up, 0.0) + np.maximum(-edge, 0.0) * 0.25 + np.maximum(-m7, 0.0) * 50.0
    return (long_edge * rr_scale).astype(np.float32), (short_edge * rr_scale).astype(np.float32)


def _weak_prior_action(side: int, hold_bars: int, ev_long: float, ev_short: float, *, edge_threshold: float) -> int:
    long_edge = float(ev_long)
    short_edge = float(ev_short)
    margin = abs(long_edge - short_edge)
    if int(side) == 0:
        if margin < float(edge_threshold):
            return ActionSpace.FLAT
        return ActionSpace.OPEN_LONG if long_edge > short_edge else ActionSpace.OPEN_SHORT
    if int(side) > 0:
        if short_edge > long_edge + float(edge_threshold):
            return ActionSpace.CLOSE
        return ActionSpace.HOLD_LONG
    if long_edge > short_edge + float(edge_threshold):
        return ActionSpace.CLOSE
    return ActionSpace.HOLD_SHORT


def _parse_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        horizons.append(max(int(part), 1))
    return sorted(set(horizons)) or [12, 24, 48, 96, 288]


def _future_q_labels(
    close: np.ndarray,
    *,
    horizons: list[int],
    fee: float,
    slip: float,
    label_scale: float,
    terminal_weight: float,
    mfe_weight: float,
    adverse_weight: float,
    entry_hurdle: float,
    clip_value: float,
) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    q = np.zeros((n, ActionSpace.N_ACTIONS), dtype=np.float32)
    round_trip_cost = (float(fee) + float(slip)) * 2.0
    for i in range(n - 1):
        px = max(float(close[i]), 1e-12)
        best_long = -round_trip_cost
        best_short = -round_trip_cost
        for h in horizons:
            end = min(i + int(h), n - 1)
            if end <= i:
                continue
            rel = close[i + 1 : end + 1] / px - 1.0
            rel = rel[np.isfinite(rel)]
            if rel.size == 0:
                continue
            long_terminal = float(rel[-1])
            long_mfe = float(np.max(rel))
            long_mae = float(np.min(rel))
            short_path = -rel
            short_terminal = float(short_path[-1])
            short_mfe = float(np.max(short_path))
            short_mae = float(np.min(short_path))
            long_value = (
                float(terminal_weight) * long_terminal
                + float(mfe_weight) * long_mfe
                + float(adverse_weight) * long_mae
                - round_trip_cost
            )
            short_value = (
                float(terminal_weight) * short_terminal
                + float(mfe_weight) * short_mfe
                + float(adverse_weight) * short_mae
                - round_trip_cost
            )
            best_long = max(best_long, long_value)
            best_short = max(best_short, short_value)
        long_q = float(np.clip(best_long * float(label_scale) - float(entry_hurdle), -float(clip_value), float(clip_value)))
        short_q = float(np.clip(best_short * float(label_scale) - float(entry_hurdle), -float(clip_value), float(clip_value)))
        q[i, ActionSpace.FLAT] = 0.0
        q[i, ActionSpace.OPEN_LONG] = long_q
        q[i, ActionSpace.OPEN_SHORT] = short_q
        q[i, ActionSpace.CLOSE] = 0.0
        q[i, ActionSpace.HOLD_LONG] = float(np.clip(long_q + round_trip_cost * float(label_scale) * 0.5, -float(clip_value), float(clip_value)))
        q[i, ActionSpace.HOLD_SHORT] = float(np.clip(short_q + round_trip_cost * float(label_scale) * 0.5, -float(clip_value), float(clip_value)))
    return q


def _atr_pct(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    else:
        ret = np.abs(np.diff(close, prepend=close[0]) / np.maximum(np.roll(close, 1), 1e-12))
        tr = ret * close
    alpha = 2.0 / (float(period) + 1.0)
    atr = np.asarray(tr, dtype=np.float64).copy()
    for i in range(1, len(atr)):
        atr[i] = alpha * atr[i] + (1.0 - alpha) * atr[i - 1]
    return np.clip(atr / np.maximum(close, 1e-12), 1e-5, 0.10).astype(np.float32)


def _path_sharpe(path_ret: np.ndarray) -> float:
    if path_ret.size < 2:
        return 0.0
    mean = float(np.mean(path_ret))
    std = float(np.std(path_ret)) + 1e-8
    return float(np.clip(mean / std * np.sqrt(float(path_ret.size)), -5.0, 5.0))


def _barrier_ret(rel: np.ndarray, tp: float, sl: float) -> float:
    if rel.size == 0:
        return 0.0
    hit_tp = np.flatnonzero(rel >= float(tp))
    hit_sl = np.flatnonzero(rel <= -float(sl))
    first_tp = int(hit_tp[0]) if hit_tp.size else 10**9
    first_sl = int(hit_sl[0]) if hit_sl.size else 10**9
    if first_tp < first_sl:
        return float(tp)
    if first_sl < first_tp:
        return -float(sl)
    return float(rel[-1])


def _ensemble_q_labels(
    df: pd.DataFrame,
    regime: np.ndarray,
    *,
    horizons: list[int],
    fee: float,
    slip: float,
    label_scale: float,
    entry_hurdle: float,
    clip_value: float,
    confidence_min: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    n = len(close)
    max_h = max(horizons)
    min_hold = min(6, max_h)
    cost = (float(fee) + float(slip)) * 2.0
    atr = _atr_pct(df)
    probs = np.asarray(regime, dtype=np.float32)
    if probs.shape[0] != n:
        raise ValueError("regime rows do not match df rows for ensemble labels")

    q = np.zeros((n, ActionSpace.N_ACTIONS), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    consensus_arr = np.zeros(n, dtype=np.float32)
    action_votes = np.zeros((n, 3), dtype=np.float32)
    component_weights = {
        "triple_barrier": 0.30,
        "oracle_exit": 0.35,
        "sharpe": 0.20,
        "mae_mfe": 0.15,
    }

    for i in range(n - 1):
        end = min(i + max_h, n - 1)
        if end <= i:
            continue
        px = max(float(close[i]), 1e-12)
        future = close[i + 1 : end + 1]
        rel = future / px - 1.0
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        short_rel = -rel
        step = np.diff(np.concatenate([[px], future[: rel.size]])) / np.maximum(np.concatenate([[px], future[: rel.size - 1]]), 1e-12)

        # 1) ATR triple barrier.
        tp = float(np.clip(2.0 * atr[i], 0.0025, 0.0500))
        sl = float(np.clip(1.0 * atr[i], 0.0015, 0.0300))
        tb_long = (_barrier_ret(rel, tp, sl) - cost) * float(label_scale)
        tb_short = (_barrier_ret(short_rel, tp, sl) - cost) * float(label_scale)

        # 2) Directional oracle exit: best possible exit inside the horizon.
        start = max(min_hold - 1, 0)
        oracle_long = ((float(np.max(rel[start:])) if rel.size > start else float(rel[-1])) - cost) * float(label_scale)
        oracle_short = ((float(np.max(short_rel[start:])) if short_rel.size > start else float(short_rel[-1])) - cost) * float(label_scale)

        # 3) Risk-adjusted path label.
        sharpe_long = _path_sharpe(step)
        sharpe_short = _path_sharpe(-step)
        sh_long = sharpe_long * 0.60
        sh_short = sharpe_short * 0.60

        # 4) MAE/MFE quality label.
        long_mfe = float(np.max(rel))
        long_mae = abs(float(np.min(rel)))
        short_mfe = float(np.max(short_rel))
        short_mae = abs(float(np.min(short_rel)))
        long_ratio = long_mfe / (long_mae + 1e-6)
        short_ratio = short_mfe / (short_mae + 1e-6)
        mm_long = (float(rel[-1]) + 0.35 * long_mfe - 1.20 * long_mae - cost) * float(label_scale)
        mm_short = (float(short_rel[-1]) + 0.35 * short_mfe - 1.20 * short_mae - cost) * float(label_scale)
        if long_ratio < 1.20:
            mm_long -= 0.25
        if short_ratio < 1.20:
            mm_short -= 0.25

        components = {
            "triple_barrier": (tb_long, tb_short),
            "oracle_exit": (oracle_long, oracle_short),
            "sharpe": (sh_long, sh_short),
            "mae_mfe": (mm_long, mm_short),
        }
        votes = np.zeros(3, dtype=np.float32)
        ens_long = 0.0
        ens_short = 0.0
        for name, (ql, qs) in components.items():
            w = float(component_weights[name])
            ql = float(np.clip(ql, -float(clip_value), float(clip_value)))
            qs = float(np.clip(qs, -float(clip_value), float(clip_value)))
            ens_long += w * ql
            ens_short += w * qs
            vote = int(np.argmax([0.0, ql, qs]))
            votes[vote] += w

        bull_p, bear_p, chop_p, whip_p = [float(x) for x in probs[i, :4]]
        trend_bias = 0.15 * (bull_p - bear_p)
        ens_long *= 1.0 + trend_bias
        ens_short *= 1.0 - trend_bias
        noisy_penalty = 0.20 * chop_p + 0.35 * whip_p
        ens_long -= noisy_penalty
        ens_short -= noisy_penalty

        sorted_votes = np.sort(votes)[::-1]
        consensus = float(sorted_votes[0] - sorted_votes[1]) if sorted_votes.size >= 2 else 0.0
        separation = float(abs(ens_long - ens_short))
        conf = float(np.clip(0.70 * consensus + 0.30 * min(separation / max(float(clip_value), 1e-6), 1.0), 0.0, 1.0))
        if conf < float(confidence_min):
            ens_long = min(ens_long, -0.05)
            ens_short = min(ens_short, -0.05)

        open_long = float(np.clip(ens_long - float(entry_hurdle), -float(clip_value), float(clip_value)))
        open_short = float(np.clip(ens_short - float(entry_hurdle), -float(clip_value), float(clip_value)))
        hold_long = float(np.clip(ens_long + cost * float(label_scale) * 0.5, -float(clip_value), float(clip_value)))
        hold_short = float(np.clip(ens_short + cost * float(label_scale) * 0.5, -float(clip_value), float(clip_value)))
        q[i, ActionSpace.FLAT] = 0.0
        q[i, ActionSpace.OPEN_LONG] = open_long
        q[i, ActionSpace.OPEN_SHORT] = open_short
        q[i, ActionSpace.CLOSE] = 0.0
        q[i, ActionSpace.HOLD_LONG] = hold_long
        q[i, ActionSpace.HOLD_SHORT] = hold_short
        confidence[i] = conf
        consensus_arr[i] = consensus
        action_votes[i] = votes

    # Keep all samples, but down-weight low-consensus labels instead of dropping coverage.
    sample_weight = np.clip((confidence - float(confidence_min)) / max(1.0 - float(confidence_min), 1e-6), 0.10, 1.0).astype(np.float32)
    flat_best = np.argmax(q[:, [ActionSpace.FLAT, ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT]], axis=1)
    report = {
        "labeler": "ensemble",
        "horizons": horizons,
        "component_weights": component_weights,
        "confidence_min": float(confidence_min),
        "confidence_mean": float(np.mean(confidence)),
        "consensus_mean": float(np.mean(consensus_arr)),
        "sample_weight_mean": float(np.mean(sample_weight)),
        "high_conf_ratio": float(np.mean(confidence >= float(confidence_min))),
        "flat_best_counts": {
            "flat": int(np.sum(flat_best == 0)),
            "open_long": int(np.sum(flat_best == 1)),
            "open_short": int(np.sum(flat_best == 2)),
        },
        "vote_means": {
            "flat": float(np.mean(action_votes[:, 0])),
            "open_long": float(np.mean(action_votes[:, 1])),
            "open_short": float(np.mean(action_votes[:, 2])),
        },
        "open_long_q_mean": float(np.mean(q[:, ActionSpace.OPEN_LONG])),
        "open_short_q_mean": float(np.mean(q[:, ActionSpace.OPEN_SHORT])),
        "open_long_q_p95": float(np.quantile(q[:, ActionSpace.OPEN_LONG], 0.95)),
        "open_short_q_p95": float(np.quantile(q[:, ActionSpace.OPEN_SHORT], 0.95)),
    }
    return q, sample_weight, report


def _pretrain_supervised_q(
    trainer: ConditionedDQNTrainer,
    market: np.ndarray,
    regime: np.ndarray,
    q_labels: np.ndarray,
    q_weights: np.ndarray | None = None,
    *,
    steps: int,
    batch_size: int,
    ce_weight: float,
    max_hold_bars: int,
) -> list[float]:
    if int(steps) <= 0:
        return []
    losses: list[float] = []
    valid_by_side = {
        0: [ActionSpace.FLAT, ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT],
        1: [ActionSpace.CLOSE, ActionSpace.HOLD_LONG],
        -1: [ActionSpace.CLOSE, ActionSpace.HOLD_SHORT],
    }
    sides = np.asarray([0, 1, -1], dtype=np.int64)
    for step in range(int(steps)):
        idx = np.random.randint(0, len(market), size=int(batch_size))
        side_sample = np.random.choice(sides, size=int(batch_size), p=np.asarray([0.50, 0.25, 0.25]))
        context = np.zeros((int(batch_size), 7), dtype=np.float32)
        context[:, 0] = side_sample.astype(np.float32)
        context[:, 2] = np.where(side_sample == 0, 0.0, 6.0 / max(float(max_hold_bars), 1.0)).astype(np.float32)
        context[:, 5] = 1.0
        market_t = torch.as_tensor(market[idx], dtype=torch.float32, device=trainer.device)
        regime_t = torch.as_tensor(regime[idx], dtype=torch.float32, device=trainer.device)
        context_t = torch.as_tensor(context, dtype=torch.float32, device=trainer.device)
        q_pred = trainer.online(market_t, regime_t, context_t, action_mask=None)
        q_target_np = q_labels[idx].astype(np.float32, copy=True)
        q_target_t = torch.as_tensor(q_target_np, dtype=torch.float32, device=trainer.device)
        weight_np = np.ones(int(batch_size), dtype=np.float32) if q_weights is None else q_weights[idx].astype(np.float32, copy=False)
        weight_t = torch.as_tensor(weight_np, dtype=torch.float32, device=trainer.device)
        loss_terms = []
        ce_terms = []
        label_counts: dict[int, int] = {}
        for side in (0, 1, -1):
            row_mask_np = side_sample == side
            if not bool(np.any(row_mask_np)):
                continue
            valid = valid_by_side[int(side)]
            row_mask = torch.as_tensor(row_mask_np, dtype=torch.bool, device=trainer.device)
            pred_valid = q_pred[row_mask][:, valid]
            target_valid = q_target_t[row_mask][:, valid]
            row_weight = weight_t[row_mask]
            best_local = torch.argmax(target_valid, dim=1)
            q_loss = torch.nn.functional.smooth_l1_loss(pred_valid, target_valid, reduction="none").mean(dim=1)
            loss_terms.append((q_loss * row_weight).sum() / torch.clamp(row_weight.sum(), min=1e-6))
            ce_loss = torch.nn.functional.cross_entropy(pred_valid, best_local, reduction="none")
            ce_terms.append((ce_loss * row_weight).sum() / torch.clamp(row_weight.sum(), min=1e-6))
            for action in np.asarray(valid, dtype=np.int64)[best_local.detach().cpu().numpy()]:
                label_counts[int(action)] = label_counts.get(int(action), 0) + 1
        loss = sum(loss_terms) / max(len(loss_terms), 1)
        if ce_terms and float(ce_weight) > 0.0:
            loss = loss + float(ce_weight) * (sum(ce_terms) / len(ce_terms))
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.online.parameters(), float(trainer.cfg.grad_clip))
        trainer.optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(int(steps) // 5, 1) == 0:
            print(
                json.dumps(
                    {
                        "stage": "pretrain_supervised_q",
                        "step": step + 1,
                        "steps": int(steps),
                        "loss": float(loss.detach().cpu()),
                        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    trainer.target.load_state_dict(trainer.online.state_dict())
    return losses


def _context_from_position(
    *,
    side: int,
    entry_price: float,
    peak_price: float,
    hold_bars: int,
    bars_since_last_trade: int,
    daily_trade_count: int,
    current_price: float,
    max_hold_bars: int,
) -> np.ndarray:
    if int(side) == 0 or float(entry_price) <= 0.0:
        return np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, float(bars_since_last_trade) / 50.0, float(daily_trade_count) / 20.0],
            dtype=np.float32,
        )
    entry_dist = float(current_price) / max(float(entry_price), 1e-12) - 1.0
    ret = entry_dist * float(side)
    peak_ret = (float(peak_price) / max(float(entry_price), 1e-12) - 1.0) * float(side)
    drawdown = ret - peak_ret
    return np.asarray(
        [
            float(side),
            float(np.clip(ret, -0.1, 0.1)),
            float(hold_bars) / max(float(max_hold_bars), 1.0),
            float(np.clip(entry_dist, -0.05, 0.05)),
            float(np.clip(drawdown, -0.05, 0.0)),
            float(bars_since_last_trade) / 50.0,
            float(daily_trade_count) / 20.0,
        ],
        dtype=np.float32,
    )


def _state(market: np.ndarray, regime: np.ndarray, context: np.ndarray) -> np.ndarray:
    return np.concatenate([market, regime, context]).astype(np.float32, copy=False)


def _split_state_tensor(trainer: ConditionedDQNTrainer, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = trainer.device
    m = int(trainer.cfg.market_dim)
    r = int(trainer.cfg.regime_dim)
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return s[:, :m], s[:, m : m + r], s[:, m + r :]


@torch.no_grad()
def _masked_q_values(trainer: ConditionedDQNTrainer, state: np.ndarray, position_side: int) -> np.ndarray:
    was_training = bool(trainer.online.training)
    trainer.online.eval()
    try:
        market_t, regime_t, context_t = _split_state_tensor(trainer, state)
        mask = ActionSpace.get_mask(int(position_side)).to(device=trainer.device).unsqueeze(0)
        q = trainer.online(market_t, regime_t, context_t, action_mask=mask)
        return q.detach().cpu().numpy()[0]
    finally:
        if was_training:
            trainer.online.train()


@torch.no_grad()
def _select_greedy_action(trainer: ConditionedDQNTrainer, state: np.ndarray, position_side: int) -> int:
    q = _masked_q_values(trainer, state, position_side)
    return int(np.nanargmax(q))


def _validation_score(metrics: dict[str, Any]) -> float:
    """All inputs are percentages except trades_per_day."""

    pnl = float(metrics["pnl"])
    mdd_penalty = abs(float(metrics["mdd"])) * 0.35
    overtrade_penalty = max(float(metrics["trades_per_day"]) - 8.0, 0.0) * 2.0
    undertrade_penalty = max(3.0 - float(metrics["trades_per_day"]), 0.0) * 20.0
    return float(pnl - mdd_penalty - overtrade_penalty - undertrade_penalty)


def _train_agent(
    train_df: pd.DataFrame,
    market: np.ndarray,
    regime: np.ndarray,
    *,
    val_df: pd.DataFrame | None = None,
    val_market: np.ndarray | None = None,
    val_regime: np.ndarray | None = None,
    cfg: ConditionedDQNTrainerConfig,
    episodes: int,
    train_every: int,
    min_hold_bars: int,
    hard_min_hold_bars: int,
    max_hold_bars: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    log_every: int,
    validation_every: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    lr_scheduler_patience: int,
    lr_scheduler_factor: float,
    checkpoint_dir: Path,
    unit_exposure: float,
    reward_scale: float,
    reward_trade_penalty: float,
    reward_early_close_penalty: float,
    reward_edge_bonus: float,
    reward_flat_opportunity_penalty: float,
    reward_opportunity_threshold: float,
    weak_bc_edge_threshold: float,
    pretrain_bc_steps: int,
    pretrain_bc_batch: int,
    pretrain_q_steps: int,
    pretrain_q_batch: int,
    pretrain_q_labeler: str,
    pretrain_q_horizons: str,
    pretrain_q_scale: float,
    pretrain_q_ce_weight: float,
    pretrain_q_terminal_weight: float,
    pretrain_q_mfe_weight: float,
    pretrain_q_adverse_weight: float,
    pretrain_q_entry_hurdle: float,
    pretrain_q_clip: float,
    ensemble_label_confidence_min: float,
    ensemble_label_cache: Path | None,
    entry_edge_threshold: float,
) -> tuple[ConditionedDQNTrainer, dict[str, Any]]:
    trainer = ConditionedDQNTrainer(cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode="max",
        factor=float(lr_scheduler_factor),
        patience=max(int(lr_scheduler_patience), 0),
        min_lr=1e-6,
    )
    reward_fn = RewardFunction(
        min_hold_bars=int(min_hold_bars),
        max_hold_bars=int(max_hold_bars),
        hold_penalty=float(reward_early_close_penalty),
        trade_penalty=float(reward_trade_penalty),
        reward_scale=float(reward_scale),
        edge_bonus=float(reward_edge_bonus),
        flat_opportunity_penalty=float(reward_flat_opportunity_penalty),
        opportunity_threshold=float(reward_opportunity_threshold),
    )
    close = train_df["close"].to_numpy(dtype=np.float64)
    ev_long_arr, ev_short_arr = _edge_arrays(train_df)
    ts = pd.to_datetime(train_df["timestamp"]).to_numpy()
    regime_idx = np.argmax(regime, axis=1)
    total_steps = max(len(train_df) - 1, 1) * max(int(episodes), 1)
    losses: list[float] = []
    episode_rewards: list[float] = []
    validation_history: list[dict[str, Any]] = []
    epsilon = float(epsilon_start)
    updates = 0
    best_score = -float("inf")
    best_episode = 0
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrain_q_losses: list[float] = []
    pretrain_q_report: dict[str, Any] | None = None

    if int(pretrain_q_steps) > 0:
        q_weights = None
        cache_path = Path(ensemble_label_cache) if ensemble_label_cache is not None else None
        if str(pretrain_q_labeler) == "ensemble":
            cache_payload = None
            if cache_path is not None and cache_path.exists():
                cache_payload = joblib.load(cache_path)
                q_labels = np.asarray(cache_payload["q_labels"], dtype=np.float32)
                q_weights = np.asarray(cache_payload["q_weights"], dtype=np.float32)
                pretrain_q_report = dict(cache_payload.get("report", {}))
                pretrain_q_report["cache_loaded"] = str(cache_path)
            else:
                q_labels, q_weights, pretrain_q_report = _ensemble_q_labels(
                    train_df,
                    regime,
                    horizons=_parse_horizons(pretrain_q_horizons),
                    fee=0.0005,
                    slip=0.0002,
                    label_scale=float(pretrain_q_scale),
                    entry_hurdle=float(pretrain_q_entry_hurdle),
                    clip_value=float(pretrain_q_clip),
                    confidence_min=float(ensemble_label_confidence_min),
                )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"q_labels": q_labels, "q_weights": q_weights, "report": pretrain_q_report}, cache_path)
                    pretrain_q_report["cache_saved"] = str(cache_path)
        else:
            q_labels = _future_q_labels(
                close,
                horizons=_parse_horizons(pretrain_q_horizons),
                fee=0.0005,
                slip=0.0002,
                label_scale=float(pretrain_q_scale),
                terminal_weight=float(pretrain_q_terminal_weight),
                mfe_weight=float(pretrain_q_mfe_weight),
                adverse_weight=float(pretrain_q_adverse_weight),
                entry_hurdle=float(pretrain_q_entry_hurdle),
                clip_value=float(pretrain_q_clip),
            )
            flat_best = np.argmax(q_labels[:, [ActionSpace.FLAT, ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT]], axis=1)
            pretrain_q_report = {
                "labeler": "legacy",
                "horizons": _parse_horizons(pretrain_q_horizons),
                "label_scale": float(pretrain_q_scale),
                "flat_best_counts": {
                    "flat": int(np.sum(flat_best == 0)),
                    "open_long": int(np.sum(flat_best == 1)),
                    "open_short": int(np.sum(flat_best == 2)),
                },
                "open_long_q_mean": float(np.mean(q_labels[:, ActionSpace.OPEN_LONG])),
                "open_short_q_mean": float(np.mean(q_labels[:, ActionSpace.OPEN_SHORT])),
                "open_long_q_p95": float(np.quantile(q_labels[:, ActionSpace.OPEN_LONG], 0.95)),
                "open_short_q_p95": float(np.quantile(q_labels[:, ActionSpace.OPEN_SHORT], 0.95)),
            }
        print(
            json.dumps(
                {
                    "stage": "pretrain_q_labels",
                    **(pretrain_q_report or {}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        pretrain_q_losses = _pretrain_supervised_q(
            trainer,
            market,
            regime,
            q_labels,
            q_weights=q_weights,
            steps=int(pretrain_q_steps),
            batch_size=int(pretrain_q_batch),
            ce_weight=float(pretrain_q_ce_weight),
            max_hold_bars=int(max_hold_bars),
        )

    if int(pretrain_bc_steps) > 0:
        flat_context = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        labels = np.asarray(
            [_weak_prior_action(0, 0, float(a), float(b), edge_threshold=float(weak_bc_edge_threshold)) for a, b in zip(ev_long_arr, ev_short_arr)],
            dtype=np.int64,
        )
        flat_mask = ActionSpace.get_mask(0).to(trainer.device).unsqueeze(0)
        for step in range(int(pretrain_bc_steps)):
            idx = np.random.randint(0, len(train_df), size=int(pretrain_bc_batch))
            market_t = torch.as_tensor(market[idx], dtype=torch.float32, device=trainer.device)
            regime_t = torch.as_tensor(regime[idx], dtype=torch.float32, device=trainer.device)
            context_t = torch.as_tensor(np.repeat(flat_context[None, :], len(idx), axis=0), dtype=torch.float32, device=trainer.device)
            label_t = torch.as_tensor(labels[idx], dtype=torch.int64, device=trainer.device)
            q = trainer.online(market_t, regime_t, context_t, action_mask=flat_mask.repeat(len(idx), 1))
            loss = torch.nn.functional.cross_entropy(q, label_t)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.online.parameters(), float(cfg.grad_clip))
            trainer.optimizer.step()
            if (step + 1) % max(int(pretrain_bc_steps) // 5, 1) == 0:
                counts = {str(a): int((labels == a).sum()) for a in (ActionSpace.FLAT, ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT)}
                print(json.dumps({"stage": "pretrain_bc", "step": step + 1, "steps": int(pretrain_bc_steps), "loss": float(loss.detach().cpu()), "label_counts": counts}, ensure_ascii=False), flush=True)
        trainer.target.load_state_dict(trainer.online.state_dict())

    def _save_checkpoint(ep: int, score: float | None, reason: str) -> None:
        torch.save(
            {
                "model_state_dict": trainer.online.state_dict(),
                "target_state_dict": trainer.target.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "episode": int(ep),
                "updates": int(updates),
                "score": None if score is None else float(score),
                "reason": str(reason),
                "epsilon": float(epsilon),
                "per_beta": float(trainer.buffer.beta),
                "per_beta_start": float(trainer.buffer.beta_start),
                "per_beta_max": float(trainer.buffer.beta_max),
                "replay_entries": int(trainer.buffer.tree.n_entries),
                "replay_capacity": int(trainer.buffer.capacity),
            },
            checkpoint_dir / "single_conditioned_dqn_checkpoint.pt",
        )

    for ep in range(int(episodes)):
        side = 0
        entry_price = 0.0
        peak_price = 0.0
        hold_bars = 0
        bars_since_last_trade = 0
        daily_trade_count = 0
        last_day: Any = None
        episode_reward = 0.0
        entries = closes = 0
        for i in range(len(train_df) - 1):
            day = pd.Timestamp(ts[i]).date()
            if last_day is None:
                last_day = day
            elif day != last_day:
                last_day = day
                daily_trade_count = 0
            if side != 0:
                hold_bars += 1
                peak_price = max(peak_price, close[i]) if side > 0 else min(peak_price, close[i])
            bars_since_last_trade += 1

            context = _context_from_position(
                side=side,
                entry_price=entry_price,
                peak_price=peak_price,
                hold_bars=hold_bars,
                bars_since_last_trade=bars_since_last_trade,
                daily_trade_count=daily_trade_count,
                current_price=float(close[i]),
                max_hold_bars=int(max_hold_bars),
            )
            state = _state(market[i], regime[i], context)
            raw_action = trainer.online.select_action(
                market[i],
                regime[i],
                context,
                position_side=side,
                epsilon=float(epsilon),
                temperature=0.50,
                device=trainer.device,
            )
            action = ActionSpace.apply(raw_action, side, min_hold_bars=int(hard_min_hold_bars), hold_bars=int(hold_bars))
            position_map = {
                "side": float(side),
                "unrealized_pnl": float(context[1]),
                "hold_bars": float(hold_bars),
                "entry_price_dist": float(context[3]),
                "drawdown_from_peak": float(context[4]),
            }
            reward = reward_fn.compute(
                action=action,
                position=position_map,
                current_price=float(close[i]),
                next_price=float(close[i + 1]),
                regime=str(REGIMES[int(regime_idx[i])]),
                daily_trades=int(daily_trade_count),
                ev_long=float(ev_long_arr[i]),
                ev_short=float(ev_short_arr[i]),
            )

            next_side = ActionSpace.next_side(side, action)
            if int(action) in (ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT):
                side = 1 if int(action) == ActionSpace.OPEN_LONG else -1
                entry_price = float(close[i + 1])
                peak_price = float(close[i + 1])
                hold_bars = 0
                bars_since_last_trade = 0
                daily_trade_count += 1
                entries += 1
            elif int(action) == ActionSpace.CLOSE:
                side = 0
                entry_price = 0.0
                peak_price = 0.0
                hold_bars = 0
                bars_since_last_trade = 0
                daily_trade_count += 1
                closes += 1

            next_context = _context_from_position(
                side=side,
                entry_price=entry_price,
                peak_price=peak_price,
                hold_bars=hold_bars,
                bars_since_last_trade=bars_since_last_trade,
                daily_trade_count=daily_trade_count,
                current_price=float(close[i + 1]),
                max_hold_bars=int(max_hold_bars),
            )
            next_state = _state(market[i + 1], regime[i + 1], next_context)
            done = bool(i == len(train_df) - 2)
            bc_action = _weak_prior_action(
                int(position_map["side"]),
                int(hold_bars),
                float(ev_long_arr[i]),
                float(ev_short_arr[i]),
                edge_threshold=float(weak_bc_edge_threshold),
            )
            trainer.push_experience(state, int(action), float(reward), next_state, done, int(position_map["side"]), int(next_side), int(bc_action))
            if i % max(int(train_every), 1) == 0:
                loss = trainer.train_step(total_steps)
                if loss is not None:
                    losses.append(float(loss))
                    updates += 1
            episode_reward += float(reward)
            if int(log_every) > 0 and (i + 1) % int(log_every) == 0:
                tail = float(np.mean(losses[-100:])) if losses else 0.0
                print(
                    json.dumps(
                        {
                            "stage": "train_progress",
                            "episode": ep + 1,
                            "episodes": int(episodes),
                            "bar": i + 1,
                            "bars": len(train_df),
                            "epsilon": epsilon,
                            "loss_100": tail,
                            "updates": updates,
                            "reward": episode_reward,
                            "entries": entries,
                            "closes": closes,
                            "entry_rate": float(entries / max(i + 1, 1)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        episode_rewards.append(float(episode_reward))
        epsilon = max(float(epsilon_end), float(epsilon) * float(epsilon_decay))
        val_summary: dict[str, Any] | None = None
        if (
            val_df is not None
            and val_market is not None
            and val_regime is not None
            and int(validation_every) > 0
            and ((ep + 1) % int(validation_every) == 0)
        ):
            val_cost1 = _run_backtest(
                val_df,
                val_market,
                val_regime,
                trainer,
                fee=0.0005,
                slip=0.0002,
                unit_exposure=float(unit_exposure),
                min_hold_bars=int(min_hold_bars),
                hard_min_hold_bars=int(hard_min_hold_bars),
                max_hold_bars=int(max_hold_bars),
                entry_edge_threshold=float(entry_edge_threshold),
                log_name=f"train_val_ep{ep + 1}",
                log_every=0,
            )
            score = _validation_score(val_cost1)
            scheduler.step(score)
            lr_now = float(trainer.optimizer.param_groups[0]["lr"])
            improved = score > best_score + float(early_stop_min_delta)
            if improved:
                best_score = score
                best_episode = ep + 1
                best_state = {
                    "online": copy.deepcopy(trainer.online.state_dict()),
                    "target": copy.deepcopy(trainer.target.state_dict()),
                    "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
                    "score": float(score),
                    "episode": int(ep + 1),
                }
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": best_state["online"],
                        "target_state_dict": best_state["target"],
                        "optimizer_state_dict": best_state["optimizer"],
                        "scheduler_state_dict": scheduler.state_dict(),
                        "cfg": cfg.__dict__,
                        "episode": int(ep + 1),
                        "score": float(score),
                        "metrics": val_cost1,
                        "epsilon": float(epsilon),
                        "per_beta": float(trainer.buffer.beta),
                        "replay_entries": int(trainer.buffer.tree.n_entries),
                    },
                    checkpoint_dir / "single_conditioned_dqn_best.pt",
                )
            else:
                bad_epochs += 1
            val_summary = {
                "episode": int(ep + 1),
                "score": float(score),
                "best_score": float(best_score),
                "best_episode": int(best_episode),
                "improved": bool(improved),
                "bad_epochs": int(bad_epochs),
                "lr": lr_now,
                "epsilon": float(epsilon),
                "per_beta": float(trainer.buffer.beta),
                "replay_entries": int(trainer.buffer.tree.n_entries),
                "q_mean": float(val_cost1.get("q_mean", 0.0)),
                "q_std": float(val_cost1.get("q_std", 0.0)),
                "action_counts": val_cost1.get("action_counts", {}),
                "metrics": {k: val_cost1[k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
            }
            validation_history.append(val_summary)
            print(json.dumps({"stage": "validation", **val_summary}, ensure_ascii=False, default=_json_default), flush=True)
            _save_checkpoint(ep + 1, score, "validation")

        print(
            json.dumps(
                {
                    "stage": "episode_done",
                    "episode": ep + 1,
                    "reward": episode_reward,
                    "epsilon_next": epsilon,
                    "loss_100": float(np.mean(losses[-100:])) if losses else 0.0,
                    "updates": updates,
                    "entries": entries,
                    "closes": closes,
                    "entry_rate": float(entries / max(len(train_df) - 1, 1)),
                    "validation": val_summary,
                },
                ensure_ascii=False,
                default=_json_default,
            ),
            flush=True,
        )
        if int(early_stop_patience) > 0 and bad_epochs >= int(early_stop_patience):
            print(
                json.dumps(
                    {
                        "stage": "early_stop",
                        "episode": ep + 1,
                        "bad_epochs": int(bad_epochs),
                        "best_episode": int(best_episode),
                        "best_score": float(best_score),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    if best_state is not None:
        trainer.online.load_state_dict(best_state["online"])
        trainer.target.load_state_dict(best_state["target"])
        trainer.optimizer.load_state_dict(best_state["optimizer"])
        restored_best = True
    else:
        _save_checkpoint(len(episode_rewards), None, "final_no_validation_best")
        restored_best = False

    meta = {
        "episodes": int(episodes),
        "episodes_completed": int(len(episode_rewards)),
        "updates": int(updates),
        "loss_tail_100": float(np.mean(losses[-100:])) if losses else 0.0,
        "pretrain_q_loss_tail_100": float(np.mean(pretrain_q_losses[-100:])) if pretrain_q_losses else 0.0,
        "pretrain_q_report": pretrain_q_report,
        "episode_rewards": episode_rewards,
        "replay_entries": int(trainer.buffer.tree.n_entries),
        "validation_history": validation_history,
        "best_score": float(best_score) if np.isfinite(best_score) else None,
        "best_episode": int(best_episode),
        "restored_best": bool(restored_best),
        "final_lr": float(trainer.optimizer.param_groups[0]["lr"]),
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "max",
            "factor": float(lr_scheduler_factor),
            "patience": int(lr_scheduler_patience),
        },
        "early_stop": {
            "patience": int(early_stop_patience),
            "min_delta": float(early_stop_min_delta),
            "triggered": bool(int(early_stop_patience) > 0 and bad_epochs >= int(early_stop_patience)),
        },
    }
    return trainer, meta


def _run_backtest(
    frame: pd.DataFrame,
    market: np.ndarray,
    regime: np.ndarray,
    trainer: ConditionedDQNTrainer,
    *,
    fee: float,
    slip: float,
    unit_exposure: float,
    min_hold_bars: int,
    hard_min_hold_bars: int,
    max_hold_bars: int,
    entry_edge_threshold: float,
    log_name: str,
    log_every: int,
) -> dict[str, Any]:
    close = frame["close"].to_numpy(dtype=np.float64)
    ev_long_arr, ev_short_arr = _edge_arrays(frame)
    cash = 1.0
    peak_equity = 1.0
    mdd = 0.0
    side = 0
    entry_price = 0.0
    entry_equity = 1.0
    peak_price = 0.0
    hold_bars = 0
    bars_since_last_trade = 10_000
    daily_trade_count = 0
    last_day: Any = None
    trades = wins = long_entries = short_entries = 0
    action_counts = {name: 0 for name in ("flat", "open_long", "open_short", "close", "hold_long", "hold_short")}
    exits: dict[str, int] = {}
    q_means: list[float] = []
    q_stds: list[float] = []
    exposure = float(unit_exposure)

    def mark(i: int) -> float:
        if side == 0:
            return cash
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry_price, entry_equity, peak_price, hold_bars, bars_since_last_trade, daily_trade_count, cash, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_price = _fill_price(frame, fill_i, side, slip, entry=True)
        peak_price = float(entry_price)
        entry_equity = cash
        hold_bars = 0
        bars_since_last_trade = 0
        daily_trade_count += 1
        cash -= cash * float(fee) * exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_position(i: int, reason: str) -> None:
        nonlocal side, entry_price, peak_price, hold_bars, bars_since_last_trade, daily_trade_count, cash, trades, wins
        fill_i = min(i + 1, len(frame) - 1)
        exit_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry_price = 0.0
        peak_price = 0.0
        hold_bars = 0
        bars_since_last_trade = 0
        daily_trade_count += 1

    for i in range(len(frame) - 2):
        day = pd.Timestamp(frame["timestamp"].iloc[i]).date()
        if last_day is None:
            last_day = day
        elif day != last_day:
            last_day = day
            daily_trade_count = 0
        if side != 0:
            hold_bars += 1
            peak_price = max(peak_price, close[i]) if side > 0 else min(peak_price, close[i])
        bars_since_last_trade += 1

        eq = mark(i)
        peak_equity = max(peak_equity, eq)
        mdd = min(mdd, eq / max(peak_equity, 1e-12) - 1.0)
        context = _context_from_position(
            side=side,
            entry_price=entry_price,
            peak_price=peak_price,
            hold_bars=hold_bars,
            bars_since_last_trade=bars_since_last_trade,
            daily_trade_count=daily_trade_count,
            current_price=float(close[i]),
            max_hold_bars=int(max_hold_bars),
        )
        state = _state(market[i], regime[i], context)
        q = _masked_q_values(trainer, state, side)
        finite_q = q[np.isfinite(q)]
        if finite_q.size:
            q_means.append(float(np.mean(finite_q)))
            q_stds.append(float(np.std(finite_q)))
        raw_action = int(np.nanargmax(q))
        action = ActionSpace.apply(raw_action, side, min_hold_bars=int(hard_min_hold_bars), hold_bars=int(hold_bars))
        if int(side) == 0 and int(action) in (ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT):
            edge = float(ev_long_arr[i]) if int(action) == ActionSpace.OPEN_LONG else float(ev_short_arr[i])
            if edge < float(entry_edge_threshold):
                action = ActionSpace.FLAT
        action_name = {
            ActionSpace.FLAT: "flat",
            ActionSpace.OPEN_LONG: "open_long",
            ActionSpace.OPEN_SHORT: "open_short",
            ActionSpace.CLOSE: "close",
            ActionSpace.HOLD_LONG: "hold_long",
            ActionSpace.HOLD_SHORT: "hold_short",
        }[int(action)]
        action_counts[action_name] += 1
        if side == 0 and int(action) == ActionSpace.OPEN_LONG:
            enter(i, 1)
        elif side == 0 and int(action) == ActionSpace.OPEN_SHORT:
            enter(i, -1)
        elif side != 0 and int(action) == ActionSpace.CLOSE:
            exit_position(i, "model_close")
        if int(log_every) > 0 and (i + 1) % int(log_every) == 0:
            print(json.dumps({"stage": "backtest_progress", "name": log_name, "bar": i + 1, "bars": len(frame), "cash": cash, "trades": trades}, ensure_ascii=False), flush=True)

    if side != 0:
        exit_position(len(frame) - 2, "end_of_data")
    eq = mark(len(frame) - 1)
    peak_equity = max(peak_equity, eq)
    mdd = min(mdd, eq / max(peak_equity, 1e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float((long_entries + short_entries) * exposure / max(len(frame), 1)),
        "q_mean": float(np.mean(q_means)) if q_means else 0.0,
        "q_std": float(np.mean(q_stds)) if q_stds else 0.0,
        "action_counts": action_counts,
        "exits": exits,
    }


def _metrics(frame: pd.DataFrame, market: np.ndarray, regime: np.ndarray, trainer: ConditionedDQNTrainer, *, fee: float, slip: float, unit_exposure: float, min_hold_bars: int, hard_min_hold_bars: int, max_hold_bars: int, entry_edge_threshold: float, name: str, log_every: int) -> dict[str, Any]:
    return {
        f"cost{mult}": _run_backtest(
            frame,
            market,
            regime,
            trainer,
            fee=float(fee) * float(mult),
            slip=float(slip) * float(mult),
            unit_exposure=float(unit_exposure),
            min_hold_bars=int(min_hold_bars),
            hard_min_hold_bars=int(hard_min_hold_bars),
            max_hold_bars=int(max_hold_bars),
            entry_edge_threshold=float(entry_edge_threshold),
            log_name=f"{name}_cost{mult}",
            log_every=int(log_every),
        )
        for mult in (1, 2, 3)
    }


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        cost: {k: metrics[cost][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")}
        for cost in ("cost1", "cost2", "cost3")
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate a single HMM-conditioned Dueling DQN + PER agent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=5418)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--train-every", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.92)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--min-hold-bars", type=int, default=12)
    p.add_argument("--hard-min-hold-bars", type=int, default=0)
    p.add_argument("--max-hold-bars", type=int, default=48)
    p.add_argument("--min-buffer-size", type=int, default=10_000)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--reward-trade-penalty", type=float, default=0.010)
    p.add_argument("--reward-early-close-penalty", type=float, default=0.020)
    p.add_argument("--reward-edge-bonus", type=float, default=0.20)
    p.add_argument("--reward-flat-opportunity-penalty", type=float, default=0.12)
    p.add_argument("--reward-opportunity-threshold", type=float, default=0.08)
    p.add_argument("--bc-weight", type=float, default=0.03)
    p.add_argument("--weak-bc-edge-threshold", type=float, default=0.10)
    p.add_argument("--pretrain-bc-steps", type=int, default=0)
    p.add_argument("--pretrain-bc-batch", type=int, default=512)
    p.add_argument("--pretrain-q-steps", type=int, default=0)
    p.add_argument("--pretrain-q-batch", type=int, default=512)
    p.add_argument("--pretrain-q-labeler", choices=("legacy", "ensemble"), default="legacy")
    p.add_argument("--pretrain-q-horizons", type=str, default="12,24,48,96,288")
    p.add_argument("--pretrain-q-scale", type=float, default=50.0)
    p.add_argument("--pretrain-q-ce-weight", type=float, default=0.20)
    p.add_argument("--pretrain-q-terminal-weight", type=float, default=0.45)
    p.add_argument("--pretrain-q-mfe-weight", type=float, default=0.65)
    p.add_argument("--pretrain-q-adverse-weight", type=float, default=0.80)
    p.add_argument("--pretrain-q-entry-hurdle", type=float, default=0.70)
    p.add_argument("--pretrain-q-clip", type=float, default=3.0)
    p.add_argument("--ensemble-label-confidence-min", type=float, default=0.30)
    p.add_argument("--ensemble-label-cache", type=Path, default=None)
    p.add_argument("--entry-edge-threshold", type=float, default=0.0)
    p.add_argument("--explore-flat-weight", type=float, default=0.85)
    p.add_argument("--explore-open-long-weight", type=float, default=0.075)
    p.add_argument("--explore-open-short-weight", type=float, default=0.075)
    p.add_argument("--explore-close-long-weight", type=float, default=0.10)
    p.add_argument("--explore-close-short-weight", type=float, default=0.10)
    p.add_argument("--include-future-regime-pred", action="store_true")
    p.add_argument("--feature-top-k", type=int, default=0)
    p.add_argument("--feature-select-horizon", type=int, default=48)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.97)
    p.add_argument("--validation-every", type=int, default=5)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=0.5)
    p.add_argument("--lr-scheduler-patience", type=int, default=5)
    p.add_argument("--lr-scheduler-factor", type=float, default=0.7)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--max-val-rows", type=int, default=0)
    p.add_argument("--max-eval-rows", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    audit = _verify_state24_sticky090_inputs(train_all, eval_df, DEFAULT_PREPROCESS_MANIFEST, DEFAULT_CLEAN4_REPORT)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    if int(args.max_train_rows) > 0:
        train_df = train_df.tail(int(args.max_train_rows)).reset_index(drop=True)
    if int(args.max_val_rows) > 0:
        val_df = val_df.tail(int(args.max_val_rows)).reset_index(drop=True)
    if int(args.max_eval_rows) > 0:
        eval_df = eval_df.head(int(args.max_eval_rows)).reset_index(drop=True)

    market_cols = _feature_cols(
        train_all,
        eval_df,
        include_future_regime_pred=bool(args.include_future_regime_pred),
        feature_top_k=int(args.feature_top_k),
        feature_select_horizon=int(args.feature_select_horizon),
    )
    bad = [
        c
        for c in market_cols
        if (c.startswith(FORBIDDEN_PREFIXES) and not (bool(args.include_future_regime_pred) and c.startswith("regime4_pred_")))
        or c in ROUTER_PROB_SET
        or c in FORBIDDEN_EXACT
    ]
    if bad:
        raise ValueError("invalid market feature leakage: " + ", ".join(bad[:20]))
    if not bool(args.include_future_regime_pred) and any(c.startswith("regime4_pred_") for c in market_cols):
        raise ValueError("future regime4_pred leaked")
    scaler = _fit_market_scaler(train_df, market_cols)
    train_market = _transform_market(train_df, market_cols, scaler)
    val_market = _transform_market(val_df, market_cols, scaler)
    eval_market = _transform_market(eval_df, market_cols, scaler)
    train_regime = _regime_matrix(train_df)
    val_regime = _regime_matrix(val_df)
    eval_regime = _regime_matrix(eval_df)

    cfg = ConditionedDQNTrainerConfig(
        state_dim=len(market_cols) + len(ROUTER_COLS) + 7,
        market_dim=len(market_cols),
        regime_dim=len(ROUTER_COLS),
        context_dim=7,
        action_dim=ActionSpace.N_ACTIONS,
        lr=float(args.lr),
        gamma=float(args.gamma),
        batch_size=int(args.batch_size),
        target_update=int(args.target_update),
        buffer_size=int(args.buffer_size),
        min_buffer_size=int(args.min_buffer_size),
        bc_weight=float(args.bc_weight),
    )
    ActionSpace.configure_exploration(
        flat_weight=float(args.explore_flat_weight),
        open_long_weight=float(args.explore_open_long_weight),
        open_short_weight=float(args.explore_open_short_weight),
        close_long_weight=float(args.explore_close_long_weight),
        close_short_weight=float(args.explore_close_short_weight),
    )
    print(
        json.dumps(
            {
                "stage": "single_conditioned_dqn_start",
                "model_id": MODEL_ID,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "market_dim": len(market_cols),
                "include_future_regime_pred": bool(args.include_future_regime_pred),
                "feature_top_k": int(args.feature_top_k),
                "feature_select_horizon": int(args.feature_select_horizon),
                "regime_dim": len(ROUTER_COLS),
                "context_dim": 7,
                "raw_state_dim": len(market_cols) + len(ROUTER_COLS) + 7,
                "internal_backbone_dim": len(market_cols) + 16 + 7,
                "train_rows": len(train_df),
                "selection_rows": len(val_df),
                "oos_rows": len(eval_df),
                "episodes": int(args.episodes),
                "train_every": int(args.train_every),
                "min_buffer_size": int(args.min_buffer_size),
                "soft_min_hold_bars": int(args.min_hold_bars),
                "hard_min_hold_bars": int(args.hard_min_hold_bars),
                "reward_scale": float(args.reward_scale),
                "reward_trade_penalty": float(args.reward_trade_penalty),
                "reward_early_close_penalty": float(args.reward_early_close_penalty),
                "reward_edge_bonus": float(args.reward_edge_bonus),
                "reward_flat_opportunity_penalty": float(args.reward_flat_opportunity_penalty),
                "reward_opportunity_threshold": float(args.reward_opportunity_threshold),
                "bc_weight": float(args.bc_weight),
                "weak_bc_edge_threshold": float(args.weak_bc_edge_threshold),
                "pretrain_bc_steps": int(args.pretrain_bc_steps),
                "pretrain_bc_batch": int(args.pretrain_bc_batch),
                "pretrain_q_steps": int(args.pretrain_q_steps),
                "pretrain_q_batch": int(args.pretrain_q_batch),
                "pretrain_q_labeler": str(args.pretrain_q_labeler),
                "pretrain_q_horizons": str(args.pretrain_q_horizons),
                "pretrain_q_scale": float(args.pretrain_q_scale),
                "pretrain_q_ce_weight": float(args.pretrain_q_ce_weight),
                "ensemble_label_confidence_min": float(args.ensemble_label_confidence_min),
                "ensemble_label_cache": None if args.ensemble_label_cache is None else str(args.ensemble_label_cache),
                "entry_edge_threshold": float(args.entry_edge_threshold),
                "exploration_weights": {
                    "flat": {
                        "flat": float(args.explore_flat_weight),
                        "open_long": float(args.explore_open_long_weight),
                        "open_short": float(args.explore_open_short_weight),
                    },
                    "long": {"close": float(args.explore_close_long_weight), "hold_long": max(1.0 - float(args.explore_close_long_weight), 0.0)},
                    "short": {"close": float(args.explore_close_short_weight), "hold_short": max(1.0 - float(args.explore_close_short_weight), 0.0)},
                },
                "validation_every": int(args.validation_every),
                "early_stop_patience": int(args.early_stop_patience),
                "lr_scheduler": {
                    "type": "ReduceLROnPlateau",
                    "patience": int(args.lr_scheduler_patience),
                    "factor": float(args.lr_scheduler_factor),
                },
                "audit": {
                    "expected_model": audit.get("expected_model"),
                    "expected_model_found_in_manifest": audit.get("expected_model_found_in_manifest"),
                    "report_states": audit.get("report_states"),
                    "report_sticky": audit.get("report_sticky"),
                    "legacy_v4_count": audit.get("legacy_v4_count"),
                    "future_regime4_common_count": audit.get("future_regime4_common_count"),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    trainer, train_meta = _train_agent(
        train_df,
        train_market,
        train_regime,
        val_df=val_df,
        val_market=val_market,
        val_regime=val_regime,
        cfg=cfg,
        episodes=int(args.episodes),
        train_every=int(args.train_every),
        min_hold_bars=int(args.min_hold_bars),
        hard_min_hold_bars=int(args.hard_min_hold_bars),
        max_hold_bars=int(args.max_hold_bars),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay=float(args.epsilon_decay),
        log_every=int(args.log_every),
        validation_every=int(args.validation_every),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        lr_scheduler_patience=int(args.lr_scheduler_patience),
        lr_scheduler_factor=float(args.lr_scheduler_factor),
        checkpoint_dir=args.out_dir,
        unit_exposure=float(args.unit_exposure),
        reward_scale=float(args.reward_scale),
        reward_trade_penalty=float(args.reward_trade_penalty),
        reward_early_close_penalty=float(args.reward_early_close_penalty),
        reward_edge_bonus=float(args.reward_edge_bonus),
        reward_flat_opportunity_penalty=float(args.reward_flat_opportunity_penalty),
        reward_opportunity_threshold=float(args.reward_opportunity_threshold),
        weak_bc_edge_threshold=float(args.weak_bc_edge_threshold),
        pretrain_bc_steps=int(args.pretrain_bc_steps),
        pretrain_bc_batch=int(args.pretrain_bc_batch),
        pretrain_q_steps=int(args.pretrain_q_steps),
        pretrain_q_batch=int(args.pretrain_q_batch),
        pretrain_q_labeler=str(args.pretrain_q_labeler),
        pretrain_q_horizons=str(args.pretrain_q_horizons),
        pretrain_q_scale=float(args.pretrain_q_scale),
        pretrain_q_ce_weight=float(args.pretrain_q_ce_weight),
        pretrain_q_terminal_weight=float(args.pretrain_q_terminal_weight),
        pretrain_q_mfe_weight=float(args.pretrain_q_mfe_weight),
        pretrain_q_adverse_weight=float(args.pretrain_q_adverse_weight),
        pretrain_q_entry_hurdle=float(args.pretrain_q_entry_hurdle),
        pretrain_q_clip=float(args.pretrain_q_clip),
        ensemble_label_confidence_min=float(args.ensemble_label_confidence_min),
        ensemble_label_cache=args.ensemble_label_cache,
        entry_edge_threshold=float(args.entry_edge_threshold),
    )

    fee = 0.0005
    slip = 0.0002
    val_metrics = _metrics(
        val_df,
        val_market,
        val_regime,
        trainer,
        fee=fee,
        slip=slip,
        unit_exposure=float(args.unit_exposure),
        min_hold_bars=int(args.min_hold_bars),
        hard_min_hold_bars=int(args.hard_min_hold_bars),
        max_hold_bars=int(args.max_hold_bars),
        entry_edge_threshold=float(args.entry_edge_threshold),
        name="selection",
        log_every=int(args.log_every),
    )
    eval_metrics = _metrics(
        eval_df,
        eval_market,
        eval_regime,
        trainer,
        fee=fee,
        slip=slip,
        unit_exposure=float(args.unit_exposure),
        min_hold_bars=int(args.min_hold_bars),
        hard_min_hold_bars=int(args.hard_min_hold_bars),
        max_hold_bars=int(args.max_hold_bars),
        entry_edge_threshold=float(args.entry_edge_threshold),
        name="oos",
        log_every=int(args.log_every),
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Single HMM-conditioned Dueling DQN + PER. HMM Regime4 probabilities are state features, not routing gates. One agent consumes market features, 4 current regime probabilities, and 7 position context features. Action space has flat/open_long/open_short/close/hold_long/hold_short.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "feature_contract": {
            "market_dim": len(market_cols),
            "regime_dim": len(ROUTER_COLS),
            "context_dim": 7,
            "raw_state_dim": len(market_cols) + len(ROUTER_COLS) + 7,
            "internal_backbone_dim": len(market_cols) + 16 + 7,
            "market_cols": market_cols,
            "regime_cols": ROUTER_COLS,
            "legacy_clean_v4_count": int(sum(c.startswith("clean_regime_2024_unsup_v4_") for c in market_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in market_cols)),
            "include_future_regime_pred": bool(args.include_future_regime_pred),
            "feature_top_k": int(args.feature_top_k),
            "feature_select_horizon": int(args.feature_select_horizon),
            "router_prob_in_market_count": int(sum(c in ROUTER_PROB_SET for c in market_cols)),
            "clean4_aux_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in market_cols)),
        },
        "state24_sticky090_feature_audit": audit,
        "train_meta": train_meta,
        "config": vars(args),
        "training_techniques": {
            "validation_score": "pnl_pct - 0.35*abs(mdd_pct) - 2.0*max(trades_per_day-8,0) - 20.0*max(3-trades_per_day,0)",
            "validation_every": int(args.validation_every),
            "best_snapshot_restore": True,
            "checkpoint_contains": [
                "model",
                "target_model",
                "optimizer",
                "lr_scheduler",
                "epsilon",
                "per_beta",
                "replay_entries",
            ],
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "max",
                "patience": int(args.lr_scheduler_patience),
                "factor": float(args.lr_scheduler_factor),
            },
            "early_stop": {
                "patience": int(args.early_stop_patience),
                "min_delta": float(args.early_stop_min_delta),
            },
            "epsilon_schedule": {
                "type": "episode_decay",
                "start": float(args.epsilon_start),
                "end": float(args.epsilon_end),
                "decay": float(args.epsilon_decay),
            },
            "exploration_policy": {
                "flat": {
                    "flat": float(args.explore_flat_weight),
                    "open_long": float(args.explore_open_long_weight),
                    "open_short": float(args.explore_open_short_weight),
                },
                "long": {"close": float(args.explore_close_long_weight), "hold_long": max(1.0 - float(args.explore_close_long_weight), 0.0)},
                "short": {"close": float(args.explore_close_short_weight), "hold_short": max(1.0 - float(args.explore_close_short_weight), 0.0)},
            },
            "environment_rules": {
                "min_hold_mode": "soft_reward_penalty_only" if int(args.hard_min_hold_bars) <= 0 else "soft_reward_penalty_plus_hard_action_guard",
                "soft_min_hold_bars": int(args.min_hold_bars),
                "hard_min_hold_bars": int(args.hard_min_hold_bars),
                "train_every": int(args.train_every),
                "min_buffer_size": int(args.min_buffer_size),
                "reward_scale": float(args.reward_scale),
                "reward_trade_penalty": float(args.reward_trade_penalty),
                "reward_early_close_penalty": float(args.reward_early_close_penalty),
                "reward_edge_bonus": float(args.reward_edge_bonus),
                "reward_flat_opportunity_penalty": float(args.reward_flat_opportunity_penalty),
                "reward_opportunity_threshold": float(args.reward_opportunity_threshold),
                "bc_weight": float(args.bc_weight),
                "weak_bc_edge_threshold": float(args.weak_bc_edge_threshold),
                "pretrain_bc_steps": int(args.pretrain_bc_steps),
                "pretrain_bc_batch": int(args.pretrain_bc_batch),
                "pretrain_q_steps": int(args.pretrain_q_steps),
                "pretrain_q_batch": int(args.pretrain_q_batch),
                "pretrain_q_labeler": str(args.pretrain_q_labeler),
                "pretrain_q_horizons": str(args.pretrain_q_horizons),
                "pretrain_q_scale": float(args.pretrain_q_scale),
                "pretrain_q_ce_weight": float(args.pretrain_q_ce_weight),
                "pretrain_q_terminal_weight": float(args.pretrain_q_terminal_weight),
                "pretrain_q_mfe_weight": float(args.pretrain_q_mfe_weight),
                "pretrain_q_adverse_weight": float(args.pretrain_q_adverse_weight),
                "pretrain_q_entry_hurdle": float(args.pretrain_q_entry_hurdle),
                "pretrain_q_clip": float(args.pretrain_q_clip),
                "ensemble_label_confidence_min": float(args.ensemble_label_confidence_min),
                "ensemble_label_cache": None if args.ensemble_label_cache is None else str(args.ensemble_label_cache),
                "entry_edge_threshold": float(args.entry_edge_threshold),
            },
            "recommended_profiles": {
                "quick": "--episodes 30 --train-every 32 --validation-every 5 --early-stop-patience 10 --epsilon-decay 0.97 --min-hold-bars 12 --hard-min-hold-bars 0 --reward-early-close-penalty 0.020 --min-buffer-size 10000",
                "full": "--episodes 200 --train-every 32 --validation-every 5 --early-stop-patience 25 --early-stop-min-delta 0.3 --lr-scheduler-patience 8 --lr-scheduler-factor 0.5 --epsilon-decay 0.97 --min-hold-bars 12 --hard-min-hold-bars 0 --reward-early-close-penalty 0.020 --min-buffer-size 10000",
            },
        },
        "validation_metrics": val_metrics,
        "metrics": eval_metrics,
        "selection_score": _score(val_metrics["cost1"], val_metrics["cost2"], val_metrics["cost3"]),
        "selected_metrics": _compact(eval_metrics),
        "artifacts": {
            "model": str(args.out_dir / "single_conditioned_dqn_agent.pt"),
            "best_model": str(args.out_dir / "single_conditioned_dqn_best.pt"),
            "checkpoint": str(args.out_dir / "single_conditioned_dqn_checkpoint.pt"),
            "scaler": str(args.out_dir / "single_conditioned_dqn_scaler.joblib"),
            "summary": str(args.out_dir / "alpha5_4_single_conditioned_dqn_summary.json"),
        },
    }
    agent_path = args.out_dir / "single_conditioned_dqn_agent.pt"
    best_path = args.out_dir / "single_conditioned_dqn_best.pt"
    torch.save({"model_state_dict": trainer.online.state_dict(), "target_state_dict": trainer.target.state_dict(), "cfg": cfg.__dict__}, agent_path)

    def _reload_oos_cost1(path: Path) -> dict[str, Any]:
        ckpt = torch.load(path, map_location=trainer.device)
        reload_cfg = ConditionedDQNTrainerConfig(**{**cfg.__dict__, **dict(ckpt.get("cfg", {}))})
        reload_trainer = ConditionedDQNTrainer(reload_cfg, device=trainer.device)
        reload_trainer.online.load_state_dict(ckpt["model_state_dict"])
        reload_trainer.target.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        reload_trainer.online.eval()
        reload_trainer.target.eval()
        return _run_backtest(
            eval_df,
            eval_market,
            eval_regime,
            reload_trainer,
            fee=fee,
            slip=slip,
            unit_exposure=float(args.unit_exposure),
            min_hold_bars=int(args.min_hold_bars),
            hard_min_hold_bars=int(args.hard_min_hold_bars),
            max_hold_bars=int(args.max_hold_bars),
            entry_edge_threshold=float(args.entry_edge_threshold),
            log_name=f"reload_{path.stem}_oos_cost1",
            log_every=0,
        )

    reload_checks: dict[str, Any] = {}
    for label, path in (("agent", agent_path), ("best", best_path)):
        if path.exists():
            metrics = _reload_oos_cost1(path)
            reload_checks[label] = {
                "path": str(path),
                "metrics": {k: metrics[k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr")},
                "diff_vs_report_cost1": {
                    "pnl": float(metrics["pnl"] - eval_metrics["cost1"]["pnl"]),
                    "mdd": float(metrics["mdd"] - eval_metrics["cost1"]["mdd"]),
                    "trades": int(metrics["trades"] - eval_metrics["cost1"]["trades"]),
                },
                "passed": bool(
                    abs(float(metrics["pnl"] - eval_metrics["cost1"]["pnl"])) < 1e-9
                    and abs(float(metrics["mdd"] - eval_metrics["cost1"]["mdd"])) < 1e-9
                    and int(metrics["trades"]) == int(eval_metrics["cost1"]["trades"])
                ),
            }
    report["reload_verification"] = reload_checks
    joblib.dump({"market_cols": market_cols, "regime_cols": ROUTER_COLS, "scaler": scaler}, args.out_dir / "single_conditioned_dqn_scaler.joblib")
    (args.out_dir / "alpha5_4_single_conditioned_dqn_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    rows = [
        {
            "name": "single_conditioned_dqn",
            "selection_score": report["selection_score"],
            "val_cost1_pnl": val_metrics["cost1"]["pnl"],
            "val_cost1_mdd": val_metrics["cost1"]["mdd"],
            "val_trades": val_metrics["cost1"]["trades"],
            "eval_cost1_pnl": eval_metrics["cost1"]["pnl"],
            "eval_cost1_mdd": eval_metrics["cost1"]["mdd"],
            "eval_trades": eval_metrics["cost1"]["trades"],
        }
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "alpha5_4_single_conditioned_dqn_grid.csv", index=False)
    print(json.dumps({"stage": "complete", "summary": report["artifacts"]["summary"], "selected_metrics": report["selected_metrics"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
