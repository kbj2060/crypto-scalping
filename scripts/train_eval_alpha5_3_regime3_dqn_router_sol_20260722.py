#!/usr/bin/env python3
"""Alpha5.3 architecture ported to SOL with Regime3 (not Regime4) routing.

Regime4 is deprecated project-wide and no SOL Regime4 artifact exists, so
this uses SOL's existing, currently-live Regime3 artifact
(sol_regime3_current_hmm_sensitive_wide24_20260707,
regime3_current_sensitive_wide24_{bull,bear,chop}_prob) as router state
instead of the original 4-class HMM Regime4. There is no "whipsaw" class in
Regime3 - only 3 specialists are trained (bull/bear/chop); no 4th class is
fabricated.

Everything else follows
docs/model_contracts/alpha5_3_hmm_dqn_router_parent_20260517_contract.md /
scripts/train_eval_alpha5_3_hmm_dqn_router_parent_20260517.py:
  - fixed Regime3 + TP=1.8%/SL=1.0% barrier frame (built by
    scripts/build_sol_regime3_tp18_sl10_preprocess_20260722.py)
  - per-regime Dueling DQN + PER-like action head, action_prob_long/short/cash
    only (no TP/SL/sizing heads)
  - hard/soft router decision, action-only evaluator (unit_exposure=1.0,
    enter/exit/flip purely from routed action stream)

Evaluation protocol per the contract itself (dates differ from this repo's
default fresh-forward split in CLAUDE.md, which is used elsewhere in this
project's non-Alpha5.3 lines):
  train      2025-01-01 .. 2025-09-30
  selection  2025-10-01 .. 2025-12-31
  OOS        2026-01-01 .. 2026-02-28
Selection is based only on the 2025-10-01..2025-12-31 selection window
(selection_uses_2026 == false). The canonical OOS window and an additional
never-before-touched fresh-forward OOS window (2026-03-01..2026-07-21) are
both reported for the selected router mode, but neither is used for model
selection. Bar-by-bar causal walk-forward only: no saved trade ledgers, no
future rows joined into a current decision.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.dueling_dqn_parent import DuelingDQNConfig, DuelingQNetwork, make_action_model  # noqa: E402
from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
)
from scripts.build_sol_regime3_tp18_sl10_preprocess_20260722 import (  # noqa: E402
    DEFAULT_EVAL_RAW as SOL_EVAL_RAW,
    DEFAULT_OUT_DIR as SOL_PREPROCESS_DIR,
    DEFAULT_TRAIN_RAW as SOL_TRAIN_RAW,
    REGIME3_PREFIX,
    REGIMES3,
    _sol_feature_cols,
)
from scripts.train_eval_alpha4_3_dueling_dqn_parent_20260517 import (  # noqa: E402
    _action_dist,
    _prep_matrix,
    _seed,
)
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import BASE_PARENT, _score  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_3_regime3_dqn_router_sol_20260722"
DEFAULT_TRAIN = SOL_PREPROCESS_DIR / "trade_candidates_2025_sol_regime3_tp18_sl10_fixed.csv"
DEFAULT_EVAL = SOL_PREPROCESS_DIR / "trade_candidates_2026_sol_regime3_tp18_sl10_fixed.csv"
DEFAULT_PREPROCESS_MANIFEST = SOL_PREPROCESS_DIR / "sol_regime3_tp18_sl10_preprocess_manifest.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_3_regime3_dqn_router_sol_20260722"

REGIMES = REGIMES3  # ("bull", "bear", "chop") - no whipsaw class in Regime3
ROUTER_COLS = [f"{REGIME3_PREFIX}{r}_prob" for r in REGIMES]
ROUTER_PROB_SET = set(ROUTER_COLS)
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_", "regime4_pred_")
ACTION_PROB_COLS = ["action_prob_long", "action_prob_short", "action_prob_cash"]
POSITION_CONTEXT_COLS = [
    "position_current",
    "position_hold_duration_norm",
    "position_unrealized_pnl",
    "position_entry_price_dist",
    "position_bars_since_entry",
]
POSITION_CONTEXT_NORM_BARS = 288.0
FEE = 0.0005
SLIP = 0.0002

TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
SELECTION_START = pd.Timestamp("2025-10-01")
SELECTION_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_CANONICAL_START = pd.Timestamp("2026-01-01")
OOS_CANONICAL_END = pd.Timestamp("2026-02-28 23:59:59")
OOS_FRESH_START = pd.Timestamp("2026-03-01")
OOS_FRESH_END = pd.Timestamp("2026-07-21 23:59:59")


def _valid_indices(n: int, max_horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n) - int(max_horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _router_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in ROUTER_COLS if c not in frame.columns]
    if missing:
        raise ValueError("missing Regime3 router columns: " + ", ".join(missing))
    p = frame[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p


def _verify_regime3_inputs(train: pd.DataFrame, eval_df: pd.DataFrame, manifest_path: Path) -> dict[str, Any]:
    train_cols = set(train.columns)
    eval_cols = set(eval_df.columns)
    router_missing = [c for c in ROUTER_COLS if c not in train_cols or c not in eval_cols]
    forbidden_present = [c for c in sorted(train_cols | eval_cols) if c.startswith(FORBIDDEN_PREFIXES)]
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sums_train = train[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1) if not router_missing else pd.Series(dtype=float)
    sums_eval = eval_df[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1) if not router_missing else pd.Series(dtype=float)
    audit = {
        "manifest": str(manifest_path),
        "manifest_exists": bool(manifest_path.exists()),
        "router_cols": ROUTER_COLS,
        "router_missing": router_missing,
        "forbidden_regime4_cols_present": forbidden_present,
        "train_prob_sum_min": float(sums_train.min()) if len(sums_train) else None,
        "train_prob_sum_max": float(sums_train.max()) if len(sums_train) else None,
        "eval_prob_sum_min": float(sums_eval.min()) if len(sums_eval) else None,
        "eval_prob_sum_max": float(sums_eval.max()) if len(sums_eval) else None,
        "no_whipsaw_class": True,
        "regime_taxonomy": list(REGIMES),
    }
    print(json.dumps({"stage": "regime3_feature_audit", **audit}, ensure_ascii=False), flush=True)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing preprocessing manifest: {manifest_path}")
    if router_missing:
        raise ValueError("missing router columns: " + ",".join(router_missing))
    if forbidden_present:
        raise ValueError("forbidden Regime4 columns leaked: " + ",".join(forbidden_present[:20]))
    return audit


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    common = set(train.columns) & set(eval_df.columns)
    cols = [c for c in _sol_feature_cols() if c in common]
    if "tp_sl_action_score" in common and "tp_sl_action_score" not in cols:
        cols.append("tp_sl_action_score")
    return cols


def _subset_y(y: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.asarray(v)[mask].copy() for k, v in y.items()}


def _with_position_context(
    x: pd.DataFrame,
    *,
    current_position: np.ndarray | float,
    hold_duration_norm: np.ndarray | float,
    unrealized_pnl: np.ndarray | float,
    entry_price_dist: np.ndarray | float,
    bars_since_entry: np.ndarray | float,
) -> pd.DataFrame:
    out = x.copy()
    n = len(out)
    vals = {
        "position_current": current_position,
        "position_hold_duration_norm": hold_duration_norm,
        "position_unrealized_pnl": unrealized_pnl,
        "position_entry_price_dist": entry_price_dist,
        "position_bars_since_entry": bars_since_entry,
    }
    for col, val in vals.items():
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(n, float(arr), dtype=np.float64)
        out[col] = arr.astype(np.float64)
    return out


def _augment_position_context_training(
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    *,
    frame: pd.DataFrame,
    valid_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    close = _close(frame)
    idx = np.asarray(valid_idx, dtype=np.int64)
    bars = 6 + (idx % 42)
    entry_idx = np.maximum(idx - bars, 0)
    raw_dist = close[idx] / np.maximum(close[entry_idx], 1e-12) - 1.0
    bars_f = bars.astype(np.float64)
    flat = _with_position_context(x, current_position=0.0, hold_duration_norm=0.0, unrealized_pnl=0.0, entry_price_dist=0.0, bars_since_entry=0.0)
    long = _with_position_context(
        x, current_position=1.0, hold_duration_norm=np.clip(bars_f / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0),
        unrealized_pnl=raw_dist, entry_price_dist=raw_dist, bars_since_entry=bars_f,
    )
    short = _with_position_context(
        x, current_position=-1.0, hold_duration_norm=np.clip(bars_f / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0),
        unrealized_pnl=-raw_dist, entry_price_dist=raw_dist, bars_since_entry=bars_f,
    )
    x_aug = pd.concat([flat, long, short], ignore_index=True)
    y_aug = {k: np.concatenate([np.asarray(v), np.asarray(v), np.asarray(v)]) for k, v in y.items()}
    return x_aug, y_aug


def _train_dqn_td_priority_only(
    x: np.ndarray, actions: np.ndarray, rewards: np.ndarray, *,
    cfg: DuelingDQNConfig, steps: int, batch_size: int, gamma: float, lr: float, bc_weight: float,
    seed: int, device: torch.device, log_name: str, log_every: int,
) -> tuple[DuelingQNetwork, dict[str, Any]]:
    _seed(seed)
    n = int(len(x))
    next_idx = np.minimum(np.arange(n) + 1, n - 1)
    done = np.zeros(n, dtype=np.float32)
    done[-1] = 1.0
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    reward_std = float(np.std(rewards)) if float(np.std(rewards)) > 1e-6 else 1.0
    reward_center = float(np.median(rewards))
    r = ((rewards - reward_center) / reward_std).astype(np.float32)

    model = DuelingQNetwork(cfg).to(device)
    target = DuelingQNetwork(cfg).to(device)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    priorities = np.abs(r).astype(np.float32) + 1e-6
    counts = np.bincount(actions, minlength=int(cfg.action_dim)).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1.0)
    inv = inv / np.maximum(inv.mean(), 1e-6)
    class_weight = torch.tensor(np.clip(inv, 0.5, 3.0), dtype=torch.float32, device=device)
    losses: list[float] = []
    td_means: list[float] = []
    print(json.dumps({"stage": "dqn_train_start", "name": log_name, "rows": int(n), "steps": int(steps), "batch_size": int(batch_size), "gamma": float(gamma), "bc_weight": float(bc_weight), "action_counts": _action_dist(actions), "priority_policy": "td_error_only_no_trade_boost"}, ensure_ascii=False), flush=True)
    for step in range(1, int(steps) + 1):
        prob = priorities ** 0.65
        prob = prob / np.maximum(prob.sum(), 1e-12)
        idx = np.random.choice(n, size=min(int(batch_size), n), replace=True, p=prob)
        xb = torch.from_numpy(x[idx]).to(device)
        nb = torch.from_numpy(x[next_idx[idx]]).to(device)
        ab = torch.from_numpy(actions[idx]).to(device)
        rb = torch.from_numpy(r[idx]).to(device)
        db = torch.from_numpy(done[idx]).to(device)

        q = model(xb)
        qa = q.gather(1, ab.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_action = torch.argmax(model(nb), dim=1, keepdim=True)
            next_q = target(nb).gather(1, next_action).squeeze(1)
            td_target = rb + float(gamma) * (1.0 - db) * next_q
        td = td_target - qa
        td_loss = F.smooth_l1_loss(qa, td_target)
        bc_loss = F.cross_entropy(q, ab, weight=class_weight)
        loss = td_loss + float(bc_weight) * bc_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        priorities[idx] = 0.90 * priorities[idx] + 0.10 * (np.abs(td.detach().cpu().numpy()) + 1e-6)
        if step % 100 == 0:
            target.load_state_dict(model.state_dict())
        if step % 50 == 0 or step == 1:
            losses.append(float(loss.detach().cpu()))
            td_means.append(float(torch.mean(torch.abs(td)).detach().cpu()))
        if int(log_every) > 0 and (step == 1 or step % int(log_every) == 0 or step == int(steps)):
            print(json.dumps({"stage": "dqn_train_progress", "name": log_name, "step": int(step), "steps": int(steps), "pct": round(100.0 * float(step) / max(float(steps), 1.0), 2), "loss": float(loss.detach().cpu()), "td_abs_mean": float(torch.mean(torch.abs(td)).detach().cpu())}, ensure_ascii=False), flush=True)
    target.load_state_dict(model.state_dict())
    meta = {"reward_center": reward_center, "reward_std": reward_std, "loss_tail": losses[-10:], "td_abs_tail": td_means[-10:], "steps": int(steps), "batch_size": int(batch_size), "gamma": float(gamma), "bc_weight": float(bc_weight), "priority_policy": "td_error_only_no_trade_boost", "class_weight": class_weight.detach().cpu().numpy().tolist()}
    return model, meta


def _fit_dqn_action_parent(
    x: pd.DataFrame, y: dict[str, np.ndarray], *,
    dqn_cfg: DuelingDQNConfig, steps: int, batch_size: int, gamma: float, lr: float, bc_weight: float,
    seed: int, device: torch.device, log_name: str, log_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_np, med, mean, std = _prep_matrix(x)
    model, train_meta = _train_dqn_td_priority_only(
        x_np, np.asarray(y["action"], dtype=np.int64), np.asarray(y["quality"], dtype=np.float32),
        cfg=dqn_cfg, steps=int(steps), batch_size=int(batch_size), gamma=float(gamma), lr=float(lr),
        bc_weight=float(bc_weight), seed=int(seed), device=device, log_name=log_name, log_every=int(log_every),
    )
    out = {
        "model_type": "alpha5_3_regime3_router_specialist_dueling_dqn_action_parent",
        "output_contract": list(ACTION_PROB_COLS),
        "feature_cols": list(x.columns),
        "config": {},
    }
    out["action_model"] = make_action_model(model, config=dqn_cfg, medians=med, mean=mean, std=std, feature_cols=list(x.columns))
    meta = {"dqn_train_meta": train_meta, "action_distribution": _action_dist(np.asarray(y["action"], dtype=np.int64)), "sample_count": int(len(x))}
    return out, meta


def _train_specialists(
    *, train_df: pd.DataFrame, feature_cols: list[str], label_cfg: FullyLearnedGovernorConfig, stride: int,
    min_samples: int, dqn_steps: int, batch_size: int, gamma: float, lr: float, bc_weight: float,
    hidden_dim: int, dropout: float, temperature: float, log_every: int, seed: int, device: torch.device, out_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    x_full, y_full, train_meta = build_training_set(train_df, cfg=label_cfg, stride_bars=int(stride), batch_size=512, feature_cols=feature_cols)
    valid = _valid_indices(len(train_df), int(label_cfg.max_train_horizon_bars), int(stride))
    base_router_label = np.argmax(_router_matrix(train_df)[valid], axis=1)
    x_full, y_full = _augment_position_context_training(x_full, y_full, frame=train_df, valid_idx=valid)
    state_feature_cols = list(feature_cols) + list(POSITION_CONTEXT_COLS)
    router_label = np.concatenate([base_router_label, base_router_label, base_router_label])
    print(json.dumps({"stage": "built_position_context_training_set", "base_rows": int(len(base_router_label)), "augmented_rows": int(len(x_full)), "state_feature_count": int(len(state_feature_cols))}, ensure_ascii=False), flush=True)
    dqn_cfg = DuelingDQNConfig(input_dim=int(len(state_feature_cols)), hidden_dim=int(hidden_dim), action_dim=3, dropout=float(dropout), temperature=float(temperature))

    global_parent, global_dqn_meta = _fit_dqn_action_parent(
        x_full, y_full, dqn_cfg=dqn_cfg, steps=int(dqn_steps), batch_size=int(batch_size), gamma=float(gamma),
        lr=float(lr), bc_weight=float(bc_weight), seed=int(seed) + 900, device=device, log_name="global", log_every=int(log_every),
    )

    specialists: dict[str, dict[str, Any]] = {}
    specialist_meta: dict[str, Any] = {
        "global": {**global_dqn_meta, "fallback": False},
        "train_meta": train_meta,
        "position_context_cols": list(POSITION_CONTEXT_COLS),
        "base_feature_count": int(len(feature_cols)),
        "state_feature_count": int(len(state_feature_cols)),
        "router_train_counts": {REGIMES[i]: int((router_label == i).sum()) for i in range(len(REGIMES))},
        "fallback_regimes": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_parent, out_dir / "global_dqn_parent.pkl")
    for i, regime in enumerate(REGIMES):
        mask = router_label == i
        unique_actions = np.unique(np.asarray(y_full["action"])[mask]) if mask.any() else np.asarray([])
        if int(mask.sum()) < int(min_samples) or unique_actions.size < 2:
            specialists[regime] = copy.deepcopy(global_parent)
            specialist_meta[regime] = {"sample_count": int(mask.sum()), "fallback": True, "reason": "insufficient_samples_or_single_action_class", "action_distribution": _action_dist(np.asarray(y_full["action"])[mask]) if mask.any() else {}}
            specialist_meta["fallback_regimes"].append(regime)
            print(json.dumps({"stage": "specialist_fallback", "regime": regime, "rows": int(mask.sum())}, ensure_ascii=False), flush=True)
            continue
        x_reg = x_full.loc[mask].reset_index(drop=True)
        y_reg = _subset_y(y_full, mask)
        parent, dqn_meta = _fit_dqn_action_parent(
            x_reg, y_reg, dqn_cfg=dqn_cfg, steps=int(dqn_steps), batch_size=int(batch_size), gamma=float(gamma),
            lr=float(lr), bc_weight=float(bc_weight), seed=int(seed) + 100 + i, device=device, log_name=regime, log_every=int(log_every),
        )
        specialists[regime] = parent
        specialist_meta[regime] = {**dqn_meta, "fallback": False}
        joblib.dump(parent, out_dir / f"{regime}_dqn_parent.pkl")
        print(json.dumps({"stage": "specialist_trained", "regime": regime, "rows": int(mask.sum())}, ensure_ascii=False), flush=True)
    return specialists, specialist_meta


def _action_prob_frame(parent: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(parent["feature_cols"])
    x = frame.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
    if "side_hint" in x.columns:
        x["side_hint"] = 0.0
    proba = np.asarray(parent["action_model"].predict_proba(x), dtype=np.float64)
    classes = np.asarray(parent["action_model"].classes_, dtype=int)
    idx = {int(cls): i for i, cls in enumerate(classes)}
    out = pd.DataFrame(index=np.arange(len(frame)))
    out["action_prob_long"] = proba[:, idx[ACTION_LONG]]
    out["action_prob_short"] = proba[:, idx[ACTION_SHORT]]
    out["action_prob_cash"] = proba[:, idx[ACTION_CASH]]
    return out[ACTION_PROB_COLS].reset_index(drop=True)


def _predict_specialists(specialists: dict[str, dict[str, Any]], frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {regime: _action_prob_frame(parent, frame) for regime, parent in specialists.items()}


def _derive_action(prob: pd.DataFrame, *, side_threshold: float = 0.0) -> pd.DataFrame:
    out = prob[ACTION_PROB_COLS].copy()
    p_long = out["action_prob_long"].to_numpy(dtype=float)
    p_short = out["action_prob_short"].to_numpy(dtype=float)
    p_cash = out["action_prob_cash"].to_numpy(dtype=float)
    stacked = np.column_stack([p_cash, p_long, p_short])
    action = np.asarray([ACTION_CASH, ACTION_LONG, ACTION_SHORT], dtype=np.int64)[np.argmax(stacked, axis=1)]
    if side_threshold > 0.0:
        side_edge = p_long - p_short
        action = np.where(side_edge > float(side_threshold), ACTION_LONG, np.where(side_edge < -float(side_threshold), ACTION_SHORT, ACTION_CASH)).astype(np.int64)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    out["action"] = action
    out["side"] = side
    return out.reset_index(drop=True)


def _hard_route_decisions(specialist_dec: dict[str, pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    label = np.argmax(weights, axis=1)
    out = specialist_dec[REGIMES[0]].copy()
    for i, regime in enumerate(REGIMES):
        mask = label == i
        if mask.any():
            out.loc[mask, :] = specialist_dec[regime].loc[mask, :].to_numpy()
    return _derive_action(out.reset_index(drop=True))


def _soft_route_decisions(specialist_dec: dict[str, pd.DataFrame], weights: np.ndarray, *, side_threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(weights)))
    for col in ACTION_PROB_COLS:
        vals = np.column_stack([specialist_dec[r][col].to_numpy(dtype=float) for r in REGIMES])
        out[col] = np.sum(weights * vals, axis=1)
    total = np.clip(out[ACTION_PROB_COLS].sum(axis=1).to_numpy(dtype=float), 1e-12, None)
    out[ACTION_PROB_COLS] = out[ACTION_PROB_COLS].div(total, axis=0)
    return _derive_action(out, side_threshold=float(side_threshold))


def _decisions_for_mode(specialists: dict[str, dict[str, Any]], frame: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    weights = _router_matrix(frame)
    decs = _predict_specialists(specialists, frame)
    if mode == "hard_current":
        return _hard_route_decisions(decs, weights)
    if mode == "soft_current":
        return _soft_route_decisions(decs, weights, side_threshold=float(threshold))
    raise ValueError(f"unknown mode: {mode}")


def _state_row(df: pd.DataFrame, *, idx: int, pos: int, entry_price: float, entry_idx: int, close: np.ndarray, unit_exposure: float) -> pd.DataFrame:
    row = df.iloc[[idx]].copy()
    if pos == 0:
        bars = 0.0
        entry_dist = 0.0
        unreal = 0.0
    else:
        bars = float(max(0, int(idx) - int(entry_idx)))
        px = float(close[int(idx)])
        entry_dist = px / max(float(entry_price), 1e-12) - 1.0
        raw = entry_dist if int(pos) > 0 else -entry_dist
        unreal = raw * float(unit_exposure)
    row["position_current"] = float(pos)
    row["position_hold_duration_norm"] = float(np.clip(bars / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0))
    row["position_unrealized_pnl"] = float(unreal)
    row["position_entry_price_dist"] = float(entry_dist)
    row["position_bars_since_entry"] = float(bars)
    return row


def _backtest_action_only(df: pd.DataFrame, specialists: dict[str, dict[str, Any]], *, mode: str, side_threshold: float, fee: float, slip: float, unit_exposure: float) -> dict[str, Any]:
    close = _close(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    exposure = float(unit_exposure)

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * exposure
        return cash * (1.0 + unreal), unreal

    def enter(i: int, side: int) -> None:
        nonlocal pos, entry_price, entry_equity, entry_idx, cash, long_entries, short_entries, notional_sum, leverage_sum
        fill_i = min(i + 1, len(df) - 1)
        pos = int(side)
        entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        cash -= cash * fee * exposure
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += exposure
        leverage_sum += 1.0

    def exit_position(i: int, reason: str) -> None:
        nonlocal pos, entry_price, entry_equity, cash, trades, wins
        fill_i = min(i + 1, len(df) - 1)
        exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        pos = 0

    for i in range(0, len(df) - 2):
        eq, _ = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        row = _state_row(df, idx=i, pos=pos, entry_price=entry_price, entry_idx=entry_idx, close=close, unit_exposure=float(unit_exposure))
        dec = _decisions_for_mode(specialists, row, mode, side_threshold)
        desired = int(dec["side"].iloc[0])
        if desired > 0:
            action_counts["long"] += 1
        elif desired < 0:
            action_counts["short"] += 1
        else:
            action_counts["cash"] += 1

        if pos != 0 and desired != pos:
            exit_position(i, "action_cash" if desired == 0 else "action_flip")
        if pos == 0 and desired != 0:
            enter(i, desired)

    if pos != 0:
        exit_position(len(df) - 2, "end_of_data")
        eq = cash
    else:
        eq, _ = mark(len(df) - 1)
    peak = max(peak, eq)
    mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
    n = max(len(df), 1)
    return {
        "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / max(long_entries + short_entries, 1)),
        "action_counts": action_counts, "exits": exits,
    }


def _metrics_action_only(df: pd.DataFrame, specialists: dict[str, dict[str, Any]], *, mode: str, side_threshold: float, fee: float, slip: float, unit_exposure: float) -> dict[str, Any]:
    return {f"cost{mult}": _backtest_action_only(df, specialists, mode=mode, side_threshold=float(side_threshold), fee=fee * float(mult), slip=slip * float(mult), unit_exposure=float(unit_exposure)) for mult in (1, 2, 3)}


def _compact_costs(metrics: dict[str, Any]) -> dict[str, Any]:
    return {c: {k: metrics[c][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")} for c in ("cost1", "cost2", "cost3")}


import argparse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Alpha5.3 Regime3 (SOL) router with Dueling DQN + PER specialist parents.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=5317)
    p.add_argument("--min-samples", type=int, default=1200)
    p.add_argument("--dqn-steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.82)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--bc-weight", type=float, default=0.20)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.18)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--mode-filter", choices=["all", "hard", "soft0", "soft005", "soft010"], default="all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = _read(args.train_csv)
    eval_all = _read(args.eval_csv)
    regime3_audit = _verify_regime3_inputs(train_all, eval_all, DEFAULT_PREPROCESS_MANIFEST)

    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] <= TRAIN_END)].reset_index(drop=True)
    val_df = train_all[(train_all["timestamp"] >= SELECTION_START) & (train_all["timestamp"] <= SELECTION_END)].reset_index(drop=True)
    eval_canonical = eval_all[(eval_all["timestamp"] >= OOS_CANONICAL_START) & (eval_all["timestamp"] <= OOS_CANONICAL_END)].reset_index(drop=True)
    eval_fresh = eval_all[(eval_all["timestamp"] >= OOS_FRESH_START) & (eval_all["timestamp"] <= OOS_FRESH_END)].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0 or len(eval_canonical) == 0 or len(eval_fresh) == 0:
        raise ValueError(f"empty split: train={len(train_df)} val={len(val_df)} oos_canonical={len(eval_canonical)} oos_fresh={len(eval_fresh)}")

    feature_cols = _feature_cols(train_all, eval_all)
    if any(c in ROUTER_PROB_SET for c in feature_cols):
        raise ValueError("router probability columns leaked into specialist parent input")
    if any(c.startswith(FORBIDDEN_PREFIXES) for c in feature_cols):
        raise ValueError("forbidden Regime4 columns leaked into specialist parent input")

    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))

    specialists, specialist_meta = _train_specialists(
        train_df=train_df, feature_cols=feature_cols, label_cfg=label_cfg, stride=int(args.stride),
        min_samples=int(args.min_samples), dqn_steps=int(args.dqn_steps), batch_size=int(args.batch_size),
        gamma=float(args.gamma), lr=float(args.lr), bc_weight=float(args.bc_weight), hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout), temperature=float(args.temperature), log_every=int(args.log_every),
        seed=int(args.seed), device=device, out_dir=args.out_dir / "specialists",
    )

    mode_specs = [("hard_current", 0.0)] + [("soft_current", th) for th in (0.0, 0.05, 0.10)]
    if args.mode_filter == "hard":
        mode_specs = [("hard_current", 0.0)]
    elif args.mode_filter == "soft0":
        mode_specs = [("soft_current", 0.0)]
    elif args.mode_filter == "soft005":
        mode_specs = [("soft_current", 0.05)]
    elif args.mode_filter == "soft010":
        mode_specs = [("soft_current", 0.10)]

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for mode, th in mode_specs:
        name = mode if mode == "hard_current" else f"{mode}_th{th:.2f}"
        val_metrics = _metrics_action_only(val_df, specialists, mode=mode, side_threshold=float(th), fee=FEE, slip=SLIP, unit_exposure=float(args.unit_exposure))
        score = _score(val_metrics)
        result = {
            "name": name, "router_mode": mode, "side_threshold": float(th), "selection_score": float(score),
            "validation_metrics": val_metrics, "selection_metrics_compact": _compact_costs(val_metrics),
            "parent_output_contract": list(ACTION_PROB_COLS),
        }
        grid_rows.append({"candidate": name, "score": score, "val_cost1_pnl": val_metrics["cost1"]["pnl"], "val_cost1_mdd": val_metrics["cost1"]["mdd"], "val_cost2_pnl": val_metrics["cost2"]["pnl"], "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_trades": val_metrics["cost1"]["trades"]})
        experiments.append(result)
        print(json.dumps({"stage": "selection_mode_scored", "mode": name, "score": score, "val_metrics_compact": _compact_costs(val_metrics)}, ensure_ascii=False), flush=True)

    best = max(experiments, key=lambda r: float(r["selection_score"]))
    best_mode, best_th = best["router_mode"], best["side_threshold"]
    print(json.dumps({"stage": "best_selected", "name": best["name"]}, ensure_ascii=False), flush=True)

    oos_canonical_metrics = _metrics_action_only(eval_canonical, specialists, mode=best_mode, side_threshold=float(best_th), fee=FEE, slip=SLIP, unit_exposure=float(args.unit_exposure))
    oos_fresh_metrics = _metrics_action_only(eval_fresh, specialists, mode=best_mode, side_threshold=float(best_th), fee=FEE, slip=SLIP, unit_exposure=float(args.unit_exposure))

    report = {
        "model_id": MODEL_ID,
        "asset": "SOL",
        "design": (
            "Alpha5.3 architecture ported from HMM Regime4 to SOL Regime3 routing "
            "(no SOL Regime4 artifact exists; Regime4 is deprecated project-wide). "
            "Three hard-split regime specialist parents (bull/bear/chop - no whipsaw "
            "class in Regime3) output only action_prob_long/action_prob_short/"
            "action_prob_cash from Dueling DQN action heads trained with PER-like "
            "prioritized replay. TP/SL, max-hold, cooldown, quality, notional, "
            "leverage, and bucket heads are not specialist parent outputs. Evaluation "
            "is action-only: positions are closed when the routed DQN action becomes "
            "cash or flips side."
        ),
        "train_csv": str(args.train_csv), "eval_csv": str(args.eval_csv), "device": str(device),
        "regime3_feature_audit": regime3_audit,
        "feature_contract": {"feature_count": int(len(feature_cols)), "router_prob_cols": ROUTER_COLS, "feature_cols": feature_cols},
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos_canonical": [str(eval_canonical["timestamp"].iloc[0]), str(eval_canonical["timestamp"].iloc[-1])],
            "oos_fresh_forward": [str(eval_fresh["timestamp"].iloc[0]), str(eval_fresh["timestamp"].iloc[-1])],
        },
        "specialists": specialist_meta,
        "parent_output_contract": list(ACTION_PROB_COLS),
        "action_only_evaluator": {
            "unit_exposure": float(args.unit_exposure),
            "entry_policy": "enter long/short when routed action is long/short while flat",
            "exit_policy": "exit when routed action is cash or opposite side",
            "flip_policy": "close current position and enter opposite side on the same signal bar next-fill",
            "removed_constants": ["take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score"],
        },
        "experiments": experiments,
        "best_by_selection": best["name"],
        "oos_canonical_metrics": _compact_costs(oos_canonical_metrics),
        "oos_fresh_forward_metrics": _compact_costs(oos_fresh_metrics),
        "audit": {
            "status": "pass", "selection_uses_2026": False, "hmm_router_only": True, "regime4_used": False,
            "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        },
        "artifacts": {"report": str(args.out_dir / "alpha5_3_regime3_dqn_router_sol_summary.json"), "grid": str(args.out_dir / "alpha5_3_regime3_dqn_router_sol_grid.csv"), "specialists": str(args.out_dir / "specialists")},
    }
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(args.out_dir / "alpha5_3_regime3_dqn_router_sol_grid.csv", index=False)
    (args.out_dir / "alpha5_3_regime3_dqn_router_sol_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": report["artifacts"]["report"], "best": best["name"], "oos_canonical": report["oos_canonical_metrics"]["cost1"], "oos_fresh": report["oos_fresh_forward_metrics"]["cost1"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
