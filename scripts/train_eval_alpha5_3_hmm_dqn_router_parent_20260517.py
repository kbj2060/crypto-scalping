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
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_alpha4_3_dueling_dqn_parent_20260517 import (  # noqa: E402
    _action_dist,
    _prep_matrix,
    _seed,
)
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    BASE_PARENT,
    OLD_CLEAN_PREFIX,
    _compact_costs,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_3_state24_sticky090_hmm_dqn_router_parent_20260517"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_3_state24_sticky090_hmm_dqn_router_parent_20260517"
DEFAULT_TRAIN = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_PREPROCESS_MANIFEST = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_manifest.json"
DEFAULT_CLEAN4_REPORT = ROOT / "data/ensemble/reports/clean_regime4_state24_sticky090_v2_20260517_report.json"
EXPECTED_CLEAN4_MODEL = "clean_regime4_state24_sticky090_v2_2024.joblib"
REGIMES = ("bull", "bear", "chop", "whipsaw")
CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
ROUTER_COLS = [f"{CLEAN4_PREFIX}{r}_prob" for r in REGIMES]
ROUTER_PROB_SET = set(ROUTER_COLS)
FORBIDDEN_PREFIXES = (OLD_CLEAN_PREFIX, "regime4_pred_")
FORBIDDEN_EXACT = {
    "clean_regime4_2024_unsup_v1_cluster",
    "clean_regime4_2024_unsup_v1_state_code",
}
ACTION_PROB_COLS = ["action_prob_long", "action_prob_short", "action_prob_cash"]
POSITION_CONTEXT_COLS = [
    "position_current",
    "position_hold_duration_norm",
    "position_unrealized_pnl",
    "position_entry_price_dist",
    "position_bars_since_entry",
]
POSITION_CONTEXT_NORM_BARS = 288.0


def _valid_indices(n: int, max_horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, max(0, int(n) - int(max_horizon) - 1), max(1, int(stride)), dtype=np.int64)


def _router_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in ROUTER_COLS if c not in frame.columns]
    if missing:
        raise ValueError("missing HMM Regime4 router columns: " + ", ".join(missing))
    p = frame[ROUTER_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    common = set(train.columns) & set(eval_df.columns)
    cols = [
        c
        for c in list(parent_ref["feature_cols"])
        if not c.startswith(FORBIDDEN_PREFIXES)
        and c not in ROUTER_PROB_SET
        and c not in FORBIDDEN_EXACT
    ]
    if "tp_sl_action_score" in common and "tp_sl_action_score" not in cols:
        cols.append("tp_sl_action_score")
    for col in sorted(c for c in common if c.startswith(CLEAN4_PREFIX)):
        if col in ROUTER_PROB_SET or col in FORBIDDEN_EXACT:
            continue
        if "cluster" in col or "state_code" in col:
            continue
        if col not in cols:
            cols.append(col)
    return cols


def _verify_state24_sticky090_inputs(train: pd.DataFrame, eval_df: pd.DataFrame, manifest_path: Path, report_path: Path) -> dict[str, Any]:
    train_cols = set(train.columns)
    eval_cols = set(eval_df.columns)
    current_cols = [c for c in sorted(train_cols & eval_cols) if c.startswith(CLEAN4_PREFIX)]
    router_missing = [c for c in ROUTER_COLS if c not in train_cols or c not in eval_cols]
    old_legacy = [c for c in sorted(train_cols | eval_cols) if c.startswith(OLD_CLEAN_PREFIX)]
    future_cols = [c for c in sorted(train_cols & eval_cols) if c.startswith("regime4_pred_")]
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_text = json.dumps(manifest, ensure_ascii=False)
    report_text = json.dumps(report, ensure_ascii=False)
    model_ok = EXPECTED_CLEAN4_MODEL in manifest_text or EXPECTED_CLEAN4_MODEL in report_text
    source_ok = "clean_regime4_state24_sticky090_v2_20260517" in manifest_text
    audit = {
        "manifest": str(manifest_path),
        "manifest_exists": bool(manifest_path.exists()),
        "report": str(report_path),
        "report_exists": bool(report_path.exists()),
        "expected_model": EXPECTED_CLEAN4_MODEL,
        "expected_model_found_in_manifest": bool(model_ok),
        "report_model_path": report.get("model_path"),
        "report_model_id": report.get("model_id"),
        "report_states": report.get("states"),
        "report_sticky": report.get("sticky"),
        "state24_source_found_in_manifest": bool(source_ok),
        "clean4_prefix": CLEAN4_PREFIX,
        "clean4_common_count": int(len(current_cols)),
        "router_missing": router_missing,
        "legacy_v4_count": int(len(old_legacy)),
        "future_regime4_common_count": int(len(future_cols)),
        "clean4_common_cols": current_cols,
    }
    print(json.dumps({"stage": "state24_sticky090_feature_audit", **audit}, ensure_ascii=False), flush=True)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing preprocessing manifest: {manifest_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"missing clean regime report: {report_path}")
    if not model_ok or not source_ok:
        raise ValueError(f"state24 sticky090 v2 model not verified in manifest: {manifest_path}")
    if router_missing:
        raise ValueError("missing router columns: " + ",".join(router_missing))
    if old_legacy:
        raise ValueError("legacy clean_regime_2024_unsup_v4 columns present")
    return audit


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
    flat = _with_position_context(
        x,
        current_position=0.0,
        hold_duration_norm=0.0,
        unrealized_pnl=0.0,
        entry_price_dist=0.0,
        bars_since_entry=0.0,
    )
    long = _with_position_context(
        x,
        current_position=1.0,
        hold_duration_norm=np.clip(bars_f / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0),
        unrealized_pnl=raw_dist,
        entry_price_dist=raw_dist,
        bars_since_entry=bars_f,
    )
    short = _with_position_context(
        x,
        current_position=-1.0,
        hold_duration_norm=np.clip(bars_f / POSITION_CONTEXT_NORM_BARS, 0.0, 1.0),
        unrealized_pnl=-raw_dist,
        entry_price_dist=raw_dist,
        bars_since_entry=bars_f,
    )
    x_aug = pd.concat([flat, long, short], ignore_index=True)
    y_aug = {k: np.concatenate([np.asarray(v), np.asarray(v), np.asarray(v)]) for k, v in y.items()}
    return x_aug, y_aug


def _train_dqn_td_priority_only(
    x: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    cfg: DuelingDQNConfig,
    steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    bc_weight: float,
    seed: int,
    device: torch.device,
    log_name: str,
    log_every: int,
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
    print(
        json.dumps(
            {
                "stage": "dqn_train_start",
                "name": log_name,
                "rows": int(n),
                "steps": int(steps),
                "batch_size": int(batch_size),
                "gamma": float(gamma),
                "bc_weight": float(bc_weight),
                "action_counts": _action_dist(actions),
                "priority_policy": "td_error_only_no_trade_boost",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
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
            print(
                json.dumps(
                    {
                        "stage": "dqn_train_progress",
                        "name": log_name,
                        "step": int(step),
                        "steps": int(steps),
                        "pct": round(100.0 * float(step) / max(float(steps), 1.0), 2),
                        "loss": float(loss.detach().cpu()),
                        "td_abs_mean": float(torch.mean(torch.abs(td)).detach().cpu()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    target.load_state_dict(model.state_dict())
    meta = {
        "reward_center": reward_center,
        "reward_std": reward_std,
        "loss_tail": losses[-10:],
        "td_abs_tail": td_means[-10:],
        "steps": int(steps),
        "batch_size": int(batch_size),
        "gamma": float(gamma),
        "bc_weight": float(bc_weight),
        "priority_policy": "td_error_only_no_trade_boost",
        "class_weight": class_weight.detach().cpu().numpy().tolist(),
    }
    return model, meta


def _fit_dqn_action_parent(
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    *,
    dqn_cfg: DuelingDQNConfig,
    steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    bc_weight: float,
    seed: int,
    device: torch.device,
    log_name: str,
    log_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_np, med, mean, std = _prep_matrix(x)
    model, train_meta = _train_dqn_td_priority_only(
        x_np,
        np.asarray(y["action"], dtype=np.int64),
        np.asarray(y["quality"], dtype=np.float32),
        cfg=dqn_cfg,
        steps=int(steps),
        batch_size=int(batch_size),
        gamma=float(gamma),
        lr=float(lr),
        bc_weight=float(bc_weight),
        seed=int(seed),
        device=device,
        log_name=log_name,
        log_every=int(log_every),
    )
    out = {
        "model_type": "alpha5_3_hmm_router_specialist_dueling_dqn_action_parent",
        "output_contract": list(ACTION_PROB_COLS),
        "feature_cols": list(x.columns),
        "config": {},
    }
    out["action_model"] = make_action_model(model, config=dqn_cfg, medians=med, mean=mean, std=std, feature_cols=list(x.columns))
    meta = {
        "dqn_train_meta": train_meta,
        "action_distribution": _action_dist(np.asarray(y["action"], dtype=np.int64)),
        "sample_count": int(len(x)),
    }
    return out, meta


def _train_specialists(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_cfg: FullyLearnedGovernorConfig,
    stride: int,
    min_samples: int,
    dqn_steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    bc_weight: float,
    hidden_dim: int,
    dropout: float,
    temperature: float,
    log_every: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    x_full, y_full, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=int(stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    valid = _valid_indices(len(train_df), int(label_cfg.max_train_horizon_bars), int(stride))
    base_router_label = np.argmax(_router_matrix(train_df)[valid], axis=1)
    x_full, y_full = _augment_position_context_training(x_full, y_full, frame=train_df, valid_idx=valid)
    state_feature_cols = list(feature_cols) + list(POSITION_CONTEXT_COLS)
    router_label = np.concatenate([base_router_label, base_router_label, base_router_label])
    print(
        json.dumps(
            {
                "stage": "built_position_context_training_set",
                "base_rows": int(len(base_router_label)),
                "augmented_rows": int(len(x_full)),
                "state_feature_count": int(len(state_feature_cols)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dqn_cfg = DuelingDQNConfig(
        input_dim=int(len(state_feature_cols)),
        hidden_dim=int(hidden_dim),
        action_dim=3,
        dropout=float(dropout),
        temperature=float(temperature),
    )

    global_parent, global_dqn_meta = _fit_dqn_action_parent(
        x_full,
        y_full,
        dqn_cfg=dqn_cfg,
        steps=int(dqn_steps),
        batch_size=int(batch_size),
        gamma=float(gamma),
        lr=float(lr),
        bc_weight=float(bc_weight),
        seed=int(seed) + 900,
        device=device,
        log_name="global",
        log_every=int(log_every),
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
            specialist_meta[regime] = {
                "sample_count": int(mask.sum()),
                "fallback": True,
                "reason": "insufficient_samples_or_single_action_class",
                "action_distribution": _action_dist(np.asarray(y_full["action"])[mask]) if mask.any() else {},
            }
            specialist_meta["fallback_regimes"].append(regime)
            print(json.dumps({"stage": "specialist_fallback", "regime": regime, "rows": int(mask.sum())}, ensure_ascii=False), flush=True)
            continue
        x_reg = x_full.loc[mask].reset_index(drop=True)
        y_reg = _subset_y(y_full, mask)
        parent, dqn_meta = _fit_dqn_action_parent(
            x_reg,
            y_reg,
            dqn_cfg=dqn_cfg,
            steps=int(dqn_steps),
            batch_size=int(batch_size),
            gamma=float(gamma),
            lr=float(lr),
            bc_weight=float(bc_weight),
            seed=int(seed) + 100 + i,
            device=device,
            log_name=regime,
            log_every=int(log_every),
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


def _state_row(
    df: pd.DataFrame,
    *,
    idx: int,
    pos: int,
    entry_price: float,
    entry_idx: int,
    close: np.ndarray,
    unit_exposure: float,
) -> pd.DataFrame:
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


def _backtest_action_only(
    df: pd.DataFrame,
    specialists: dict[str, dict[str, Any]],
    *,
    mode: str,
    side_threshold: float,
    fee: float,
    slip: float,
    unit_exposure: float,
) -> dict[str, Any]:
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
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / max(long_entries + short_entries, 1)),
        "action_counts": action_counts,
        "exits": exits,
    }


def _metrics_action_only(
    df: pd.DataFrame,
    specialists: dict[str, dict[str, Any]],
    *,
    mode: str,
    side_threshold: float,
    fee: float,
    slip: float,
    unit_exposure: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": _backtest_action_only(
            df,
            specialists,
            mode=mode,
            side_threshold=float(side_threshold),
            fee=fee * float(mult),
            slip=slip * float(mult),
            unit_exposure=float(unit_exposure),
        )
        for mult in (1, 2, 3)
    }


def _evaluate_action_only_decisions(
    *,
    name: str,
    mode: str,
    side_threshold: float,
    specialists: dict[str, dict[str, Any]],
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    fee: float,
    slip: float,
    unit_exposure: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    val_metrics = _metrics_action_only(
        val_df,
        specialists,
        mode=mode,
        side_threshold=float(side_threshold),
        fee=fee,
        slip=slip,
        unit_exposure=float(unit_exposure),
    )
    eval_metrics = _metrics_action_only(
        eval_df,
        specialists,
        mode=mode,
        side_threshold=float(side_threshold),
        fee=fee,
        slip=slip,
        unit_exposure=float(unit_exposure),
    )
    score = _score(val_metrics)
    rows = [
        {
            "candidate": name,
            "runner_config": "action_only_no_runner",
            "score": score,
            "val_cost1_pnl": val_metrics["cost1"]["pnl"],
            "val_cost1_mdd": val_metrics["cost1"]["mdd"],
            "val_cost2_pnl": val_metrics["cost2"]["pnl"],
            "val_cost3_pnl": val_metrics["cost3"]["pnl"],
            "val_trades": val_metrics["cost1"]["trades"],
        }
    ]
    return {
        "name": name,
        "selection_score": float(score),
        "validation_metrics": val_metrics,
        "metrics": eval_metrics,
        "selected_runner_config": {"name": "action_only_no_runner"},
        "artifacts": {},
    }, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Alpha5.3 HMM Regime4 router with Dueling DQN + PER specialist parents.")
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
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--max-val-rows", type=int, default=0)
    p.add_argument("--max-eval-rows", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    state24_audit = _verify_state24_sticky090_inputs(train_all, eval_df, DEFAULT_PREPROCESS_MANIFEST, DEFAULT_CLEAN4_REPORT)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    if int(args.max_train_rows) > 0:
        train_df = train_df.tail(int(args.max_train_rows)).reset_index(drop=True)
    if int(args.max_val_rows) > 0:
        val_df = val_df.tail(int(args.max_val_rows)).reset_index(drop=True)
    if int(args.max_eval_rows) > 0:
        eval_df = eval_df.head(int(args.max_eval_rows)).reset_index(drop=True)

    feature_cols = _feature_cols(train_all, eval_df)
    old_leaks = [c for c in feature_cols if c.startswith(OLD_CLEAN_PREFIX)]
    if old_leaks:
        raise ValueError("legacy clean_regime_2024_unsup_v4 leaked: " + ",".join(old_leaks[:20]))
    if any(c.startswith("regime4_pred_") for c in feature_cols):
        raise ValueError("future TFT regime4_pred leaked into HMM-only specialist input")
    if any(c in ROUTER_PROB_SET for c in feature_cols):
        raise ValueError("router probability columns leaked into specialist parent input")

    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)

    specialists, specialist_meta = _train_specialists(
        train_df=train_df,
        feature_cols=feature_cols,
        label_cfg=label_cfg,
        stride=int(args.stride),
        min_samples=int(args.min_samples),
        dqn_steps=int(args.dqn_steps),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        lr=float(args.lr),
        bc_weight=float(args.bc_weight),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        temperature=float(args.temperature),
        log_every=int(args.log_every),
        seed=int(args.seed),
        device=device,
        out_dir=args.out_dir / "specialists",
    )

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    mode_specs = [("hard_current", 0.0)] + [("soft_current", th) for th in (0.0, 0.05, 0.10)]
    if args.mode_filter == "hard":
        mode_specs = [("hard_current", 0.0)]
    elif args.mode_filter == "soft0":
        mode_specs = [("soft_current", 0.0)]
    elif args.mode_filter == "soft005":
        mode_specs = [("soft_current", 0.05)]
    elif args.mode_filter == "soft010":
        mode_specs = [("soft_current", 0.10)]
    for mode, th in mode_specs:
        name = mode if mode == "hard_current" else f"{mode}_th{th:.2f}"
        result, rows = _evaluate_action_only_decisions(
            name=name,
            mode=mode,
            side_threshold=float(th),
            specialists=specialists,
            val_df=val_df,
            eval_df=eval_df,
            fee=fee,
            slip=slip,
            unit_exposure=float(args.unit_exposure),
        )
        result["router_mode"] = mode
        result["side_threshold"] = float(th)
        result["parent_output_contract"] = list(ACTION_PROB_COLS)
        result["action_only_evaluator"] = {"unit_exposure": float(args.unit_exposure), "exit_policy": "cash_or_flip_action"}
        result["selected_metrics_compact"] = _compact_costs(result["metrics"])
        experiments.append(result)
        grid_rows.extend(rows)
        print(json.dumps({"mode": name, "best": result["name"], "metrics": result["selected_metrics_compact"]}, ensure_ascii=False, default=_json_default), flush=True)

    best = max(experiments, key=lambda r: float(r["selection_score"]))
    report = {
        "model_id": MODEL_ID,
        "design": "HMM Regime4 is promoted from parent input to router state. Four hard-split regime specialist parents output only action_prob_long/action_prob_short/action_prob_cash from Dueling DQN action heads trained with PER-like prioritized replay. TP/SL, max-hold, cooldown, quality, notional, leverage, and bucket heads are not specialist parent outputs. Evaluation is action-only: positions are closed when the routed DQN action becomes cash or flips side. No legacy clean_regime_2024_unsup_v4_* and no TFT future regime features are used.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "device": str(device),
        "state24_sticky090_feature_audit": state24_audit,
        "feature_contract": {
            "feature_count": int(len(feature_cols)),
            "legacy_clean_v4_count": int(sum(c.startswith(OLD_CLEAN_PREFIX) for c in feature_cols)),
            "router_prob_cols": ROUTER_COLS,
            "router_prob_in_specialist_input_count": int(sum(c in ROUTER_PROB_SET for c in feature_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
            "current_regime4_aux_count": int(sum(c.startswith(CLEAN4_PREFIX) for c in feature_cols)),
            "contains_tp_sl_action_score": "tp_sl_action_score" in feature_cols,
            "feature_cols": feature_cols,
        },
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
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
        "selected_metrics": _compact_costs(best["metrics"]),
        "audit": {
            "status": "pass",
            "selection_uses_2026": False,
            "teacher_layer": "disabled",
            "deep_scout": "disabled",
            "legacy_clean_v4_allowed": False,
            "normal_class": "disabled",
            "hmm_router_only": True,
        },
        "artifacts": {
            "report": str(args.out_dir / "alpha5_3_hmm_dqn_router_parent_summary.json"),
            "grid": str(args.out_dir / "alpha5_3_hmm_dqn_router_parent_grid.csv"),
            "specialists": str(args.out_dir / "specialists"),
        },
    }
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(args.out_dir / "alpha5_3_hmm_dqn_router_parent_grid.csv", index=False)
    (args.out_dir / "alpha5_3_hmm_dqn_router_parent_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": report["artifacts"]["report"], "best": best["name"], "metrics": report["selected_metrics"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
