#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.train_alpha6_dsac_ensemble_router_20260523 import (  # noqa: E402
    MODEL_SPECS,
    DiscreteSAC,
    Replay,
    RouterData,
    _load_router_data,
    _load_router_data_oof,
)


@dataclass(frozen=True)
class RiskTemplate:
    name: str
    notional: float
    tp: float
    sl: float
    max_hold: int


RISK_TEMPLATES = [
    RiskTemplate("micro_tight", 0.15, 0.0035, 0.0025, 6),
    RiskTemplate("scalp_balanced", 0.25, 0.0060, 0.0040, 12),
    RiskTemplate("day_balanced", 0.35, 0.0100, 0.0065, 24),
    RiskTemplate("trend_wide", 0.45, 0.0180, 0.0100, 48),
    RiskTemplate("conviction_runner", 0.60, 0.0300, 0.0150, 72),
]


def _cost(notional: float, fee: float, slip: float) -> float:
    return 2.0 * (float(fee) + float(slip)) * float(notional)


def _future_side_quality(
    frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    min_edge: float,
    horizons: tuple[int, ...] = (6, 12, 24, 48, 96),
) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    n = len(frame)
    action = np.zeros(n, dtype=np.int64)
    quality = np.zeros(n, dtype=np.float64)
    for i in range(n - max(horizons) - 2):
        entry = max(float(close[i + 1]), 1e-12)
        best_side = 0
        best_score = 0.0
        for h in horizons:
            path = close[i + 1 : i + h + 2] / entry - 1.0
            for side, cls in ((1.0, 1), (-1.0, 2)):
                signed = path * side
                terminal = float(signed[-1]) * 0.25
                mfe = max(0.0, float(np.max(signed))) * 0.25
                mae = max(0.0, -float(np.min(signed))) * 0.25
                vol = float(np.nanstd(signed)) * 0.25
                score = terminal + 0.20 * mfe - 0.95 * mae - 0.03 * vol - _cost(0.25, fee, slip) - 0.00035 * (h / 96.0)
                if score > best_score:
                    best_score = score
                    best_side = cls
        action[i] = best_side if best_score >= float(min_edge) else 0
        quality[i] = best_score
    return action, quality


def _path_score(
    close: np.ndarray,
    idx: int,
    *,
    side_cls: int,
    horizon: int,
    fee: float,
    slip: float,
    notional: float = 0.25,
) -> float:
    if side_cls == 0 or idx + int(horizon) + 1 >= len(close):
        return 0.0
    side = 1.0 if int(side_cls) == 1 else -1.0
    entry = max(float(close[idx + 1]), 1e-12)
    path = (close[idx + 1 : idx + int(horizon) + 2] / entry - 1.0) * side
    terminal = float(path[-1]) * notional
    mfe = max(0.0, float(np.max(path))) * notional
    mae = max(0.0, -float(np.min(path))) * notional
    vol = float(np.nanstd(path)) * notional
    return terminal + 0.20 * mfe - 0.95 * mae - 0.03 * vol - _cost(notional, fee, slip) - 0.00035 * (int(horizon) / 96.0)


def _candidate_side_quality(data: RouterData, *, fee: float, slip: float, min_edge: float) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(data.frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    n = len(data.frame)
    labels = np.zeros(n, dtype=np.int64)
    quality = np.zeros(n, dtype=np.float64)
    actions = np.vstack([p["action"].to_numpy(dtype=np.int64) for p in data.preds]).T
    qualities = np.vstack([p["quality"].to_numpy(dtype=np.float64) for p in data.preds]).T
    horizons = np.vstack([p["target_horizon"].to_numpy(dtype=np.float64) for p in data.preds]).T
    active = (actions != 0) & (qualities >= data.thresholds.reshape(1, -1))
    for i in range(n - 98):
        if not active[i].any():
            continue
        if active[i, 0]:
            candidate = int(actions[i, 0])
        else:
            long_q = float(np.where((actions[i] == 1) & active[i], qualities[i], 0.0).sum())
            short_q = float(np.where((actions[i] == 2) & active[i], qualities[i], 0.0).sum())
            if max(long_q, short_q) <= 0.0:
                continue
            candidate = 1 if long_q >= short_q else 2
        h_mask = active[i] & (actions[i] == candidate) & (horizons[i] > 0)
        horizon = int(np.clip(np.nanmedian(horizons[i, h_mask]) if h_mask.any() else 12, 2, 96))
        score = _path_score(close, i, side_cls=candidate, horizon=horizon, fee=fee, slip=slip)
        quality[i] = score
        labels[i] = candidate if score >= float(min_edge) else 0
    return labels, quality


def _active_candidate_mask(data: RouterData) -> np.ndarray:
    actions = np.vstack([p["action"].to_numpy(dtype=np.int64) for p in data.preds]).T
    qualities = np.vstack([p["quality"].to_numpy(dtype=np.float64) for p in data.preds]).T
    return ((actions != 0) & (qualities >= data.thresholds.reshape(1, -1))).any(axis=1)


def _meta_state(data: RouterData, action_prob: np.ndarray, quality: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, list[str]]:
    onehot = np.zeros((len(action), 3), dtype=np.float32)
    onehot[np.arange(len(action)), np.clip(action.astype(int), 0, 2)] = 1.0
    q = np.asarray(quality, dtype=np.float32).reshape(-1, 1)
    x = np.hstack([data.base_x.astype(np.float32), action_prob.astype(np.float32), q, onehot]).astype(np.float32)
    names = list(data.base_names) + ["meta_cash_prob", "meta_long_prob", "meta_short_prob", "meta_quality"] + [
        "meta_action_cash",
        "meta_action_long",
        "meta_action_short",
    ]
    return x, names


def _fit_meta_models(
    x: np.ndarray,
    y_action: np.ndarray,
    y_quality: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, Any]:
    action_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=180,
        learning_rate=0.035,
        max_depth=3,
        num_leaves=7,
        min_child_samples=120,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=4.0,
        reg_lambda=12.0,
        random_state=seed,
        verbosity=-1,
    )
    quality_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=160,
        learning_rate=0.035,
        max_depth=2,
        num_leaves=4,
        min_child_samples=150,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=6.0,
        reg_lambda=16.0,
        random_state=seed + 17,
        verbosity=-1,
    )
    action_model.fit(x[train_idx], y_action[train_idx])
    quality_model.fit(x[train_idx], y_quality[train_idx])
    return action_model, quality_model


def _predict_action_proba(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for j, cls in enumerate(np.asarray(model.classes_, dtype=int)):
        if 0 <= cls <= 2:
            out[:, cls] = proba[:, j]
    return out


class RiskBanditEnv:
    def __init__(
        self,
        state_x: np.ndarray,
        frame: pd.DataFrame,
        indices: np.ndarray,
        meta_action: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        fee: float,
        slip: float,
    ) -> None:
        self.x = state_x
        self.frame = frame
        self.indices = indices.astype(np.int64)
        self.meta_action = meta_action.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
        self.fee = float(fee)
        self.slip = float(slip)
        self.close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
        self.high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
        self.low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.x.shape[1])

    def reset(self) -> np.ndarray:
        self.ptr = 0
        return self._state()

    def _state(self) -> np.ndarray:
        i = int(self.indices[min(self.ptr, len(self.indices) - 1)])
        return ((self.x[i] - self.mean) / self.std).astype(np.float32)

    def _reward_for(self, idx: int, action_id: int) -> float:
        side_cls = int(self.meta_action[idx])
        if side_cls == 0:
            return 0.0
        tpl = RISK_TEMPLATES[int(action_id)]
        side = 1 if side_cls == 1 else -1
        entry_i = min(idx + 1, len(self.close) - 1)
        entry = max(float(self.close[entry_i]), 1e-12)
        end = min(entry_i + int(tpl.max_hold), len(self.close) - 1)
        exit_ret = 0.0
        for j in range(entry_i + 1, end + 1):
            if side > 0:
                hi = float(self.high[j] / entry - 1.0)
                lo = float(self.low[j] / entry - 1.0)
            else:
                hi = float(entry / max(self.low[j], 1e-12) - 1.0)
                lo = float(entry / max(self.high[j], 1e-12) - 1.0)
            if lo <= -tpl.sl:
                exit_ret = -tpl.sl
                break
            if hi >= tpl.tp:
                exit_ret = tpl.tp
                break
        else:
            px = max(float(self.close[end]), 1e-12)
            exit_ret = (px - entry) / entry if side > 0 else (entry - px) / entry
        return float(exit_ret * tpl.notional - _cost(tpl.notional, self.fee, self.slip) - 0.00005 * (tpl.max_hold / 12.0))

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        idx = int(self.indices[self.ptr])
        reward = self._reward_for(idx, int(action_id))
        self.ptr += 1
        done = self.ptr >= len(self.indices)
        return self._state() if not done else np.zeros(self.state_dim, dtype=np.float32), reward, done, {"idx": idx}


def _train_risk_dsac(
    state_x: np.ndarray,
    frame: pd.DataFrame,
    train_candidates: np.ndarray,
    meta_action: np.ndarray,
    *,
    fee: float,
    slip: float,
    seed: int,
    device: str,
    episodes: int,
    warmup: int,
    batch_size: int,
) -> tuple[DiscreteSAC, dict[str, Any], np.ndarray, np.ndarray]:
    mean = state_x[train_candidates].mean(axis=0)
    std = state_x[train_candidates].std(axis=0)
    env = RiskBanditEnv(state_x, frame, train_candidates, meta_action, mean, std, fee=fee, slip=slip)
    agent = DiscreteSAC(env.state_dim, len(RISK_TEMPLATES), device, gamma=0.0, alpha_init=0.05, alpha_min=0.005, alpha_max=0.30)
    replay = Replay(capacity=100_000)
    step = 0
    last: dict[str, Any] = {}
    for ep in range(int(episodes)):
        s = env.reset()
        while True:
            if step < int(warmup):
                a = int(np.random.randint(len(RISK_TEMPLATES)))
            else:
                a = agent.act(s, deterministic=False)
            ns, r, done, _ = env.step(a)
            replay.add(s, a, r, ns, done)
            s = ns
            step += 1
            if step >= int(warmup):
                last = agent.update(replay, int(batch_size)) or last
            if done:
                break
        print(f"[risk-dsac] episode={ep+1}/{episodes} candidates={len(train_candidates)} last={last}", flush=True)
    return agent, last, mean, std


def _risk_action(agent: DiscreteSAC, state_x: np.ndarray, idx: int, mean: np.ndarray, std: np.ndarray) -> int:
    s = ((state_x[idx] - mean) / np.where(std <= 1e-6, 1.0, std)).astype(np.float32)
    return int(agent.act(s, deterministic=True))


def _backtest(
    data: RouterData,
    state_x: np.ndarray,
    meta_action: np.ndarray,
    meta_quality: np.ndarray,
    agent: DiscreteSAC,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    start: int,
    end: int,
    threshold: float,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    frame = data.frame
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_idx = -1
    entry_equity = 1.0
    tpl = RISK_TEMPLATES[1]
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}
    template_counts: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        raw = (close[i] - entry) / max(entry, 1e-12) if side > 0 else (entry - close[i]) / max(entry, 1e-12)
        return cash * (1.0 + raw * tpl.notional)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal cash, side, entry, entry_idx, trades, wins, tpl
        fill_i = min(i + 1, len(frame) - 1)
        px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * tpl.notional)
        cash -= before * fee * tpl.notional
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        entry_idx = -1

    for i in range(max(0, start), min(end, len(frame) - 2)):
        if side != 0:
            hold = i - entry_idx
            if side > 0:
                adverse = float(low[i] / max(entry, 1e-12) - 1.0)
                favorable = float(high[i] / max(entry, 1e-12) - 1.0)
            else:
                adverse = float(entry / max(high[i], 1e-12) - 1.0)
                favorable = float(entry / max(low[i], 1e-12) - 1.0)
            if adverse <= -tpl.sl:
                exit_pos(i, "sl")
            elif favorable >= tpl.tp:
                exit_pos(i, "tp")
            elif hold >= tpl.max_hold:
                exit_pos(i, "max_hold")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and int(meta_action[i]) != 0 and float(meta_quality[i]) >= float(threshold):
            rid = _risk_action(agent, state_x, i, mean, std)
            tpl = RISK_TEMPLATES[rid]
            side = 1 if int(meta_action[i]) == 1 else -1
            fill_i = min(i + 1, len(frame) - 1)
            entry = _fill_price(frame, fill_i, side, slip, entry=True)
            entry_idx = i
            entry_equity = cash
            cash -= cash * fee * tpl.notional
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            template_counts[tpl.name] = template_counts.get(tpl.name, 0) + 1
    if side != 0:
        exit_pos(min(end, len(frame) - 2), "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(frame.iloc[start : end + 1])),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
        "risk_templates": template_counts,
    }


def _validation_splits(idx: np.ndarray, purge: int) -> dict[str, tuple[int, int]]:
    a = int(idx.min())
    b = int(idx.max())
    n = b - a + 1
    calib_end = a + n // 2
    return {
        "calib": (a, max(a, calib_end - purge)),
        "test": (calib_end, b - 2),
        "full_val": (a, b - 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 shallow LGBM meta action/quality + DSAC risk head.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_lgbm_meta_dsac_risk_20260524")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--oof-folds", type=int, default=2)
    ap.add_argument("--oof-iterations", type=int, default=120)
    ap.add_argument("--oof-exit-iterations", type=int, default=40)
    ap.add_argument("--oof-purge-bars", type=int, default=96)
    ap.add_argument("--purge-bars", type=int, default=96)
    ap.add_argument("--label-min-edge", type=float, default=0.0010)
    ap.add_argument("--meta-label-source", choices=("candidate", "oracle"), default="candidate")
    ap.add_argument("--enforce-expert-candidate-gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--meta-active-prob-min", type=float, default=0.0)
    ap.add_argument("--meta-margin-min", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--risk-episodes", type=int, default=3)
    ap.add_argument("--risk-warmup", type=int, default=1500)
    ap.add_argument("--risk-batch-size", type=int, default=256)
    ap.add_argument("--max-risk-train-candidates", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.0004 * float(args.cost_mult)
    slip = 0.00015 * float(args.cost_mult)

    if int(args.oof_folds) > 1:
        data = _load_router_data_oof(
            args.variant,
            folds=int(args.oof_folds),
            iterations=int(args.oof_iterations),
            exit_iterations=int(args.oof_exit_iterations),
            purge_bars=int(args.oof_purge_bars),
            seed=int(args.seed),
        )
    else:
        data = _load_router_data(args.variant)
    split = data.frame["dataset_split"].astype(str).str.lower().to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split != "train")
    if str(args.meta_label_source) == "oracle":
        y_action, y_quality = _future_side_quality(data.frame, fee=fee, slip=slip, min_edge=float(args.label_min_edge))
    else:
        y_action, y_quality = _candidate_side_quality(data, fee=fee, slip=slip, min_edge=float(args.label_min_edge))
    action_model, quality_model = _fit_meta_models(data.base_x, y_action, y_quality, train_idx, seed=int(args.seed))
    action_prob = _predict_action_proba(action_model, data.base_x)
    meta_action = np.argmax(action_prob, axis=1).astype(np.int64)
    active_prob = np.maximum(action_prob[:, 1], action_prob[:, 2])
    action_margin = active_prob - np.maximum(action_prob[:, 0], np.minimum(action_prob[:, 1], action_prob[:, 2]))
    bad_conf = (active_prob < float(args.meta_active_prob_min)) | (action_margin < float(args.meta_margin_min))
    meta_action[bad_conf] = 0
    if bool(args.enforce_expert_candidate_gate):
        meta_action[~_active_candidate_mask(data)] = 0
    meta_quality = np.asarray(quality_model.predict(data.base_x), dtype=np.float64)
    state_x, state_names = _meta_state(data, action_prob, meta_quality, meta_action)

    q_candidates = meta_quality[val_idx]
    grid = sorted(set(float(x) for x in np.quantile(q_candidates, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99])))
    train_candidates = train_idx[(meta_action[train_idx] != 0) & (meta_quality[train_idx] >= np.quantile(meta_quality[train_idx], 0.50))]
    train_candidates = train_candidates[train_candidates < len(data.frame) - 100]
    if int(args.max_risk_train_candidates) > 0 and len(train_candidates) > int(args.max_risk_train_candidates):
        train_candidates = train_candidates[-int(args.max_risk_train_candidates) :]
    if len(train_candidates) < 100:
        raise RuntimeError(f"too few DSAC risk candidates: {len(train_candidates)}")
    risk_agent, risk_update, risk_mean, risk_std = _train_risk_dsac(
        state_x,
        data.frame,
        train_candidates,
        meta_action,
        fee=fee,
        slip=slip,
        seed=int(args.seed),
        device=str(args.device),
        episodes=int(args.risk_episodes),
        warmup=int(args.risk_warmup),
        batch_size=int(args.risk_batch_size),
    )

    splits = _validation_splits(val_idx, int(args.purge_bars))
    calib_rows: list[dict[str, Any]] = []
    best: tuple[float, float, dict[str, Any]] | None = None
    for th in grid:
        bt = _backtest(data, state_x, meta_action, meta_quality, risk_agent, risk_mean, risk_std, start=splits["calib"][0], end=splits["calib"][1], threshold=th, fee=fee, slip=slip)
        calib_rows.append({"threshold": th, **bt})
        score = bt["calmar"] if bt["trades"] >= 8 else -1e6 + bt["pnl"]
        if best is None or score > best[0]:
            best = (score, th, bt)
    assert best is not None
    eval_rows = []
    for name, (start, end) in splits.items():
        bt = _backtest(data, state_x, meta_action, meta_quality, risk_agent, risk_mean, risk_std, start=start, end=end, threshold=best[1], fee=fee, slip=slip)
        eval_rows.append({"split": name, "threshold": best[1], **bt})
        print(f"[bt] split={name} {bt}", flush=True)

    pd.DataFrame(calib_rows).sort_values("calmar", ascending=False).to_csv(args.out_dir / "calib_thresholds.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(args.out_dir / "eval.csv", index=False)
    bundle = {
        "model_id": "alpha6_lgbm_meta_dsac_risk_20260524",
        "variant": args.variant,
        "model_specs": [(name, str(prefix)) for name, prefix in MODEL_SPECS],
        "base_feature_names": data.base_names,
        "state_feature_names": state_names,
        "risk_templates": [t.__dict__ for t in RISK_TEMPLATES],
        "best_threshold": float(best[1]),
        "cost_mult": float(args.cost_mult),
        "label_min_edge": float(args.label_min_edge),
        "meta_label_source": str(args.meta_label_source),
        "enforce_expert_candidate_gate": bool(args.enforce_expert_candidate_gate),
        "meta_active_prob_min": float(args.meta_active_prob_min),
        "meta_margin_min": float(args.meta_margin_min),
        "fee": float(fee),
        "slip": float(slip),
        "oof_folds": int(args.oof_folds),
        "oof_iterations": int(args.oof_iterations),
        "oof_exit_iterations": int(args.oof_exit_iterations),
        "oof_purge_bars": int(args.oof_purge_bars),
        "risk_update": risk_update,
        "eval": eval_rows,
        "audit": {
            "meta_output_contract": "action_proba/action + quality only",
            "meta_label_contract": "candidate mode only labels directions already proposed by active CatBoost experts; oracle mode is direct future-side labeling and is diagnostic only.",
            "risk_head_contract": "DSAC receives the same meta input plus meta action/quality and selects discrete notional/tp/sl/max_hold templates.",
            "catboost_oof_for_meta": bool(int(args.oof_folds) > 1),
            "meta_model_regularization": "shallow LightGBM max_depth<=3 num_leaves<=7 with strong L1/L2.",
            "regime_features": "All available frame columns containing 'regime' are included by the shared base feature builder.",
        },
    }
    joblib.dump({"action_model": action_model, "quality_model": quality_model, "config": bundle}, args.out_dir / "meta_lgbm_action_quality.joblib")
    torch.save(
        {
            "actor": risk_agent.actor.state_dict(),
            "critic": risk_agent.critic.state_dict(),
            "risk_mean": risk_mean,
            "risk_std": risk_std,
            "config": bundle,
        },
        args.out_dir / "risk_dsac.pt",
    )
    (args.out_dir / "summary.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2, default=str))
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
