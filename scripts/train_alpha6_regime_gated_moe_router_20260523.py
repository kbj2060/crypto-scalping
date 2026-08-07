#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_alpha6_dsac_ensemble_router_20260523 import (  # noqa: E402
    MODEL_SPECS,
    RouterData,
    _exit_model_for,
    _load_router_data,
    _load_router_data_oof,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import _exit_close_prob, _exit_state_vec  # noqa: E402


@dataclass(frozen=True)
class Template:
    name: str
    weights: tuple[float, float, float, float, float, float]
    threshold_mult: float
    horizon_mult: float
    giveback_limit: float
    veto_strength: float
    max_horizon: int
    trade_bonus: float


TEMPLATES = [
    Template("skip", (0, 0, 0, 0, 0, 0), 999.0, 1.0, 0.0, 0.0, 0, 0.0),
    Template("primary_strict", (1.0, 0, 0, 0, 0, 0), 1.00, 1.0, 0.50, 0.0, 96, 0.0000),
    Template("primary_normal", (1.0, 0, 0, 0, 0, 0), 0.70, 1.0, 0.60, 0.0, 96, 0.0002),
    Template("primary_loose", (1.0, 0, 0, 0, 0, 0), 0.45, 0.85, 0.55, 0.0, 48, 0.0005),
    Template("coverage_loose", (0.55, 0.35, 0.04, 0.04, 0.01, 0.01), 0.45, 0.90, 0.58, 0.25, 72, 0.0005),
    Template("scalp_short_horizon", (0.60, 0.05, 0.05, 0.25, 0.025, 0.025), 0.35, 0.65, 0.42, 0.15, 24, 0.0008),
    Template("disagreement_fade", (0.36, 0.22, 0.12, 0.12, 0.09, 0.09), 0.40, 0.75, 0.45, 0.05, 36, 0.0006),
    Template("high_conviction_blend", (0.48, 0.18, 0.14, 0.12, 0.04, 0.04), 0.55, 1.0, 0.62, 0.10, 96, 0.0003),
    Template("risk_veto_soft_loose", (0.50, 0.20, 0.12, 0.10, 0.04, 0.04), 0.38, 0.75, 0.40, 0.45, 48, 0.0006),
    Template("force_top_edge", (0.42, 0.22, 0.12, 0.12, 0.06, 0.06), 0.20, 0.70, 0.38, 0.20, 24, 0.0010),
]


def _cost() -> float:
    return 2.0 * (0.0004 + 0.00015) * 0.25


def _row_arrays(data: RouterData, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray([int(p.iloc[idx]["action"]) for p in data.preds], dtype=np.int64)
    quality = np.asarray([float(p.iloc[idx]["quality"]) for p in data.preds], dtype=np.float64)
    horizons = np.asarray([int(p.iloc[idx]["target_horizon"]) for p in data.preds], dtype=np.int64)
    return actions, quality, horizons


def _template_decision(data: RouterData, idx: int, template_id: int) -> tuple[int, int, float]:
    if template_id <= 0:
        return 0, 0, 0.0
    tpl = TEMPLATES[template_id]
    w = np.asarray(tpl.weights, dtype=np.float64)
    actions, quality, horizons = _row_arrays(data, idx)
    active_quality = np.maximum(quality, 0.0)
    long_edge = float(np.sum(w * active_quality * (actions == 1)))
    short_edge = float(np.sum(w * active_quality * (actions == 2)))
    risk_long_opp = float(np.sum(active_quality[[4, 5]] * (actions[[4, 5]] == 2)))
    risk_short_opp = float(np.sum(active_quality[[4, 5]] * (actions[[4, 5]] == 1)))
    long_edge -= tpl.veto_strength * risk_long_opp
    short_edge -= tpl.veto_strength * risk_short_opp
    threshold = float(np.sum(w * data.thresholds) / max(w.sum(), 1e-9)) * tpl.threshold_mult
    if max(long_edge, short_edge) <= threshold:
        return 0, 0, max(long_edge, short_edge)
    side = 1 if long_edge > short_edge else -1
    side_action = 1 if side > 0 else 2
    selected = (actions == side_action) & (horizons > 0)
    if selected.any():
        h = int(np.average(horizons[selected], weights=np.maximum(w[selected], 1e-6)))
    else:
        h = 12
    h = int(np.clip(round(h * tpl.horizon_mult), 2, max(2, tpl.max_horizon)))
    return side, h, max(long_edge, short_edge)


def _score_entry(data: RouterData, idx: int, side: int, horizon: int, trade_bonus: float) -> float:
    close = pd.to_numeric(data.frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    if side == 0 or idx + horizon + 1 >= len(close):
        return 0.0
    entry = max(float(close[idx]), 1e-12)
    path = (close[idx : idx + horizon + 1] / entry - 1.0) * side
    terminal = float(path[-1]) * 0.25
    mfe = max(0.0, float(np.max(path))) * 0.25
    mae = max(0.0, -float(np.min(path))) * 0.25
    vol = float(np.nanstd(path)) * 0.25
    return terminal + 0.25 * mfe - 0.85 * mae - 0.05 * vol - _cost() - 0.002 * (horizon / 96.0) + trade_bonus


def _make_oracle_labels(data: RouterData, indices: np.ndarray, min_score: float) -> np.ndarray:
    labels = np.zeros(len(indices), dtype=np.int64)
    max_idx = len(data.frame) - 100
    for out_i, idx in enumerate(indices):
        if idx >= max_idx:
            continue
        best_id = 0
        best_score = float(min_score)
        for tid, tpl in enumerate(TEMPLATES[1:], start=1):
            side, horizon, _ = _template_decision(data, int(idx), tid)
            if side == 0:
                continue
            score = _score_entry(data, int(idx), side, horizon, tpl.trade_bonus)
            if score > best_score:
                best_score = score
                best_id = tid
        labels[out_i] = best_id
    return labels


class MoEPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_policy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[MoEPolicy, dict[str, Any]]:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    model = MoEPolicy(x.shape[1], len(TEMPLATES)).to(device)
    counts = np.bincount(y, minlength=len(TEMPLATES)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.clip(weights / max(weights.mean(), 1e-9), 0.25, 5.0)
    # Do not let the abundant skip label dominate the policy.
    weights[0] = min(weights[0], 0.45)
    loss_w = torch.tensor(weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    hist: list[dict[str, float]] = []
    for ep in range(int(epochs)):
        order = rng.permutation(len(x))
        total = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=loss_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += len(idx)
        hist.append({"epoch": ep + 1, "loss": total / max(seen, 1), "acc": correct / max(seen, 1)})
    return model, {"label_counts": counts.astype(int).tolist(), "class_weights": weights.tolist(), "history": hist}


def _policy_actions(model: MoEPolicy, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits = model(torch.tensor(x[start : start + 4096], dtype=torch.float32, device=device))
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def _backtest_templates(data: RouterData, indices: np.ndarray, template_ids: np.ndarray) -> dict[str, Any]:
    close = pd.to_numeric(data.frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_idx = -1
    horizon = 0
    template_id = 0
    giveback_limit = 0.5
    mae = 0.0
    mfe = 0.0
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}

    def close_pos(i: int, reason: str) -> None:
        nonlocal cash, side, entry, entry_idx, horizon, template_id, mae, mfe, trades, wins
        raw = (float(close[i]) - entry) / max(entry, 1e-12) * side
        pnl = raw * 0.25 - (0.0004 + 0.00015) * 0.25
        cash += pnl
        trades += 1
        wins += int(pnl > 0)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        entry_idx = -1
        horizon = 0
        template_id = 0
        mae = 0.0
        mfe = 0.0

    idx_to_action = {int(i): int(a) for i, a in zip(indices, template_ids)}
    idx_set = set(idx_to_action)
    for i in indices:
        i = int(i)
        if side != 0:
            raw = (float(close[i]) - entry) / max(entry, 1e-12) * side * 0.25
            mae = max(mae, max(0.0, -raw))
            mfe = max(mfe, max(0.0, raw))
            giveback = max(0.0, mfe - max(raw, 0.0))
            hold = i - entry_idx
            tpl = TEMPLATES[template_id]
            w = np.asarray(tpl.weights, dtype=np.float64)
            exit_prob = 0.0
            exit_th = float(np.sum(w * data.exit_thresholds) / max(w.sum(), 1e-9)) if w.sum() > 0 else 1.0
            if hold >= 2 and w.sum() > 0:
                state = _exit_state_vec(
                    data.frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=float(close[i]),
                    hold=hold,
                    horizon=max(horizon, 2),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=0,
                    expected_return=0.01,
                )
                probs = []
                for j in np.flatnonzero(w > 0):
                    probs.append(float(w[j]) * _exit_close_prob(_exit_model_for(data, int(j), i), data.xs[int(j)][i], state))
                exit_prob = float(np.sum(probs) / max(w.sum(), 1e-9))
            if hold >= 2 and exit_prob >= exit_th:
                close_pos(i, "exit_model")
            elif hold >= horizon:
                close_pos(i, "horizon")
            elif mfe > 0 and giveback / max(mfe, 1e-9) >= giveback_limit and hold >= 2:
                close_pos(i, "giveback")
        if side == 0 and i in idx_set:
            tid = idx_to_action[i]
            s, h, _ = _template_decision(data, i, tid)
            if s != 0:
                side = int(s)
                entry = float(close[i])
                entry_idx = i
                horizon = int(h)
                template_id = int(tid)
                giveback_limit = float(TEMPLATES[tid].giveback_limit)
                cash -= (0.0004 + 0.00015) * 0.25
                long_entries += int(side > 0)
                short_entries += int(side < 0)
        eq = cash
        if side != 0:
            eq += ((float(close[i]) - entry) / max(entry, 1e-12) * side) * 0.25
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
    if side != 0:
        close_pos(int(indices[-1]), "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
        "action_counts": {TEMPLATES[int(k)].name: int(v) for k, v in pd.Series(template_ids).value_counts().sort_index().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime-gated MoE router for Alpha6 expert signals.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_regime_gated_moe_router_20260523")
    ap.add_argument("--oof-folds", type=int, default=0)
    ap.add_argument("--oof-iterations", type=int, default=120)
    ap.add_argument("--oof-exit-iterations", type=int, default=40)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--label-min-score", type=float, default=-0.00015)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if int(args.oof_folds) > 1:
        data = _load_router_data_oof(
            args.variant,
            folds=int(args.oof_folds),
            iterations=int(args.oof_iterations),
            exit_iterations=int(args.oof_exit_iterations),
            purge_bars=96,
            seed=int(args.seed),
        )
    else:
        data = _load_router_data(args.variant)
    split = data.frame["dataset_split"].astype(str).str.lower().to_numpy()
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split != "train")
    if int(args.max_train_rows) > 0:
        train_idx = train_idx[-int(args.max_train_rows) :]
    if int(args.max_val_rows) > 0:
        val_idx = val_idx[: int(args.max_val_rows)]

    mean = data.base_x[train_idx].mean(axis=0)
    std = np.where(data.base_x[train_idx].std(axis=0) <= 1e-6, 1.0, data.base_x[train_idx].std(axis=0))
    x_train = ((data.base_x[train_idx] - mean) / std).astype(np.float32)
    x_val = ((data.base_x[val_idx] - mean) / std).astype(np.float32)
    y_train = _make_oracle_labels(data, train_idx, float(args.label_min_score))
    policy, train_meta = _train_policy(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    val_actions = _policy_actions(policy, x_val, args.device)
    train_actions = _policy_actions(policy, x_train, args.device)
    result = {
        "model_id": "alpha6_regime_gated_moe_router_20260523",
        "variant": args.variant,
        "templates": [tpl.__dict__ for tpl in TEMPLATES],
        "model_specs": [(name, str(prefix)) for name, prefix in MODEL_SPECS],
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "state_dim": int(data.base_x.shape[1]),
        "oof_folds": int(args.oof_folds),
        "oof_iterations": int(args.oof_iterations),
        "oof_exit_iterations": int(args.oof_exit_iterations),
        "label_min_score": float(args.label_min_score),
        "train_meta": train_meta,
        "train_backtest": _backtest_templates(data, train_idx, train_actions),
        "val_backtest": _backtest_templates(data, val_idx, val_actions),
        "audit": {
            "router_train_uses_expert_in_sample_predictions": False if int(args.oof_folds) > 1 else True,
            "validation_uses_full_train_expert_bundles": True,
            "policy_type": "supervised contextual-bandit MoE over dynamic threshold/exit templates",
            "fixed24_excluded": True,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    joblib.dump({"policy_state": policy.cpu().state_dict(), "mean": mean, "std": std, "result": result}, args.out_dir / "moe_router.joblib")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
