#!/usr/bin/env python3
from __future__ import annotations

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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    SPLIT_TS,
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_cash_region_dsac_fallback_selector_20260526"
S1_PARENT = ROOT / "data/ensemble/supervised/alpha7_v2_only_high_turnover_s1_live_20260526/primary_parent.pkl"
S1_CAND_SUMMARY = ROOT / "tmp/causal_regen_20260516/alpha7_v2_only_high_turnover_rebuild_20260526/t0015_c015_h030_s6/summary.json"
MODEL_ID = "alpha7_cash_region_dsac_fallback_selector_20260526"

ACTION_SKIP = 0
ACTION_FB0 = 1
ACTION_FB1 = 2
ACTION_DIM = 3


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _state_matrix(frame: pd.DataFrame, primary: pd.DataFrame, fb0: pd.DataFrame, fb1: pd.DataFrame) -> np.ndarray:
    cols: list[np.ndarray] = []
    market_cols = [
        "smart_money_flow",
        "ofi_acceleration",
        "funding_pressure",
        "crowding_pressure",
        "liquidity_vacuum",
        "trade_intensity",
        "ai_dir_edge",
        "ai_dir_entropy",
        "ai_adverse_risk",
        "tide_vol_zscore",
        "patchtst_regime_sim",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_chop_prob",
        "regime4_pred_confidence",
        "regime4_pred_trend_prob",
        "regime4_pred_whipsaw_prob",
        "tp_sl_action_score",
    ]
    for c in market_cols:
        cols.append(_safe_col(frame, c))
    # Primary context
    for c in [
        "action",
        "side",
        "quality_score",
        "confidence",
        "notional_exposure",
        "leverage",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
    ]:
        cols.append(_safe_col(primary, c))
    # Fallback candidate context
    for c in ["quality_score", "confidence", "notional_exposure", "take_profit", "stop_loss", "max_hold_bars"]:
        cols.append(_safe_col(fb0, c))
        cols.append(_safe_col(fb1, c))
    x = np.column_stack(cols).astype(np.float32, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _first_hit(path: np.ndarray, tp: float, sl: float, hold: int) -> int:
    m = min(int(max(1, hold)), len(path))
    if m <= 1:
        return 0
    p = path[:m]
    hit = np.flatnonzero((p >= float(tp)) | (p <= -abs(float(sl))))
    return int(hit[0]) if hit.size else int(m - 1)


def _decision_reward(close: np.ndarray, i: int, dec_row: pd.Series, *, fee: float, slip: float) -> float:
    action = int(dec_row.get("action", 0) or 0)
    side = int(dec_row.get("side", 0) or 0)
    if action == 0 or side == 0:
        return 0.0
    notional = float(dec_row.get("notional_exposure", 0.0) or 0.0)
    tp = float(dec_row.get("take_profit", 0.0) or 0.0)
    sl = float(dec_row.get("stop_loss", 0.0) or 0.0)
    hold = int(dec_row.get("max_hold_bars", 0) or 0)
    if notional <= 0.0 or hold <= 0:
        return 0.0
    end = min(len(close), i + hold + 1)
    if end <= i + 1:
        return 0.0
    fut = close[i + 1 : end]
    entry = max(float(close[i]), 1e-12)
    side_ret = ((fut / entry) - 1.0) * float(side)
    path = side_ret * notional
    exit_i = _first_hit(path, tp, sl, hold)
    pnl = float(path[exit_i] - 2.0 * (fee + slip) * notional)
    adverse = max(0.0, -float(np.min(path[: exit_i + 1])))
    hold_frac = float(exit_i + 1) / 288.0
    # Conservative shaped reward for selector.
    reward = pnl - 0.65 * adverse - 0.004 * hold_frac
    return float(reward)


@dataclass
class DatasetBundle:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def _build_cash_region_dataset(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    fb0: pd.DataFrame,
    fb1: pd.DataFrame,
    *,
    fee: float,
    slip: float,
) -> tuple[DatasetBundle, np.ndarray]:
    close = _safe_col(frame, "close").astype(np.float64)
    x_all = _state_matrix(frame, primary, fb0, fb1)
    primary_cash = (_safe_col(primary, "action").astype(np.int64) == 0) | (_safe_col(primary, "side").astype(np.int64) == 0)
    cash_idx = np.flatnonzero(primary_cash)
    if cash_idx.size < 3:
        raise RuntimeError("cash-region dataset too small")

    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []

    for k in range(len(cash_idx) - 1):
        i = int(cash_idx[k])
        ni = int(cash_idx[k + 1])
        s = x_all[i]
        sp = x_all[ni]
        done = 1.0 if k == len(cash_idx) - 2 else 0.0

        r0 = 0.0
        r1 = _decision_reward(close, i, fb0.iloc[i], fee=fee, slip=slip)
        r2 = _decision_reward(close, i, fb1.iloc[i], fee=fee, slip=slip)
        rewards = [r0, r1, r2]
        for a, r in enumerate(rewards):
            s_list.append(s)
            sp_list.append(sp)
            a_list.append(a)
            r_list.append(float(r))
            d_list.append(done)

    bundle = DatasetBundle(
        states=np.asarray(s_list, dtype=np.float32),
        next_states=np.asarray(sp_list, dtype=np.float32),
        actions=np.asarray(a_list, dtype=np.int64),
        rewards=np.asarray(r_list, dtype=np.float32),
        dones=np.asarray(d_list, dtype=np.float32),
    )
    return bundle, cash_idx


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dsac_offline(
    data: DatasetBundle,
    *,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    steps: int = 7000,
    batch_size: int = 512,
    gamma: float = 0.995,
    tau: float = 0.01,
    lr: float = 3e-4,
) -> dict[str, Any]:
    actor = Actor(state_dim, action_dim).to(device)
    q1 = Critic(state_dim, action_dim).to(device)
    q2 = Critic(state_dim, action_dim).to(device)
    tq1 = Critic(state_dim, action_dim).to(device)
    tq2 = Critic(state_dim, action_dim).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.10), device=device, requires_grad=True)
    target_entropy = -math.log(float(action_dim))

    opt_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=lr)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=lr)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)

    ds = TensorDataset(
        torch.from_numpy(data.states),
        torch.from_numpy(data.next_states),
        torch.from_numpy(data.actions),
        torch.from_numpy(data.rewards),
        torch.from_numpy(data.dones),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    last = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0}

    for step in range(1, steps + 1):
        try:
            s, sp, a, r, d = next(it)
        except StopIteration:
            it = iter(dl)
            s, sp, a, r, d = next(it)
        s = s.to(device)
        sp = sp.to(device)
        a = a.to(device)
        r = r.to(device)
        d = d.to(device)

        with torch.no_grad():
            next_logits = actor(sp)
            next_logp = F.log_softmax(next_logits, dim=-1)
            next_pi = next_logp.exp()
            alpha = log_alpha.exp()
            next_q = torch.min(tq1(sp), tq2(sp))
            v_next = (next_pi * (next_q - alpha * next_logp)).sum(dim=-1)
            y = r + (1.0 - d) * gamma * v_next

        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.mse_loss(qa1, y) + F.mse_loss(qa2, y)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        opt_q1.step()
        opt_q2.step()

        logits = actor(s)
        logp = F.log_softmax(logits, dim=-1)
        pi = logp.exp()
        alpha = log_alpha.exp()
        q_min = torch.min(q1(s), q2(s))
        actor_loss = (pi * (alpha * logp - q_min)).sum(dim=-1).mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        opt_actor.step()

        entropy = -(pi * logp).sum(dim=-1).mean().detach()
        alpha_loss = -(log_alpha * (entropy - target_entropy)).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(5.0))

        with torch.no_grad():
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        if step % 250 == 0:
            last = {
                "q_loss": float(q_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha": float(log_alpha.exp().item()),
                "entropy": float(entropy.item()),
                "step": int(step),
            }

    return {
        "actor": actor.cpu(),
        "q1": q1.cpu(),
        "q2": q2.cpu(),
        "train_diag": last,
    }


def _policy_action(actor: nn.Module, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    with torch.no_grad():
        x = torch.from_numpy(states).to(device)
        logits = actor(x)
        act = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
    return act


def _compose_final_decisions(
    primary: pd.DataFrame,
    fb0: pd.DataFrame,
    fb1: pd.DataFrame,
    actions: np.ndarray,
) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    pa = pd.to_numeric(primary["action"], errors="coerce").fillna(0).astype(int).to_numpy()
    ps = pd.to_numeric(primary["side"], errors="coerce").fillna(0).astype(int).to_numpy()
    primary_active = (pa != 0) & (ps != 0)
    for i in range(len(out)):
        if primary_active[i]:
            continue
        a = int(actions[i])
        if a == ACTION_FB0:
            src = fb0.iloc[i]
        elif a == ACTION_FB1:
            src = fb1.iloc[i]
        else:
            continue
        sa = int(src.get("action", 0) or 0)
        ss = int(src.get("side", 0) or 0)
        if sa == 0 or ss == 0:
            continue
        out.iloc[i] = src
    return out


def _extract_runtime(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    target = str(s.get("best_by_selection", ""))
    for e in s.get("experiments", []):
        if target and str(e.get("name", "")) != target:
            continue
        rt = e.get("selected_parent_scale_runtime")
        if rt:
            return alpha2.Alpha2Runtime(
                name=str(rt["name"]),
                confidence=float(rt["confidence"]),
                parent_notional_scale=float(rt["parent_notional_scale"]),
                max_notional=float(rt["max_notional"]),
            )
    return None


def main() -> int:
    _seed_everything(260526)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_all = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    fb0_parent = joblib.load(FALLBACK_PARENT)
    fb1_parent = joblib.load(S1_PARENT)

    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fb0_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    fb1_rt = _extract_runtime(S1_CAND_SUMMARY)

    ref = joblib.load(ROOT / "data/ensemble/ckpt/best_dsac_agents.pth") if False else None
    fee = 0.0005
    slip = 0.0002

    # Precompute decisions for all splits.
    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    fb0_train = _predict_scaled(fb0_parent, train_df, fb0_rt)
    fb0_val = _predict_scaled(fb0_parent, val_df, fb0_rt)
    fb0_eval = _predict_scaled(fb0_parent, eval_df, fb0_rt)

    fb1_train = _predict_scaled(fb1_parent, train_df, fb1_rt)
    fb1_val = _predict_scaled(fb1_parent, val_df, fb1_rt)
    fb1_eval = _predict_scaled(fb1_parent, eval_df, fb1_rt)

    ds_train, _ = _build_cash_region_dataset(train_df, p_train, fb0_train, fb1_train, fee=fee, slip=slip)
    state_dim = int(ds_train.states.shape[1])
    trained = _train_dsac_offline(ds_train, state_dim=state_dim, action_dim=ACTION_DIM, device=device)

    actor: nn.Module = trained["actor"]
    actions_train = _policy_action(actor, _state_matrix(train_df, p_train, fb0_train, fb1_train), device=device)
    actions_val = _policy_action(actor, _state_matrix(val_df, p_val, fb0_val, fb1_val), device=device)
    actions_eval = _policy_action(actor, _state_matrix(eval_df, p_eval, fb0_eval, fb1_eval), device=device)

    dec_train = _compose_final_decisions(p_train, fb0_train, fb1_train, actions_train)
    dec_val = _compose_final_decisions(p_val, fb0_val, fb1_val, actions_val)
    dec_eval = _compose_final_decisions(p_eval, fb0_eval, fb1_eval, actions_eval)

    m_train = _combo_metrics(train_df, dec_train)
    m_val = _combo_metrics(val_df, dec_val)
    m_eval = _combo_metrics(eval_df, dec_eval)

    baseline_val = _combo_metrics(val_df, _combine_primary_fallback(p_val, fb0_val))
    baseline_eval = _combo_metrics(eval_df, _combine_primary_fallback(p_eval, fb0_eval))

    counts_eval = {
        "skip": int(np.sum(actions_eval == ACTION_SKIP)),
        "fallback_alpha43": int(np.sum(actions_eval == ACTION_FB0)),
        "fallback_s1": int(np.sum(actions_eval == ACTION_FB1)),
    }
    primary_cash_eval = (
        (_safe_col(p_eval, "action").astype(np.int64) == 0)
        | (_safe_col(p_eval, "side").astype(np.int64) == 0)
    )
    counts_eval_cash = {
        "cash_rows": int(primary_cash_eval.sum()),
        "skip_on_cash": int(np.sum((actions_eval == ACTION_SKIP) & primary_cash_eval)),
        "fb0_on_cash": int(np.sum((actions_eval == ACTION_FB0) & primary_cash_eval)),
        "fb1_on_cash": int(np.sum((actions_eval == ACTION_FB1) & primary_cash_eval)),
    }

    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dim": state_dim,
            "action_dim": ACTION_DIM,
            "actor_state_dict": actor.state_dict(),
            "train_diag": trained["train_diag"],
        },
        OUT_DIR / "cash_region_dsac_selector.pt",
    )

    summary = {
        "model_id": MODEL_ID,
        "design": "Cash-region-only DSAC fallback selector (skip / current alpha43 fallback / s1 fallback) with counterfactual replay.",
        "state_dim": state_dim,
        "action_dim": ACTION_DIM,
        "train_diag": trained["train_diag"],
        "train_metrics": m_train["cost3"],
        "val_metrics": m_val["cost3"],
        "oos_metrics": m_eval["cost3"],
        "baseline_val_metrics": baseline_val["cost3"],
        "baseline_oos_metrics": baseline_eval["cost3"],
        "delta_vs_baseline": {
            "val_cost3_pnl": float(m_val["cost3"]["pnl"] - baseline_val["cost3"]["pnl"]),
            "val_cost3_trades": int(m_val["cost3"]["trades"] - baseline_val["cost3"]["trades"]),
            "oos_cost3_pnl": float(m_eval["cost3"]["pnl"] - baseline_eval["cost3"]["pnl"]),
            "oos_cost3_trades": int(m_eval["cost3"]["trades"] - baseline_eval["cost3"]["trades"]),
        },
        "action_usage_eval_all_rows": counts_eval,
        "action_usage_eval_cash_rows": counts_eval_cash,
        "artifacts": {
            "selector_ckpt": str((OUT_DIR / "cash_region_dsac_selector.pt").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
