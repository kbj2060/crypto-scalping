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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import _json_default  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import ACTIVE_SCALES, ACTIVE_TEMPLATE  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_risk_allocator_20260602 import (  # noqa: E402
    ACTION_LOOKUP,
    OUT_DIR as RISK_OUT_DIR,
    _active,
    _apply_norm,
    _build_state_frame,
    _fast_replay_metrics,
    _fit_norm,
    _load_variant_frames,
    _numeric_feature_cols,
    _simulate_action,
    _to_decisions,
    _zero_row,
)


MODEL_ID = "omega1_expertdq_dsac_veto_filter_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ALLOW_ACTION = 1
VETO_ACTION = 0


@dataclass
class VetoDataset:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    best_actions: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VetoActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.logits(x))
        action = dist.sample()
        return action, dist.log_prob(action)

    def allow_prob(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(x), dim=-1)[:, ALLOW_ACTION]


class VetoCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, 192),
            nn.SiLU(),
            nn.Linear(192, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_veto_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_active_rows: int,
) -> tuple[VetoDataset, dict[str, Any]]:
    active_idxs = np.flatnonzero(_active(dec) & (np.arange(len(frame)) < len(frame) - 3))
    rng = np.random.default_rng(260602)
    total_active_rows = int(len(active_idxs))
    if int(max_active_rows) > 0 and len(active_idxs) > int(max_active_rows):
        active_idxs = np.sort(rng.choice(active_idxs, size=int(max_active_rows), replace=False))
    arrays = {k: pd.to_numeric(frame[k], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close")}
    template_id = ACTION_LOOKUP[(0, 0, 0, 0, 0, 0, 0)]

    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    best_list: list[int] = []
    allow_nets: list[float] = []
    best_counts = {VETO_ACTION: 0, ALLOW_ACTION: 0}

    for i in active_idxs:
        allow_net, _ = _simulate_action(
            frame,
            arrays,
            int(i),
            dec.iloc[int(i)],
            template_id,
            fee=fee,
            slip=slip,
            cost_mult=cost_mult,
        )
        best = ALLOW_ACTION if allow_net > 0.0 else VETO_ACTION
        best_counts[best] += 1
        allow_nets.append(float(allow_net))
        for action, reward in ((VETO_ACTION, 0.0), (ALLOW_ACTION, float(allow_net))):
            s_list.append(states[int(i)])
            sp_list.append(states[min(int(i) + 1, len(states) - 1)])
            a_list.append(action)
            r_list.append(float(reward))
            d_list.append(1.0)
            best_list.append(best)

    rewards = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards = np.clip(rewards / scale, -8.0, 8.0).astype(np.float32)
    allow_arr = np.asarray(allow_nets, dtype=np.float64)
    diag = {
        "active_rows": int(len(active_idxs)),
        "total_active_rows": total_active_rows,
        "sample_count": int(len(rewards)),
        "reward_scale": float(scale),
        "oracle_best_counts": {"veto": int(best_counts[VETO_ACTION]), "allow": int(best_counts[ALLOW_ACTION])},
        "allow_net_mean": float(np.mean(allow_arr)) if len(allow_arr) else 0.0,
        "allow_net_median": float(np.median(allow_arr)) if len(allow_arr) else 0.0,
        "allow_net_win_rate": float(np.mean(allow_arr > 0.0)) if len(allow_arr) else 0.0,
    }
    return (
        VetoDataset(
            states=np.asarray(s_list, dtype=np.float32),
            next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64),
            rewards=rewards,
            dones=np.asarray(d_list, dtype=np.float32),
            best_actions=np.asarray(best_list, dtype=np.int64),
        ),
        diag,
    )


def _train_dsac(
    data: VetoDataset,
    *,
    state_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    bc_coef: float,
) -> tuple[VetoActor, dict[str, Any]]:
    actor = VetoActor(state_dim).to(device)
    q1 = VetoCritic(state_dim).to(device)
    q2 = VetoCritic(state_dim).to(device)
    tq1 = VetoCritic(state_dim).to(device)
    tq2 = VetoCritic(state_dim).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.10), device=device, requires_grad=True)
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=lr, weight_decay=1e-5)
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=lr, weight_decay=1e-5)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    ds = TensorDataset(
        torch.from_numpy(data.states),
        torch.from_numpy(data.next_states),
        torch.from_numpy(data.actions),
        torch.from_numpy(data.rewards),
        torch.from_numpy(data.dones),
        torch.from_numpy(data.best_actions),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            s, sp, a, r, d, best_a = next(it)
        except StopIteration:
            it = iter(dl)
            s, sp, a, r, d, best_a = next(it)
        s = s.to(device)
        sp = sp.to(device)
        a = a.to(device)
        r = r.to(device)
        d = d.to(device)
        best_a = best_a.to(device)
        with torch.no_grad():
            na, nlogp = actor.sample(sp)
            next_q = torch.min(tq1(sp), tq2(sp)).gather(1, na.view(-1, 1)).squeeze(1)
            y = r + (1.0 - d) * 0.995 * (next_q - log_alpha.exp() * nlogp)
        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.smooth_l1_loss(qa1, y) + F.smooth_l1_loss(qa2, y)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()

        pa, plogp = actor.sample(s)
        pq = torch.min(q1(s), q2(s)).gather(1, pa.view(-1, 1)).squeeze(1)
        bc_loss = F.cross_entropy(actor.logits(s), best_a)
        actor_loss = (log_alpha.exp() * plogp - pq).mean() + float(bc_coef) * bc_loss
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = (-plogp).mean().detach()
        alpha_loss = -(log_alpha * (entropy - 0.45)).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(2.0))
        with torch.no_grad():
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(0.99).add_(0.01 * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(0.99).add_(0.01 * p.data)
        if step % 250 == 0:
            last = {
                "step": int(step),
                "q_loss": float(q_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "alpha": float(log_alpha.exp().detach().cpu()),
                "entropy": float(entropy.cpu()),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), last


def _allow_prob(actor: VetoActor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            out.append(actor.allow_prob(x).cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _compose_veto_decisions(base: pd.DataFrame, allow: np.ndarray) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    active = _active(out)
    deny = active & (~np.asarray(allow, dtype=bool))
    if bool(np.any(deny)):
        out.loc[deny, "action"] = 0
        out.loc[deny, "side"] = 0
        out.loc[deny, "notional_exposure"] = 0.0
        out.loc[deny, "position_fraction"] = 0.0
        out.loc[deny, "take_profit"] = 0.0
        out.loc[deny, "stop_loss"] = 0.0
        out.loc[deny, "max_hold_bars"] = 0
        out.loc[deny, "cooldown_bars"] = 0
        out.loc[deny, "leverage"] = 1.0
    return out


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def _metrics_row(split: str, variant: str, frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    metrics = _fast_replay_metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)
    row = {"split": split, "variant": variant, "cost": 3, **metrics}
    row["selection_score"] = _score(pd.Series(row))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p05")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.18)
    ap.add_argument("--max-active-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(260602)
    out_dir = OUT_DIR / str(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    train_df, val_df, oos_df, train_src, val_src, oos_src, overlay = _load_variant_frames(str(args.variant))
    train_dec = _to_decisions(train_src, oof=True)
    val_dec = _to_decisions(val_src, oof=True)
    oos_dec = _to_decisions(oos_src, oof=False)

    feature_cols = _numeric_feature_cols(train_df)
    s_train = _build_state_frame(train_df, train_dec, train_src, oof=True, feature_cols=feature_cols)
    s_val = _build_state_frame(val_df, val_dec, val_src, oof=True, feature_cols=feature_cols)
    s_oos = _build_state_frame(oos_df, oos_dec, oos_src, oof=False, feature_cols=feature_cols)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_oos = _apply_norm(s_oos, norm)

    parent_cfg = joblib.load(v31.DEFAULT_PARENT)["config"]
    fee = float(parent_cfg["fee"])
    slip = float(parent_cfg["slip"])
    dataset, data_diag = _build_veto_dataset(
        train_df,
        x_train,
        train_dec,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_active_rows=int(args.max_active_rows),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "variant": args.variant,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "action_dim": 2,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(oos_df)),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    actor, train_diag = _train_dsac(
        dataset,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        bc_coef=float(args.bc_coef),
    )
    p_train = _allow_prob(actor, x_train, device=device)
    p_val = _allow_prob(actor, x_val, device=device)
    p_oos = _allow_prob(actor, x_oos, device=device)

    thresholds = [round(float(x), 2) for x in np.linspace(0.05, 0.95, 19)]
    rows: list[dict[str, Any]] = []
    rows.append(_metrics_row("val", "fixed_omega1_template", val_df, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)))
    rows.append(_metrics_row("oos", "fixed_omega1_template", oos_df, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)))
    for th in thresholds:
        val_veto = _compose_veto_decisions(val_dec, p_val >= th)
        oos_veto = _compose_veto_decisions(oos_dec, p_oos >= th)
        rows.append(_metrics_row("val", f"dsac_veto_th{th:.2f}", val_df, val_veto, fee=fee, slip=slip, cost_mult=float(args.cost_mult)))
        rows.append(_metrics_row("oos", f"dsac_veto_th{th:.2f}", oos_df, oos_veto, fee=fee, slip=slip, cost_mult=float(args.cost_mult)))

    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)
    val_rank = grid[grid["split"] == "val"].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"])
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    fixed_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_omega1_template")].iloc[0].to_dict()

    model_path = out_dir / "omega1_expertdq_dsac_veto_filter.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": str(args.variant),
            "state_dim": int(x_train.shape[1]),
            "action_dim": 2,
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "actor_state_dict": actor.state_dict(),
            "selected_threshold": float(selected_variant.rsplit("th", 1)[1]) if selected_variant.startswith("dsac_veto_th") else None,
        },
        model_path,
    )
    selected_val = val_rank.iloc[0].to_dict()
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Omega1 supervised expert-local decision/quality is frozen. DSAC only decides allow/veto; risk template remains fixed.",
        "selection_basis": "2025Q4 validation Cost3 fast replay only; 2026 OOS is report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "risk_template": ACTIVE_TEMPLATE,
        "expert_scales": ACTIVE_SCALES,
        "feature_cols": feature_cols,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": 2,
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "cost_mult": float(args.cost_mult),
            "reward_label": "allow=current_template_complete_trade_net_pnl_after_entry_exit_fee_slippage; veto=0",
            "data_diag": data_diag,
            "train_diag": train_diag,
            "allow_prob_stats": {
                "train_mean": float(np.mean(p_train)),
                "val_mean": float(np.mean(p_val)),
                "oos_mean": float(np.mean(p_oos)),
                "train_p10_p50_p90": [float(x) for x in np.quantile(p_train, [0.1, 0.5, 0.9])],
                "val_p10_p50_p90": [float(x) for x in np.quantile(p_val, [0.1, 0.5, 0.9])],
                "oos_p10_p50_p90": [float(x) for x in np.quantile(p_oos, [0.1, 0.5, 0.9])],
            },
        },
        "selected_by_val_cost3": {"variant": selected_variant, "val_cost3": selected_val, "oos_cost3": selected_oos},
        "fixed_template_oos_cost3": fixed_oos,
        "delta_selected_vs_fixed_oos_cost3_pnl": float(selected_oos["pnl"]) - float(fixed_oos["pnl"]),
        "overlay": overlay,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path), "model": str(model_path), "prior_risk_allocator_dir": str(RISK_OUT_DIR)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "selected": summary["selected_by_val_cost3"], "fixed_oos_cost3": fixed_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
