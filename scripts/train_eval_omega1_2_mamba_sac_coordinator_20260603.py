#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from mamba_ssm import Mamba  # noqa: E402

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402


MODEL_ID = "omega1_2_mamba_sac_head_coordinator_20260603"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_EXIT_HEAD = (
    ROOT
    / "tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_exit_head_nohold_20260603_veto_edge200_full_seed260604/exit_head_bundle.pt"
)
SPLIT_TS = pd.Timestamp("2025-10-01")
ACTION_CASH = omega.ACTION_CASH


@dataclass(frozen=True)
class CoordAction:
    enter: float
    exit_thr: float
    exposure_mult: float
    emergency_sl: float


ACTION_GRID = [
    CoordAction(0.0, 0.995, 0.0, 0.05),
    *[
        CoordAction(1.0, exit_thr, mult, sl)
        for exit_thr in (0.50, 0.70, 0.90, 0.98)
        for mult in (0.50, 0.75, 1.00, 1.20)
        for sl in (0.025, 0.040, 0.065)
    ],
]

FAST_ACTION_GRID = [
    CoordAction(0.0, 0.995, 0.0, 0.05),
    *[
        CoordAction(1.0, exit_thr, mult, sl)
        for exit_thr in (0.70, 0.90)
        for mult in (0.60, 1.00)
        for sl in (0.035, 0.065)
    ],
]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _action_to_vec(a: CoordAction) -> np.ndarray:
    return np.asarray(
        [
            float(a.enter),
            (float(a.exit_thr) - 0.50) / (0.995 - 0.50),
            (float(a.exposure_mult) - 0.0) / 1.20,
            (float(a.emergency_sl) - 0.025) / (0.065 - 0.025),
        ],
        dtype=np.float32,
    )


def _vec_to_action(v: np.ndarray) -> CoordAction:
    x = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return CoordAction(
        enter=float(x[0]),
        exit_thr=float(0.50 + x[1] * (0.995 - 0.50)),
        exposure_mult=float(x[2] * 1.20),
        emergency_sl=float(0.025 + x[3] * (0.065 - 0.025)),
    )


def _prepare_frames() -> dict[str, Any]:
    # No TP/SL/max-hold/cooldown. Direction/Quality are frozen soft_floor_0p00.
    frames = exit_head._prepare_frames(disable_tp_sl=True)
    return frames


def _fit_norm(df: pd.DataFrame) -> dict[str, Any]:
    arr = df.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": list(df.columns), "median": med.astype(np.float32), "scale": scale.astype(np.float32)}


def _apply_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mamba-SAC state missing columns: {missing[:20]}")
    arr = df[cols].to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    return np.tanh(np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _rolling_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    pad = np.repeat(arr[:1], max(int(seq_len) - 1, 0), axis=0)
    padded = np.concatenate([pad, arr], axis=0)
    view = np.lib.stride_tricks.sliding_window_view(padded, int(seq_len), axis=0)
    return np.swapaxes(view, 1, 2).copy().astype(np.float32)


def _load_exit_models(path: Path, *, device: torch.device) -> dict[str, tuple[exit_head.ExitTabMClassifier, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "exit_head_models" not in bundle:
        raise RuntimeError(f"{path} missing exit_head_models")
    return exit_head._load_exit_heads(bundle["exit_head_models"], device=device)


def _simulate_coord_action(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    loaded_exit: dict[str, tuple[exit_head.ExitTabMClassifier, dict[str, Any]]],
    signal_i: int,
    coord: CoordAction,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    max_sim_bars: int,
) -> tuple[float, dict[str, Any]]:
    if coord.enter < 0.50:
        return 0.0, {"active": 0, "exit_reason": "skip", "exit_i": int(signal_i), "net": 0.0}
    drow = dec.iloc[int(signal_i)]
    action = int(drow.get("action", 0) or 0)
    side = int(drow.get("side", 0) or 0)
    if action == ACTION_CASH or side == 0:
        return 0.0, {"active": 0, "exit_reason": "cash_source", "exit_i": int(signal_i), "net": 0.0}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    filled, entry_price, entry_fee, _route = omega._try_execution(arrays, int(signal_i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return 0.0, {"active": 0, "exit_reason": "entry_miss", "exit_i": int(signal_i), "net": 0.0}
    entry_i = min(int(signal_i) + 1, len(frame) - 1)
    entry_state = state.iloc[int(signal_i)]
    base_notional = float(drow.get("notional_exposure", 0.0) or 0.0)
    notional = float(np.clip(base_notional * float(coord.exposure_mult), 0.0, 1.20))
    leverage = float(drow.get("leverage", 1.0) or 1.0)
    if notional <= 0.0:
        return 0.0, {"active": 0, "exit_reason": "zero_exposure", "exit_i": int(signal_i), "net": 0.0}
    cash = 1.0 - 1.0 * entry_fee * notional
    mfe = 0.0
    mae = 0.0
    exit_reason = "forced_end"
    exit_i = len(frame) - 1
    exit_fill: float | None = None
    exit_fee = fee_eff
    last_i = len(frame) - 2 if int(max_sim_bars) <= 0 else min(len(frame) - 2, entry_i + int(max_sim_bars))
    for i in range(entry_i, last_i + 1):
        px = float(arrays["close"][i])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        if unreal <= -abs(float(coord.emergency_sl)):
            _, exit_fill, exit_fee, _ = omega._try_execution(arrays, int(i), side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "emergency_sl"
            exit_i = int(i)
            break
        x_exit = pd.DataFrame(
            [
                exit_head._position_feature_row(
                    state,
                    entry_state,
                    row_i=i,
                    side=side,
                    entry_price=float(entry_price),
                    entry_i=entry_i,
                    notional=notional,
                    leverage=leverage,
                    take_profit=0.0,
                    stop_loss=0.0,
                    mfe=mfe,
                    mae=mae,
                    unreal=unreal,
                )
            ]
        )
        prob = float(exit_head._predict_loaded_exit_prob(loaded_exit, x_exit, frame.iloc[[i]].reset_index(drop=True), device=device)[0])
        if prob >= float(coord.exit_thr):
            _, exit_fill, exit_fee, _ = omega._try_execution(arrays, int(i), side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "exit_head"
            exit_i = int(i)
            break
    if exit_fill is None:
        exit_reason = "sim_horizon" if int(max_sim_bars) > 0 else "forced_end"
        exit_fill = omega._fill_price(arrays, min(last_i + 1, len(frame) - 1), side, slip_eff, entry=False)
        exit_i = int(last_i)
    raw_exit = (exit_fill - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_fill) / max(entry_price, 1e-12)
    before = cash
    cash = cash * (1.0 + raw_exit * notional)
    cash -= before * exit_fee * notional
    net = float(cash - 1.0)
    reward = net - 0.10 * max(0.0, -mae - 0.035) * notional * leverage
    return reward, {"active": 1, "exit_reason": exit_reason, "exit_i": int(exit_i), "net": net, "win": int(net > 0.0)}


@dataclass
class OfflineData:
    seq: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    weights: np.ndarray


def _build_offline_data(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    seq: np.ndarray,
    dec: pd.DataFrame,
    loaded_exit: dict[str, tuple[exit_head.ExitTabMClassifier, dict[str, Any]]],
    *,
    max_entries: int,
    seed: int,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    action_grid: list[CoordAction],
    max_sim_bars: int,
    min_trade_edge: float,
) -> tuple[OfflineData, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    active_idx = np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - 3))
    if max_entries > 0 and len(active_idx) > max_entries:
        active_idx = np.sort(rng.choice(active_idx, size=int(max_entries), replace=False))
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    xs: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    rewards: list[float] = []
    weights: list[float] = []
    reason_counts: dict[str, int] = {}
    best_rewards: list[float] = []
    for idx in active_idx:
        local: list[tuple[float, CoordAction, dict[str, Any]]] = []
        for action in action_grid:
            reward, meta = _simulate_coord_action(
                frame,
                state,
                arrays,
                dec,
                loaded_exit,
                int(idx),
                action,
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
                device=device,
                max_sim_bars=int(max_sim_bars),
            )
            local.append((float(reward), action, meta))
            reason = str(meta.get("exit_reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        local.sort(key=lambda x: x[0], reverse=True)
        best_rewards.append(float(local[0][0]))
        skip = next(x for x in local if x[1].enter < 0.5)
        if float(local[0][0]) <= float(min_trade_edge):
            # If no counterfactual action clears the minimum edge, teach the coordinator to veto.
            keep = [skip]
        else:
            keep = [x for x in local[:4] if float(x[0]) > float(min_trade_edge)]
            if not any(k[1].enter < 0.5 for k in keep):
                keep.append(skip)
        scale = max(float(np.std([x[0] for x in local])), 1e-4)
        baseline = float(np.median([x[0] for x in local]))
        for reward, action, _meta in keep:
            xs.append(seq[int(idx)])
            acts.append(_action_to_vec(action))
            rewards.append(float(reward))
            weights.append(float(np.exp(np.clip((float(reward) - baseline) / scale, -4.0, 4.0))))
    if not xs:
        raise RuntimeError("empty Mamba-SAC offline dataset")
    r = np.asarray(rewards, dtype=np.float32)
    return (
        OfflineData(
            seq=np.asarray(xs, dtype=np.float32),
            actions=np.asarray(acts, dtype=np.float32),
            rewards=r,
            weights=np.asarray(weights, dtype=np.float32),
        ),
        {
            "used_entries": int(len(active_idx)),
            "samples": int(len(xs)),
            "best_reward_mean": float(np.mean(best_rewards)) if best_rewards else 0.0,
            "reward_mean": float(np.mean(r)),
            "reward_std": float(np.std(r)),
            "counterfactual_exit_reasons": reason_counts,
        },
    )


class MambaEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, emb_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.mamba(h)
        return self.out(self.norm(h[:, -1, :]))


class CoordinatorActor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.enc = MambaEncoder(input_dim, d_model, emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim, 96), nn.SiLU(), nn.Linear(96, 4))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.enc(seq)))


class CoordinatorCritic(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.enc = MambaEncoder(input_dim, d_model, emb_dim)
        self.q = nn.Sequential(nn.Linear(emb_dim + 4, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.enc(seq), action], dim=1)).squeeze(1)


def _train_coordinator(data: OfflineData, *, device: torch.device, steps: int, batch_size: int, lr: float) -> tuple[CoordinatorActor, dict[str, Any]]:
    actor = CoordinatorActor(data.seq.shape[-1]).to(device)
    q1 = CoordinatorCritic(data.seq.shape[-1]).to(device)
    q2 = CoordinatorCritic(data.seq.shape[-1]).to(device)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=2e-5)
    opt_q = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=lr, weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.seq), torch.from_numpy(data.actions), torch.from_numpy(data.rewards), torch.from_numpy(data.weights))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            seq_b, act_b, rew_b, w_b = next(it)
        except StopIteration:
            it = iter(dl)
            seq_b, act_b, rew_b, w_b = next(it)
        seq_b = seq_b.to(device)
        act_b = act_b.to(device)
        rew_b = rew_b.to(device)
        w_b = w_b.to(device)
        q1_pred = q1(seq_b, act_b)
        q2_pred = q2(seq_b, act_b)
        q_loss = torch.nn.functional.smooth_l1_loss(q1_pred, rew_b) + torch.nn.functional.smooth_l1_loss(q2_pred, rew_b)
        opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 3.0)
        opt_q.step()
        pred_act = actor(seq_b)
        q_val = torch.minimum(q1(seq_b, pred_act), q2(seq_b, pred_act))
        bc = (((pred_act - act_b) ** 2).mean(dim=1) * w_b).sum() / torch.clamp(w_b.sum(), min=1.0)
        # SAC-style actor: maximize conservative Q, AWAC/BC anchors the policy to profitable counterfactuals.
        a_loss = -q_val.mean() + 0.35 * bc
        opt_a.zero_grad(set_to_none=True)
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 3.0)
        opt_a.step()
        if step % 250 == 0:
            last = {"step": int(step), "critic_loss": float(q_loss.detach().cpu()), "actor_loss": float(a_loss.detach().cpu()), "bc_loss": float(bc.detach().cpu())}
    return actor.cpu(), last


@torch.no_grad()
def _policy_actions(actor: CoordinatorActor, seq: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(seq), 2048):
        xb = torch.from_numpy(seq[start : start + 2048]).to(device)
        out.append(actor(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _replay_policy(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    seq: np.ndarray,
    dec: pd.DataFrame,
    actor: CoordinatorActor,
    loaded_exit: dict[str, tuple[exit_head.ExitTabMClassifier, dict[str, Any]]],
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    max_sim_bars: int,
) -> dict[str, Any]:
    actions = _policy_actions(actor, seq, device=device)
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}
    active = omega._active(dec)
    i = 0
    while i < len(frame) - 2:
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if not bool(active[i]):
            i += 1
            continue
        coord = _vec_to_action(actions[i])
        reward, meta = _simulate_coord_action(
            frame,
            state,
            arrays,
            dec,
            loaded_exit,
            i,
            coord,
            fee=fee,
            slip=slip,
            cost_mult=cost_mult,
            device=device,
            max_sim_bars=int(max_sim_bars),
        )
        if int(meta.get("active", 0)) == 0:
            i += 1
            continue
        before = cash
        cash = cash * (1.0 + float(meta["net"]))
        trades += 1
        wins += int(cash > before)
        side = int(dec.iloc[i].get("side", 0) or 0)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        reason = str(meta.get("exit_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
        i = max(int(meta.get("exit_i", i)) + 1, i + 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--exit-head-bundle", type=Path, default=DEFAULT_EXIT_HEAD)
    ap.add_argument("--fast-grid", action="store_true")
    ap.add_argument("--train-max-sim-bars", type=int, default=384)
    ap.add_argument("--eval-max-sim-bars", type=int, default=0)
    ap.add_argument("--min-trade-edge", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=260603)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames()
    fee, slip = omega._load_fee_slip()
    state_cols = [c for c in frames["s_train"].columns if c not in {"timestamp"}]
    norm = _fit_norm(frames["s_train"][state_cols])
    x_train = _apply_norm(frames["s_train"][state_cols], norm)
    x_val = _apply_norm(frames["s_val"][state_cols], norm)
    x_oos = _apply_norm(frames["s_oos"][state_cols], norm)
    seq_train = _rolling_sequences(x_train, int(args.seq_len))
    seq_val = _rolling_sequences(x_val, int(args.seq_len))
    seq_oos = _rolling_sequences(x_oos, int(args.seq_len))
    loaded_exit = _load_exit_models(Path(args.exit_head_bundle), device=device)
    action_grid = FAST_ACTION_GRID if bool(args.fast_grid) else ACTION_GRID
    data, data_diag = _build_offline_data(
        frames["train_df"],
        frames["s_train"],
        seq_train,
        frames["train_fixed"],
        loaded_exit,
        max_entries=int(args.max_train_entries),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        action_grid=action_grid,
        max_sim_bars=int(args.train_max_sim_bars),
        min_trade_edge=float(args.min_trade_edge),
    )
    print(json.dumps({"stage": "coordinator_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    actor, train_diag = _train_coordinator(data, device=device, steps=int(args.steps), batch_size=int(args.batch_size), lr=float(args.lr))
    val_metrics = _replay_policy(
        frames["val_df"],
        frames["s_val"],
        seq_val,
        frames["val_fixed"],
        actor,
        loaded_exit,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        max_sim_bars=int(args.eval_max_sim_bars),
    )
    oos_metrics = _replay_policy(
        frames["oos_df"],
        frames["s_oos"],
        seq_oos,
        frames["oos_fixed"],
        actor,
        loaded_exit,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        max_sim_bars=int(args.eval_max_sim_bars),
    )
    no_risk_report = ROOT / "tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_exit_head_nohold_20260603_no_risk_template_edge100_seed260603/report.json"
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen soft_floor_0p00 Decision/Quality + frozen cost-aware Exit Head feed a Mamba sequence encoder. Offline SAC/AWAC coordinator outputs entry veto, exit threshold, exposure multiplier, and emergency SL. TP/SL/max-hold/cooldown risk template is removed; notional/leverage are only accounting exposure bases.",
        "state_columns": list(norm["columns"]),
        "seq_len": int(args.seq_len),
        "action": {"dims": ["enter_gate", "exit_threshold", "exposure_multiplier", "emergency_sl"], "grid_size_for_counterfactual": len(action_grid), "fast_grid": bool(args.fast_grid)},
        "exit_head_bundle": str(args.exit_head_bundle),
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "train_max_sim_bars": int(args.train_max_sim_bars),
            "eval_max_sim_bars": int(args.eval_max_sim_bars),
            "min_trade_edge": float(args.min_trade_edge),
            "data_diag": data_diag,
            "train_diag": train_diag,
        },
        "results": {"validation": val_metrics, "oos": oos_metrics},
        "baseline_note": {"no_risk_template_exit_head_report": str(no_risk_report)},
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult)},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "model": str(out_dir / "mamba_sac_coordinator.pt")},
    }
    torch.save({"actor_state_dict": actor.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": list(norm["columns"])}, out_dir / "mamba_sac_coordinator.pt")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
