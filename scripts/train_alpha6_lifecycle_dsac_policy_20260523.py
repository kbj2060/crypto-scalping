#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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

from scripts.train_alpha6_dsac_ensemble_router_20260523 import (  # noqa: E402
    DiscreteSAC,
    Replay,
    RouterData,
    _load_router_data,
    _load_router_data_oof,
)


ACTION_NAMES = [
    "flat",
    "long_small",
    "long_normal",
    "short_small",
    "short_normal",
    "hold",
    "reduce",
    "close",
    "reverse_long",
    "reverse_short",
]


class LifecycleEnv:
    """Direct lifecycle policy: DSAC controls position changes, experts are state only."""

    def __init__(
        self,
        data: RouterData,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        fee: float = 0.0004,
        slip: float = 0.00015,
        small_notional: float = 0.10,
        normal_notional: float = 0.25,
        max_notional: float = 0.35,
        entry_bonus: float = 0.003,
        flat_edge_penalty: float = 0.00025,
        hold_theta: float = 0.00004,
        reduce_bonus: float = 0.0005,
    ) -> None:
        self.data = data
        self.indices = indices.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
        self.fee = float(fee)
        self.slip = float(slip)
        self.small_notional = float(small_notional)
        self.normal_notional = float(normal_notional)
        self.max_notional = float(max_notional)
        self.entry_bonus = float(entry_bonus)
        self.flat_edge_penalty = float(flat_edge_penalty)
        self.hold_theta = float(hold_theta)
        self.reduce_bonus = float(reduce_bonus)
        self.close = pd.to_numeric(data.frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.data.base_x.shape[1] + 10)

    def reset(self) -> np.ndarray:
        self.ptr = 0
        self.i = int(self.indices[self.ptr])
        self.cash = 1.0
        self.peak = 1.0
        self.mdd = 0.0
        self.side = 0
        self.notional = 0.0
        self.entry_px = 0.0
        self.entry_idx = -1
        self.mae = 0.0
        self.mfe = 0.0
        self.trades = 0
        self.wins = 0
        self.long_entries = 0
        self.short_entries = 0
        self.delta_turnover = 0.0
        self.action_hist: list[int] = []
        return self._state()

    def _edge_strength(self, idx: int) -> float:
        names = self.data.base_names
        if "quality_top" not in names:
            return 0.0
        return float(self.data.base_x[idx, names.index("quality_top")])

    def _equity(self, idx: int | None = None) -> float:
        if idx is None:
            idx = self.i
        if self.side == 0 or self.notional <= 0:
            return float(self.cash)
        raw = (float(self.close[idx]) - self.entry_px) / max(self.entry_px, 1e-12)
        return float(self.cash + raw * self.side * self.notional)

    def _position_features(self) -> np.ndarray:
        if self.side == 0:
            return np.zeros(10, dtype=np.float32)
        raw = (float(self.close[self.i]) - self.entry_px) / max(self.entry_px, 1e-12) * self.side
        hold = max(0, self.i - self.entry_idx)
        giveback = max(0.0, self.mfe - max(raw * self.notional, 0.0))
        return np.asarray(
            [
                self.side,
                self.notional,
                hold / 96.0,
                raw,
                raw / max(float(self.data.frame.get("atr14_pct", pd.Series([0.003])).iloc[self.i]), 1e-9),
                self.mae,
                self.mfe,
                giveback / max(self.mfe, 1e-9),
                self._edge_strength(self.i),
                self.delta_turnover,
            ],
            dtype=np.float32,
        )

    def _state(self) -> np.ndarray:
        base = (self.data.base_x[self.i] - self.mean) / self.std
        return np.concatenate([base, self._position_features()]).astype(np.float32)

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        if self.side == 0:
            mask[[0, 1, 2, 3, 4]] = True
        elif self.side > 0:
            mask[[0, 1, 2, 5, 6, 7, 9]] = True
        else:
            mask[[0, 3, 4, 5, 6, 7, 8]] = True
        return mask

    def _realize_reduce(self, reduce_notional: float, px: float) -> float:
        reduce_notional = float(np.clip(reduce_notional, 0.0, self.notional))
        if reduce_notional <= 0 or self.side == 0:
            return 0.0
        raw = (px - self.entry_px) / max(self.entry_px, 1e-12) * self.side
        pnl = raw * reduce_notional
        cost = (self.fee + self.slip) * reduce_notional
        self.cash += pnl - cost
        self.notional -= reduce_notional
        self.delta_turnover += reduce_notional
        if reduce_notional > 0:
            self.trades += 1
            self.wins += int(pnl - cost > 0)
        if self.notional <= 1e-9:
            self.side = 0
            self.notional = 0.0
            self.entry_px = 0.0
            self.entry_idx = -1
            self.mae = 0.0
            self.mfe = 0.0
        return pnl - cost

    def _set_position(self, target_side: int, target_notional: float, px: float) -> float:
        target_notional = float(np.clip(target_notional, 0.0, self.max_notional))
        reward_bonus = 0.0
        if target_side == 0 or target_notional <= 0:
            if self.side != 0:
                self._realize_reduce(self.notional, px)
            return reward_bonus
        if self.side != 0 and self.side != target_side:
            self._realize_reduce(self.notional, px)
        if self.side == 0:
            self.side = int(target_side)
            self.notional = target_notional
            self.entry_px = float(px)
            self.entry_idx = int(self.i)
            self.cash -= (self.fee + self.slip) * target_notional
            self.delta_turnover += target_notional
            self.long_entries += int(self.side > 0)
            self.short_entries += int(self.side < 0)
            reward_bonus += self.entry_bonus
            return reward_bonus
        if self.side == target_side:
            if target_notional > self.notional:
                add = target_notional - self.notional
                self.entry_px = (self.entry_px * self.notional + px * add) / max(target_notional, 1e-12)
                self.cash -= (self.fee + self.slip) * add
                self.delta_turnover += add
                self.notional = target_notional
                reward_bonus += self.entry_bonus * 0.35
            elif target_notional < self.notional:
                realized = self._realize_reduce(self.notional - target_notional, px)
                if realized > 0:
                    reward_bonus += self.reduce_bonus
        return reward_bonus

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = int(np.clip(action, 0, len(ACTION_NAMES) - 1))
        mask = self.valid_action_mask()
        if not mask[action]:
            action = 0
        before = self._equity(self.i)
        px = float(self.close[self.i])
        self.delta_turnover = 0.0
        bonus = 0.0
        if action == 0:
            if self.side == 0 and self._edge_strength(self.i) > 0:
                bonus -= self.flat_edge_penalty
            else:
                bonus += self._set_position(0, 0.0, px)
        elif action == 1:
            bonus += self._set_position(1, self.small_notional, px)
        elif action == 2:
            bonus += self._set_position(1, self.normal_notional, px)
        elif action == 3:
            bonus += self._set_position(-1, self.small_notional, px)
        elif action == 4:
            bonus += self._set_position(-1, self.normal_notional, px)
        elif action == 6:
            bonus += self._set_position(self.side, self.notional * 0.5, px)
        elif action == 7:
            bonus += self._set_position(0, 0.0, px)
        elif action == 8:
            bonus += self._set_position(1, self.normal_notional, px)
        elif action == 9:
            bonus += self._set_position(-1, self.normal_notional, px)
        self.ptr += 1
        done = self.ptr >= len(self.indices) - 1
        if not done:
            self.i = int(self.indices[self.ptr])
        if self.side != 0:
            raw_pnl = (float(self.close[self.i]) - self.entry_px) / max(self.entry_px, 1e-12) * self.side * self.notional
            self.mae = max(self.mae, max(0.0, -raw_pnl))
            self.mfe = max(self.mfe, max(0.0, raw_pnl))
        if done and self.side != 0:
            self._realize_reduce(self.notional, float(self.close[self.i]))
        after = self._equity(self.i)
        self.peak = max(self.peak, after)
        self.mdd = min(self.mdd, after / max(self.peak, 1e-12) - 1.0)
        hold_pen = self.hold_theta * self.notional if self.side != 0 else 0.0
        reward = float(np.clip((after - before) * 100.0, -2.0, 2.0) + bonus - hold_pen)
        self.action_hist.append(action)
        return self._state(), reward, done, {"actual_action": action}

    def summary(self) -> dict[str, Any]:
        return {
            "pnl": float((self.cash - 1.0) * 100.0),
            "mdd": float(self.mdd * 100.0),
            "trades": int(self.trades),
            "wr": float(self.wins / max(self.trades, 1)),
            "long_entries": int(self.long_entries),
            "short_entries": int(self.short_entries),
            "turnover": float(self.delta_turnover),
            "action_counts": {ACTION_NAMES[int(k)]: int(v) for k, v in pd.Series(self.action_hist).value_counts().sort_index().items()},
        }


def _run(env: LifecycleEnv, agent: DiscreteSAC | None, fixed: int | None = None) -> dict[str, Any]:
    s = env.reset()
    while True:
        if fixed is not None:
            a = fixed
        elif agent is None:
            a = 0
        else:
            a = agent.act(s, deterministic=True, mask=env.valid_action_mask())
        ns, _, done, _ = env.step(a)
        s = ns
        if done:
            break
    return env.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description="Lifecycle DSAC policy over Alpha6 expert state features.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_lifecycle_dsac_policy_20260523")
    ap.add_argument("--oof-folds", type=int, default=0)
    ap.add_argument("--oof-iterations", type=int, default=80)
    ap.add_argument("--oof-exit-iterations", type=int, default=40)
    ap.add_argument("--max-train-rows", type=int, default=20000)
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--updates-per-step", type=int, default=1)
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
    std = data.base_x[train_idx].std(axis=0)
    train_env = LifecycleEnv(data, train_idx, mean, std)
    agent = DiscreteSAC(train_env.state_dim, len(ACTION_NAMES), args.device, alpha_init=0.03, alpha_min=0.003, alpha_max=0.20)
    replay = Replay(capacity=300_000)
    step = 0
    last: dict[str, float] = {}
    for ep in range(int(args.episodes)):
        s = train_env.reset()
        while True:
            if step < int(args.warmup):
                valid = np.flatnonzero(train_env.valid_action_mask())
                a = int(np.random.choice(valid)) if len(valid) else 0
            else:
                a = agent.act(s, deterministic=False, mask=train_env.valid_action_mask())
            ns, r, done, info = train_env.step(a)
            replay.add(s, int(info.get("actual_action", a)), r, ns, done)
            s = ns
            step += 1
            if step >= int(args.warmup):
                for _ in range(int(args.updates_per_step)):
                    update = agent.update(replay, int(args.batch_size))
                    if update:
                        last = update
            if done:
                break
        print(f"[lifecycle-dsac] episode={ep+1}/{args.episodes} train={train_env.summary()} update={last}", flush=True)
    val_env = LifecycleEnv(data, val_idx, mean, std)
    result = {
        "model_id": "alpha6_lifecycle_dsac_policy_20260523",
        "variant": args.variant,
        "action_names": ACTION_NAMES,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "state_dim": int(train_env.state_dim),
        "oof_folds": int(args.oof_folds),
        "oof_iterations": int(args.oof_iterations),
        "oof_exit_iterations": int(args.oof_exit_iterations),
        "episodes": int(args.episodes),
        "last_update": last,
        "val_backtest": _run(val_env, agent),
        "audit": {
            "router_train_uses_expert_in_sample_predictions": False if int(args.oof_folds) > 1 else True,
            "policy_type": "direct lifecycle discrete SAC; experts are state features only",
            "position_delta_cost": "fee+slip charged on every notional delta, including resize/reduce/reverse",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    joblib.dump({"actor": agent.actor.cpu().state_dict(), "critic": agent.critic.cpu().state_dict(), "result": result}, args.out_dir / "lifecycle_dsac.joblib")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
