#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FEATURE_COLS  # noqa: E402
from ensemble.train_rl_dsac_agent import DSACAgent  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    _exit_probability_vec,
    _feature_vec_fast,
    backtest_no_limit_exit,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)


DEFAULT_CKPT_DIR = ROOT / "data/ensemble/ckpt/dsac_layer_rl"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_layer_rl_2026.json"
REGIME_NAMES = ("bull", "bear", "chop", "whipsaw", "normal")
RL_CONTEXT_COLS = (
    "rl_pos_side",
    "rl_age_frac",
    "rl_age_log",
    "rl_unrealized",
    "rl_peak_unrealized",
    "rl_dd_from_peak",
    "rl_notional_frac",
    "rl_leverage_frac",
    "rl_entry_quality",
    "rl_entry_confidence",
    "rl_current_same_side",
    "rl_current_opposite_side",
    "rl_current_quality",
    "rl_current_confidence",
)


@dataclass
class LayerSpec:
    name: str
    state_dim: int
    uses_dsac_exit: bool = False
    uses_dsac_entry: bool = False
    uses_base_entry: bool = True


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _classes_proba(model: Any, arr: np.ndarray, cls: int = 1) -> float:
    proba = model.predict_proba(arr.reshape(1, -1).astype(np.float32, copy=False))
    classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
    if int(cls) not in classes:
        return 0.0
    return float(proba[0, int(np.flatnonzero(classes == int(cls))[0])])


def _fit_feature_norm(train_feat: pd.DataFrame, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = train_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if int(state_dim) > len(FEATURE_COLS):
        extra = int(state_dim) - len(FEATURE_COLS)
        mean = np.concatenate([mean, np.zeros(extra, dtype=np.float32)])
        std = np.concatenate([std, np.ones(extra, dtype=np.float32)])
    return mean, std


class DSACLayerEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        policy: dict[str, Any],
        exit_model: Any,
        *,
        layer: str,
        entry_cfg: dict[str, Any],
        risk_cfg: dict[str, Any],
        exit_cfg: dict[str, Any],
        mean: np.ndarray,
        std: np.ndarray,
        fee: float,
        slip: float,
        reward_scale: float,
        episode_len: int | None,
        random_start: bool,
        seed: int,
    ):
        self.df = df.reset_index(drop=True)
        self.policy = policy
        self.exit_model = exit_model
        self.layer = str(layer)
        self.entry_cfg = dict(entry_cfg)
        self.risk_cfg = dict(risk_cfg)
        self.exit_cfg = dict(exit_cfg)
        self.mean = mean.astype(np.float32)
        self.std = np.where(std.astype(np.float32) < 1e-6, 1.0, std.astype(np.float32))
        self.fee = float(fee)
        self.slip = float(slip)
        self.reward_scale = float(reward_scale)
        self.episode_len = None if episode_len is None or episode_len <= 0 else int(episode_len)
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(int(seed))

        self.base_feat, self.decisions, self.close, self.fill_px = _base_frame(self.df, self.policy, self.entry_cfg)
        self.base_values = self.base_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.base_values_model = self.base_feat.to_numpy(dtype=np.float32, copy=False)
        self.actions = self.decisions["action"].astype(int).to_numpy()
        self.sides = self.decisions["side"].astype(int).to_numpy()
        self.notionals = pd.to_numeric(self.decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.leverages = pd.to_numeric(self.decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        self.cooldowns = pd.to_numeric(self.decisions["cooldown_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        self.qualities = pd.to_numeric(self.decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.confs = pd.to_numeric(self.decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.day_codes = (
            pd.to_datetime(self.df["timestamp"], errors="coerce").dt.floor("D").astype("int64").to_numpy()
            if "timestamp" in self.df.columns
            else (np.arange(len(self.df), dtype=np.int64) // 288).astype(np.int64)
        )
        active_n = self.notionals[self.notionals > 0.0]
        active_l = self.leverages[self.notionals > 0.0]
        self.default_notional = float(np.median(active_n)) if active_n.size else 0.55
        self.default_leverage = float(np.median(active_l)) if active_l.size else 1.5
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(len(self.mean))

    def reset(self) -> np.ndarray:
        max_start = max(0, len(self.df) - (self.episode_len or len(self.df)) - 3)
        self.start = int(self.rng.integers(0, max_start + 1)) if self.random_start and max_start > 0 else 0
        self.end = min(len(self.df) - 3, self.start + (self.episode_len or len(self.df) - 3))
        self.i = int(self.start)
        self.cash = 1.0
        self.peak = 1.0
        self.mdd = 0.0
        self.pos = 0
        self.entry_price = 0.0
        self.entry_equity = 1.0
        self.entry_idx = 0
        self.notional = 0.0
        self.leverage = 1.0
        self.model_cooldown = 0
        self.cooldown_left = 0
        self.loss_cooldown_left = 0
        self.loss_streak = 0
        self.peak_unrealized = 0.0
        self.entry_quality = 0.0
        self.entry_confidence = 0.0
        self.trades = self.wins = self.long_entries = self.short_entries = 0
        self.notional_sum = self.leverage_sum = 0.0
        self.exits: dict[str, int] = {}
        self.entry_blocks: dict[str, int] = {}
        self.day_key = None
        self.daily_start_cash = 1.0
        self.daily_peak_eq = 1.0
        self.daily_trades = 0
        return self._state()

    def _days(self) -> float:
        if "timestamp" not in self.df.columns or self.end <= self.start:
            return max((self.end - self.start + 1) / 288.0, 1e-8)
        a = self.df["timestamp"].iloc[self.start]
        b = self.df["timestamp"].iloc[self.end]
        return max((b - a).total_seconds() / 86400.0, 1e-8)

    def _block(self, reason: str) -> None:
        self.entry_blocks[reason] = self.entry_blocks.get(reason, 0) + 1

    def _fill_price(self, idx: int, side: int, *, entry: bool) -> float:
        px = float(self.fill_px[int(np.clip(idx, 0, len(self.fill_px) - 1))])
        if side > 0:
            return px * (1.0 + self.slip if entry else 1.0 - self.slip)
        return px * (1.0 - self.slip if entry else 1.0 + self.slip)

    def _mark(self, idx: int) -> tuple[float, float]:
        if self.pos == 0:
            return self.cash, 0.0
        px = float(self.close[int(np.clip(idx, 0, len(self.close) - 1))])
        if self.pos > 0:
            raw = (px * (1.0 - self.slip) - self.entry_price) / max(self.entry_price, 1e-12)
        else:
            raw = (self.entry_price - px * (1.0 + self.slip)) / max(self.entry_price, 1e-12)
        unreal = raw * self.notional
        return self.cash * (1.0 + unreal), float(unreal)

    def _regime(self, idx: int) -> str:
        vals = {}
        row = self.df.iloc[int(np.clip(idx, 0, len(self.df) - 1))]
        for r in REGIME_NAMES:
            vals[r] = _safe_float(row.get(f"regime_{r}_id", row.get(f"regime_{r}", 0.0)), 0.0)
        if max(abs(v) for v in vals.values()) <= 1e-12:
            return "normal"
        return max(vals, key=vals.get)

    def _state_raw(self) -> np.ndarray:
        idx = int(np.clip(self.i, 0, len(self.base_values) - 1))
        feat = self.base_values[idx]
        if self.state_dim <= len(FEATURE_COLS):
            return feat.astype(np.float32, copy=False)
        current_side = int(self.sides[idx])
        age = max(0, self.i - self.entry_idx) if self.pos != 0 else 0
        _, unreal = self._mark(idx)
        ctx = np.asarray(
            [
                float(self.pos),
                float(np.clip(age / 288.0, 0.0, 1.0)),
                float(np.log1p(age) / np.log1p(288.0)),
                float(np.clip(unreal, -1.0, 1.0)),
                float(np.clip(self.peak_unrealized, -1.0, 1.0)),
                float(np.clip(self.peak_unrealized - unreal, 0.0, 1.0)),
                float(np.clip(self.notional / 3.6, 0.0, 2.0)),
                float(np.clip(self.leverage / 5.0, 0.0, 2.0)),
                float(self.entry_quality),
                float(self.entry_confidence),
                float(current_side == self.pos and self.pos != 0),
                float(current_side == -self.pos and self.pos != 0),
                float(self.qualities[idx]),
                float(self.confs[idx]),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat, ctx], axis=0).astype(np.float32, copy=False)

    def _state(self) -> np.ndarray:
        raw = self._state_raw()
        return np.nan_to_num((raw - self.mean) / self.std, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def _base_exit_proba(self, idx: int, unreal: float) -> float:
        age = max(0, int(idx - self.entry_idx))
        row = _feature_vec_fast(
            self.base_values_model,
            self.sides,
            self.qualities,
            self.confs,
            i=int(idx),
            side=int(self.pos),
            age=int(age),
            unrealized=float(unreal),
            peak_unrealized=float(self.peak_unrealized),
            notional=float(self.notional),
            leverage=float(self.leverage),
            entry_quality=float(self.entry_quality),
            entry_confidence=float(self.entry_confidence),
        )
        return _exit_probability_vec(self.exit_model, row)

    def _close_position(self, idx: int, reason: str) -> None:
        exit_price = self._fill_price(min(idx + 1, len(self.df) - 1), self.pos, entry=False)
        raw = (exit_price - self.entry_price) / max(self.entry_price, 1e-12) if self.pos > 0 else (self.entry_price - exit_price) / max(self.entry_price, 1e-12)
        before = self.cash
        self.cash = self.cash * (1.0 + raw * self.notional)
        self.cash -= before * self.fee * self.notional
        self.trades += 1
        self.daily_trades += 1
        is_win = self.cash > self.entry_equity
        self.wins += int(is_win)
        self.loss_streak = 0 if is_win else self.loss_streak + 1
        if not is_win:
            self.loss_cooldown_left = max(self.loss_cooldown_left, int(self.risk_cfg.get("loss_cooldown_bars", 0)))
        self.exits[reason] = self.exits.get(reason, 0) + 1
        self.pos = 0
        self.entry_price = 0.0
        self.notional = 0.0
        self.leverage = 1.0
        self.cooldown_left = int(self.model_cooldown)
        self.model_cooldown = 0
        self.peak_unrealized = 0.0
        self.entry_quality = 0.0
        self.entry_confidence = 0.0

    def _risk_allows_entry(self, eq: float, daily_realized: float, daily_dd: float) -> bool:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            self._block("model_cooldown")
            return False
        if self.loss_cooldown_left > 0:
            self.loss_cooldown_left -= 1
            self._block("loss_cooldown")
            return False
        if self.daily_trades >= int(self.risk_cfg.get("max_daily_trades", 999999)):
            self._block("daily_trade_budget")
            return False
        if daily_realized <= -abs(float(self.risk_cfg.get("daily_loss_limit", 0.0))):
            self._block("daily_loss_lock")
            return False
        if daily_dd >= abs(float(self.risk_cfg.get("daily_dd_limit", 0.0))):
            self._block("daily_dd_lock")
            return False
        return True

    def _open_position(self, idx: int, side: int, notional: float, leverage: float, account_dd: float, daily_realized: float) -> None:
        n = float(notional)
        if account_dd >= float(self.risk_cfg.get("global_dd_cut", 999.0)):
            n *= float(self.risk_cfg.get("global_dd_mult", 1.0))
        if self.loss_streak >= int(self.risk_cfg.get("loss_streak_soft", 999999)):
            steps = self.loss_streak - int(self.risk_cfg.get("loss_streak_soft", 999999)) + 1
            n *= float(self.risk_cfg.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(self.risk_cfg.get("daily_profit_boost_start", 999.0)):
            n *= float(self.risk_cfg.get("daily_profit_boost_mult", 1.0))
        n = float(np.clip(n, 0.0, float(self.risk_cfg.get("max_notional", 3.6))))
        if n <= 1e-8 or side == 0:
            self._block("zero_notional")
            return
        self.pos = int(side)
        self.entry_price = self._fill_price(min(idx + 1, len(self.df) - 1), self.pos, entry=True)
        self.entry_equity = self.cash
        self.entry_idx = int(idx)
        self.notional = n
        self.leverage = float(leverage)
        self.model_cooldown = int(self.cooldowns[idx]) if idx < len(self.cooldowns) else 0
        self.cash -= self.cash * self.fee * self.notional
        self.long_entries += int(self.pos > 0)
        self.short_entries += int(self.pos < 0)
        self.notional_sum += self.notional
        self.leverage_sum += self.leverage
        self.peak_unrealized = 0.0
        self.entry_quality = float(self.qualities[idx]) if idx < len(self.qualities) else 0.0
        self.entry_confidence = float(self.confs[idx]) if idx < len(self.confs) else 0.0

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        idx = int(self.i)
        eq0, unreal0 = self._mark(idx)
        key = int(self.day_codes[idx])
        if key != self.day_key:
            self.day_key = key
            self.daily_start_cash = max(eq0, 1e-12)
            self.daily_peak_eq = max(eq0, 1e-12)
            self.daily_trades = 0
        self.peak = max(self.peak, eq0)
        self.daily_peak_eq = max(self.daily_peak_eq, eq0)
        self.mdd = min(self.mdd, eq0 / max(self.peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq0 / max(self.peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq0 / max(self.daily_peak_eq, 1e-12))
        daily_realized = self.cash / max(self.daily_start_cash, 1e-12) - 1.0
        a = float(np.clip(action, -1.0, 1.0))

        if self.pos != 0:
            self.peak_unrealized = max(self.peak_unrealized, unreal0)
            age = idx - self.entry_idx
            reason = ""
            if self.layer in {"exit_replace", "full_lifecycle"}:
                if self.layer == "exit_replace":
                    if a > 0.20 and age >= int(self.exit_cfg.get("min_exit_age", 6)):
                        reason = "dsac_exit"
                else:
                    if (self.pos > 0 and a < -0.20) or (self.pos < 0 and a > 0.20):
                        if age >= int(self.exit_cfg.get("min_exit_age", 6)):
                            reason = "dsac_lifecycle_exit"
            else:
                if age >= int(self.exit_cfg.get("min_exit_age", 6)):
                    p_exit = self._base_exit_proba(idx, unreal0)
                    if p_exit >= float(self.exit_cfg.get("exit_threshold", 0.45)):
                        reason = "exit_governor"
            if reason:
                self._close_position(idx, reason)
        else:
            if self._risk_allows_entry(eq0, daily_realized, daily_dd):
                base_active = int(self.actions[idx]) != ACTION_CASH and int(self.sides[idx]) != 0 and float(self.notionals[idx]) > 0.0
                side = 0
                notional = 0.0
                leverage = self.default_leverage
                if self.layer == "entry_replace":
                    if a > 0.20:
                        side = 1
                    elif a < -0.20:
                        side = -1
                    if side:
                        notional = max(float(self.notionals[idx]), self.default_notional * np.clip(abs(a), 0.25, 1.0))
                        leverage = float(self.leverages[idx]) if float(self.leverages[idx]) > 0 else self.default_leverage
                elif self.layer == "entry_filter":
                    if base_active:
                        base_side = int(self.sides[idx])
                        if a * base_side >= -0.10:
                            side = base_side
                            notional = float(self.notionals[idx])
                            leverage = float(self.leverages[idx])
                        else:
                            self._block("dsac_filter_block")
                    else:
                        self._block("cash_signal")
                elif self.layer == "exposure_scaler":
                    if base_active:
                        mult = float(np.clip(((a + 1.0) / 2.0) * 1.25, 0.0, 1.25))
                        side = int(self.sides[idx]) if mult > 0.05 else 0
                        notional = float(self.notionals[idx]) * mult
                        leverage = float(self.leverages[idx])
                    else:
                        self._block("cash_signal")
                elif self.layer == "exit_replace":
                    if base_active:
                        side = int(self.sides[idx])
                        notional = float(self.notionals[idx])
                        leverage = float(self.leverages[idx])
                    else:
                        self._block("cash_signal")
                elif self.layer == "full_lifecycle":
                    if a > 0.20:
                        side = 1
                    elif a < -0.20:
                        side = -1
                    if side:
                        notional = max(float(self.notionals[idx]), self.default_notional * np.clip(abs(a), 0.25, 1.0))
                        leverage = float(self.leverages[idx]) if float(self.leverages[idx]) > 0 else self.default_leverage
                self._open_position(idx, side, notional, leverage, account_dd, daily_realized)

        self.i += 1
        done = bool(self.i >= self.end or self.cash <= 0.05)
        if done and self.pos != 0:
            self._close_position(self.i, "forced_end")
        eq1, _ = self._mark(self.i)
        reward = float(np.clip((eq1 / max(eq0, 1e-12) - 1.0) * self.reward_scale, -5.0, 5.0))
        return self._state(), reward, done, {"equity": float(eq1), "regime": self._regime(idx)}

    def metrics(self) -> dict[str, Any]:
        entries = max(self.long_entries + self.short_entries, 1)
        return {
            "pnl": float((self.cash - 1.0) * 100.0),
            "mdd": float(self.mdd * 100.0),
            "trades": int(self.trades),
            "wr": float(self.wins / max(self.trades, 1)),
            "trades_per_day": float(self.trades / self._days()),
            "long_entries": int(self.long_entries),
            "short_entries": int(self.short_entries),
            "avg_notional": float(self.notional_sum / entries),
            "avg_leverage": float(self.leverage_sum / entries),
            "entry_blocks": dict(self.entry_blocks),
            "exits": dict(self.exits),
        }


def _run_eval(env: DSACLayerEnv, agent: DSACAgent) -> dict[str, Any]:
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, deterministic=True)
        state, _, done, _ = env.step(action)
    return env.metrics()


def _train_one(
    layer: str,
    train_env: DSACLayerEnv,
    eval_env: DSACLayerEnv,
    *,
    episodes: int,
    batch_size: int,
    min_buffer: int,
    update_freq: int,
    hidden_dim: int,
    device: str,
    seed: int,
) -> tuple[DSACAgent, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    agent = DSACAgent(
        state_dim=train_env.state_dim,
        hidden_dim=int(hidden_dim),
        gamma=0.99,
        n_quantiles=32,
        cvar_frac=0.40,
        pessimism_min_weight=0.65,
        dynamic_entropy=True,
        anti_flat_lambda=0.04,
        anti_flat_min_abs=0.10,
        direction_reg_lambda=0.03,
        side_balance_lambda=0.04,
        device=device,
    )
    best = {"score": -1e18, "eval": None, "episode": 0}
    global_step = 0
    last_stats: dict[str, Any] = {}
    for ep in range(1, int(episodes) + 1):
        s = train_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if len(agent.memory) < int(min_buffer):
                a = float(np.random.uniform(-1.0, 1.0))
            else:
                a = agent.act(s, deterministic=False)
            ns, r, done, info = train_env.step(a)
            progress = (train_env.i - train_env.start) / max(train_env.end - train_env.start, 1)
            agent.memory.push(s, a, r, ns, done, regime=str(info.get("regime", "normal")), progress=progress)
            s = ns
            ep_reward += float(r)
            global_step += 1
            if global_step % int(update_freq) == 0 and len(agent.memory) >= int(min_buffer):
                last_stats = agent.update(int(batch_size))
        ev = _run_eval(eval_env, agent)
        score = float(ev["pnl"]) + 0.5 * float(ev["mdd"]) + 0.02 * min(float(ev["trades"]), 800.0)
        if score > best["score"]:
            best = {"score": score, "eval": ev, "episode": ep}
            best_actor = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        if ep == 1 or ep == int(episodes) or ep % max(1, int(episodes) // 4) == 0:
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "episode": ep,
                        "reward": round(ep_reward, 4),
                        "eval_pnl": round(float(ev["pnl"]), 4),
                        "eval_mdd": round(float(ev["mdd"]), 4),
                        "trades": int(ev["trades"]),
                        "memory": len(agent.memory),
                        "stats": {k: round(float(v), 5) for k, v in (last_stats or {}).items() if isinstance(v, (int, float))},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if "best_actor" in locals():
        agent.actor.load_state_dict(best_actor)
    final_eval = _run_eval(eval_env, agent)
    return agent, {"best": best, "final_eval": final_eval, "updates": int(agent._updates), "memory": int(len(agent.memory))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True DSAC retraining for layer replacement candidates under the current governor accounting.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--candidates", default="entry_filter,exit_replace,entry_replace,exposure_scaler,full_lifecycle")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--episode-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-buffer", type=int, default=2048)
    p.add_argument("--update-freq", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--reward-scale", type=float, default=50.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_feat, _, _, _ = _base_frame(train_df, policy, entry_cfg)

    candidate_names = [x.strip() for x in str(args.candidates).split(",") if x.strip()]
    state_dims = {
        "entry_filter": len(FEATURE_COLS),
        "entry_replace": len(FEATURE_COLS),
        "exposure_scaler": len(FEATURE_COLS),
        "exit_replace": len(FEATURE_COLS) + len(RL_CONTEXT_COLS),
        "full_lifecycle": len(FEATURE_COLS) + len(RL_CONTEXT_COLS),
    }
    report: dict[str, Any] = {
        "type": "dsac_layer_rl_2026",
        "note": "True DSAC run: Gaussian actor + distributional twin critic + replay buffer + environment step. The current HF no-limit accounting/risk controls are preserved inside the layer environment.",
        "policy": str(args.policy),
        "exit_bundle": str(args.exit_bundle),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "device": device,
        "episodes": int(args.episodes),
        "episode_len": int(args.episode_len),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "candidates": {},
    }

    baseline_env = DSACLayerEnv(
        eval_df,
        policy,
        exit_model,
        layer="exit_replace",
        entry_cfg=entry_cfg,
        risk_cfg=risk_cfg,
        exit_cfg=exit_cfg,
        mean=_fit_feature_norm(train_feat, len(FEATURE_COLS))[0],
        std=_fit_feature_norm(train_feat, len(FEATURE_COLS))[1],
        fee=float(args.fee),
        slip=float(args.slip),
        reward_scale=float(args.reward_scale),
        episode_len=None,
        random_start=False,
        seed=int(args.seed),
    )
    # Baseline is evaluated through the canonical backtester to avoid any drift from the training env.
    base_feat, base_dec, close, fill = _base_frame(eval_df, policy, entry_cfg)
    base_bt = backtest_no_limit_exit(
        eval_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=(base_feat, base_dec, close, fill),
    )
    report["baseline"] = _compact(base_bt)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    for idx, layer in enumerate(candidate_names):
        if layer not in state_dims:
            raise ValueError(f"unknown candidate: {layer}")
        mean, std = _fit_feature_norm(train_feat, state_dims[layer])
        train_env = DSACLayerEnv(
            train_df,
            policy,
            exit_model,
            layer=layer,
            entry_cfg=entry_cfg,
            risk_cfg=risk_cfg,
            exit_cfg=exit_cfg,
            mean=mean,
            std=std,
            fee=float(args.fee),
            slip=float(args.slip),
            reward_scale=float(args.reward_scale),
            episode_len=int(args.episode_len),
            random_start=True,
            seed=int(args.seed) + idx,
        )
        eval_env = DSACLayerEnv(
            eval_df,
            policy,
            exit_model,
            layer=layer,
            entry_cfg=entry_cfg,
            risk_cfg=risk_cfg,
            exit_cfg=exit_cfg,
            mean=mean,
            std=std,
            fee=float(args.fee),
            slip=float(args.slip),
            reward_scale=float(args.reward_scale),
            episode_len=None,
            random_start=False,
            seed=int(args.seed) + 100 + idx,
        )
        agent, meta = _train_one(
            layer,
            train_env,
            eval_env,
            episodes=int(args.episodes),
            batch_size=int(args.batch_size),
            min_buffer=int(args.min_buffer),
            update_freq=int(args.update_freq),
            hidden_dim=int(args.hidden_dim),
            device=device,
            seed=int(args.seed) + idx,
        )
        ckpt_path = args.ckpt_dir / f"{layer}.pth"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "state_dim": int(state_dims[layer]),
                "state_cols": list(FEATURE_COLS) + (list(RL_CONTEXT_COLS) if state_dims[layer] > len(FEATURE_COLS) else []),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "layer": layer,
                "meta": meta,
            },
            ckpt_path,
        )
        report["candidates"][layer] = {"ckpt": str(ckpt_path), **meta}
        args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ranked = sorted(
        ((name, data["final_eval"]) for name, data in report["candidates"].items()),
        key=lambda x: float(x[1].get("pnl", -1e18)),
        reverse=True,
    )
    report["ranked"] = [{"name": name, **ev} for name, ev in ranked]
    report["decision"] = {
        "best_name": ranked[0][0] if ranked else None,
        "best_pnl": ranked[0][1].get("pnl") if ranked else None,
        "baseline_pnl": report["baseline"].get("pnl"),
        "delta_vs_baseline": float((ranked[0][1].get("pnl") if ranked else 0.0) - report["baseline"].get("pnl", 0.0)),
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "baseline": report["baseline"], "ranked": report["ranked"], "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
