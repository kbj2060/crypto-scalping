"""Shared continuous-action environment pieces for SAC/DSAC variants."""

from __future__ import annotations

import random
from collections import deque
import os

import numpy as np
import pandas as pd

try:
    from ensemble.train_rl_agent import (
        HMM_DIM,
        REGIME_COLS,
        STACK_N,
        STATE_ALPHA,
        STATE_CONF,
        STATE_DIM,
        STATE_ELITE,
        STATE_PRED,
        STATE_SYNTH,
        MultiTimeframeFeatures,
    )
except Exception:
    from ensemble.rl_runtime_primitives import (
        HMM_DIM,
        REGIME_COLS,
        STACK_N,
        STATE_ALPHA,
        STATE_CONF,
        STATE_DIM,
        STATE_ELITE,
        STATE_PRED,
        STATE_SYNTH,
        MultiTimeframeFeatures,
    )

_POS_THRESH = 0.15
_CLOSE_THRESH = 0.05


class SACTradingEnv:
    """Continuous-action trading environment shared by SAC/DSAC variants."""

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        fee=0.0005,
        slip=0.0002,
        phase="train",
        hmm_detector=None,
        mtf_features=None,
        side_mode="both",
        reward_beta=None,
        specialist_pos_thresh=None,
        specialist_close_thresh=None,
        specialist_min_opportunity_move=None,
        specialist_min_breakout=None,
        specialist_idle_penalty=None,
        terminal_reward_scale: float = 1.0,
        terminal_quality_win: float = 0.15,
        terminal_quality_loss: float = 0.05,
    ):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.hmm_detector = hmm_detector
        self.side_mode = "both"
        self.terminal_reward_scale = float(terminal_reward_scale)
        self.terminal_quality_win = float(terminal_quality_win)
        self.terminal_quality_loss = float(terminal_quality_loss)
        # NOTE:
        # - train_rl_dsac_agent.py passes specialist_pos/close thresholds expecting
        #   them to be reflected by the environment execution logic.
        # - Previously, step() used module-level constants only, so these knobs
        #   were effectively ignored.
        self.pos_thresh = float(specialist_pos_thresh) if specialist_pos_thresh is not None else float(_POS_THRESH)
        self.close_thresh = float(specialist_close_thresh) if specialist_close_thresh is not None else float(_CLOSE_THRESH)
        self.specialist_idle_penalty = (
            float(specialist_idle_penalty) if specialist_idle_penalty is not None else None
        )
        # Long-hold shaping (mainly for DSAC both-side): penalize stagnant/adverse long holds.
        self.hold_plateau_start = int(os.getenv("RL_HOLD_PLATEAU_START", "24"))
        self.hold_plateau_pnl_abs = float(os.getenv("RL_HOLD_PLATEAU_PNL_ABS", "0.003"))
        self.hold_plateau_penalty = float(os.getenv("RL_HOLD_PLATEAU_PENALTY", "0.003"))
        self.adverse_hold_start = int(os.getenv("RL_ADVERSE_HOLD_START", "24"))
        self.adverse_hold_pnl_th = float(os.getenv("RL_ADVERSE_HOLD_PNL_TH", "0.004"))
        self.adverse_hold_penalty = float(os.getenv("RL_ADVERSE_HOLD_PENALTY", "0.010"))

        if mtf_features is not None:
            self.mtf = mtf_features
        else:
            self.mtf = MultiTimeframeFeatures(self.df["close"].values.astype(np.float32))

        self.MAX_EPISODE_STEPS = 4096 if phase == "train" else len(self.df) - 1

        feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
        feat_df = (
            self.df.reindex(columns=feat_cols, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        self._feat_np = feat_df.to_numpy(dtype=np.float32)
        self._close_np = self.df["close"].values.astype(np.float32)
        self._high_np = (
            self.df["high"].values.astype(np.float32)
            if "high" in self.df.columns else self._close_np.copy()
        )
        self._low_np = (
            self.df["low"].values.astype(np.float32)
            if "low" in self.df.columns else self._close_np.copy()
        )
        self._open_np = (
            self.df["open"].values.astype(np.float32)
            if "open" in self.df.columns else self._close_np.copy()
        )
        self._m7_tp_price_np = (
            pd.to_numeric(self.df.get("m7_tp_price", 0.0), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._m7_sl_price_np = (
            pd.to_numeric(self.df.get("m7_sl_price", 0.0), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._m7_target_hold_np = (
            pd.to_numeric(self.df.get("m7_target_hold", 0.0), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._n_pred = len(STATE_PRED)
        self._n_conf = len(STATE_CONF)
        self._n_elite = len(STATE_ELITE)
        self._n_alpha = len(STATE_ALPHA)
        self._n_regime = len(REGIME_COLS)
        self._n_synth = len(STATE_SYNTH)
        self._frame_stack = deque(maxlen=STACK_N)

        hmm_cols = ["log_return", "garch_vol_z", "oi_change_rate"]
        self._hmm_obs_np = {
            col: self.df[col].fillna(0).values.astype(np.float32)
            if col in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
            for col in hmm_cols
        }
        self._train_start_by_regime = self._build_train_start_buckets()
        self.reset()

    def _build_train_start_buckets(self):
        buckets = {k: [] for k in ["bull", "bear", "chop", "whipsaw", "normal"]}
        if self.phase != "train":
            return buckets
        max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        if max_start <= 0 or not all(c in self.df.columns for c in REGIME_COLS):
            return buckets
        regime_mat = self.df.loc[:max_start, REGIME_COLS].to_numpy(dtype=np.float32)
        for idx, row in enumerate(regime_mat):
            reg_i = int(np.argmax(row))
            reg_name = REGIME_COLS[reg_i].replace("regime_", "")
            if reg_name in buckets:
                buckets[reg_name].append(idx)
        return buckets

    def _sample_train_start(self, max_start: int) -> int:
        return random.randint(0, max_start)

    def reset(self, start_idx=None):
        if self.phase == "train":
            max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
            self.start_step = start_idx if start_idx is not None else random.randint(0, max_start)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.MAX_EPISODE_STEPS, len(self.df) - 1)

        self.balance = self.initial_balance
        self.pos = None
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.hold_count = 0

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = False
        self._last_closed_side = ""
        self._last_closed_hold_count = 0

        if self.hmm_detector is not None:
            self.hmm_detector.reset_episode()

        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step))

    def step(self, action: float):
        action = float(np.clip(action, -1.0, 1.0))
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        abs_action = abs(action)
        leverage_rate = abs_action
        force_close = bool(self.pos is not None and self.unrealized_pnl <= -0.025)

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_adjusting = False

        if force_close:
            is_closing = True
        elif self.pos is None:
            if action > self.pos_thresh:
                is_entering_long = True
            elif action < -self.pos_thresh:
                is_entering_short = True
        else:
            if abs_action < self.close_thresh:
                is_closing = True
            elif self.pos == "LONG" and action < -self.pos_thresh:
                is_closing = True
            elif self.pos == "SHORT" and action > self.pos_thresh:
                is_closing = True
            else:
                is_adjusting = True

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close
        self._last_closed_side = ""
        self._last_closed_hold_count = 0

        if is_entering_long:
            self.pos = "LONG"
            self.entry_price = fill_price * (1.0 + self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_entering_short:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1.0 - self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_adjusting and self.pos is not None:
            old_lev = self.current_leverage
            new_lev = leverage_rate
            lev_delta = abs(new_lev - old_lev)
            if lev_delta > 0.05:
                self.balance -= self.balance * self.fee * lev_delta
                self.current_leverage = new_lev
        elif is_closing and self.pos is not None:
            closed_side = str(self.pos)
            closed_hold_count = int(self.hold_count)
            base_balance = self.balance
            if self.pos == "LONG":
                realized_pnl = (fill_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - fill_price * (1.0 + self.slip)) / self.entry_price
            realized_pnl *= self.current_leverage
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self._last_closed_side = closed_side
            self._last_closed_hold_count = closed_hold_count
            self.pos = None
            self.current_leverage = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == "LONG":
                raw_pnl = (next_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                raw_pnl = (self.entry_price - next_price * (1.0 + self.slip)) / self.entry_price
            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -0.01:
            dd_ratio = abs(self.unrealized_pnl) / 0.025
            r2_drawdown = -0.1 * (dd_ratio ** 2)

        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                r3_quality = 0.15 * min(self._last_realized_pnl / 0.01, 1.0)
            else:
                r3_quality = -0.08 if self.side_mode == "both" else -0.05

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > self.hold_plateau_start:
            if abs(float(self.unrealized_pnl)) < self.hold_plateau_pnl_abs:
                r4_time_decay = -self.hold_plateau_penalty * float(
                    np.clip((self.hold_count - self.hold_plateau_start) / 96.0, 0.0, 1.0)
                )

        r7_adverse_hold = 0.0
        if (
            self.pos is not None
            and self.unrealized_pnl < -abs(self.adverse_hold_pnl_th)
            and self.hold_count > self.adverse_hold_start
        ):
            r7_adverse_hold = -self.adverse_hold_penalty * float(
                np.clip(abs(self.unrealized_pnl) / 0.02, 0.0, 1.0)
            )

        r5_idle = 0.0
        if self.pos is None:
            regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
            regime_raw = self._feat_np[regime_step]
            o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
            regime_vec = regime_raw[o : o + self._n_regime]
            regime_idx = int(np.argmax(regime_vec))
            if self.specialist_idle_penalty is not None:
                r5_idle = float(self.specialist_idle_penalty)
            else:
                # 0=chop, 1=whipsaw, 2=bull, 3=bear, 4=normal
                if regime_idx in (2, 3):
                    r5_idle = -0.003
                elif regime_idx in (0, 1):
                    r5_idle = -0.0003
                else:
                    r5_idle = -0.001

        # 실제 fee/slippage는 balance 변화로 이미 반영됨.
        # 추가 진입 고정 페널티는 이중 비용이 되어 진입 억제를 과도하게 키울 수 있어 제거.
        r6_trade_cost = 0.0

        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost + r7_adverse_hold
        # Keep reward linear around 0 and avoid double tanh saturation.
        reward = float(np.clip(raw_reward, -2.0, 2.0))

        if done and self.pos is not None:
            base_balance = self.balance
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            if self.pos == "LONG":
                ep_realized = (ep_end_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1.0 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            terminal_r = float(np.tanh(ep_realized * 50.0)) * self.terminal_reward_scale
            if ep_realized > 0:
                terminal_r += self.terminal_quality_win * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= self.terminal_quality_loss
            reward = float(np.clip(raw_reward + terminal_r, -2.0, 2.0))
            self.pos = None

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
            "force_closed": bool(self._just_closed and self._was_force_closed),
            "closed_side": self._last_closed_side,
            "closed_hold_count": int(self._last_closed_hold_count),
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    @property
    def win_rate(self):
        return self.win_trades / max(1, self.total_trades)

    def _get_stacked_state(self, raw_state):
        self._frame_stack.append(raw_state)
        pad = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        return np.concatenate(frames)

    def _build_state(self, idx):
        if idx < 0 or idx >= len(self._feat_np):
            return np.zeros(STATE_DIM, dtype=np.float32)
        row = self._feat_np[idx]
        o = 0
        preds = row[o:o + self._n_pred]; o += self._n_pred
        confs = row[o:o + self._n_conf]; o += self._n_conf
        signal = preds * confs
        elite = row[o:o + self._n_elite]; o += self._n_elite
        alpha6 = row[o:o + self._n_alpha]; o += self._n_alpha
        regime_raw = row[o:o + self._n_regime]; o += self._n_regime
        regime_idx = np.array([float(np.argmax(regime_raw))], dtype=np.float32)
        synth2 = row[o:o + self._n_synth]

        close = self._close_np[idx]
        pos_features = np.array([
            1.0 if self.pos == "LONG" else (-1.0 if self.pos == "SHORT" else 0.0),
            self.entry_price / close - 1 if self.pos is not None else 0.0,
            np.tanh(self.unrealized_pnl / 0.02),
            np.clip(self.max_drawdown / 0.05, -1.0, 1.0),
            self.hold_count / 144,
        ], dtype=np.float32)

        if self.hmm_detector is not None:
            row_dict = {col: float(self._hmm_obs_np[col][idx]) for col in self._hmm_obs_np}
            hmm_feat = self.hmm_detector.get_features(row_dict)
        else:
            hmm_feat = np.zeros(HMM_DIM, dtype=np.float32)

        mtf_feat = self.mtf.get(idx)

        return np.nan_to_num(
            np.concatenate([signal, elite, alpha6, regime_idx, hmm_feat, synth2, pos_features, mtf_feat]),
            0.0,
        )


class ReplayBuffer:
    """Simple replay buffer for continuous-action agents."""

    def __init__(self, capacity=500000):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._s = None
        self._a = np.empty(capacity, np.float32)
        self._r = np.empty(capacity, np.float32)
        self._ns = None
        self._d = np.empty(capacity, np.bool_)

    def push(self, state, action, reward, next_state, done):
        if self._s is None:
            sdim = len(state)
            self._s = np.empty((self._cap, sdim), np.float32)
            self._ns = np.empty((self._cap, sdim), np.float32)
        p = self._ptr
        self._s[p] = state
        self._a[p] = action
        self._r[p] = reward
        self._ns[p] = next_state
        self._d[p] = done
        self._ptr = (p + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self._s[idx],
            self._a[idx],
            self._r[idx],
            self._ns[idx],
            self._d[idx].astype(np.float32),
        )

    def __len__(self):
        return self._size
