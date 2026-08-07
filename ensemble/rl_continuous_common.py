"""Shared continuous-action environment pieces for SAC/DSAC variants."""

from __future__ import annotations

import random
from collections import deque
import os

import numpy as np
import pandas as pd

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
_DEFAULT_MAX_LEVERAGE = 5.0
_DEFAULT_MIN_TRADE_EXPOSURE = 0.10
_DEFAULT_RISK_MAX_MARGIN = 0.35
_DEFAULT_NET_EDGE_MIN = 0.0015
_DEFAULT_EARLY_EXIT_WINDOW = 3
_DEFAULT_EARLY_EXIT_PENALTY = 0.035


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


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
        dd_penalty_coeff=None,
        kelly_align_bonus=None,
        kelly_chop_loss_penalty=None,
        adverse_hold_enable=None,
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
        self.side_mode = str(side_mode or "both")
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
        self.max_leverage = max(1.0, float(os.getenv("RL_MAX_LEVERAGE", str(_DEFAULT_MAX_LEVERAGE))))
        self.min_trade_exposure = float(
            np.clip(float(os.getenv("RL_MIN_TRADE_EXPOSURE", str(_DEFAULT_MIN_TRADE_EXPOSURE))), 0.0, self.max_leverage)
        )
        self.risk_max_margin = float(np.clip(float(os.getenv("RL_RISK_MAX_MARGIN", str(_DEFAULT_RISK_MAX_MARGIN))), 0.01, 1.0))
        self.net_edge_min = max(0.0, float(os.getenv("RL_NET_EDGE_MIN", str(_DEFAULT_NET_EDGE_MIN))))
        self.early_exit_window = max(0, int(os.getenv("RL_EARLY_EXIT_WINDOW", str(_DEFAULT_EARLY_EXIT_WINDOW))))
        self.early_exit_penalty = max(
            0.0, float(os.getenv("RL_EARLY_EXIT_PENALTY", str(_DEFAULT_EARLY_EXIT_PENALTY)))
        )
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
        # Phase-2 reward shaping knobs.
        self.dd_soft_start = float(os.getenv("RL_DD_SOFT_START", "0.01"))
        self.dd_hard_scale = float(os.getenv("RL_DD_HARD_SCALE", "0.025"))
        self.dd_penalty_coeff = (
            float(dd_penalty_coeff)
            if dd_penalty_coeff is not None
            else float(os.getenv("RL_DD_PENALTY_COEFF", "0.10"))
        )
        self.kelly_align_bonus = (
            float(kelly_align_bonus)
            if kelly_align_bonus is not None
            else float(os.getenv("RL_KELLY_ALIGN_BONUS", "0.20"))
        )
        self.kelly_chop_loss_penalty = (
            float(kelly_chop_loss_penalty)
            if kelly_chop_loss_penalty is not None
            else float(os.getenv("RL_KELLY_CHOP_LOSS_PENALTY", "2.00"))
        )
        self.adverse_hold_enable = (
            bool(adverse_hold_enable)
            if adverse_hold_enable is not None
            else _env_flag("RL_ADVERSE_HOLD_ENABLE", True)
        )
        self.force_close_enable = _env_flag("RL_FORCE_CLOSE_ENABLE", True)
        self.force_close_th = float(os.getenv("RL_FORCE_CLOSE_TH", "-0.025"))

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

        def _num_col_or_default(col: str, default: float = 0.0) -> np.ndarray:
            if col in self.df.columns:
                s = self.df[col]
            else:
                s = pd.Series(default, index=self.df.index, dtype="float64")
            s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
            return s.to_numpy(dtype=np.float32)

        self._m7_tp_price_np = _num_col_or_default("m7_tp_price", 0.0)
        self._m7_sl_price_np = _num_col_or_default("m7_sl_price", 0.0)
        self._m7_target_hold_np = _num_col_or_default("m7_target_hold", 0.0)
        self._m7_q50_np = _num_col_or_default("m7_q50", 0.0)
        self._ai_vol_pct_np = _num_col_or_default("ai_vol_regime_pct", 0.5)
        self._ai_adverse_np = _num_col_or_default("ai_adverse_risk", 0.0)
        self._ai_reward_np = _num_col_or_default("ai_reward_risk", 0.0)
        self._ai_flow_flip_np = _num_col_or_default("ai_flow_flip_prob", 0.5)
        self._m7_qwidth_np = _num_col_or_default("m7_qwidth", 0.0)
        self._garch_vol_z_np = _num_col_or_default("garch_vol_z", 0.0)
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

    def regime_bucket(self, idx: int | None = None) -> str:
        if idx is None:
            idx = int(self.current_step)
        idx = min(max(int(idx), 0), len(self._feat_np) - 1)
        row = self._feat_np[idx]
        o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
        regime_vec = np.nan_to_num(row[o : o + self._n_regime], nan=0.0)
        if regime_vec.size == 0:
            return "normal"
        reg_idx = int(np.argmax(regime_vec))
        if reg_idx == 0:
            return "chop"
        if reg_idx == 1:
            return "whipsaw"
        if reg_idx == 2:
            return "bull"
        if reg_idx == 3:
            return "bear"
        return "normal"

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
        self.current_margin_fraction = 0.0
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

    def _risk_overlay(self, idx: int, direction: float, conviction: float) -> tuple[float, float, float]:
        """Convert policy conviction into futures margin/leverage via risk controls."""
        if abs(direction) <= self.pos_thresh or conviction <= 0.0:
            return 0.0, 1.0, 0.0

        idx = min(max(int(idx), 0), len(self._close_np) - 1)
        row = self._feat_np[min(idx, len(self._feat_np) - 1)]
        o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
        regime_vec = np.nan_to_num(row[o : o + self._n_regime], nan=0.0)
        regime_idx = int(np.argmax(regime_vec)) if regime_vec.size else 4

        # Regime caps are a risk overlay, not a learned action. They keep the
        # policy from solving alpha and liquidation risk in one unstable output.
        if regime_idx == 0:  # chop
            lev_cap = min(self.max_leverage, float(os.getenv("RL_RISK_CHOP_LEV_CAP", "1.4")))
        elif regime_idx == 1:  # whipsaw
            lev_cap = min(self.max_leverage, float(os.getenv("RL_RISK_WHIPSAW_LEV_CAP", "1.2")))
        elif regime_idx in (2, 3):  # bull/bear trend
            lev_cap = min(self.max_leverage, float(os.getenv("RL_RISK_TREND_LEV_CAP", str(self.max_leverage))))
        else:
            lev_cap = min(self.max_leverage, float(os.getenv("RL_RISK_NORMAL_LEV_CAP", "2.5")))
        lev_cap = max(1.0, lev_cap)

        vol_pct = float(np.clip(self._ai_vol_pct_np[idx], 0.0, 1.0))
        adverse = float(np.clip(np.tanh(max(0.0, float(self._ai_adverse_np[idx])) / 0.010), 0.0, 1.0))
        reward_edge = float(np.clip(np.tanh(max(0.0, float(self._ai_reward_np[idx])) / 4.0), 0.0, 1.0))
        flow_flip = float(np.clip(self._ai_flow_flip_np[idx], 0.0, 1.0))
        q50_abs = abs(float(self._m7_q50_np[idx]))
        qwidth = max(0.0, float(self._m7_qwidth_np[idx]))
        if qwidth <= 1e-12:
            qwidth = max(abs(float(self._garch_vol_z_np[idx])) * 0.002, 5e-4)
        uncertainty = float(np.clip(qwidth / 0.015, 0.0, 1.0))

        # Use q50 as an opportunity magnitude gate only. In the current dataset
        # M7 directional outputs can invert out-of-sample, so direction remains
        # the policy's job while this overlay only blocks low-edge/noisy periods.
        edge_buffer = self.net_edge_min * (1.0 + 0.35 * vol_pct + 0.35 * flow_flip + 0.20 * uncertainty)
        if q50_abs <= edge_buffer:
            return 0.0, 1.0, 0.0

        risk_damp = (
            (1.0 - 0.55 * vol_pct)
            * (1.0 - 0.45 * adverse)
            * (1.0 - 0.35 * flow_flip)
            * (1.0 - 0.25 * uncertainty)
        )
        risk_damp = float(np.clip(risk_damp, 0.10, 1.0))
        opportunity = float(np.clip(0.55 + 0.45 * reward_edge, 0.55, 1.0))
        effective = float(np.clip(conviction * risk_damp * opportunity, 0.0, 1.0))

        leverage = float(1.0 + (lev_cap - 1.0) * effective)
        exposure_cap = float(max(self.min_trade_exposure, lev_cap * self.risk_max_margin))
        exposure_cap = min(exposure_cap, lev_cap)
        exposure = float(self.min_trade_exposure + (exposure_cap - self.min_trade_exposure) * effective)
        if effective <= 1e-8:
            exposure = 0.0
        margin_fraction = float(np.clip(exposure / max(leverage, 1e-8), 0.0, self.risk_max_margin))
        exposure = float(margin_fraction * leverage)
        return margin_fraction, leverage, exposure

    def _decode_action(self, action, idx: int | None = None) -> tuple[float, float, float, float]:
        """Decode 2D policy action into direction plus risk-overlay exposure."""
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size <= 1:
            direction = float(np.clip(arr[0] if arr.size else 0.0, -1.0, 1.0))
            # Legacy scalar DSAC mode: action magnitude is directly interpreted as
            # exposure strength. This matches train_rl_dsac_agent.py behaviour.
            exposure = float(abs(direction)) if abs(direction) > self.pos_thresh else 0.0
            margin_fraction = exposure
            leverage = 1.0 if exposure > 0.0 else 0.0
            return direction, margin_fraction, leverage, exposure
        else:
            direction = float(np.clip(arr[0], -1.0, 1.0))
            dir_intensity = float(
                np.clip((abs(direction) - float(self.pos_thresh)) / max(1.0 - float(self.pos_thresh), 1e-6), 0.0, 1.0)
            )
            raw_conviction = float(np.clip((arr[1] + 1.0) * 0.5, 0.0, 1.0))
            conviction = dir_intensity * raw_conviction
        if abs(direction) <= self.pos_thresh:
            conviction = 0.0
        margin_fraction, leverage, exposure = self._risk_overlay(self.current_step if idx is None else idx, direction, conviction)
        return direction, margin_fraction, leverage, exposure

    def step(self, action):
        decision_step = self.current_step
        direction_action, margin_fraction, leverage_rate, exposure_rate = self._decode_action(action, decision_step)
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        abs_action = abs(direction_action)
        force_close = bool(
            self.force_close_enable
            and self.pos is not None
            and self.unrealized_pnl <= self.force_close_th
        )

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_adjusting = False
        exposure_turnover = 0.0

        if force_close:
            is_closing = True
        elif self.pos is None:
            if direction_action > self.pos_thresh and exposure_rate > 0.0:
                is_entering_long = True
            elif direction_action < -self.pos_thresh and exposure_rate > 0.0:
                is_entering_short = True
        else:
            if abs_action < self.close_thresh:
                is_closing = True
            elif self.pos == "LONG" and direction_action < -self.pos_thresh:
                is_closing = True
            elif self.pos == "SHORT" and direction_action > self.pos_thresh:
                is_closing = True
            else:
                is_adjusting = True
            if is_adjusting and exposure_rate <= 0.0:
                is_adjusting = False
                is_closing = True

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
            self.current_margin_fraction = margin_fraction
            exposure_turnover = exposure_rate
            self.balance -= self.balance * self.fee * exposure_rate
        elif is_entering_short:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1.0 - self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.current_margin_fraction = margin_fraction
            exposure_turnover = exposure_rate
            self.balance -= self.balance * self.fee * exposure_rate
        elif is_adjusting and self.pos is not None:
            old_exposure = self.current_margin_fraction * self.current_leverage
            new_lev = leverage_rate
            new_margin = margin_fraction
            new_exposure = new_margin * new_lev
            exposure_delta = abs(new_exposure - old_exposure)
            if exposure_delta > 0.05:
                exposure_turnover = exposure_delta
                # Realize PnL at the old exposure before resizing. Otherwise a later
                # leverage increase would retroactively amplify the whole past move.
                base_balance = self.balance
                if self.pos == "LONG":
                    realized_pnl = (fill_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
                    self.entry_price = fill_price * (1.0 + self.slip)
                else:
                    realized_pnl = (self.entry_price - fill_price * (1.0 + self.slip)) / self.entry_price
                    self.entry_price = fill_price * (1.0 - self.slip)
                realized_pnl *= old_exposure
                self.balance = base_balance * (1.0 + realized_pnl)
                self.balance -= base_balance * self.fee * exposure_delta
                self.current_leverage = new_lev
                self.current_margin_fraction = new_margin
                self.entry_idx = fill_step
                self.hold_count = 0
                self.unrealized_pnl = 0.0
                self.peak_pnl = 0.0
                self.max_drawdown = 0.0
        elif is_closing and self.pos is not None:
            closed_side = str(self.pos)
            closed_hold_count = int(self.hold_count)
            base_balance = self.balance
            close_exposure = self.current_margin_fraction * self.current_leverage
            if self.pos == "LONG":
                realized_pnl = (fill_price * (1.0 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - fill_price * (1.0 + self.slip)) / self.entry_price
            realized_pnl *= self.current_margin_fraction * self.current_leverage
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * (self.current_margin_fraction * self.current_leverage)
            exposure_turnover = close_exposure
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self._last_closed_side = closed_side
            self._last_closed_hold_count = closed_hold_count
            self.pos = None
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
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
            self.unrealized_pnl = raw_pnl * (self.current_margin_fraction * self.current_leverage)
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
        regime_raw = self._feat_np[regime_step]
        o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
        regime_vec = regime_raw[o : o + self._n_regime]
        regime_idx = int(np.argmax(regime_vec))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -abs(self.dd_soft_start):
            dd_excess = abs(self.unrealized_pnl) - abs(self.dd_soft_start)
            dd_den = max(abs(self.dd_hard_scale) - abs(self.dd_soft_start), 1e-6)
            dd_ratio = np.clip(dd_excess / dd_den, 0.0, 3.0)
            r2_drawdown = -self.dd_penalty_coeff * float(dd_ratio ** 2)

        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                # Reward meaningful wins, not tiny fee-sensitive scalps.
                r3_quality = 0.08 * float(np.clip((self._last_realized_pnl - 0.002) / 0.012, 0.0, 1.0))
            else:
                loss_ratio = float(np.clip(abs(self._last_realized_pnl) / 0.012, 0.0, 2.0))
                r3_quality = -(0.10 + 0.10 * loss_ratio) if self.side_mode == "both" else -(0.07 + 0.08 * loss_ratio)

        r9_early_exit = 0.0
        if (
            self._just_closed
            and not self._was_force_closed
            and self.early_exit_window > 0
            and self._last_closed_hold_count < self.early_exit_window
        ):
            shortfall = float((self.early_exit_window - self._last_closed_hold_count) / max(self.early_exit_window, 1))
            # Do not punish a genuinely strong fast take-profit as much; punish
            # weak scalps and early losses because they usually lose after costs.
            strong_profit_credit = 0.0
            if self._last_realized_pnl > 0.0:
                strong_profit_credit = float(np.clip((self._last_realized_pnl - 0.002) / 0.006, 0.0, 1.0))
            loss_multiplier = 1.0
            if self._last_realized_pnl < 0.0:
                loss_multiplier += float(np.clip(abs(self._last_realized_pnl) / 0.010, 0.0, 1.0))
            r9_early_exit = -self.early_exit_penalty * shortfall * (1.0 - strong_profit_credit) * loss_multiplier

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > self.hold_plateau_start:
            if abs(float(self.unrealized_pnl)) < self.hold_plateau_pnl_abs:
                r4_time_decay = -self.hold_plateau_penalty * float(
                    np.clip((self.hold_count - self.hold_plateau_start) / 96.0, 0.0, 1.0)
                )

        r7_adverse_hold = 0.0
        if (
            self.adverse_hold_enable
            and
            self.pos is not None
            and self.unrealized_pnl < -abs(self.adverse_hold_pnl_th)
            and self.hold_count > self.adverse_hold_start
        ):
            r7_adverse_hold = -self.adverse_hold_penalty * float(
                np.clip(abs(self.unrealized_pnl) / 0.02, 0.0, 1.0)
            )

        r5_idle = 0.0
        if self.pos is None:
            if self.specialist_idle_penalty is not None:
                r5_idle = float(self.specialist_idle_penalty)

        # Balance already reflects fee/slippage; this shaping term discourages
        # hyperactive resizing that wins often but loses net expectancy.
        r6_trade_cost = -0.004 * float(np.clip(exposure_turnover, 0.0, 5.0))

        r8_kelly_regime = 0.0
        if self.pos is not None:
            step_ret = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
            lev = float(np.clip((self.current_margin_fraction * self.current_leverage) / max(self.max_leverage, 1e-6), 0.0, 1.0))
            is_aligned = (self.pos == "LONG" and regime_idx == 2) or (self.pos == "SHORT" and regime_idx == 3)
            if step_ret > 0.0 and is_aligned:
                r8_kelly_regime += self.kelly_align_bonus * lev * float(np.clip(step_ret / 0.002, 0.0, 1.0))
            if step_ret < 0.0 and regime_idx in (0, 1):
                # chop/whipsaw에서 고레버리지 손실을 더 강하게 벌점
                extra = max(self.kelly_chop_loss_penalty - 1.0, 0.0)
                r8_kelly_regime -= extra * abs(r1_pnl) * lev

        raw_reward = (
            r1_pnl
            + r2_drawdown
            + r3_quality
            + r4_time_decay
            + r5_idle
            + r6_trade_cost
            + r7_adverse_hold
            + r8_kelly_regime
            + r9_early_exit
        )
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
            ep_realized *= self.current_margin_fraction * self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * (self.current_margin_fraction * self.current_leverage)
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
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
            "force_closed": bool(self._just_closed and self._was_force_closed),
            "closed_side": self._last_closed_side,
            "closed_hold_count": int(self._last_closed_hold_count),
            "regime_bucket": self.regime_bucket(decision_step),
            "leverage": float(self.current_leverage),
            "margin_fraction": float(self.current_margin_fraction),
            "notional_exposure": float(self.current_margin_fraction * self.current_leverage),
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
        self._a = None
        self._r = np.empty(capacity, np.float32)
        self._ns = None
        self._d = np.empty(capacity, np.bool_)

    def push(self, state, action, reward, next_state, done):
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        if self._s is None:
            self._s = np.empty((self._cap, *state_arr.shape), np.float32)
            self._ns = np.empty((self._cap, *next_state_arr.shape), np.float32)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._a is None:
            self._a = np.empty((self._cap, int(action_arr.size)), np.float32)
        p = self._ptr
        self._s[p] = state_arr
        self._a[p] = action_arr
        self._r[p] = reward
        self._ns[p] = next_state_arr
        self._d[p] = done
        self._ptr = (p + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        actions = self._a[idx]
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions[:, 0]
        return (
            self._s[idx],
            actions,
            self._r[idx],
            self._ns[idx],
            self._d[idx].astype(np.float32),
        )

    def __len__(self):
        return self._size
