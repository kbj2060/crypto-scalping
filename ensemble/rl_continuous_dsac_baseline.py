"""Baseline continuous-action environment for DSAC compact agents.

This preserves the older scalar-action execution/reward behavior used by
train_rl_dsac_agent.py before the shared environment accumulated futures-specific
resize logic and extra reward shaping.
"""

from __future__ import annotations

import os
import random
from collections import deque

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
_REV_EXIT_THRESH = 0.08


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _unlevered_pnl_from_exposure(pnl: float, exposure: float) -> float:
    exp = max(float(exposure or 0.0), 1e-8)
    return float(pnl) / exp


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


class SACTradingEnv:
    """Scalar-action trading env used by the original DSAC compact agent."""

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
        specialist_force_close_th=None,
        specialist_rev_exit_thresh=None,
        dd_penalty_coeff=None,
        kelly_align_bonus=None,
        kelly_chop_loss_penalty=None,
        adverse_hold_enable=None,
        terminal_reward_scale: float = 1.0,
        terminal_quality_win: float = 0.15,
        terminal_quality_loss: float = 0.05,
        focus_regime: str | None = None,
        terminate_on_regime_change: bool = True,
        focus_segments: list[tuple[int, int]] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.hmm_detector = hmm_detector
        self.side_mode = str(side_mode or "both")
        self.focus_regime = str(focus_regime or "").strip().lower() or None
        if self.focus_regime not in {"bull", "bear", "chop", "whipsaw", "normal"}:
            self.focus_regime = None
        self.terminate_on_regime_change = bool(terminate_on_regime_change)
        self.focus_segments = [
            (int(max(0, s)), int(max(0, e)))
            for s, e in (focus_segments or [])
            if int(e) > int(s)
        ]
        self._active_focus_segment = None
        self.terminal_reward_scale = float(terminal_reward_scale)
        self.terminal_quality_win = float(terminal_quality_win)
        self.terminal_quality_loss = float(terminal_quality_loss)
        self.pos_thresh = float(specialist_pos_thresh) if specialist_pos_thresh is not None else float(_POS_THRESH)
        self.close_thresh = float(specialist_close_thresh) if specialist_close_thresh is not None else float(_CLOSE_THRESH)
        self.specialist_idle_penalty = (
            float(specialist_idle_penalty) if specialist_idle_penalty is not None else None
        )
        self.hold_plateau_start = int(os.getenv("RL_HOLD_PLATEAU_START", "24"))
        self.hold_plateau_pnl_abs = float(os.getenv("RL_HOLD_PLATEAU_PNL_ABS", "0.003"))
        self.hold_plateau_penalty = float(os.getenv("RL_HOLD_PLATEAU_PENALTY", "0.003"))
        self.adverse_hold_start = int(os.getenv("RL_ADVERSE_HOLD_START", "24"))
        self.adverse_hold_pnl_th = float(os.getenv("RL_ADVERSE_HOLD_PNL_TH", "0.004"))
        self.adverse_hold_penalty = float(os.getenv("RL_ADVERSE_HOLD_PENALTY", "0.010"))
        self.dd_soft_start = float(os.getenv("RL_DD_SOFT_START", "0.01"))
        self.dd_hard_scale = float(os.getenv("RL_DD_HARD_SCALE", "0.025"))
        self.dd_penalty_coeff = (
            float(dd_penalty_coeff) if dd_penalty_coeff is not None else float(os.getenv("RL_DD_PENALTY_COEFF", "0.10"))
        )
        self.kelly_align_bonus = (
            float(kelly_align_bonus)
            if kelly_align_bonus is not None else float(os.getenv("RL_KELLY_ALIGN_BONUS", "0.20"))
        )
        self.kelly_chop_loss_penalty = (
            float(kelly_chop_loss_penalty)
            if kelly_chop_loss_penalty is not None else float(os.getenv("RL_KELLY_CHOP_LOSS_PENALTY", "2.00"))
        )
        self.force_close_enable = _env_flag("RL_FORCE_CLOSE_ENABLE", True)
        self.force_close_th = (
            float(specialist_force_close_th)
            if specialist_force_close_th is not None
            else float(os.getenv("RL_FORCE_CLOSE_TH", "-0.025"))
        )
        self.rev_exit_thresh = (
            float(specialist_rev_exit_thresh)
            if specialist_rev_exit_thresh is not None
            else float(os.getenv("RL_REV_EXIT_THRESH", str(_REV_EXIT_THRESH)))
        )
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
            self.df["high"].values.astype(np.float32) if "high" in self.df.columns else self._close_np.copy()
        )
        self._low_np = (
            self.df["low"].values.astype(np.float32) if "low" in self.df.columns else self._close_np.copy()
        )
        self._open_np = (
            self.df["open"].values.astype(np.float32) if "open" in self.df.columns else self._close_np.copy()
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

    def _base_lot_total_exposure(self) -> float:
        return float(sum(float(x.get("exposure", 0.0) or 0.0) for x in getattr(self, "_lots", [])))

    def _base_lot_weighted_entry(self) -> float:
        lots = getattr(self, "_lots", [])
        total = self._base_lot_total_exposure()
        if total <= 1e-12 or not lots:
            return 0.0
        return float(
            sum(float(x.get("entry_price", 0.0) or 0.0) * float(x.get("exposure", 0.0) or 0.0) for x in lots)
            / total
        )

    def _base_lot_oldest_entry_idx(self) -> int:
        lots = getattr(self, "_lots", [])
        if not lots:
            return int(self.current_step)
        return int(min(int(x.get("entry_idx", self.current_step)) for x in lots))

    def _base_lot_trade_pnl(self, side: str, entry_price: float, exit_price: float, exposure: float) -> float:
        exp = max(float(exposure), 0.0)
        if exp <= 1e-12:
            return 0.0
        if str(side).upper() == "LONG":
            raw = (float(exit_price) * (1.0 - self.slip) - float(entry_price)) / max(float(entry_price), 1e-8)
        else:
            raw = (float(entry_price) - float(exit_price) * (1.0 + self.slip)) / max(float(entry_price), 1e-8)
        return float(raw * exp)

    def _base_lot_mark_pnl(self, price: float) -> float:
        total = 0.0
        for lot in getattr(self, "_lots", []):
            total += self._base_lot_trade_pnl(str(lot.get("side", "")), float(lot.get("entry_price", 0.0) or 0.0), float(price), float(lot.get("exposure", 0.0) or 0.0))
        return float(total)

    def _base_sync_position_from_lots(self) -> None:
        lots = getattr(self, "_lots", [])
        total = self._base_lot_total_exposure()
        if not lots or total <= 1e-12:
            self.pos = None
            self.entry_price = 0.0
            self.entry_idx = int(self.current_step)
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
            self.current_notional_exposure = 0.0
            self.unrealized_pnl = 0.0
            self.hold_count = 0
            return
        self.pos = str(lots[0].get("side", "")).upper()
        self.entry_price = self._base_lot_weighted_entry()
        self.entry_idx = self._base_lot_oldest_entry_idx()
        lev_num = 0.0
        margin_num = 0.0
        for lot in lots:
            exp = float(lot.get("exposure", 0.0) or 0.0)
            lev_num += exp * float(lot.get("leverage", exp) or exp)
            margin_num += exp * float(lot.get("margin_fraction", exp) or exp)
        weighted_lev = lev_num / max(total, 1e-12)
        weighted_margin = margin_num / max(total, 1e-12)
        self.current_notional_exposure = float(max(total, 0.0))
        self.current_leverage = float(max(weighted_lev, 0.0))
        if weighted_margin > 0.0:
            self.current_margin_fraction = float(np.clip(weighted_margin, 0.0, 1.0))
        else:
            self.current_margin_fraction = float(np.clip(total / max(weighted_lev, 1e-8), 0.0, 1.0))

    def _base_lot_add(
        self,
        side: str,
        entry_price: float,
        exposure: float,
        entry_idx: int,
        leverage: float | None = None,
        margin_fraction: float | None = None,
    ) -> None:
        exp = max(float(exposure), 0.0)
        if exp <= 1e-12:
            return
        lev = max(float(leverage if leverage is not None else exp), 0.0)
        margin = max(float(margin_fraction if margin_fraction is not None else exp), 0.0)
        self._lots.append(
            {
                "side": str(side).upper(),
                "entry_price": float(entry_price),
                "exposure": float(exp),
                "leverage": float(lev),
                "margin_fraction": float(margin),
                "entry_idx": int(entry_idx),
            }
        )
        self._base_sync_position_from_lots()

    def set_next_risk_decision(self, decision: dict | None) -> None:
        self._next_risk_decision = dict(decision or {})

    def _consume_next_risk_decision(self) -> dict:
        decision = dict(getattr(self, "_next_risk_decision", {}) or {})
        self._next_risk_decision = {}
        return decision

    def _base_lot_reduce(self, close_exposure: float, exit_price: float) -> float:
        remaining = float(max(close_exposure, 0.0))
        realized = 0.0
        while remaining > 1e-12 and getattr(self, "_lots", []):
            lot = self._lots[0]
            lot_exp = float(lot.get("exposure", 0.0) or 0.0)
            take = min(lot_exp, remaining)
            realized += self._base_lot_trade_pnl(str(lot.get("side", "")), float(lot.get("entry_price", 0.0) or 0.0), float(exit_price), float(take))
            lot_exp -= take
            remaining -= take
            if lot_exp <= 1e-12:
                self._lots.pop(0)
            else:
                lot["exposure"] = float(lot_exp)
        self._base_sync_position_from_lots()
        return float(realized)

    def _incremental_lot_value(self, current_total: float, current_weighted: float, target_total: float, target_weighted: float) -> float:
        delta = float(target_total) - float(current_total)
        if delta <= 1e-12:
            return float(target_weighted)
        value = (float(target_weighted) * float(target_total) - float(current_weighted) * float(current_total)) / delta
        return float(max(value, 0.0))

    def _base_lot_resize_to(
        self,
        target_exposure: float,
        fill_price: float,
        fill_step: int,
        *,
        target_leverage: float | None = None,
        target_margin_fraction: float | None = None,
    ) -> dict:
        current = self._base_lot_total_exposure()
        target = max(float(target_exposure), 0.0)
        out = {
            "resized": False,
            "resize_side": "",
            "resize_delta": 0.0,
            "resize_realized_pnl": 0.0,
            "resize_fee_paid": 0.0,
            "resize_prev_notional": float(current),
            "resize_target_notional": float(current),
        }
        if self.pos is None or current <= 1e-12:
            return out
        delta = target - current
        if abs(delta) <= 1e-12:
            return out

        if delta > 0.0:
            side = str(self.pos).upper()
            entry_fill = float(fill_price) * (1.0 + self.slip if side == "LONG" else 1.0 - self.slip)
            next_total = current + delta
            lev = self._incremental_lot_value(
                current,
                float(getattr(self, "current_leverage", current)),
                next_total,
                float(target_leverage if target_leverage is not None else getattr(self, "current_leverage", next_total)),
            )
            margin = self._incremental_lot_value(
                current,
                float(getattr(self, "current_margin_fraction", current)),
                next_total,
                float(target_margin_fraction if target_margin_fraction is not None else getattr(self, "current_margin_fraction", next_total)),
            )
            fee_paid = float(self.balance * self.fee * delta)
            self.balance -= fee_paid
            self._base_lot_add(side, entry_fill, delta, fill_step, leverage=lev, margin_fraction=margin)
            out.update(
                {
                    "resized": True,
                    "resize_side": "add",
                    "resize_delta": float(delta),
                    "resize_fee_paid": float(fee_paid),
                    "resize_target_notional": float(self._base_lot_total_exposure()),
                }
            )
            return out

        close_exp = min(-delta, current)
        if close_exp >= current - 1e-12:
            return out
        base_balance = float(self.balance)
        realized = self._base_lot_reduce(close_exp, float(fill_price))
        self.balance = base_balance * (1.0 + realized)
        fee_paid = float(base_balance * self.fee * close_exp)
        self.balance -= fee_paid
        out.update(
            {
                "resized": True,
                "resize_side": "reduce",
                "resize_delta": float(-close_exp),
                "resize_realized_pnl": float(realized),
                "resize_fee_paid": float(fee_paid),
                "resize_target_notional": float(self._base_lot_total_exposure()),
            }
        )
        return out

    def _build_train_start_buckets(self):
        buckets = {k: [] for k in ["bull", "bear", "chop", "whipsaw", "normal"]}
        if self.phase != "train":
            return buckets
        max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        if max_start <= 0:
            return buckets
        if not all(c in self.df.columns for c in REGIME_COLS):
            return buckets
        regime_mat = self.df.loc[:max_start, REGIME_COLS].to_numpy(dtype=np.float32)
        for idx, row in enumerate(regime_mat):
            reg_i = int(np.argmax(row))
            reg_name = REGIME_COLS[reg_i].replace("regime_", "")
            if reg_name in buckets:
                buckets[reg_name].append(idx)
        return buckets

    def _sample_train_start(self, max_start: int) -> int:
        if self.focus_segments:
            seg_start, seg_end = random.choice(self.focus_segments)
            max_local_start = max(seg_start, seg_end - 1)
            if max_local_start <= seg_start:
                self._active_focus_segment = (int(seg_start), int(seg_end))
                return int(seg_start)
            start = random.randint(int(seg_start), int(max_local_start))
            self._active_focus_segment = (int(seg_start), int(seg_end))
            return int(start)
        if self.focus_regime:
            focused = list(self._train_start_by_regime.get(self.focus_regime, []))
            if focused:
                return int(random.choice(focused))
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

    def regime_context(self, idx: int | None = None) -> dict[str, float]:
        if idx is None:
            idx = int(self.current_step)
        idx = min(max(int(idx), 0), len(self._feat_np) - 1)
        return {
            "confidence": 1.0,
            "margin": 1.0,
            "segment_len": 1.0,
            "age": 0.0,
            "age_frac": 0.0,
            "stable": 1.0,
        }

    def reset(self, start_idx=None):
        if self.phase == "train":
            max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
            self._active_focus_segment = None
            self.start_step = start_idx if start_idx is not None else self._sample_train_start(max_start)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.MAX_EPISODE_STEPS, len(self.df) - 1)
        if self.phase == "train" and self._active_focus_segment is not None:
            _, seg_end = self._active_focus_segment
            self.end_step = min(int(seg_end), int(self.end_step))

        self.balance = self.initial_balance
        self.pos = None
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0
        self.current_margin_fraction = 0.0
        self.current_notional_exposure = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.hold_count = 0
        self._lots = []

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._last_net_realized_pnl = 0.0
        self._was_force_closed = False
        self._last_closed_side = ""
        self._last_closed_hold_count = 0
        self._next_risk_decision = {}
        self._last_resized = False
        self._last_resize_side = ""
        self._last_resize_delta = 0.0
        self._last_resize_realized_pnl = 0.0
        self._last_resize_fee_paid = 0.0

        if self.hmm_detector is not None:
            self.hmm_detector.reset_episode()

        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step))

    def step(self, action: float):
        action = float(np.clip(action, -1.0, 1.0))
        risk_decision = self._consume_next_risk_decision()
        if self.pos is None and risk_decision and not bool(risk_decision.get("allow_entry", True)):
            action = 0.0
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step
        allow_long = self.side_mode in {"both", "long"}
        allow_short = self.side_mode in {"both", "short"}

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        abs_action = abs(action)
        leverage_rate = 1.0 if abs_action > self.pos_thresh else 0.0
        entry_leverage = leverage_rate
        entry_margin_fraction = leverage_rate
        if self.pos is None and leverage_rate > 0.0 and risk_decision:
            leverage_rate = float(max(risk_decision.get("notional_exposure", leverage_rate), 0.0))
            entry_leverage = float(max(risk_decision.get("leverage", leverage_rate), 0.0))
            entry_margin_fraction = float(max(risk_decision.get("position_fraction", leverage_rate), 0.0))
        force_close = bool(
            self.force_close_enable
            and self.pos is not None
            and self.unrealized_pnl <= self.force_close_th
        )

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_same_side_hold = False
        if force_close:
            is_closing = True
        elif self.pos is None:
            if allow_long and action > self.pos_thresh:
                is_entering_long = True
            elif allow_short and action < -self.pos_thresh:
                is_entering_short = True
        else:
            if abs_action < self.close_thresh:
                is_closing = True
            elif self.pos == "LONG" and action < -self.rev_exit_thresh:
                is_closing = True
            elif self.pos == "SHORT" and action > self.rev_exit_thresh:
                is_closing = True
            elif (self.pos == "LONG" and action > self.pos_thresh) or (self.pos == "SHORT" and action < -self.pos_thresh):
                is_same_side_hold = True

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._last_net_realized_pnl = 0.0
        self._was_force_closed = force_close
        self._last_closed_side = ""
        self._last_closed_hold_count = 0
        self._last_resized = False
        self._last_resize_side = ""
        self._last_resize_delta = 0.0
        self._last_resize_realized_pnl = 0.0
        self._last_resize_fee_paid = 0.0

        if is_entering_long:
            entry_fill = fill_price * (1.0 + self.slip)
            self.balance -= self.balance * self.fee * leverage_rate
            self._base_lot_add("LONG", entry_fill, leverage_rate, fill_step, leverage=entry_leverage, margin_fraction=entry_margin_fraction)
        elif is_entering_short:
            entry_fill = fill_price * (1.0 - self.slip)
            self.balance -= self.balance * self.fee * leverage_rate
            self._base_lot_add("SHORT", entry_fill, leverage_rate, fill_step, leverage=entry_leverage, margin_fraction=entry_margin_fraction)
        elif is_closing and self.pos is not None:
            closed_side = str(self.pos)
            closed_hold_count = int(self.hold_count)
            base_balance = self.balance
            close_exp = self._base_lot_total_exposure()
            realized_pnl = self._base_lot_reduce(close_exp, fill_price)
            # Use a conservative round-trip fee estimate for trade-level win
            # accounting. Balance already paid entry fees earlier and pays exit
            # fees below, but gross realized_pnl alone overstates win rate for
            # tiny scalps.
            net_realized_pnl = realized_pnl - (2.0 * self.fee * close_exp)
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * close_exp
            self.total_trades += 1
            if net_realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self._last_net_realized_pnl = net_realized_pnl
            self._last_closed_side = closed_side
            self._last_closed_hold_count = closed_hold_count
            self._lots = []
            self.pos = None
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
            self.current_notional_exposure = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0
        elif is_same_side_hold and risk_decision and bool(risk_decision.get("allow_resize", False)):
            target_exp = float(
                max(
                    risk_decision.get("target_notional_exposure", risk_decision.get("notional_exposure", self._base_lot_total_exposure())),
                    0.0,
                )
            )
            resize_info = self._base_lot_resize_to(
                target_exp,
                fill_price,
                fill_step,
                target_leverage=float(risk_decision.get("leverage", getattr(self, "current_leverage", 0.0))),
                target_margin_fraction=float(risk_decision.get("position_fraction", getattr(self, "current_margin_fraction", 0.0))),
            )
            self._last_resized = bool(resize_info.get("resized", False))
            self._last_resize_side = str(resize_info.get("resize_side", ""))
            self._last_resize_delta = float(resize_info.get("resize_delta", 0.0))
            self._last_resize_realized_pnl = float(resize_info.get("resize_realized_pnl", 0.0))
            self._last_resize_fee_paid = float(resize_info.get("resize_fee_paid", 0.0))

        self.current_step += 1
        done = self.current_step >= self.end_step
        regime_exit = False
        if (
            (not done)
            and self.focus_regime
            and self.terminate_on_regime_change
            and self.regime_bucket(self.current_step) != self.focus_regime
        ):
            done = True
            regime_exit = True
        next_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]

        if self.pos is not None:
            self._base_sync_position_from_lots()
            self.hold_count = self.current_step - self.entry_idx
            self.unrealized_pnl = self._base_lot_mark_pnl(next_price)
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
        regime_name = self.regime_bucket(regime_step)

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
            if self.specialist_idle_penalty is not None:
                r5_idle = float(self.specialist_idle_penalty)
            else:
                r5_idle = 0.0

        r6_trade_cost = 0.0

        r8_kelly_regime = 0.0
        if self.pos is not None:
            step_ret = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
            lev = float(np.clip(self.current_leverage, 0.0, 1.5))
            is_aligned = (self.pos == "LONG" and regime_name == "bull") or (self.pos == "SHORT" and regime_name == "bear")
            if step_ret > 0.0 and is_aligned:
                r8_kelly_regime += self.kelly_align_bonus * lev * float(np.clip(step_ret / 0.002, 0.0, 1.0))
            if step_ret < 0.0 and regime_name in {"chop", "whipsaw"}:
                extra = max(self.kelly_chop_loss_penalty - 1.0, 0.0)
                r8_kelly_regime -= extra * abs(r1_pnl) * lev

        raw_reward = (
            r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost + r7_adverse_hold + r8_kelly_regime
        )
        reward = float(np.clip(raw_reward, -2.0, 2.0))

        if done and self.pos is not None:
            base_balance = self.balance
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            close_exp = self._base_lot_total_exposure()
            ep_realized = self._base_lot_reduce(close_exp, ep_end_price)
            ep_net_realized = ep_realized - (2.0 * self.fee * close_exp)
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * close_exp
            self.total_trades += 1
            if ep_net_realized > 0:
                self.win_trades += 1
            self._last_realized_pnl = ep_realized
            self._last_net_realized_pnl = ep_net_realized
            terminal_r = float(np.tanh(ep_realized * 50.0)) * self.terminal_reward_scale
            if ep_realized > 0:
                terminal_r += self.terminal_quality_win * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= self.terminal_quality_loss
            reward = float(np.clip(raw_reward + terminal_r, -2.0, 2.0))
            self._lots = []
            self.pos = None
            self.current_leverage = 0.0
            self.current_margin_fraction = 0.0
            self.current_notional_exposure = 0.0

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
            "force_closed": bool(self._just_closed and self._was_force_closed),
            "closed_side": self._last_closed_side,
            "closed_hold_count": int(self._last_closed_hold_count),
            "regime_bucket": self.regime_bucket(decision_step),
            "regime_exit": bool(regime_exit),
            "focus_regime": self.focus_regime or "",
            "leverage": float(self.current_leverage),
            "execution_leverage": float(1.0 if self.pos is not None and self.current_margin_fraction > 1e-12 else 0.0),
            "margin_fraction": float(self.current_margin_fraction),
            "notional_exposure": float(getattr(self, "current_notional_exposure", self.current_margin_fraction)),
            "exposure_bucket": float(getattr(self, "current_notional_exposure", self.current_margin_fraction)),
            "risk_decision": risk_decision,
            "resized": bool(self._last_resized),
            "resize_side": str(self._last_resize_side),
            "resize_delta": float(self._last_resize_delta),
            "resize_realized_pnl": float(self._last_resize_realized_pnl),
            "resize_fee_paid": float(self._last_resize_fee_paid),
            "net_realized_pnl": float(self._last_net_realized_pnl),
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
        preds = row[o : o + self._n_pred]
        o += self._n_pred
        confs = row[o : o + self._n_conf]
        o += self._n_conf
        signal = preds * confs
        elite = row[o : o + self._n_elite]
        o += self._n_elite
        alpha6 = row[o : o + self._n_alpha]
        o += self._n_alpha
        regime_raw = row[o : o + self._n_regime]
        o += self._n_regime
        reg_idx = float(np.argmax(regime_raw))
        regime_idx = np.array([reg_idx], dtype=np.float32)
        synth2 = row[o : o + self._n_synth]

        close = self._close_np[idx]
        current_exp = float(max(getattr(self, "current_notional_exposure", getattr(self, "current_leverage", 0.0)), 0.0))
        unrealized_unlev = _unlevered_pnl_from_exposure(self.unrealized_pnl, current_exp) if self.pos is not None else 0.0
        drawdown_unlev = _unlevered_pnl_from_exposure(self.max_drawdown, current_exp) if self.pos is not None else 0.0
        pos_features = np.array(
            [
                1.0 if self.pos == "LONG" else (-1.0 if self.pos == "SHORT" else 0.0),
                self.entry_price / close - 1 if self.pos is not None else 0.0,
                np.tanh(unrealized_unlev / 0.02),
                np.clip(drawdown_unlev / 0.05, -1.0, 1.0),
                self.hold_count / 144,
            ],
            dtype=np.float32,
        )

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
