from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(_safe_float(v, lo), float(lo), float(hi)))


def _norm01(v: float, scale: float, center: float = 0.0) -> float:
    return float(np.clip(0.5 + 0.5 * np.tanh((_safe_float(v, center) - center) / max(float(scale), 1e-8)), 0.0, 1.0))


def _row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    return _safe_float(row.get(key, default), default)


@dataclass(frozen=True)
class RiskEngineConfig:
    min_signal_abs: float = 0.10
    entry_score_threshold: float = 0.16
    trend_entry_score_threshold: float = 0.18
    fade_entry_score_threshold: float = 0.42
    fade_min_setup_score: float = 0.62
    fade_max_risk_score: float = 0.70
    hard_block_risk: float = 0.92
    exit_risk: float = 0.86
    max_leverage: float = 3.0
    min_leverage: float = 1.0
    min_margin_fraction: float = 0.04
    max_margin_fraction: float = 0.35
    min_notional_exposure: float = 0.10
    max_notional_exposure: float = 1.25
    daily_pressure_relief: float = 0.05
    max_trades_per_day: float = 20.0
    min_bars_between_entries: int = 6
    min_hold_bars_before_risk_exit: int = 3
    hard_veto_risk_add: float = 0.62
    hard_veto_entry_penalty: float = 0.18
    max_hold_bars: int = 96
    unlevered_stop: float = -0.018
    stop_loss_base: float = 0.030
    take_profit_base: float = 0.040
    trailing_stop_base: float = 0.014
    profit_lock_start: float = 0.030
    profit_lock_floor: float = 0.010
    daily_loss_limit: float = 0.060
    daily_drawdown_soft_limit: float = 0.035
    daily_drawdown_hard_limit: float = 0.070
    daily_drawdown_size_cut: float = 0.70
    loss_streak_soft_start: int = 4
    max_consecutive_losses: int = 8
    loss_streak_size_cut: float = 0.90
    loss_streak_risk_add: float = 0.0
    enable_position_resize: bool = False
    min_resize_notional_delta: float = 0.04
    min_resize_hold_bars: int = 3
    resize_cooldown_bars: int = 12
    max_resize_add_fraction: float = 0.35
    max_resize_reduce_fraction: float = 0.45
    resize_add_min_unrealized: float = -0.002
    resize_add_min_entry_score: float = 0.20
    resize_reduce_risk_score: float = 0.58


@dataclass(frozen=True)
class RiskLeverageDecision:
    allow_entry: bool
    should_exit: bool
    effective_action: float
    direction: int
    position_fraction: float
    leverage: float
    notional_exposure: float
    target_notional_exposure: float
    resize_notional_delta: float
    allow_resize: bool
    conviction: float
    risk_score: float
    entry_score: float
    quality_score: float
    event_score: float
    setup_score: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_hold_bars: int
    size_multiplier: float
    exit_reason: str
    block_reason: str
    resize_reason: str
    sizing_reason: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class RiskLeverageEngine:
    """Single point for entry blocking, capital fraction, and leverage.

    The engine intentionally treats legacy gates as features, not final truth.
    It converts expert direction plus market context into an executable risk
    decision that the trading environment can consume.
    """

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        self.config = config or RiskEngineConfig()

    def decide(
        self,
        *,
        row: Mapping[str, Any],
        raw_action: float,
        shaped_action: float,
        current_regime: str,
        candidate_regime: str,
        position_side: str | None = None,
        position_regime: str | None = None,
        unrealized_pnl: float = 0.0,
        hold_count: int = 0,
        day_progress: float = 0.0,
        daily_closed_trades: int = 0,
        daily_trade_pressure: float = 0.0,
        daily_realized_pnl: float = 0.0,
        daily_equity_drawdown: float = 0.0,
        consecutive_losses: int = 0,
        peak_unrealized_pnl: float = 0.0,
        bars_since_last_entry: int = 10**9,
        bars_since_last_resize: int = 10**9,
        current_notional_exposure: float = 0.0,
        current_margin_fraction: float = 0.0,
        current_leverage: float = 0.0,
        candidate_admitted: bool = True,
        hard_veto: bool = False,
    ) -> RiskLeverageDecision:
        cfg = self.config
        raw = _clip(raw_action, -1.0, 1.0)
        shaped = _clip(shaped_action, -1.0, 1.0)
        base_action = shaped if abs(shaped) > 1e-8 else raw
        direction = 1 if base_action > 0.0 else (-1 if base_action < 0.0 else 0)
        abs_signal = abs(base_action)
        has_position = position_side is not None
        current_notional = _clip(current_notional_exposure, 0.0, cfg.max_notional_exposure)

        event_score = self._event_score(row, direction)
        quality_score = self._quality_score(row, direction)
        flow_score = self._flow_score(row, direction)
        regime_score = self._regime_score(row, current_regime, candidate_regime, bool(candidate_admitted))
        setup_score = self._setup_score(row, direction, current_regime, candidate_regime)
        risk_score = self._risk_score(
            row=row,
            current_regime=current_regime,
            candidate_regime=candidate_regime,
            position_side=position_side,
            position_regime=position_regime,
            unrealized_pnl=unrealized_pnl,
            hold_count=hold_count,
            hard_veto=hard_veto,
        )
        daily_realized = _safe_float(daily_realized_pnl, 0.0)
        daily_dd = max(0.0, -_safe_float(daily_equity_drawdown, 0.0))
        loss_streak = int(max(0, consecutive_losses))
        loss_streak_risk = cfg.loss_streak_risk_add * float(min(loss_streak, max(int(cfg.max_consecutive_losses), 1)))
        # Account-level drawdown controls sizing and hard locks below. Feeding it
        # into the alpha risk score caused premature exits after otherwise valid
        # setups, so keep this path as a portfolio guardrail only.
        dd_risk = 0.0
        risk_score = _clip(risk_score + loss_streak_risk + dd_risk, 0.0, 1.0)

        signal_score = _norm01(abs_signal, 0.28, center=0.12)
        conviction = _clip(
            0.25 * signal_score
            + 0.20 * event_score
            + 0.17 * quality_score
            + 0.13 * flow_score
            + 0.13 * regime_score
            + 0.12 * setup_score,
            0.0,
            1.0,
        )
        is_fade = str(candidate_regime or "").lower() == "fade"
        pressure_relief = cfg.daily_pressure_relief * _clip(daily_trade_pressure, 0.0, 1.0)
        base_threshold = cfg.fade_entry_score_threshold if is_fade else max(cfg.entry_score_threshold, cfg.trend_entry_score_threshold)
        entry_threshold = max(0.02, float(base_threshold) - pressure_relief)
        entry_score = _clip(
            conviction
            - 0.64 * risk_score
            + 0.05 * signal_score
            - (cfg.hard_veto_entry_penalty if hard_veto else 0.0),
            -1.0,
            1.0,
        )
        stop_loss_pct, take_profit_pct, trailing_stop_pct, dynamic_max_hold = self._dynamic_position_controls(
            risk_score=risk_score,
            conviction=conviction,
            is_fade=is_fade,
        )
        exit_reason = self._exit_reason(
            risk_score=risk_score,
            entry_score=entry_score,
            current_regime=current_regime,
            position_regime=position_regime,
            unrealized_pnl=unrealized_pnl,
            peak_unrealized_pnl=peak_unrealized_pnl,
            hold_count=hold_count,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_bars=dynamic_max_hold,
            daily_loss_locked=bool(cfg.daily_loss_limit > 0.0 and daily_realized <= -abs(cfg.daily_loss_limit)),
            daily_drawdown_locked=bool(cfg.daily_drawdown_hard_limit > 0.0 and daily_dd >= abs(cfg.daily_drawdown_hard_limit)),
        )

        block_reason = ""
        allow_entry = not has_position
        if has_position:
            allow_entry = False
        elif direction == 0:
            allow_entry = False
            block_reason = "no_direction"
        elif abs_signal < cfg.min_signal_abs:
            allow_entry = False
            block_reason = "weak_signal"
        elif bars_since_last_entry < cfg.min_bars_between_entries:
            allow_entry = False
            block_reason = "cooldown"
        elif cfg.max_trades_per_day > 0.0 and daily_closed_trades >= cfg.max_trades_per_day:
            allow_entry = False
            block_reason = "daily_budget_full"
        elif cfg.daily_loss_limit > 0.0 and daily_realized <= -abs(cfg.daily_loss_limit):
            allow_entry = False
            block_reason = "daily_loss_lock"
        elif cfg.daily_drawdown_hard_limit > 0.0 and daily_dd >= abs(cfg.daily_drawdown_hard_limit):
            allow_entry = False
            block_reason = "daily_drawdown_lock"
        elif cfg.max_consecutive_losses > 0 and loss_streak >= int(cfg.max_consecutive_losses):
            allow_entry = False
            block_reason = "loss_streak_lock"
        elif is_fade and setup_score < cfg.fade_min_setup_score:
            allow_entry = False
            block_reason = "fade_setup_weak"
        elif is_fade and risk_score >= cfg.fade_max_risk_score:
            allow_entry = False
            block_reason = "fade_risk"
        elif risk_score >= cfg.hard_block_risk:
            allow_entry = False
            block_reason = "hard_risk"
        elif entry_score < entry_threshold:
            allow_entry = False
            block_reason = "edge_below_risk"

        should_exit = bool(exit_reason)

        if not allow_entry and not has_position:
            return RiskLeverageDecision(
                allow_entry=False,
                should_exit=bool(should_exit),
                effective_action=0.0,
                direction=int(direction),
                position_fraction=0.0,
                leverage=0.0,
                notional_exposure=0.0,
                target_notional_exposure=0.0,
                resize_notional_delta=0.0,
                allow_resize=False,
                conviction=float(conviction),
                risk_score=float(risk_score),
                entry_score=float(entry_score),
                quality_score=float(quality_score),
                event_score=float(event_score),
                setup_score=float(setup_score),
                stop_loss_pct=float(stop_loss_pct),
                take_profit_pct=float(take_profit_pct),
                trailing_stop_pct=float(trailing_stop_pct),
                max_hold_bars=int(dynamic_max_hold),
                size_multiplier=0.0,
                exit_reason=str(exit_reason),
                block_reason=block_reason,
                resize_reason="",
                sizing_reason="blocked",
            )

        size_score = _clip((entry_score - entry_threshold) / max(1.0 - entry_threshold, 1e-8), 0.0, 1.0)
        size_score = _clip(0.35 + 0.65 * size_score, 0.0, 1.0)
        risk_damp = _clip(1.0 - 0.55 * risk_score, 0.12, 1.0)
        size_multiplier = self._size_multiplier(
            daily_realized_pnl=daily_realized,
            daily_equity_drawdown=-daily_dd,
            consecutive_losses=loss_streak,
        )
        margin = cfg.min_margin_fraction + (cfg.max_margin_fraction - cfg.min_margin_fraction) * size_score * risk_damp
        margin *= size_multiplier
        leverage = cfg.min_leverage + (cfg.max_leverage - cfg.min_leverage) * size_score * (1.0 - 0.45 * risk_score)
        leverage = 1.0 + (leverage - 1.0) * _clip(0.55 + 0.45 * size_multiplier, 0.35, 1.0)
        if is_fade:
            margin *= 0.62
            leverage = min(leverage, 1.85)
        margin = _clip(margin, cfg.min_margin_fraction, cfg.max_margin_fraction)
        leverage = _clip(leverage, cfg.min_leverage, cfg.max_leverage)
        notional = _clip(margin * leverage, 0.0, cfg.max_notional_exposure)
        target_notional, resize_delta, allow_resize, resize_reason = self._resize_plan(
            has_position=has_position,
            current_notional=current_notional,
            desired_notional=notional,
            risk_score=risk_score,
            entry_score=entry_score,
            unrealized_pnl=unrealized_pnl,
            daily_realized_pnl=daily_realized,
            daily_equity_drawdown=-daily_dd,
            loss_streak=loss_streak,
            hold_count=hold_count,
            bars_since_last_resize=bars_since_last_resize,
            should_exit=should_exit,
        )
        if not has_position and notional < cfg.min_notional_exposure:
            return RiskLeverageDecision(
                allow_entry=False,
                should_exit=bool(should_exit),
                effective_action=0.0,
                direction=int(direction),
                position_fraction=0.0,
                leverage=0.0,
                notional_exposure=0.0,
                target_notional_exposure=0.0,
                resize_notional_delta=0.0,
                allow_resize=False,
                conviction=float(conviction),
                risk_score=float(risk_score),
                entry_score=float(entry_score),
                quality_score=float(quality_score),
                event_score=float(event_score),
                setup_score=float(setup_score),
                stop_loss_pct=float(stop_loss_pct),
                take_profit_pct=float(take_profit_pct),
                trailing_stop_pct=float(trailing_stop_pct),
                max_hold_bars=int(dynamic_max_hold),
                size_multiplier=float(size_multiplier),
                exit_reason=str(exit_reason),
                block_reason="size_below_min",
                resize_reason="",
                sizing_reason="blocked",
            )

        # Direction is conveyed by action sign; notional is passed separately.
        eff_abs = max(cfg.min_signal_abs + 1e-4, min(1.0, abs_signal))
        effective_action = float(direction) * eff_abs
        return RiskLeverageDecision(
            allow_entry=True,
            should_exit=bool(should_exit),
            effective_action=float(effective_action),
            direction=int(direction),
            position_fraction=float(margin),
            leverage=float(leverage),
            notional_exposure=float(notional),
            target_notional_exposure=float(target_notional),
            resize_notional_delta=float(resize_delta),
            allow_resize=bool(allow_resize),
            conviction=float(conviction),
            risk_score=float(risk_score),
            entry_score=float(entry_score),
            quality_score=float(quality_score),
            event_score=float(event_score),
            setup_score=float(setup_score),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            trailing_stop_pct=float(trailing_stop_pct),
            max_hold_bars=int(dynamic_max_hold),
            size_multiplier=float(size_multiplier),
            exit_reason=str(exit_reason),
            block_reason="",
            resize_reason=str(resize_reason),
            sizing_reason="dynamic_risk_budget",
        )

    def _resize_plan(
        self,
        *,
        has_position: bool,
        current_notional: float,
        desired_notional: float,
        risk_score: float,
        entry_score: float,
        unrealized_pnl: float,
        daily_realized_pnl: float,
        daily_equity_drawdown: float,
        loss_streak: int,
        hold_count: int,
        bars_since_last_resize: int,
        should_exit: bool,
    ) -> tuple[float, float, bool, str]:
        cfg = self.config
        desired = _clip(desired_notional, 0.0, cfg.max_notional_exposure)
        current = _clip(current_notional, 0.0, cfg.max_notional_exposure)
        if not has_position:
            return float(desired), 0.0, False, ""
        if should_exit:
            return float(current), 0.0, False, "exit_pending"
        if not cfg.enable_position_resize:
            return float(current), 0.0, False, "resize_disabled"

        account_locked = bool(
            (cfg.daily_loss_limit > 0.0 and daily_realized_pnl <= -abs(cfg.daily_loss_limit))
            or (cfg.daily_drawdown_hard_limit > 0.0 and daily_equity_drawdown <= -abs(cfg.daily_drawdown_hard_limit))
            or (cfg.max_consecutive_losses > 0 and loss_streak >= int(cfg.max_consecutive_losses))
        )
        if account_locked:
            desired = min(current, desired)

        max_add = max(current * float(cfg.max_resize_add_fraction), cfg.min_resize_notional_delta)
        max_reduce = max(current * float(cfg.max_resize_reduce_fraction), cfg.min_resize_notional_delta)
        if desired > current:
            desired = min(desired, current + max_add)
        else:
            desired = max(desired, current - max_reduce)
        if desired > 0.0:
            desired = _clip(desired, cfg.min_notional_exposure, cfg.max_notional_exposure)

        delta = float(desired - current)
        abs_delta = abs(delta)
        min_delta = max(float(cfg.min_resize_notional_delta), current * 0.08)
        if abs_delta < min_delta:
            return float(current), 0.0, False, "delta_too_small"
        if hold_count < int(cfg.min_resize_hold_bars):
            return float(current), 0.0, False, "resize_hold_warmup"
        if bars_since_last_resize < int(cfg.resize_cooldown_bars):
            return float(current), 0.0, False, "resize_cooldown"
        if delta > 0.0:
            if unrealized_pnl < cfg.resize_add_min_unrealized:
                return float(current), 0.0, False, "no_add_to_loser"
            if entry_score < cfg.resize_add_min_entry_score:
                return float(current), 0.0, False, "add_edge_weak"
            if account_locked:
                return float(current), 0.0, False, "account_locked_no_add"
            return float(desired), float(delta), True, "resize_add"

        if risk_score < cfg.resize_reduce_risk_score and entry_score >= cfg.resize_add_min_entry_score and not account_locked:
            return float(current), 0.0, False, "reduce_not_needed"
        return float(desired), float(delta), True, "resize_reduce"

    def _event_score(self, row: Mapping[str, Any], direction: int) -> float:
        if direction == 0:
            return 0.0
        pmax = _row_float(row, "evt_det_prob_max", 0.0)
        lp = _row_float(row, "evt_det_long_prob", 0.0)
        sp = _row_float(row, "evt_det_short_prob", 0.0)
        edge = _row_float(row, "evt_det_edge", 0.0)
        side_prob = lp if direction > 0 else sp
        contra_prob = sp if direction > 0 else lp
        side_edge = edge if direction > 0 else -edge
        return _clip(0.25 + 0.35 * pmax + 0.25 * (side_prob - contra_prob + 0.5) + 0.15 * _norm01(side_edge, 0.04), 0.0, 1.0)

    def _quality_score(self, row: Mapping[str, Any], direction: int) -> float:
        up = _row_float(row, "m7_trend_xgb_up", 0.0)
        dn = _row_float(row, "m7_trend_xgb_dn", 0.0)
        side_prob = up if direction > 0 else dn
        q = _norm01(_row_float(row, "m7_quality_pred", 0.0), 0.003)
        exp_ret = direction * _row_float(row, "m7_expected_ret", 0.0)
        exp_score = _norm01(exp_ret, 0.003)
        composite = _norm01(_row_float(row, "m7_composite_score", 0.0), 0.10)
        return _clip(0.30 * side_prob + 0.30 * q + 0.25 * exp_score + 0.15 * composite, 0.0, 1.0)

    def _flow_score(self, row: Mapping[str, Any], direction: int) -> float:
        if direction == 0:
            return 0.0
        flow = direction * _row_float(row, "smart_money_flow", 0.0)
        taker = direction * _row_float(row, "taker_acceleration", 0.0)
        mtf = direction * (0.55 * _row_float(row, "mtf_trend_1h", 0.0) + 0.45 * _row_float(row, "mtf_trend_4h", 0.0))
        return _clip(0.40 * _norm01(flow, 0.05) + 0.35 * _norm01(taker, 0.05) + 0.25 * _norm01(mtf, 1.0), 0.0, 1.0)

    def _regime_score(self, row: Mapping[str, Any], current_regime: str, candidate_regime: str, admitted: bool) -> float:
        same = str(candidate_regime or "").lower() == str(current_regime or "").lower()
        fade_fit = str(candidate_regime or "").lower() == "fade" and str(current_regime or "").lower() in {"chop", "whipsaw", "normal"}
        fit = 1.0 if same or fade_fit else (0.68 if admitted else 0.35)
        return _clip((1.0 if admitted else 0.65) * fit, 0.0, 1.0)

    def _setup_score(self, row: Mapping[str, Any], direction: int, current_regime: str, candidate_regime: str) -> float:
        if direction == 0:
            return 0.0
        if str(candidate_regime or "").lower() != "fade":
            return _clip(0.45 * self._event_score(row, direction) + 0.35 * self._quality_score(row, direction) + 0.20 * self._flow_score(row, direction), 0.0, 1.0)

        current = str(current_regime or "").lower()
        regime_fit = 1.0 if current in {"chop", "whipsaw", "normal"} else 0.35
        rsi = _row_float(row, "rsi", 50.0)
        log_ret = _row_float(row, "log_return", 0.0)
        taker = _row_float(row, "taker_acceleration", 0.0)
        edge = _row_float(row, "evt_det_edge", 0.0)
        pmax = _row_float(row, "evt_det_prob_max", 0.0)
        mtf = 0.55 * _row_float(row, "mtf_trend_1h", 0.0) + 0.45 * _row_float(row, "mtf_trend_4h", 0.0)

        oversold = max(0.0, (42.0 - rsi) / 18.0) + max(0.0, -log_ret / 0.0045) + max(0.0, -taker / 0.05)
        overbought = max(0.0, (rsi - 58.0) / 18.0) + max(0.0, log_ret / 0.0045) + max(0.0, taker / 0.05)
        reversion = oversold if direction > 0 else overbought
        wrong_way = overbought if direction > 0 else oversold
        spike = max(abs(edge) / 0.045, abs(log_ret) / 0.0045, abs(taker) / 0.05)
        event = 1.0 if pmax >= 0.58 or abs(edge) >= 0.035 or abs(log_ret) >= 0.004 else 0.0
        trend_against_fade = max(0.0, direction * mtf)
        raw = 0.42 * _clip(reversion / 2.2, 0.0, 1.0) + 0.24 * _clip(spike / 1.4, 0.0, 1.0) + 0.18 * event + 0.16 * _clip((reversion - wrong_way + 1.0) / 2.0, 0.0, 1.0)
        raw -= 0.22 * _clip(trend_against_fade, 0.0, 1.0)
        return _clip(raw * regime_fit, 0.0, 1.0)

    def _risk_score(
        self,
        *,
        row: Mapping[str, Any],
        current_regime: str,
        candidate_regime: str,
        position_side: str | None,
        position_regime: str | None,
        unrealized_pnl: float,
        hold_count: int,
        hard_veto: bool,
    ) -> float:
        tail = _clip(abs(_row_float(row, "m7_tail_risk", 0.0)) / 0.012, 0.0, 1.0)
        qwidth = _clip(abs(_row_float(row, "m7_qwidth", 0.0)) / 0.018, 0.0, 1.0)
        vol = _clip(abs(_row_float(row, "volatility_z", 0.0)) / 3.0, 0.0, 1.0)
        rs_vol = _clip(abs(_row_float(row, "rogers_satchell_vol", 0.0)) / 0.012, 0.0, 1.0)
        illiq = _clip(abs(_row_float(row, "amihud_illiquidity_z", 0.0)) / 4.0, 0.0, 1.0)
        vacuum = _clip(abs(_row_float(row, "liquidity_vacuum", 0.0)), 0.0, 1.0)
        crowd = _clip(abs(_row_float(row, "crowding_pressure", 0.0)), 0.0, 1.0)
        exec_bad = _clip(1.0 - _norm01(_row_float(row, "execution_quality", 0.0), 1.0), 0.0, 1.0)
        mismatch = bool(position_side and position_regime and str(position_regime).lower() != str(current_regime).lower())
        adverse = _clip(max(0.0, -float(unrealized_pnl)) / 0.03, 0.0, 1.0)
        hold_risk = _clip(float(max(hold_count, 0)) / max(float(self.config.max_hold_bars), 1.0), 0.0, 1.0)
        # A legacy veto is no longer an external gate, but it is a strong
        # warning that the alpha setup is incomplete. Keep it inside the
        # unified risk score so the engine can still override only exceptional
        # high-conviction cases.
        legacy_gate_risk = self.config.hard_veto_risk_add if hard_veto else 0.0
        position_risk = max(adverse, 0.55 * hold_risk, 0.72 if mismatch else 0.0)
        market_risk = 0.18 * tail + 0.15 * qwidth + 0.15 * vol + 0.12 * rs_vol + 0.12 * illiq + 0.10 * vacuum + 0.08 * crowd + 0.10 * exec_bad
        regime_risk = 1.0 if mismatch else 0.0
        return _clip(0.55 * market_risk + 0.25 * regime_risk + 0.20 * position_risk + legacy_gate_risk, 0.0, 1.0)

    def _dynamic_position_controls(self, *, risk_score: float, conviction: float, is_fade: bool) -> tuple[float, float, float, int]:
        cfg = self.config
        risk = _clip(risk_score, 0.0, 1.0)
        conv = _clip(conviction, 0.0, 1.0)
        stop = -_clip(cfg.stop_loss_base * (0.85 + 0.55 * risk) / max(0.65 + conv, 1e-8), 0.006, 0.024)
        take = _clip(cfg.take_profit_base * (0.65 + 1.25 * conv) * (1.0 - 0.20 * risk), 0.006, 0.035)
        trail = _clip(cfg.trailing_stop_base * (1.10 + 0.60 * risk) / max(0.80 + 0.50 * conv, 1e-8), 0.003, 0.014)
        hold_scale = _clip(1.10 + 0.28 * conv - 0.42 * risk, 0.35, 1.25)
        max_hold = int(round(float(cfg.max_hold_bars) * hold_scale))
        if is_fade:
            max_hold = min(max_hold, 48)
            take = min(take, 0.018)
            trail = min(trail, 0.009)
        return float(stop), float(take), float(trail), int(max(12, min(max_hold, max(int(cfg.max_hold_bars), 12))))

    def _size_multiplier(self, *, daily_realized_pnl: float, daily_equity_drawdown: float, consecutive_losses: int) -> float:
        cfg = self.config
        mult = 1.0
        streak = int(max(0, consecutive_losses))
        if streak >= int(cfg.loss_streak_soft_start):
            steps = streak - int(cfg.loss_streak_soft_start) + 1
            mult *= float(cfg.loss_streak_size_cut) ** float(max(0, steps))
        daily_dd = max(0.0, -_safe_float(daily_equity_drawdown, 0.0))
        if cfg.daily_drawdown_hard_limit > cfg.daily_drawdown_soft_limit > 0.0 and daily_dd > cfg.daily_drawdown_soft_limit:
            dd_frac = _clip(
                (daily_dd - cfg.daily_drawdown_soft_limit) / (cfg.daily_drawdown_hard_limit - cfg.daily_drawdown_soft_limit),
                0.0,
                1.0,
            )
            mult *= 1.0 - (1.0 - _clip(cfg.daily_drawdown_size_cut, 0.05, 1.0)) * dd_frac
        if cfg.daily_loss_limit > 0.0 and daily_realized_pnl < 0.0:
            loss_frac = _clip(abs(daily_realized_pnl) / abs(cfg.daily_loss_limit), 0.0, 1.0)
            if loss_frac > 0.35:
                mult *= 1.0 - 0.25 * _clip((loss_frac - 0.35) / 0.65, 0.0, 1.0)
        return _clip(mult, 0.15, 1.0)

    def _exit_reason(
        self,
        *,
        risk_score: float,
        entry_score: float,
        current_regime: str,
        position_regime: str | None,
        unrealized_pnl: float,
        peak_unrealized_pnl: float,
        hold_count: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        trailing_stop_pct: float,
        max_hold_bars: int,
        daily_loss_locked: bool,
        daily_drawdown_locked: bool,
    ) -> str:
        if not position_regime:
            return ""
        mismatch = str(current_regime or "").lower() != str(position_regime or "").lower()
        if daily_loss_locked and unrealized_pnl <= 0.0:
            return "daily_loss_lock"
        if daily_drawdown_locked and unrealized_pnl <= 0.0:
            return "daily_drawdown_lock"
        if unrealized_pnl <= stop_loss_pct:
            return "dynamic_stop_loss"
        if unrealized_pnl <= self.config.unlevered_stop:
            return "hard_stop_loss"
        if hold_count < self.config.min_hold_bars_before_risk_exit:
            return ""
        if unrealized_pnl >= take_profit_pct:
            return "take_profit"
        trail_floor = max(float(self.config.profit_lock_floor), float(peak_unrealized_pnl) - float(trailing_stop_pct))
        if peak_unrealized_pnl >= self.config.profit_lock_start and unrealized_pnl <= trail_floor:
            return "profit_trailing_lock"
        if risk_score >= self.config.exit_risk and unrealized_pnl <= 0.0:
            return "risk_exit"
        if mismatch and risk_score >= 0.72:
            return "regime_mismatch"
        if hold_count >= max_hold_bars and entry_score < 0.0:
            return "time_stop"
        return ""
