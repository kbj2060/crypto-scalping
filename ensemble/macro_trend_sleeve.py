from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroTrendSleeveConfig:
    lookback_bars: int = 6048
    threshold: float = 0.05
    persist_updates: int = 5
    update_bars: int = 288
    notional_exposure: float = 3.0
    leverage: float = 5.0
    min_history_bars: int = 6048
    bootstrap_current: bool = True
    take_profit: float = 1.25
    stop_loss: float = 0.0
    trailing_arm: float = 0.0
    trailing_gap: float = 0.0
    lockout_bars: int = 0
    lockout_until_signal_change: bool = True
    lockout_on_any_close: bool = False


@dataclass(frozen=True)
class MacroTrendSleeveDecision:
    allow_entry: bool
    side: str
    signal: int
    momentum: float
    notional_exposure: float
    leverage: float
    position_fraction: float
    block_reason: str
    source: str = "macro_trend_sleeve_v1"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _close_array(frame: pd.DataFrame) -> np.ndarray:
    if "close" not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return (
        pd.to_numeric(frame["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def macro_trend_signal(frame: pd.DataFrame, cfg: MacroTrendSleeveConfig | None = None) -> tuple[int, float, str]:
    cfg = cfg or MacroTrendSleeveConfig()
    close = _close_array(frame)
    n = len(close)
    lookback = int(max(1, cfg.lookback_bars))
    if n <= max(lookback, int(cfg.min_history_bars)):
        return 0, 0.0, "insufficient_history"
    if not np.isfinite(close[-1]) or not np.isfinite(close[-lookback - 1]) or close[-lookback - 1] <= 0.0:
        return 0, 0.0, "bad_close_history"

    desired = np.zeros(n, dtype=np.int8)
    mom = np.full(n, np.nan, dtype=np.float64)
    mom[lookback:] = close[lookback:] / np.maximum(close[:-lookback], 1e-12) - 1.0
    desired[mom > float(cfg.threshold)] = 1
    desired[mom < -float(cfg.threshold)] = -1

    current = 0
    pending = 0
    pending_count = 0
    update_bars = max(int(cfg.update_bars), 1)
    persist = max(int(cfg.persist_updates), 1)
    for i, raw in enumerate(desired):
        if i % update_bars != 0:
            continue
        raw_i = int(raw)
        if raw_i == current:
            pending = 0
            pending_count = 0
            continue
        if raw_i == pending:
            pending_count += 1
        else:
            pending = raw_i
            pending_count = 1
        if pending_count >= persist:
            current = raw_i
            pending = 0
            pending_count = 0

    last_mom = float(mom[-1]) if np.isfinite(mom[-1]) else 0.0
    if current == 0 and bool(cfg.bootstrap_current) and abs(last_mom) > float(cfg.threshold):
        return (1 if last_mom > 0.0 else -1), last_mom, ""
    if current == 0:
        return 0, last_mom, "momentum_not_confirmed"
    return int(current), last_mom, ""


def macro_trend_decision(frame: pd.DataFrame, cfg: MacroTrendSleeveConfig | None = None) -> MacroTrendSleeveDecision:
    cfg = cfg or MacroTrendSleeveConfig()
    signal, momentum, block_reason = macro_trend_signal(frame, cfg)
    if signal == 0:
        return MacroTrendSleeveDecision(
            False,
            "NONE",
            0,
            float(momentum),
            0.0,
            0.0,
            0.0,
            block_reason or "no_signal",
        )
    notional = float(max(cfg.notional_exposure, 0.0))
    leverage = float(max(cfg.leverage, 1.0))
    fraction = float(np.clip(notional / max(leverage, 1e-8), 0.0, 1.0))
    return MacroTrendSleeveDecision(
        True,
        "LONG" if signal > 0 else "SHORT",
        int(signal),
        float(momentum),
        notional,
        leverage,
        fraction,
        "",
    )
