#!/usr/bin/env python3
"""Priority-3 test: replace the L6 governor's rule-based NFP approximation ("first Friday of
the month") with verified real release dates for the period overlapping validation/OOS
(2025-10-01..2026-02-28), and test veto-only / haircut-only / combined modes against the frozen
v2 winner.

Verified via WebFetch (investing.com economic calendar, 2026-07-04): the 2025 government
shutdown badly disrupted the NFP release schedule in exactly the window this project's
validation/OOS periods cover:
  - September 2025 NFP: released 2025-11-20 (rule-based guess: 2025-10-03 -- wrong)
  - October 2025 NFP: released 2025-12-16 (rule-based guess: 2025-11-07 -- wrong)
  - November 2025 NFP: released 2025-12-16, same day as October's (rule-based guess:
    2025-12-05 -- wrong)
  - December 2025 NFP: released 2026-01-09 (rule-based guess: 2026-01-02 -- close but off by a
    week)
Every rule-based NFP veto window the L6 governor would have applied inside Oct 2025-Feb 2026 was
wrong; the real veto windows are at completely different dates. ISM Manufacturing/Services PMI
are private-sector releases (Institute for Supply Management), not government data, so were not
delayed by the shutdown -- left on the existing rule-based approximation. FOMC dates were already
verified in an earlier session (federalreserve.gov).

Jan-Aug 2025 NFP dates are left on the rule-based approximation for this test (lower priority --
outside the val/OOS window this project scores), with two known exceptions applied from general
knowledge of the 2025 calendar (Jan delayed a week past New Year's, Jul shifted earlier due to
the Jul 4 holiday) -- these do not affect the val/OOS scoring window either way.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
import replay_omega6_v2_l4l6_20260704 as l4l6  # noqa: E402

# Verified NFP release dates (see module docstring). Supersedes the rule-based "first Friday"
# approximation for the months that matter to this project's val/OOS window.
VERIFIED_NFP_DATES_2025_2026 = {
    "2025-01": "2025-01-10",  # delayed a week past New Year's (known 2025 calendar fact)
    "2025-02": "2025-02-07",
    "2025-03": "2025-03-07",
    "2025-04": "2025-04-04",
    "2025-05": "2025-05-02",
    "2025-06": "2025-06-06",
    "2025-07": "2025-07-03",  # shifted earlier due to Jul 4 holiday
    "2025-08": "2025-08-01",
    "2025-09": "2025-11-20",  # WebFetch-verified: shutdown-delayed
    "2025-10": "2025-12-16",  # WebFetch-verified: shutdown-delayed
    "2025-11": "2025-12-16",  # WebFetch-verified: shutdown-delayed, same day as Oct release
    "2025-12": "2026-01-09",  # WebFetch-verified
}


def _et_to_utc_naive(day: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
    ny = ZoneInfo("America/New_York")
    dt = datetime(int(day.year), int(day.month), int(day.day), int(hour), int(minute), tzinfo=ny)
    return pd.Timestamp(dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))


def verified_nfp_events() -> list[pd.Timestamp]:
    return [_et_to_utc_naive(pd.Timestamp(d), 8, 30) for d in VERIFIED_NFP_DATES_2025_2026.values()]


def build_macro_veto_mask_real_nfp(timestamps: pd.Series) -> np.ndarray:
    """Same as l4l6.build_macro_veto_mask but with the NFP portion replaced by verified dates
    instead of the rule-based first-Friday approximation. ISM/FOMC portions unchanged."""
    years = sorted({t.year for t in timestamps} | {t.year - 1 for t in timestamps} | {t.year + 1 for t in timestamps})
    all_events: list[pd.Timestamp] = []
    for y in years:
        for month in range(1, 13):
            manufacturing = l4l6._nth_weekday(y, month, 1)
            all_events.append(l4l6._et_to_utc_naive(manufacturing, 10, 0))
            services = l4l6._nth_weekday(y, month, 3)
            all_events.append(l4l6._et_to_utc_naive(services, 10, 0))
            flash = l4l6._weekday_on_or_after(y, month, 23)
            all_events.append(l4l6._et_to_utc_naive(flash, 9, 45))
        for raw in l4l6.L6_FOMC_DECISION_DATES.get(int(y), ()):
            all_events.append(l4l6._et_to_utc_naive(pd.Timestamp(raw), 14, 0))
    all_events.extend(verified_nfp_events())

    events_arr = pd.Series(all_events).sort_values().to_numpy()
    veto = np.zeros(len(timestamps), dtype=bool)
    ts_arr = timestamps.to_numpy()
    for i, ts in enumerate(ts_arr):
        ts = pd.Timestamp(ts)
        for ev in events_arr:
            ev = pd.Timestamp(ev)
            if ev - pd.Timedelta(minutes=l4l6.L6_MACRO_PRE_MINUTES) <= ts <= ev + pd.Timedelta(minutes=l4l6.L6_MACRO_POST_MINUTES):
                veto[i] = True
                break
    return veto


def run_real_calendar(tape: pd.DataFrame, *, start, end, use_veto: bool, use_haircut: bool, fee_mult: float) -> dict:
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(dtype=np.float64)
    open_ = sub["open"].to_numpy(dtype=np.float64)

    primary_side_arr = sub["primary_side"].to_numpy(dtype=np.int64)
    fallback_side_arr = sub["fallback_side"].to_numpy(dtype=np.int64)
    eff_side = np.where(primary_side_arr != 0, primary_side_arr, fallback_side_arr)
    persistence_ok = eff_side != 0
    for k in range(1, 3):
        shifted = np.roll(eff_side, k)
        shifted[:k] = 0
        persistence_ok &= shifted == eff_side

    atr_pct_arr = sub["atr_pct"].to_numpy(dtype=np.float64)
    macro_veto = build_macro_veto_mask_real_nfp(sub["timestamp"]) if use_veto else np.zeros(n, dtype=bool)
    shock_haircut = l4l6.build_shock_haircut_mask(sub) if use_haircut else np.zeros(n, dtype=bool)

    FEE = 0.00020 * fee_mult
    SLIP = 0.00050 * fee_mult
    fixed_margin, fixed_leverage = 0.30, 2.0
    tp_atr_mult, sl_atr_mult, cooldown_bars = 15.0, 5.0, 12

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    hold_start = 0
    notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 288
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or not persistence_ok[i] or eff_side[i] == 0:
                i += 1
                continue
            if macro_veto[i]:
                i += 1
                continue
            side = int(eff_side[i])
            margin, leverage = fixed_margin, fixed_leverage
            if shock_haircut[i]:
                margin *= l4l6.L6_SHOCK_NOTIONAL_SCALE
            atr = max(atr_pct_arr[i], 1e-6)
            tp, sl = tp_atr_mult * atr, sl_atr_mult * atr
            entry_price = float(open_[min(i + 1, n - 1)]) * (1.0 + SLIP if side > 0 else 1.0 - SLIP)
            pos = side
            notional = margin * leverage
            take_profit, stop_loss = tp, sl
            hold_start = i
            entry_equity = cash
            cash -= cash * FEE * notional
            i += 1
            continue
        px = close[i]
        raw = (px * (1.0 - SLIP) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + SLIP)) / max(entry_price, 1e-12)
        unreal = raw * notional
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        hold_bars = i - hold_start
        reason = ""
        if take_profit > 0.0 and unreal >= take_profit:
            reason = "take_profit"
        elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
            reason = "stop_loss"
        elif hold_bars >= max_hold:
            reason = "time_stop"
        if reason:
            exit_price = close[i] * (1.0 - SLIP if pos > 0 else 1.0 + SLIP)
            raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
            before = cash
            cash = cash * (1.0 + raw_exit * notional)
            cash -= before * FEE * notional
            trades.append({"win": bool(cash > entry_equity)})
            pos = 0
            cooldown_until = i + cooldown_bars
        i += 1
    wins = sum(1 for t in trades if t["win"])
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": len(trades), "wr": float(wins / len(trades)) if trades else 0.0}


def main() -> int:
    tape_raw = v2.load_tape()
    tape = v2.apply_quality_threshold(tape_raw, 0.58)

    scenarios = [
        ("baseline_no_l6", dict(use_veto=False, use_haircut=False)),
        ("real_nfp_veto_only", dict(use_veto=True, use_haircut=False)),
        ("shock_haircut_only", dict(use_veto=False, use_haircut=True)),
        ("real_nfp_veto_and_haircut", dict(use_veto=True, use_haircut=True)),
    ]
    for name, kw in scenarios:
        out = {}
        for tag, mult in (("cost1", 1.0), ("cost3", 3.0)):
            out[tag] = run_real_calendar(tape, start=v2.VAL_START, end=v2.VAL_END, fee_mult=mult, **kw)
        print(
            f"{name}: cost1 pnl={out['cost1']['pnl']:.2f}% mdd={out['cost1']['mdd']:.2f}% trades={out['cost1']['trades']} wr={out['cost1']['wr']:.3f} | "
            f"cost3 pnl={out['cost3']['pnl']:.2f}% mdd={out['cost3']['mdd']:.2f}% trades={out['cost3']['trades']} wr={out['cost3']['wr']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
