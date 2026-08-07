"""
VAL/OOS fresh-forward test of the weekend-gap mean-reversion pattern found in
research_market_session_effects_20260729.py (screen: ETH r=-0.43, SOL r=-0.43,
BTC r=-0.22 between weekend Fri-close->Mon-open drift and Monday's first-hour move).

Rule under test (fully mechanical, no external data needed):
  Entry:  Monday 09:30 America/New_York (US equity open), once per week.
  Direction: fade the weekend move -> short if weekend_gap_ret > 0, long if < 0.
  Filter: only take the trade if |weekend_gap_ret| > threshold (threshold chosen
          from a small SET OF ROUND, NOT DATA-FITTED, CANDIDATES: 0/0.5/1/1.5/2%,
          selected on VAL only, then frozen and replayed mechanically on OOS).
  Exit:   fixed 1-hour hold (10:30 ET), matching where the screen's correlation
          was strongest (r=-0.43/-0.43/-0.22 vs r=-0.23/-0.34/-0.17 for full day).
  Sizing: CLAUDE.md canonical futures contract -- margin_fraction=0.30, leverage=3
          -> notional=0.90 (fixed; sizing is NOT part of the hypothesis under test).
  Cost:   round-trip = 2*(fee_bps+slip_bps)/1e4 * notional, using this repo's
          existing backtest convention (fee=2bps, slip=1bps -> 6bps round trip),
          scripts/backtest_m7_signal_only.py argparse defaults.

Split (fixed per CLAUDE.md Fresh-Forward rule):
  VAL: 2025-09-01 .. 2025-12-31
  OOS: 2026-01-01 .. 2026-03-31

Flags: fresh_forward_bar_by_bar=true (mechanical week-by-week replay using only
info available at Monday 09:30 ET), trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false
(entry direction uses only the already-closed weekend gap, exit is a fixed
forward-looking hold not fit to future data).

KNOWN LIMITATIONS (explicit, not silently hidden):
  - US market holidays landing on Monday (Labor Day, MLK Day, Presidents Day,
    Memorial Day) are NOT excluded -- crypto trades those days but there's no
    real "9:30 ET open" event, so those weeks are noise in the sample, not signal.
  - n is small (~17 VAL weeks, ~13 OOS weeks) -- a handful of anomalous weeks can
    swing the result; this is a screen, not a promotion-grade test.
  - No DSR/PBO multiple-testing correction applied (core/selection_stats.py is
    only on branch claude/optimal-trading-formula-plan-2x7sml, not on main).
  - Single fixed 1-hour exit only; no TP/SL price-move contract applied.
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KLINES = REPO / "binance_data" / "klines"
ET = ZoneInfo("America/New_York")

VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE  # 0.90, per CLAUDE.md canonical example

FEE_BPS = 2.0
SLIP_BPS = 1.0
ROUND_TRIP_COST_FRAC = 2.0 * (FEE_BPS + SLIP_BPS) / 1e4 * NOTIONAL  # applied to equity

THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02]


def load_weekly_gap_table(symbol: str) -> pd.DataFrame:
    path = KLINES / symbol / f"{symbol}-5m-api.csv"
    raw = pd.read_csv(path, usecols=["timestamp", "close"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize("UTC")
    raw = raw.set_index("timestamp").sort_index()
    raw["et_time"] = raw.index.tz_convert(ET)

    dow = raw["et_time"].dt.dayofweek
    minute_of_day = raw["et_time"].dt.hour * 60 + raw["et_time"].dt.minute

    fri_close = raw[(dow == 4) & minute_of_day.between(15 * 60 + 55, 15 * 60 + 59)]
    mon_open = raw[(dow == 0) & minute_of_day.between(9 * 60 + 25, 9 * 60 + 29)]
    mon_first_hour_end = raw[(dow == 0) & minute_of_day.between(10 * 60 + 25, 10 * 60 + 29)]

    def daily_first(g):
        s = g.groupby(g.index.date)["close"].first()
        s.index = pd.to_datetime(s.index)
        return s

    fc = daily_first(fri_close)
    mo = daily_first(mon_open)
    mfh = daily_first(mon_first_hour_end)

    weeks = pd.DataFrame({"fri_close": fc})
    weeks["monday_date"] = weeks.index + pd.Timedelta(days=3)
    weeks = weeks.set_index("monday_date")
    weeks["mon_open"] = mo
    weeks["mon_first_hour_end"] = mfh
    weeks = weeks.dropna()

    weeks["weekend_gap_ret"] = np.log(weeks["mon_open"] / weeks["fri_close"])
    weeks["mon_first_hour_ret"] = np.log(weeks["mon_first_hour_end"] / weeks["mon_open"])
    weeks["symbol"] = symbol
    return weeks.reset_index()


def simulate(weeks: pd.DataFrame, threshold: float) -> dict:
    df = weeks.copy()
    df["direction"] = -np.sign(df["weekend_gap_ret"])
    df["take_trade"] = df["weekend_gap_ret"].abs() > threshold
    traded = df[df["take_trade"]].copy()

    traded["price_move"] = traded["direction"] * traded["mon_first_hour_ret"]
    traded["account_pnl_frac"] = traded["price_move"] * NOTIONAL - ROUND_TRIP_COST_FRAC

    n = len(traded)
    if n == 0:
        return dict(n_trades=0, total_return_pct=0.0, mdd_pct=0.0, win_rate=np.nan, mean_pnl_bps=np.nan)

    equity = (1.0 + traded["account_pnl_frac"]).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    mdd = drawdown.min()

    return dict(
        n_trades=n,
        total_return_pct=(equity.iloc[-1] - 1.0) * 100,
        mdd_pct=mdd * 100,
        win_rate=(traded["account_pnl_frac"] > 0).mean(),
        mean_pnl_bps=traded["account_pnl_frac"].mean() * 1e4,
    )


def main():
    print(f"NOTIONAL={NOTIONAL}, round_trip_cost={ROUND_TRIP_COST_FRAC*1e4:.2f}bps of notional\n", file=sys.stderr)

    all_val_sweeps = []
    chosen = {}

    for symbol in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
        path = KLINES / symbol / f"{symbol}-5m-api.csv"
        if not path.exists():
            continue
        weeks = load_weekly_gap_table(symbol)

        val_weeks = weeks[(weeks["monday_date"] >= VAL_START) & (weeks["monday_date"] <= VAL_END)]
        oos_weeks = weeks[(weeks["monday_date"] >= OOS_START) & (weeks["monday_date"] <= OOS_END)]

        print(f"=== {symbol}: VAL threshold sweep ({len(val_weeks)} VAL weeks available) ===")
        for th in THRESHOLDS:
            res = simulate(val_weeks, th)
            res.update(symbol=symbol, threshold=th, split="VAL")
            all_val_sweeps.append(res)
            print(f"  th={th:.3f}  n={res['n_trades']:3d}  return={res['total_return_pct']:7.2f}%  "
                  f"mdd={res['mdd_pct']:7.2f}%  win_rate={res['win_rate']}")

        # pick threshold maximizing VAL total return among thresholds with >=5 trades
        candidates = [r for r in all_val_sweeps if r["symbol"] == symbol and r["n_trades"] >= 5]
        if not candidates:
            print(f"  {symbol}: no threshold has >=5 VAL trades, skipping OOS confirmation\n")
            continue
        best = max(candidates, key=lambda r: r["total_return_pct"])
        chosen[symbol] = best["threshold"]
        print(f"  -> chosen threshold (max VAL return, n>=5): {best['threshold']:.3f}\n")

    print("\n=== OOS confirmation (threshold frozen from VAL, replayed mechanically) ===")
    print("fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, "
          "saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false\n")
    oos_rows = []
    for symbol, th in chosen.items():
        weeks = load_weekly_gap_table(symbol)
        oos_weeks = weeks[(weeks["monday_date"] >= OOS_START) & (weeks["monday_date"] <= OOS_END)]
        res = simulate(oos_weeks, th)
        res.update(symbol=symbol, threshold=th, split="OOS")
        oos_rows.append(res)
        print(f"{symbol}: threshold={th:.3f}  n_trades={res['n_trades']}  "
              f"OOS_return={res['total_return_pct']:.2f}%  OOS_mdd={res['mdd_pct']:.2f}%  "
              f"win_rate={res['win_rate']}  mean_pnl_bps={res['mean_pnl_bps']:.2f}")

    out_dir = REPO / "data" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_val_sweeps).to_csv(out_dir / "weekend_meanrev_val_sweep_20260729.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(out_dir / "weekend_meanrev_oos_confirm_20260729.csv", index=False)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
