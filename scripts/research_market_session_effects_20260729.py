"""
Free proxy test (no paid data needed) for whether a TradFi-session-aware feature
(intraday DXY/SPX/futures) could plausibly carry tradeable signal for crypto.

Rationale: if crypto reacts measurably to the US equity session in its OWN price
behavior (higher realized vol during market hours, a reaction burst right at the
9:30am ET open, or a predictable "catch-up" move after weekend equity-market-closed
gaps), that's evidence intraday TradFi data COULD help. If crypto shows no such
session structure at all, paying for intraday equity/futures/DXY data is unlikely
to reveal anything -- the reaction speed argument from the correlation screen
(macro_correlation_screen_20260729.csv) would be confirmed independently.

Uses ONLY data already on disk (binance_data/klines/*), no network calls.

Tests:
  1. Realized vol / mean move, split by session:
       us_market_hours   : Mon-Fri, 09:30-16:00 America/New_York (DST-aware)
       us_closed_weekday : Mon-Fri, outside that window
       weekend           : Sat-Sun
  2. "Open reaction": abs 5m return in the first 30 min after 09:30 ET open,
     vs a matched sample of 30-min windows during us_closed_weekday.
  3. Weekend gap -> Monday catch-up: does the crypto move accumulated over the
     weekend (Fri 16:00 ET close -> Mon 09:30 ET open) predict Monday's first-hour
     move (continuation) or day move? This is the "macro catch-up" pattern that
     would justify intraday equity data if it were real.
"""
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
KLINES = REPO / "binance_data" / "klines"
ET = ZoneInfo("America/New_York")


def load_5m(symbol: str) -> pd.DataFrame:
    path = KLINES / symbol / f"{symbol}-5m-api.csv"
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df = df.set_index("timestamp").sort_index()
    df["logret"] = np.log(df["close"]).diff()
    df["et_time"] = df.index.tz_convert(ET)
    return df.dropna(subset=["logret"])


def classify_session(et_time: pd.Series) -> pd.Series:
    dow = et_time.dt.dayofweek  # 0=Mon .. 6=Sun
    minute_of_day = et_time.dt.hour * 60 + et_time.dt.minute
    is_weekday = dow < 5
    is_market_hours = is_weekday & (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)
    seg = np.where(~is_weekday, "weekend", np.where(is_market_hours, "us_market_hours", "us_closed_weekday"))
    return pd.Series(seg, index=et_time.index)


def session_stats(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["segment"] = classify_session(df["et_time"])
    g = df.groupby("segment")["logret"]
    out = pd.DataFrame({
        "n_bars": g.count(),
        "mean_signed_ret_bps": g.mean() * 1e4,
        "std_ret_bps": g.std() * 1e4,
        "mean_abs_ret_bps": g.apply(lambda x: x.abs().mean()) * 1e4,
    })
    out.insert(0, "symbol", symbol)
    return out


def open_reaction_test(df: pd.DataFrame, symbol: str):
    et = df["et_time"]
    dow = et.dt.dayofweek
    minute_of_day = et.dt.hour * 60 + et.dt.minute
    is_weekday = dow < 5

    open_window = is_weekday & (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 10 * 60)
    open_abs_ret = df.loc[open_window, "logret"].abs() * 1e4

    closed_weekday = is_weekday & ~((minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60))
    baseline_abs_ret = df.loc[closed_weekday, "logret"].abs() * 1e4

    t, p = stats.ttest_ind(open_abs_ret, baseline_abs_ret, equal_var=False)
    return dict(
        symbol=symbol,
        open_30min_mean_abs_bps=open_abs_ret.mean(),
        open_30min_n=len(open_abs_ret),
        baseline_mean_abs_bps=baseline_abs_ret.mean(),
        baseline_n=len(baseline_abs_ret),
        t_stat=t,
        p_value=p,
    )


def weekend_catchup_test(df: pd.DataFrame, symbol: str):
    close = df["close"] if "close" in df.columns else None
    prices = df["close"] if "close" in df else None
    # reload closes aligned with returns index
    s = df["logret"]
    et = df["et_time"]
    dow = et.dt.dayofweek
    minute_of_day = et.dt.hour * 60 + et.dt.minute

    # Friday close marker: last bar before 16:00 ET on a Friday
    fri_close_mask = (dow == 4) & (minute_of_day >= 15 * 60 + 55) & (minute_of_day < 16 * 60)
    mon_open_mask = (dow == 0) & (minute_of_day >= 9 * 60 + 25) & (minute_of_day < 9 * 60 + 30)
    mon_firsthour_end_mask = (dow == 0) & (minute_of_day >= 10 * 60 + 25) & (minute_of_day < 10 * 60 + 30)
    mon_dayend_mask = (dow == 0) & (minute_of_day >= 15 * 60 + 55) & (minute_of_day < 16 * 60)

    px = df["close_px"] if "close_px" in df.columns else None
    return fri_close_mask, mon_open_mask, mon_firsthour_end_mask, mon_dayend_mask


def main():
    all_session_stats = []
    open_reactions = []
    catchup_rows = []

    for symbol in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
        path = KLINES / symbol / f"{symbol}-5m-api.csv"
        if not path.exists():
            print(f"skip {symbol}: no file", file=sys.stderr)
            continue
        raw = pd.read_csv(path, usecols=["timestamp", "close"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize("UTC")
        raw = raw.set_index("timestamp").sort_index()
        raw["close_px"] = raw["close"]
        raw["logret"] = np.log(raw["close"]).diff()
        raw["et_time"] = raw.index.tz_convert(ET)
        df = raw.dropna(subset=["logret"])

        print(f"=== {symbol}: session realized-vol/return ===", file=sys.stderr)
        all_session_stats.append(session_stats(df, symbol))
        open_reactions.append(open_reaction_test(df, symbol))

        # weekend gap -> Monday catch-up
        dow = df["et_time"].dt.dayofweek
        minute_of_day = df["et_time"].dt.hour * 60 + df["et_time"].dt.minute
        fri_close = df[(dow == 4) & (minute_of_day.between(15 * 60 + 55, 15 * 60 + 59))]
        mon_open = df[(dow == 0) & (minute_of_day.between(9 * 60 + 25, 9 * 60 + 29))]
        mon_first_hour_end = df[(dow == 0) & (minute_of_day.between(10 * 60 + 25, 10 * 60 + 29))]
        mon_day_end = df[(dow == 0) & (minute_of_day.between(15 * 60 + 55, 15 * 60 + 59))]

        def daily_first(g):
            return g.groupby(g.index.date).first()

        fc = daily_first(fri_close)["close_px"]
        mo = daily_first(mon_open)["close_px"]
        mfh = daily_first(mon_first_hour_end)["close_px"]
        med = daily_first(mon_day_end)["close_px"]

        # align by week: shift friday date index forward to the following monday
        fc.index = pd.to_datetime(fc.index)
        mo.index = pd.to_datetime(mo.index)
        mfh.index = pd.to_datetime(mfh.index)
        med.index = pd.to_datetime(med.index)

        weeks = pd.DataFrame({"fri_close": fc})
        weeks["monday_date"] = weeks.index + pd.Timedelta(days=3)  # Fri->Mon
        weeks = weeks.set_index("monday_date")
        weeks["mon_open"] = mo
        weeks["mon_first_hour_end"] = mfh
        weeks["mon_day_end"] = med
        weeks = weeks.dropna()

        weeks["weekend_gap_ret"] = np.log(weeks["mon_open"] / weeks["fri_close"])
        weeks["mon_first_hour_ret"] = np.log(weeks["mon_first_hour_end"] / weeks["mon_open"])
        weeks["mon_day_ret"] = np.log(weeks["mon_day_end"] / weeks["mon_open"])

        n = len(weeks)
        if n >= 10:
            r1, p1 = stats.pearsonr(weeks["weekend_gap_ret"], weeks["mon_first_hour_ret"])
            r2, p2 = stats.pearsonr(weeks["weekend_gap_ret"], weeks["mon_day_ret"])
        else:
            r1 = p1 = r2 = p2 = np.nan
        catchup_rows.append(dict(
            symbol=symbol, n_weeks=n,
            corr_gap_vs_mon_firsthour=r1, p_firsthour=p1,
            corr_gap_vs_mon_day=r2, p_day=p2,
        ))

    print("\n=== 1) Realized vol / mean return by session ===")
    sess_df = pd.concat(all_session_stats)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 140):
        print(sess_df.to_string())

    print("\n=== 2) US market-open reaction (first 30min after 9:30 ET) vs weekday-closed baseline ===")
    open_df = pd.DataFrame(open_reactions)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(open_df.to_string(index=False))

    print("\n=== 3) Weekend gap -> Monday catch-up (does weekend equity-implied drift predict Monday crypto move?) ===")
    catchup_df = pd.DataFrame(catchup_rows)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(catchup_df.to_string(index=False))

    out_dir = REPO / "data" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_df.to_csv(out_dir / "market_session_stats_20260729.csv")
    open_df.to_csv(out_dir / "market_open_reaction_20260729.csv", index=False)
    catchup_df.to_csv(out_dir / "weekend_catchup_20260729.csv", index=False)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
