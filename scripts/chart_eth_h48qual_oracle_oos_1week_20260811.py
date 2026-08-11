"""Plot one representative week of ORACLE trades for ETH h48qual's two real head labels, within
the labels' available OOS coverage (2026-01-01..02-28):

  - direction_head target: zigzag_action (tmp/causal_regen_20260516/zigzag_action_labels_20260531,
    build_wave3_action_labels_20260531.py) -- contiguous LONG/SHORT runs, one oracle trade per run.
  - quality_head target: h48_conservative barrier formula (build_omega1_2_triple_barrier_labels_
    20260619.py, tp_mult=1.2, sl_mult=0.8, min_tp=0.006, min_sl=0.004) but with horizon=384bar(32h)
    instead of the deployed 48bar -- per sweep_h48qual_horizon_wide_20260811.py, 384bar is where
    zigzag-direction agreement first peaks (92.1% vs 89.5% at 48bar) while specificity nearly
    doubles (65.1% vs 34.2%); recomputed fresh here on the same 2026 OHLCV, one oracle trade per
    active (tb_action != 0) signal bar, entry at next-bar open, SL-priority barrier resolution.

Both panels share the same selected week and price axis for direct visual comparison.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_KR_FONT = ROOT / "tmp/fonts/NotoSansKR-VF.ttf"
if _KR_FONT.exists():
    fm.fontManager.addfont(str(_KR_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_KR_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False
LABEL_PATH = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2026.csv"
OUT_PNG = ROOT / "tmp/eth_h48qual_oracle_label_check_20260811/oos_1week_h48qual_oracle_trades_h384.png"

FEE_RATE, SLIP_RATE = 0.0005, 0.0002
FEE_COST = (FEE_RATE + SLIP_RATE) * 2.0 * 3.0  # 0.0042, matches build_omega1_2_triple_barrier_labels_20260619.py
H48_HORIZON, H48_TP_MULT, H48_SL_MULT, H48_MIN_TP, H48_MIN_SL = 384, 1.2, 0.8, 0.006, 0.004


def _atr_price_move(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr / np.where(close != 0, close, np.nan)).rolling(96, min_periods=24).mean().shift(1)
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _reason_and_return(side: int, entry: float, future_high: np.ndarray, future_low: np.ndarray,
                        future_close: np.ndarray, tp_move: float, sl_move: float):
    if entry <= 0.0:
        return 0.0, "invalid_entry", 0.0
    if side > 0:
        tp_level, sl_level = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        rel_low = future_low / entry - 1.0
        mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
        for b, (hi, lo) in enumerate(zip(future_high, future_low), start=1):
            if lo <= sl_level:
                return -float(sl_move), "sl", mae
            if hi >= tp_level:
                return float(tp_move), "tp", mae
        return float(future_close[-1] / entry - 1.0), "timeout", mae
    tp_level, sl_level = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
    rel_high = 1.0 - future_low / entry
    mae = float(np.nanmin(rel_high)) if len(rel_high) else 0.0
    for b, (hi, lo) in enumerate(zip(future_high, future_low), start=1):
        if hi >= sl_level:
            return -float(sl_move), "sl", mae
        if lo <= tp_level:
            return float(tp_move), "tp", mae
    return float(1.0 - future_close[-1] / entry), "timeout", mae


def build_h48_conservative_trades(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    open_px = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    ts = df["timestamp"]
    atr = _atr_price_move(df)

    rows = []
    last_i = n - H48_HORIZON - 2
    for i in range(max(last_i, 0)):
        entry_i = i + 1
        end_i = entry_i + H48_HORIZON
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(H48_MIN_TP, H48_TP_MULT * vol)
        sl_move = max(H48_MIN_SL, H48_SL_MULT * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, long_mae = _reason_and_return(1, entry, fh, fl, fc, tp_move, sl_move)
        short_ret, short_reason, short_mae = _reason_and_return(-1, entry, fh, fl, fc, tp_move, sl_move)
        long_q = long_ret - FEE_COST - 0.20 * max(-long_mae, 0.0) - 0.003 * int(long_reason == "sl")
        short_q = short_ret - FEE_COST - 0.20 * max(-short_mae, 0.0) - 0.003 * int(short_reason == "sl")
        if long_q > 0.0 and long_q >= short_q:
            side, ret, reason, exit_bars = 1, long_ret, long_reason, (
                np.argmax((fl <= entry * (1 - sl_move)) | (fh >= entry * (1 + tp_move))) + 1
                if ((fl <= entry * (1 - sl_move)) | (fh >= entry * (1 + tp_move))).any() else H48_HORIZON
            )
        elif short_q > 0.0:
            side, ret, reason, exit_bars = -1, short_ret, short_reason, (
                np.argmax((fh >= entry * (1 + sl_move)) | (fl <= entry * (1 - tp_move))) + 1
                if ((fh >= entry * (1 + sl_move)) | (fl <= entry * (1 - tp_move))).any() else H48_HORIZON
            )
        else:
            continue
        exit_i = min(entry_i + int(exit_bars) - 1, n - 1)
        rows.append({"entry_ts": ts.iloc[i], "exit_ts": ts.iloc[exit_i], "side": side,
                     "entry_px": entry, "exit_px": float(close[exit_i]), "trade_return": float(ret),
                     "reason": reason})
    return pd.DataFrame(rows)


def _contiguous_runs(action: np.ndarray):
    n = len(action)
    i = 0
    while i < n:
        a = action[i]
        j = i
        while j + 1 < n and action[j + 1] == a:
            j += 1
        if a != 0:
            yield i, j, a
        i = j + 1


def build_zigzag_trades(df: pd.DataFrame) -> pd.DataFrame:
    action = df["zigzag_action"].to_numpy()
    ts = df["timestamp"]
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    rows = []
    for i, j, a in _contiguous_runs(action):
        side = 1 if a == 1 else -1
        entry_px, exit_px = open_[i], close[j]
        ret = side * (exit_px / entry_px - 1.0)
        rows.append({"entry_ts": ts.iloc[i], "exit_ts": ts.iloc[j], "side": side,
                     "entry_px": entry_px, "exit_px": exit_px, "trade_return": ret, "reason": "run_end"})
    return pd.DataFrame(rows)


def _plot_panel(ax, price_week: pd.DataFrame, week_ledger: pd.DataFrame, title: str) -> None:
    ax.plot(price_week["timestamp"], price_week["close"], color="#888888", linewidth=0.8, zorder=1)
    for _, row in week_ledger.iterrows():
        marker = "^" if row["side"] == 1 else "v"
        c = "#2ca02c" if row["trade_return"] > 0 else "#d62728"
        ax.scatter(row["entry_ts"], row["entry_px"], marker=marker, color=c, s=60,
                   zorder=3, edgecolors="black", linewidths=0.5)
        ax.plot([row["entry_ts"], row["exit_ts"]], [row["entry_px"], row["exit_px"]],
               color=c, linewidth=1.1, linestyle="--", alpha=0.7, zorder=2)
    win_rate = float((week_ledger["trade_return"] > 0).mean()) if len(week_ledger) else 0.0
    sum_ret = float(week_ledger["trade_return"].sum() * 100)
    n_long = int((week_ledger["side"] == 1).sum())
    n_short = int((week_ledger["side"] == -1).sum())
    ax.set_title(title)
    ax.set_ylabel("ETH close (USDT)")
    ax.grid(alpha=0.25)
    ax.text(0.01, 0.02,
            f"n={len(week_ledger)} (long={n_long} short={n_short})  win_rate={win_rate:.1%}  sum_ret={sum_ret:+.2f}%",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def main() -> int:
    df = pd.read_csv(LABEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    zz_trades = build_zigzag_trades(df)
    h48_trades = build_h48_conservative_trades(df)
    print(f"zigzag_action 오라클 거래 수: {len(zz_trades)}   h48_conservative 오라클 거래 수: {len(h48_trades)}")

    week_key = zz_trades["entry_ts"].dt.to_period("W-SUN")
    best_week = week_key.value_counts().idxmax()
    week_start, week_end = best_week.start_time, best_week.end_time
    print(f"선택된 주: {week_start} .. {week_end} (zigzag_action 거래 수 기준 최다)")

    price_week = df[(df["timestamp"] >= week_start) & (df["timestamp"] <= week_end + pd.Timedelta(hours=6))]
    zz_week = zz_trades[(zz_trades["entry_ts"] >= week_start) & (zz_trades["entry_ts"] <= week_end)].copy()
    h48_week = h48_trades[(h48_trades["entry_ts"] >= week_start) & (h48_trades["entry_ts"] <= week_end)].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    _plot_panel(ax1, price_week, zz_week,
                f"direction_head 타겟 -- zigzag_action (스윙 오라클)\n{week_start.date()} ~ {week_end.date()}")
    _plot_panel(ax2, price_week, h48_week,
                f"quality_head 타겟 -- h48_conservative 배리어 공식, horizon=384bar(32h) 재검토판\n{week_start.date()} ~ {week_end.date()}")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    fig.autofmt_xdate()
    fig.suptitle("ETH 5m -- h48qual 실제 헤드 라벨 OOS 1주일 오라클 점검 (up=long down=short, green=win red=loss)",
                 fontsize=11)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"저장: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
