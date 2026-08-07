"""
Fresh-Forward bar-by-bar walk-forward validation of the full BTC candidate
stack (Layer 1 CUSUM gate -> Layer 2 feature context -> Layer 3 LightGBM
quality/direction -> Layer 4 RL exit-timing policy), per this project's
Fresh-Forward Validation/OOS/Test Rule:

  - fixed split: VAL 2025-09-01..2025-12-31, OOS 2026-01-01..2026-03-31
  - single continuous causal scan, bar index by bar index, in chronological
    order, through the ENTIRE VAL+OOS window
  - at each bar, only information available up to and including that bar is
    used (CUSUM state carried forward incrementally; Layer 3 models trained
    only on data before VAL_START; Layer 4 RL policy likewise)
  - NO saved trade ledger, candidate-event ledger, or future-row join is
    used anywhere -- every entry/exit decision and PnL figure here is
    generated fresh in this single pass
  - one position at a time (matches this project's single-position
    execution discipline)

Report explicitly declares: fresh_forward_bar_by_bar=true,
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false.

Research/dev-score only until BTC-specific slippage is audited (see prior
session finding: only ETH's live cost-calibration has been validated).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import train_btc_exit_stopping_rl_20260803 as M  # noqa: E402

ACTOR_PATH = ROOT / "data/ensemble/ckpt/btc_exit_stopping_rl_actor_seed270705_20260803.pth"
OUT_CSV = ROOT / "tmp/btc_fresh_forward_walkforward_trades_20260803.csv"


def main():
    frame = pd.read_parquet(M.FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    ts = frame["timestamp"]
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    atr = M._atr_price_move(frame)  # already causal (rolling+shift(1)), computed once, same formula live would use

    with open(M.MODEL_DIR / "btc_cusum_trailing_final_long.pkl", "rb") as f:
        long_model = pickle.load(f)
    with open(M.MODEL_DIR / "btc_cusum_trailing_final_short.pkl", "rb") as f:
        short_model = pickle.load(f)
    feat_cols = long_model.feature_name_

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = M.DSACAgent(state_dim=M.STATE_DIM, hidden_dim=128, n_quantiles=25, device=device)
    agent.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    agent.actor.eval()

    # scan window: start scanning from the first bar >= VAL_START so CUSUM
    # state accumulates causally from the start of the evaluation window
    # (mirrors how a live process would only start accumulating state once
    # it begins running -- we don't reach back into training-period CUSUM
    # state, keeping VAL/OOS strictly self-contained).
    start_i = int(np.searchsorted(ts.to_numpy(), np.datetime64(M.VAL_START)))
    end_i = int(np.searchsorted(ts.to_numpy(), np.datetime64(M.OOS_END)))

    s_pos = s_neg = 0.0
    logret = np.diff(np.log(close), prepend=np.log(close[0]))

    trades = []
    open_ep = None  # active ExitEpisode, or None
    open_entry_i = None
    cur_state = None

    for i in range(start_i, min(end_i, n - 2)):
        # --- Layer 4: if a position is open, step it forward one bar ---
        if open_ep is not None:
            a = agent.act(cur_state, deterministic=True)
            a_scalar = float(np.asarray(a).reshape(-1)[0])
            cur_state, r, done, info = open_ep.step(a_scalar)
            if done:
                trades.append({
                    "entry_ts": ts.iloc[open_entry_i], "exit_ts": ts.iloc[min(open_entry_i + open_ep.bar, n - 1)],
                    "side": open_ep.side, "net": info["net"], "bars": open_ep.bar,
                })
                open_ep = None
                open_entry_i = None
                cur_state = None
            continue  # one position at a time -- don't also evaluate a new entry this bar

        # --- Layer 1: CUSUM gate (state carried forward bar-by-bar) ---
        thresh = max(float(atr[i]), 0.001) * 2.0
        s_pos = max(0.0, s_pos + logret[i])
        s_neg = min(0.0, s_neg + logret[i])
        fired = s_pos > thresh or s_neg < -thresh
        if not fired:
            continue
        s_pos = s_neg = 0.0

        entry_i = i + 1
        if entry_i + 1 >= n:
            continue

        # --- Layer 2/3: predict quality/direction from THIS bar's features only ---
        x = frame.loc[[i], feat_cols]
        pl = float(long_model.predict(x)[0])
        ps = float(short_model.predict(x)[0])
        if pl >= M.ENTRY_THRESHOLD and pl >= ps:
            side, conv = 1, pl
        elif ps >= M.ENTRY_THRESHOLD:
            side, conv = 2, ps
        else:
            continue

        # --- open position, hand off to Layer 4 for subsequent bars ---
        open_ep = M.ExitEpisode(frame, i, side, conv, close, high, low, open_px, atr)
        open_entry_i = entry_i
        cur_state = open_ep.reset()

    trades = pd.DataFrame(trades)
    trades.to_csv(OUT_CSV, index=False)

    val = trades[trades["entry_ts"] < M.OOS_START]
    oos = trades[trades["entry_ts"] >= M.OOS_START]

    def summarize(df, label):
        if len(df) == 0:
            print(f"{label}: n=0")
            return
        print(f"{label}: n={len(df)} win%={100*(df['net']>0).mean():.1f} "
              f"mean_net={100*df['net'].mean():.3f}% sum_net={100*df['net'].sum():.2f}% "
              f"mean_hold={df['bars'].mean():.1f}bar")

    print("\n=== Fresh-Forward bar-by-bar walk-forward: Layer1(CUSUM)->Layer2/3(quality)->Layer4(RL exit) ===")
    print(f"scan window: {ts.iloc[start_i]} .. {ts.iloc[min(end_i, n-1)]}, {end_i-start_i} bars")
    summarize(val, "VAL  (2025-09-01..2025-12-31)")
    summarize(oos, "OOS  (2026-01-01..2026-03-31)")
    print("\ncompliance flags:")
    print("  fresh_forward_bar_by_bar=true")
    print("  trade_ledgers_used_as_input=false")
    print("  saved_parent_exit_timestamps_used=false")
    print("  future_rows_used_for_entry=false")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
