"""
DIAGNOSTIC-ONLY (per CLAUDE.md Fresh-Forward rule: saved ledgers may be used for
diagnostic/accounting-audit/historical-reproduction purposes, never as promotion
or live-performance evidence).

User hypothesis: ETH Omega4.6.1's wide SL/TP band means trades that go into
profit mid-flight sometimes round-trip back to a loss before the wide exit
triggers, and a CVD/OI-based exit gate could lock in profit earlier.

Project memory context this test must be read against: 21+ rounds of ETH
Omega4.6.1 exit-logic experiments already failed, including a PERFECT-FORESIGHT
"prefer fast exit" variant that was catastrophic OOS — because the model's real
edge structure is long-hold (OOS TP-exit median hold ~1162 bars vs SL-exit ~484
bars). This script quantifies, using the actual saved VAL/OOS trade ledgers,
(a) how often a trade was ever meaningfully in profit before closing at a net
price loss, and (b) what a CVD/OI reversal-based early-exit gate would have done
to total ledger PnL (unlevered, price-move terms) if applied on top of the
existing ledger's entries/exits.

Ledger sources:
  VAL: tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv
  OOS: tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv
       (spans 2026-01-01..2026-06-30; filtered here to 2026-01-01..2026-03-31)
Price path: data/training_features_5m.csv (open/close/cvd_slope_12/oi_change_rate),
the same causal 5m feature file used elsewhere this session. Entry price = open
of the entry_timestamp bar, exit price = close of the exit_timestamp bar, per
this repo's confirmed close-based / next-bar-open live-fill convention.
"""
import pandas as pd

VAL_LEDGER = "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv"
OOS_LEDGER = "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"

PROFIT_LOCK_THRESH = 0.004  # 0.4% unlevered price-move profit before the gate arms


def load_price_index():
    df = pd.read_csv("data/training_features_5m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "close", "cvd_slope_12", "oi_change_rate"]]


def analyze_trade(row, prices: pd.DataFrame):
    entry_ts = pd.Timestamp(row["entry_timestamp"])
    exit_ts = pd.Timestamp(row["exit_timestamp"])
    side = int(row["side"])

    if entry_ts not in prices.index or exit_ts not in prices.index:
        return None

    entry_price = prices.loc[entry_ts, "open"]
    path = prices.loc[entry_ts:exit_ts]
    if len(path) < 2:
        return None

    path_pnl = side * (path["close"] / entry_price - 1.0)
    mfe = path_pnl.max()
    final_pnl = path_pnl.iloc[-1]

    # gate: once running favorable excursion clears the lock threshold, exit on
    # the first bar where CVD/OI show a reversal against the position
    gated_pnl = final_pnl
    gated_exit_ts = exit_ts
    armed = False
    for ts, pnl_t in path_pnl.items():
        if not armed and pnl_t >= PROFIT_LOCK_THRESH:
            armed = True
        if armed:
            cvd_s = prices.loc[ts, "cvd_slope_12"]
            oi_c = prices.loc[ts, "oi_change_rate"]
            reversal = (side == 1 and cvd_s < 0 and oi_c > 0) or (side == -1 and cvd_s > 0 and oi_c > 0)
            if reversal:
                gated_pnl = pnl_t
                gated_exit_ts = ts
                break

    return {
        "entry_timestamp": entry_ts, "exit_timestamp": exit_ts, "gated_exit_timestamp": gated_exit_ts,
        "side": side, "reason": row["reason"], "ledger_trade_return": row["trade_return"],
        "mfe_pct": mfe * 100, "final_price_pnl_pct": final_pnl * 100, "gated_pnl_pct": gated_pnl * 100,
        "gate_fired": gated_exit_ts != exit_ts,
        "profitable_then_lost": (mfe >= PROFIT_LOCK_THRESH) and (final_pnl < 0),
    }


def summarize(trades: pd.DataFrame, label: str):
    print(f"\n=== {label}: {len(trades)} trades ===")
    print(f"  sum(original price-move PnL, unlevered): {trades['final_price_pnl_pct'].sum():.2f}%")
    print(f"  sum(gated price-move PnL, unlevered):     {trades['gated_pnl_pct'].sum():.2f}%")
    n_fired = trades["gate_fired"].sum()
    print(f"  gate fired on {n_fired}/{len(trades)} trades")
    n_profitable_then_lost = trades["profitable_then_lost"].sum()
    print(f"  trades that were >={PROFIT_LOCK_THRESH*100:.1f}% favorable at some point but closed at a net price loss: "
          f"{n_profitable_then_lost}/{len(trades)}")
    if n_fired > 0:
        fired = trades[trades["gate_fired"]]
        delta = (fired["gated_pnl_pct"] - fired["final_price_pnl_pct"])
        print(f"  of the {n_fired} gated trades: helped {int((delta > 0).sum())}, hurt {int((delta < 0).sum())}, "
              f"net delta from gating = {delta.sum():.2f}%")
        print(fired[["entry_timestamp", "exit_timestamp", "gated_exit_timestamp", "side", "reason",
                      "mfe_pct", "final_price_pnl_pct", "gated_pnl_pct"]].to_string(index=False))


def main():
    prices = load_price_index()

    val = pd.read_csv(VAL_LEDGER)
    oos_full = pd.read_csv(OOS_LEDGER)
    oos_full["entry_timestamp"] = pd.to_datetime(oos_full["entry_timestamp"])
    oos = oos_full[(oos_full["entry_timestamp"] >= "2026-01-01") & (oos_full["entry_timestamp"] <= "2026-03-31")]

    for label, ledger in [("VAL (2025-10-01..2025-12-31)", val), ("OOS (2026-01-01..2026-03-31)", oos)]:
        results = [analyze_trade(row, prices) for _, row in ledger.iterrows()]
        results = [r for r in results if r is not None]
        trades = pd.DataFrame(results)
        summarize(trades, label)


if __name__ == "__main__":
    main()
