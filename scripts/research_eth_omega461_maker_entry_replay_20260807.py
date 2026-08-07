"""Maker-entry replay at Omega4.6.1's ACTUAL combined-router entry decisions (ETH).

Question (Kappa1 closure asset #1 applied to ETH): the 2026-07-18 placement study showed
maker join entry improves ~+1.28bps with 99.8% fill on 86k UNCONDITIONAL per-minute intents.
Does that survive at Omega4.6.1's ~63 actual entries, which are momentum-flavoured and where
the classic failure is adverse selection (fills on retracing losers, misses on runaway winners)?

DIAGNOSTIC per CLAUDE.md Fresh-Forward Rule: the saved combined-router ledgers are used ONLY
to look up historical entry decision timestamps, sides, and realized outcomes -- execution
microstructure replay, not a model-performance claim. No promotion claim is made here.

Rule (matches the 2026-07-18 study + Kappa1 Stage-0 v2): post-only limit at the entry 1m
bar's open; fill requires strict trade-through (buy: later 1m low < limit; sell: high >
limit) within a deadline K; unfilled -> fallback taker at open[entry+K].
improvement_bps = +2.5 (taker 4.5 - maker 2.0 fee) if maker-filled,
                  -side*(open[entry+K]/limit - 1)*1e4 if fallback taker (fee saving 0).
Also reported: no-fallback policy (unfilled = trade skipped) PnL impact, and the
winner-vs-loser fill split that detects adverse selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
LEDGERS = {"VAL": LEDGER_DIR / "combined_router_ledger_VAL.csv",
           "OOS_extended": LEDGER_DIR / "combined_router_ledger_extended.csv"}
M1_PATH = ROOT / "data/training_features_1m.csv"
OUT = ROOT / "docs/experiments/eth_omega461_maker_entry_replay_20260807.json"

FEE_SAVING_BPS = 2.5  # taker 4.5 - maker 2.0
DEADLINES_MIN = [5, 15, 30]


def main() -> None:
    m1 = pd.read_csv(M1_PATH, usecols=["timestamp", "open", "high", "low", "close"])
    m1["timestamp"] = pd.to_datetime(m1["timestamp"])
    m1 = m1.sort_values("timestamp").reset_index(drop=True)
    ts = m1["timestamp"].to_numpy()
    open_ = m1["open"].to_numpy()
    high = m1["high"].to_numpy()
    low = m1["low"].to_numpy()
    n = len(m1)

    report = {"fee_saving_bps_when_maker_filled": FEE_SAVING_BPS,
              "fill_rule": "post-only at entry 1m open; strict trade-through; fallback taker at open[entry+K]",
              "ledger_role": "diagnostic lookup of entry timestamps/sides/outcomes only",
              "splits": {}}
    for split, path in LEDGERS.items():
        ledger = pd.read_csv(path)
        ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
        rows = []
        for _, trade in ledger.iterrows():
            pos = np.searchsorted(ts, np.datetime64(trade["entry_timestamp"]))
            if pos >= n or ts[pos] != np.datetime64(trade["entry_timestamp"]):
                rows.append({"matched": False})
                continue
            side = int(trade["side"])
            limit = open_[pos]
            row = {"matched": True, "side": side, "win": int(trade["win"]),
                   "trade_return": float(trade["trade_return"]),
                   "source": trade.get("source_alias", ""), "fill_min": None}
            for k in DEADLINES_MIN:
                end = min(pos + k, n - 1)
                filled = None
                for j in range(pos, end):
                    through = low[j] < limit if side == 1 else high[j] > limit
                    if through:
                        filled = j - pos + 1
                        break
                row[f"filled_{k}m"] = filled is not None
                if filled is not None and row["fill_min"] is None:
                    row["fill_min"] = filled
                if filled is not None:
                    row[f"improve_{k}m_bps"] = FEE_SAVING_BPS
                else:
                    fallback = open_[end]
                    row[f"improve_{k}m_bps"] = -side * (fallback / limit - 1) * 1e4
            rows.append(row)
        frame = pd.DataFrame([r for r in rows if r.get("matched")])
        split_out = {"trades_in_ledger": int(len(ledger)), "trades_matched_1m": int(len(frame))}
        for k in DEADLINES_MIN:
            filled = frame[f"filled_{k}m"]
            imp = frame[f"improve_{k}m_bps"]
            unfilled = frame[~filled]
            split_out[f"deadline_{k}m"] = {
                "fill_rate": float(filled.mean()),
                "improvement_mean_bps": float(imp.mean()),
                "improvement_min_bps": float(imp.min()),
                "fallback_slippage_mean_bps": float(imp[~filled].mean()) if (~filled).any() else 0.0,
                "unfilled_trades": int((~filled).sum()),
                "unfilled_trade_returns": [round(float(v), 4) for v in unfilled["trade_return"]],
                "pnl_missed_if_no_fallback": float(unfilled["trade_return"].sum()),
                "fill_rate_winners": float(filled[frame["win"] == 1].mean()) if (frame["win"] == 1).any() else None,
                "fill_rate_losers": float(filled[frame["win"] == 0].mean()) if (frame["win"] == 0).any() else None,
            }
        total_return = float(frame["trade_return"].sum())
        split_out["ledger_total_trade_return"] = total_return
        split_out["per_trade_return_mean_bps"] = float(frame["trade_return"].mean() * 1e4)
        report["splits"][split] = split_out
        print(f"{split}: matched {len(frame)}/{len(ledger)}")
        for k in DEADLINES_MIN:
            d = split_out[f"deadline_{k}m"]
            print(f"  K={k}m fill={d['fill_rate']:.1%} improve={d['improvement_mean_bps']:+.2f}bps "
                  f"unfilled={d['unfilled_trades']} missedPnL(no-fallback)={d['pnl_missed_if_no_fallback']:+.4f} "
                  f"fillW={d['fill_rate_winners']} fillL={d['fill_rate_losers']}")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
