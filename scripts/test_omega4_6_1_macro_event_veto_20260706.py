#!/usr/bin/env python3
"""Test: does adding a macro-event entry veto (skip entries in the immediate aftermath of major
US index releases: NFP/ISM-manufacturing/ISM-services/S&P Global PMI/FOMC) improve
omega4_6_1_duration_ou_halflife_risk_gate's extended Jan-Jun 2026 OOS result?

Design chosen by user: veto window is SHORTER than Omega5's original -30min/+120min (which fully
blocks ~2.5h around every event) -- here it's -30min/+15min (skip entries in the immediate
pre/post-release window only), and the strategy resumes completely normally afterward (no
sizing boost, no haircut -- pure entry veto). Reuses the exact rule-based event-date calendar
(NFP=1st Friday 8:30am ET, ISM mfg/services=1st/3rd business day 10am ET, flash PMI=on/after the
23rd 9:45am ET, FOMC=static verified 2026 dates) from trading_bot_modules/omega5_live.py, applied
post-hoc to the already-computed combined/gated ledger from combine_omega4_6_1_extended_oos_20260706.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "trading_bot_modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "trading_bot_modules"))

from omega5_live import Omega5LiveAdapter  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
import os
VETO_PRE_MIN = int(os.environ.get("VETO_PRE_MIN", "30"))
VETO_POST_MIN = int(os.environ.get("VETO_POST_MIN", "15"))


def build_event_calendar(years: list[int]) -> list[tuple[str, pd.Timestamp]]:
    events = []
    for y in years:
        events.extend(Omega5LiveAdapter._macro_events_for_year(y))
    return events


def in_veto_window(ts: pd.Timestamp, events: list[tuple[str, pd.Timestamp]]) -> tuple[bool, str]:
    for name, event_ts in events:
        start = event_ts - pd.Timedelta(minutes=VETO_PRE_MIN)
        end = event_ts + pd.Timedelta(minutes=VETO_POST_MIN)
        if start <= ts <= end:
            return True, name
    return False, ""


def summarize(ledger: pd.DataFrame, label: str) -> dict:
    active = ledger[ledger["notional"].astype(float) > 1e-12]
    if active.empty:
        return {"label": label, "pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    wins = (active["trade_return"].astype(float) > 0).sum() if "win" not in active.columns else active["win"].astype(int).sum()
    return {
        "label": label,
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(dd.min() * 100.0),
        "trades": int(len(active)),
        "wr": float(wins / len(active)),
    }


def main() -> int:
    events = build_event_calendar([2025, 2026, 2027])
    print(f"macro event calendar: {len(events)} events across 2025-2027 (NFP/ISM-mfg/ISM-svc/flash-PMI/FOMC)", flush=True)

    for tag, path in (("pre-duration-gate (router combine)", "combined_router_ledger_extended.csv"),
                       ("post-duration-gate (final)", "combined_router_duration_gated_ledger_extended.csv")):
        ledger = pd.read_csv(OUT_DIR / path)
        ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
        active = ledger[ledger["notional"].astype(float) > 1e-12].copy()
        hits = active["entry_timestamp_dt"].apply(lambda ts: in_veto_window(ts, events))
        active["macro_veto_hit"] = [h[0] for h in hits]
        active["macro_veto_event"] = [h[1] for h in hits]

        baseline = summarize(ledger, f"{tag}: baseline (no veto)")
        vetoed = ledger.copy()
        veto_mask = ledger["entry_timestamp_dt"].isin(active.loc[active["macro_veto_hit"], "entry_timestamp_dt"])
        vetoed.loc[veto_mask, "notional"] = 0.0
        with_veto = summarize(vetoed, f"{tag}: WITH macro-event veto (-{VETO_PRE_MIN}min/+{VETO_POST_MIN}min)")

        n_hit = int(active["macro_veto_hit"].sum())
        print(f"\n=== {tag} ===", flush=True)
        print(f"  {n_hit}/{len(active)} trades landed inside the veto window:", flush=True)
        for _, row in active.loc[active["macro_veto_hit"]].iterrows():
            print(f"    entry={row['entry_timestamp']} side={row['side']} event={row['macro_veto_event']} "
                  f"trade_return={row['trade_return']*100:.2f}%", flush=True)
        print(f"  baseline:  pnl={baseline['pnl']:.2f}% mdd={baseline['mdd']:.2f}% trades={baseline['trades']} wr={baseline['wr']:.3f}", flush=True)
        print(f"  with veto: pnl={with_veto['pnl']:.2f}% mdd={with_veto['mdd']:.2f}% trades={with_veto['trades']} wr={with_veto['wr']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
