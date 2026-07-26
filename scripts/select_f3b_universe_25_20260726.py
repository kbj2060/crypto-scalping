#!/usr/bin/env python3
"""Mechanically select the 25-coin universe for the F3-B cross-sectional momentum
redesign (docs/mechanical_trading_research_synthesis_20260726.md S5.1.2).

The prior F3-B/F3-B-LF experiments used ETH/BTC/SOL only and were killed by lack
of statistical power (a 3-name cross-section carries too much idiosyncratic vol
for a 1.2-year sample to distinguish a plausible effect size from zero). A wider
universe reduces cross-sectional noise via basket averaging. Picking that
universe by hand after looking at momentum results would be the data-snooping
this repo's anti-fishing rule exists to prevent, so the selection here uses a
public, mechanical, performance-blind rule fixed BEFORE any return is computed:

  1. All Binance USDT-M perpetual futures with status=TRADING.
  2. onboardDate <= 2024-01-01 -- guarantees >=2 years of history for every
     name and lets the whole 25-asset cross-section share one fixed start date
     with no "join late" logic needed for lookback warm-up.
  3. Ranked by 24h quote volume at the moment this script is run (a liquidity
     snapshot, not a performance signal), and the top 25 are taken verbatim.

Uses fapi.binance.com public endpoints only -- no API key, no account state,
matching the existing convention in download_klines_1m_20260716.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
N_UNIVERSE = 25
ONBOARD_CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")
OUT = ROOT / "data/ensemble/metrics/f3b_universe25_selection_20260726.json"


def main() -> None:
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=15).json()
    perp_usdt = [
        s for s in info["symbols"]
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]
    tick = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=15).json()
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in tick}

    candidates = []
    for s in perp_usdt:
        sym = s["symbol"]
        onboard = pd.Timestamp(s["onboardDate"], unit="ms", tz="UTC") if s.get("onboardDate") else None
        if sym in vol and onboard is not None and onboard <= ONBOARD_CUTOFF:
            candidates.append({"symbol": sym, "quote_volume_24h": vol[sym], "onboard_date": str(onboard.date())})
    candidates.sort(key=lambda r: -r["quote_volume_24h"])

    selected = candidates[:N_UNIVERSE]
    report = {
        "selected_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "rule": "TRADING USDT-M perpetuals, onboardDate<=2024-01-01, ranked by 24h quote volume, top 25",
        "n_candidates_meeting_onboard_cutoff": len(candidates),
        "n_selected": len(selected),
        "universe": [r["symbol"] for r in selected],
        "detail": selected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
