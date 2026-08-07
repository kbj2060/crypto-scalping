"""Stage 0 (Rho1 panel design, docs/btc_panel_crossasset_architecture_design_20260804.md):
select the 40-60 symbol USDT-perp universe for the cross-sectional panel.

Selection rule: currently-TRADING USDT-margined PERPETUAL contracts on Binance futures
(fapi.binance.com, public REST, no account credentials -- same deliberate choice as
scripts/download_klines_sol_20260707.py) whose onboardDate is before 2024-01-01 UTC,
ranked by current 24h quote volume, top N taken.

KNOWN CAVEATS (documented, not silently hidden):
1. Liquidity-lookahead: ranking by TODAY's 24h volume to pick a historical-training
   universe is itself a mild lookahead for panel *composition* -- a symbol popular today
   may not have been liquid in 2024. This is acceptable for Stage 0 (raw data acquisition)
   but the eventual training pipeline must reconstruct a rolling, point-in-time liquidity
   universe (e.g. trailing 30d quote volume as of each bar) rather than use this fixed list
   as a per-bar membership mask. This script's output is a DOWNLOAD LIST, not a per-bar
   causal universe definition.
2. Survivorship bias: this script only lists symbols that are still TRADING now. Perpetuals
   delisted between 2024-01-01 and today are NOT included -- Binance's exchangeInfo endpoint
   does not expose delisted symbols. Full survivorship-bias mitigation (including delisted
   contracts) requires a different data source (e.g. a historical symbol-list snapshot or
   third-party archive) and is deferred; flagged here as a known gap in Stage 0's universe,
   per docs/btc_panel_crossasset_architecture_design_20260804.md section 5 risk #2.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
TICKER_24H_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
ONBOARD_CUTOFF_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_universe(top_n: int) -> dict:
    info = requests.get(EXCHANGE_INFO_URL, timeout=30).json()
    tickers = requests.get(TICKER_24H_URL, timeout=30).json()
    vol_by_symbol = {t["symbol"]: float(t["quoteVolume"]) for t in tickers}

    candidates = [
        s for s in info["symbols"]
        if s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
        and s["onboardDate"] < ONBOARD_CUTOFF_MS
    ]
    ranked = sorted(candidates, key=lambda s: -vol_by_symbol.get(s["symbol"], 0.0))
    selected = ranked[:top_n]

    rows = []
    for s in selected:
        rows.append({
            "symbol": s["symbol"],
            "onboard_date": datetime.fromtimestamp(s["onboardDate"] / 1000, tz=timezone.utc).date().isoformat(),
            "quote_volume_24h_at_selection": vol_by_symbol.get(s["symbol"], 0.0),
        })
    return {
        "schema_version": "panel_universe_symbols_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "USDT-margined PERPETUAL, status=TRADING, onboardDate<2024-01-01, "
                           "ranked by 24h quote volume at selection time, top_n taken",
        "caveats": [
            "liquidity_lookahead: ranked by TODAY's 24h volume, not point-in-time historical "
            "liquidity -- this is a download list, not a per-bar causal universe mask",
            "survivorship_bias: only currently-TRADING symbols considered; perpetuals delisted "
            "since 2024-01-01 are excluded (exchangeInfo does not expose delisted symbols)",
        ],
        "candidates_considered": len(candidates),
        "top_n": top_n,
        "symbols": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=60)
    args = ap.parse_args()

    manifest = fetch_universe(args.top_n)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    print(f"selected {len(manifest['symbols'])} symbols (of {manifest['candidates_considered']} candidates)")
    print(f"wrote {OUT_PATH}")
    for row in manifest["symbols"]:
        print(f"  {row['symbol']:16s} onboard={row['onboard_date']} vol24h={row['quote_volume_24h_at_selection']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
