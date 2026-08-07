"""Stage 0 (Rho1 panel design): download daily metrics + monthly funding zips for the full
panel universe, reusing scripts/download_metrics_funding_generic_20260713.py's download
functions unchanged (data.binance.vision, immutable per-day/per-month zip files, idempotent
skip-if-exists).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_metrics_funding_generic_20260713 import download_funding, download_metrics  # noqa: E402

UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]
    print(f"downloading metrics+funding for {len(symbols)} symbols, {start}..{end}...", flush=True)

    for i, sym in enumerate(symbols, 1):
        m_ok, m_missing = download_metrics(sym, start, end)
        f_ok, f_missing = download_funding(sym, start, end)
        print(f"[{i}/{len(symbols)}] {sym:16s} metrics={m_ok}ok/{m_missing}miss  funding={f_ok}ok/{f_missing}miss", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
