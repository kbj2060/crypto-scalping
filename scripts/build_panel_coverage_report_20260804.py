"""Stage 0 (Rho1 panel design) close-out: per-symbol coverage/gap report for the panel
universe's klines/metrics/funding, plus registration of the newly-downloaded metrics/funding
zip files into binance_data/RAW_SOURCE_MANIFEST.json (same schema and fail-fast-on-drift
semantics as scripts/update_features.py's _verify_and_register_raw_source).

Does NOT register the per-symbol combined 5m kline CSVs into RAW_SOURCE_MANIFEST -- those are
intentionally mutable/growing files (extended on every re-run), same as the existing BTC/ETH/SOL
*-5m-api.csv files, which are also not manifest-pinned. Only the immutable daily-metrics and
monthly-funding zip files downloaded from data.binance.vision are hash-pinned.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
KLINES_DIR = ROOT / "binance_data/klines"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
RAW_SOURCE_MANIFEST = ROOT / "binance_data/RAW_SOURCE_MANIFEST.json"
REPORT_PATH = ROOT / "docs/panel_universe_coverage_report_20260804.md"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if RAW_SOURCE_MANIFEST.exists():
        return json.loads(RAW_SOURCE_MANIFEST.read_text())
    return {"schema_version": "raw_source_manifest_v1", "files": {}}


def _register(manifest: dict, rel_path: str, abs_path: Path) -> str:
    """Returns 'new', 'unchanged', or raises on drift."""
    entry = manifest["files"].get(rel_path)
    digest = _sha256_file(abs_path)
    if entry is None:
        manifest["files"][rel_path] = {
            "sha256": digest,
            "size_bytes": abs_path.stat().st_size,
            "first_seen": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return "new"
    if entry["sha256"] != digest:
        raise RuntimeError(
            f"Raw source drift detected: {rel_path} content changed since first registered "
            f"({entry['first_seen']}, sha256={entry['sha256']}) -- now sha256={digest}."
        )
    return "unchanged"


def register_zips(symbols: list[str]) -> dict:
    manifest = _load_manifest()
    counts = {"new": 0, "unchanged": 0}
    for sym in symbols:
        for zf in sorted(METRICS_DIR.glob(f"{sym}-metrics-*.zip")):
            status = _register(manifest, f"metrics/{zf.name}", zf)
            counts[status] += 1
        for zf in sorted(FUNDING_DIR.glob(f"{sym}-fundingRate-*.zip")):
            status = _register(manifest, f"funding_rate_other/{zf.name}", zf)
            counts[status] += 1
    RAW_SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return counts


def kline_coverage(sym: str) -> dict:
    path = KLINES_DIR / sym / f"{sym}-5m-api.csv"
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path, usecols=["timestamp"], low_memory=False)
    ts = pd.to_datetime(df["timestamp"])
    ts = ts.sort_values()
    expected = pd.date_range(ts.iloc[0], ts.iloc[-1], freq="5min")
    n_missing = len(expected) - len(ts)
    gaps = ts.diff().dropna()
    max_gap_bars = int((gaps.max() / pd.Timedelta(minutes=5))) if len(gaps) else 0
    return {
        "exists": True,
        "rows": len(ts),
        "ts_min": str(ts.iloc[0]),
        "ts_max": str(ts.iloc[-1]),
        "expected_bars": len(expected),
        "missing_bars": int(n_missing),
        "missing_pct": round(100.0 * n_missing / len(expected), 3) if len(expected) else None,
        "max_gap_bars": max_gap_bars,
    }


def zip_coverage(sym: str) -> dict:
    n_metrics = len(list(METRICS_DIR.glob(f"{sym}-metrics-*.zip")))
    n_funding = len(list(FUNDING_DIR.glob(f"{sym}-fundingRate-*.zip")))
    return {"n_metrics_zips": n_metrics, "n_funding_zips": n_funding}


def main() -> int:
    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]

    print(f"registering raw zip files for {len(symbols)} symbols into RAW_SOURCE_MANIFEST.json...")
    reg_counts = register_zips(symbols)
    print(f"  {reg_counts}")

    rows = []
    for sym in symbols:
        kc = kline_coverage(sym)
        zc = zip_coverage(sym)
        rows.append({"symbol": sym, **kc, **zc})

    lines = [
        "# Panel Universe Coverage Report (Stage 0)\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        f"Symbols: {len(symbols)}\n",
        "Zip registration into RAW_SOURCE_MANIFEST.json: "
        f"{reg_counts['new']} newly registered, {reg_counts['unchanged']} already-registered/unchanged.\n",
        "\n| symbol | rows | ts_min | ts_max | missing_bars | missing_pct | max_gap_bars | metrics_zips | funding_zips |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r.get("exists"):
            lines.append(f"| {r['symbol']} | MISSING | - | - | - | - | - | {r['n_metrics_zips']} | {r['n_funding_zips']} |")
            continue
        lines.append(
            f"| {r['symbol']} | {r['rows']} | {r['ts_min']} | {r['ts_max']} | "
            f"{r['missing_bars']} | {r['missing_pct']}% | {r['max_gap_bars']} | "
            f"{r['n_metrics_zips']} | {r['n_funding_zips']} |"
        )

    missing_klines = [r["symbol"] for r in rows if not r.get("exists")]
    high_gap = [r["symbol"] for r in rows if r.get("exists") and r.get("missing_pct", 0) and r["missing_pct"] > 1.0]
    lines.append("\n## Flags\n")
    lines.append(f"- Symbols with no klines file at all: {missing_klines or 'none'}")
    lines.append(f"- Symbols with >1% missing 5m bars: {high_gap or 'none'}")
    lines.append(
        "\nCaveats carried over from data/splits/panel_universe_symbols_20260804.json: "
        "liquidity-lookahead in universe selection (ranked by today's volume) and "
        "survivorship bias (delisted symbols excluded) -- see "
        "docs/btc_panel_crossasset_architecture_design_20260804.md section 5."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(f"missing klines: {missing_klines}")
    print(f"high-gap symbols: {high_gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
