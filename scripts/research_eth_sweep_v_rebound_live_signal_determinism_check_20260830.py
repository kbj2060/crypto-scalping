#!/usr/bin/env python3
"""Diagnose user report: "why does the liquidity_sweep V-rebound dashboard indicator's value
change on every refresh?" live_eth_sweep_v_rebound_signal_20260829.py::compute_eth_sweep_v_rebound_
signal() re-fits+re-predicts TabPFN from scratch on EVERY call (no per-event proba caching beyond
dashboard/server.py's 60s response cache) -- fetches fresh klines, recomputes features, and calls
TabPFNClassifier(device="cuda", random_state=20260829).fit(train, ...).predict_proba(sweeps[...]).

Two independent candidate explanations to isolate:
  (A) Legitimate: new 5m bars keep arriving in real time, so a genuinely NEW sweep event replaces
      the old one, or ACTIVE_WINDOW_MINUTES=30 expires -- the underlying thing being reported
      actually changed. Not a bug.
  (B) Suspect: for the SAME underlying event (same sweep_ts_utc, same feature row, fixed
      random_state), does TabPFN's predict_proba output DRIFT between separate calls -- either from
      (B1) plain GPU floating-point run-to-run variation on an IDENTICAL input batch, or
      (B2) sensitivity to the surrounding test-batch COMPOSITION (the live script scores ALL
      currently-qualifying sweep rows in one predict_proba call together, and which historical
      sweep events fall inside the rolling 1500-bar/5.2-day fetch window shifts every call as time
      passes -- if TabPFN's predict_proba is not strictly row-independent, this alone could move
      the SAME row's own predicted probability with no market-data change at all).

This script isolates B1 vs B2 directly instead of guessing from documentation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import live_eth_sweep_v_rebound_signal_20260829 as live  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402


def main() -> int:
    print("fetching live klines once (shared across all sub-tests below)...")
    kl = live._fetch_klines()
    assert kl is not None and len(kl) >= 900, "kline fetch failed"
    frame = live._build_features(kl)
    sweeps = live._sweep_rows(frame).dropna(subset=live.FEATURES)
    print(f"qualifying sweep rows in current {live.FETCH_LIMIT}-bar window: {len(sweeps)}")
    if sweeps.empty:
        print("no active sweep rows right now -- cannot test, re-run when a sweep is active")
        return 1

    train = live._load_train_context()
    latest_row = sweeps.tail(1)
    latest_ts = latest_row["timestamp"].iloc[0]
    print(f"most recent qualifying event: {latest_ts} (testing THIS row's stability)")

    # --- Test B1: identical input batch (the real `sweeps` frame), called twice in a row ---
    print("\n=== Test B1: same TabPFNClassifier call repeated 5x on the IDENTICAL full batch ===")
    full_probas = []
    for i in range(5):
        clf = TabPFNClassifier(device="cuda", random_state=20260829)
        clf.fit(train[live.FEATURES], train["label"].to_numpy())
        proba = clf.predict_proba(sweeps[live.FEATURES])[:, 1]
        latest_idx = sweeps.index.get_loc(latest_row.index[0])
        p_latest = float(proba[latest_idx])
        full_probas.append(p_latest)
        print(f"  run {i+1}: proba_rebound for latest event = {p_latest:.6f}")
    spread_b1 = max(full_probas) - min(full_probas)
    print(f"  -> spread across 5 identical-input runs: {spread_b1:.6f}")

    # --- Test B2: same latest row, but scored alongside DIFFERENT-sized surrounding batches ---
    print("\n=== Test B2: same latest row, scored inside batches of different size/composition ===")
    batch_variants = {
        "solo (just this 1 row)": sweeps.tail(1),
        "last 3 rows": sweeps.tail(3),
        "last 10 rows": sweeps.tail(min(10, len(sweeps))),
        "full window (all rows)": sweeps,
    }
    batch_probas = {}
    for name, batch in batch_variants.items():
        clf = TabPFNClassifier(device="cuda", random_state=20260829)
        clf.fit(train[live.FEATURES], train["label"].to_numpy())
        proba = clf.predict_proba(batch[live.FEATURES])[:, 1]
        p_latest = float(proba[batch.index.get_loc(latest_row.index[0])]) if latest_row.index[0] in batch.index else None
        batch_probas[name] = p_latest
        print(f"  batch='{name}' (n={len(batch)}): proba_rebound for latest event = {p_latest}")
    vals = [v for v in batch_probas.values() if v is not None]
    spread_b2 = max(vals) - min(vals)
    print(f"  -> spread across batch-composition variants: {spread_b2:.6f}")

    print("\n=== VERDICT ===")
    print(f"B1 (identical-input GPU run-to-run spread): {spread_b1:.6f}")
    print(f"B2 (same-row, different-batch-composition spread): {spread_b2:.6f}")
    if spread_b1 < 1e-6 and spread_b2 < 1e-6:
        print("Both near-zero -> TabPFN itself is stable here; refresh-to-refresh changes the user")
        print("sees must come from genuinely NEW market data (new bar arriving / new sweep event /")
        print("30-min active window expiring), not from model nondeterminism.")
    elif spread_b1 >= 1e-6 and spread_b2 < 1e-6:
        print("B1 dominates -> raw GPU/TabPFN inference is not bit-stable call-to-call even for the")
        print("exact same input. Real, but check magnitude vs the displayed rounding (x100, integer%).")
    elif spread_b2 >= 1e-6:
        print("B2 dominates -> TabPFN's predict_proba for a given row is sensitive to which OTHER")
        print("rows are batched alongside it. Since the live script rebuilds `sweeps` fresh from a")
        print("rolling kline fetch every call, the batch composition legitimately shifts call-to-call")
        print("even when the reported event itself hasn't changed -- this alone can move the displayed")
        print("probability with zero real market change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
