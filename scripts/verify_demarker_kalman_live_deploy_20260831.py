#!/usr/bin/env python3
"""Deployment verification for demarker_extreme/kalman_deviation_meanrev (2026-08-31) -- same
two-step check every prior Homer signal deploy used: (1) reproduce compute_evidence_signal_
metalabels() directly on fresh live data, on the server, before touching the public site, and
confirm the new signals return a sane fired/proba; (2) (done separately, after restart) curl the
public API. Prints the MOST RECENT bar where each new signal actually fired (not just whatever the
current instant looks like), so a real fired=True/proba path gets exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import live_evidence_signal_metalabel_20260829 as metalabel_mod  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals, fetch_klines  # noqa: E402
from live_evidence_signal_metalabel_20260829 import compute_evidence_signal_metalabels  # noqa: E402


def main() -> None:
    df = fetch_klines()
    print(f"{len(df)} bars fetched, latest={df['timestamp'].iloc[-1]}")
    sig = compute_signals(df)
    print(f"SIGNAL_ORDER count check: demarker_extreme/kalman_deviation_meanrev present in sig columns: "
          f"{'bottom_demarker_extreme' in sig.columns}/{'bottom_kalman_deviation_meanrev' in sig.columns}")

    for name in ("demarker_extreme", "kalman_deviation_meanrev"):
        fired_mask = sig[f"bottom_{name}"].fillna(False) | sig[f"top_{name}"].fillna(False)
        n_fired = int(fired_mask.sum())
        print(f"\n{name}: {n_fired} raw fires in fetched window")
        if n_fired == 0:
            print("  (no fires in this window -- cannot exercise the fired=True path right now)")
            continue
        last_fire_pos = fired_mask.to_numpy().nonzero()[0][-1]
        print(f"  most recent fire at pos={last_fire_pos}, ts={sig['timestamp'].iloc[last_fire_pos]}, "
              f"side={'bottom' if sig[f'bottom_{name}'].iloc[last_fire_pos] else 'top'}")
        # Reproduce compute_evidence_signal_metalabels() AS OF that fire bar (truncate df/sig to it)
        metalabel_mod._LAST_FIRE_CACHE.clear()  # force a genuinely fresh computation, not a carried-over cache hit
        df_trunc = df.iloc[:last_fire_pos + 1].reset_index(drop=True)
        sig_trunc = compute_signals(df_trunc)
        result = compute_evidence_signal_metalabels(df_trunc, sig_trunc.iloc[-1])
        print(f"  compute_evidence_signal_metalabels() result: {result[name]}")
        proba = result[name]["proba"]
        assert result[name]["fired"] is True, f"{name}: expected fired=True at its own known fire bar"
        assert proba is not None and 0.0 <= proba <= 1.0, f"{name}: proba out of range or None: {proba}"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
