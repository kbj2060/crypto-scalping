"""Sigma6 fresh-window confirmation (2026-07-20).

Sigma6 (docs/model_contracts/sigma6_regime_trend_20260705_contract.md) is this project's
strongest-generalizing verified result (OOS cost1 +45.9% lev4 / +16.6% lev3), but its
2026-03-02..06-30 OOS window has been re-examined repeatedly by this project (Sigma6 itself,
Sigma8-11, the F4-B dated-ledger rework), so a genuinely fresh, never-before-seen window is needed
before treating it as confirmed. This script re-runs the exact frozen production configs
(CONFIGS below, taken verbatim from research_f4b_sigma6_dated_ledger_20260719.py, which itself
matches the contract doc) against the newly extended tape/regime data (raw 5m features, regime3
wide24 HMM, CryptoMamba-h6 stability sidecar, sigma3-1h ensemble tape all extended through
2026-07-20 this session) over two windows:
  (a) full extended OOS 2026-03-02..07-20, for continuity with the original contract's numbers.
  (b) fresh-only 2026-07-01..07-20 -- bars this model's development has NEVER seen before.
Uses run_sigma6_regime_trend_20260705.py's own load_tape_with_regime()/backtest logic unchanged
(imported, not copied) plus research_f4b_sigma6_dated_ledger_20260719.py's dated-ledger variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_f4b_sigma6_dated_ledger_20260719 as f4b  # noqa: E402
import run_sigma6_regime_trend_20260705 as sigma6  # noqa: E402
import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma6_fresh_window_20260720"

FULL_OOS_START, FULL_OOS_END = sigma6.OOS_START, pd.Timestamp("2026-07-20 23:59:59")
FRESH_START, FRESH_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-20 23:59:59")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = sigma6.load_tape_with_regime()
    print(f"tape range: {raw['timestamp'].min()} .. {raw['timestamp'].max()} ({len(raw)} rows)", flush=True)

    report = {}
    for name, cfg0 in f4b.CONFIGS.items():
        cfg = dict(cfg0)
        thr = cfg.pop("thr")
        tape = v2.apply_quality_threshold(raw, thr)

        full_result, full_ledger = f4b.backtest_with_dates(tape, start=FULL_OOS_START, end=FULL_OOS_END, **cfg)
        fresh_result, fresh_ledger = f4b.backtest_with_dates(tape, start=FRESH_START, end=FRESH_END, **cfg)

        full_ledger.to_csv(OUT_DIR / f"{name}_full_extended_oos_ledger.csv", index=False)
        fresh_ledger.to_csv(OUT_DIR / f"{name}_fresh_only_ledger.csv", index=False)

        report[name] = {"full_extended_oos_2026_03_02_to_07_20": full_result,
                         "fresh_only_2026_07_01_to_07_20": fresh_result}
        print(f"\n=== {name} ===", flush=True)
        print(f"  full extended OOS (2026-03-02..07-20): {full_result}", flush=True)
        print(f"  fresh-only (2026-07-01..07-20):        {fresh_result}", flush=True)
        if not fresh_ledger.empty:
            print(fresh_ledger[["entry_timestamp", "exit_timestamp", "reason", "ret", "win"]].to_string(index=False), flush=True)

    import json
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
