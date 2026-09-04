#!/usr/bin/env python3
"""Follow-up to backtest_eth_evidence_signal_regime_entry_exit_20260827.py after diagnosing WHY its
regime_stop had ~zero net effect (memory eth_regime_entry_exit_backtest_diagnosis_20260827): only
3.5% of trades were ever touched, and of those, 77% (36/47) correctly cut a would-be-SL trade short
~4.6 bars early (+0.293%p avg) but 23% (11/47) were single-bar trend_prob spikes that bounced back
and would have hit TP anyway (-0.837%p avg) -- a false-alarm cost ~3x the size of a correct catch's
benefit, washing out the net effect.

Fix tested here: require the underwater+trend_prob>=theta condition to hold for `regime_persist_bars`
CONSECUTIVE bars (not just 1) before firing -- see _resolve_trade_regime_stop's regime_persist_bars
param, added this session. k_entry is fixed at 1 (the best-performing entry setting from the prior
grid; higher k_entry consistently worsened total_return there, an unrelated finding not re-tested
here). Only the exit side varies: theta_exit x regime_persist_bars, on the same 2 signals confirmed
regime-robust today (orthogonal_combo bottom, short_term_return_z top).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_evidence_signal_regime_entry_exit_20260827 import (  # noqa: E402
    PRIMARY_CANDIDATES, TP_ATR_MULT, SL_ATR_MULT, HORIZON_BARS, LEVERAGE, MARGIN_FRACTION,
    ROUNDTRIP_COST_RATE, _compute_frame, run_window,
)

OUT_DIR = ROOT / "tmp/eth_evidence_signal_regime_stop_persistence_20260827"
K_ENTRY = 1
THETA_EXIT_GRID = [0.5, 0.6, 0.7]
PERSIST_GRID = [1, 2, 3]


def log(msg: str) -> None:
    print(f"[regime_stop_persistence] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Building 2025/2026 frames...")
    frames = {"2025": _compute_frame(gate.sweep.BASE_2025), "2026": _compute_frame(gate.sweep.BASE_2026)}

    report: dict[str, Any] = {"config": {"tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT,
                                          "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE,
                                          "margin_fraction": MARGIN_FRACTION,
                                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE, "k_entry": K_ENTRY},
                               "results": {}}

    # baseline: no regime stop at all, for direct comparison
    for name, side in PRIMARY_CANDIDATES:
        bcol = f"bottom_{name}" if side == "bottom" else None
        tcol = f"top_{name}" if side == "top" else None
        baseline_windows = {}
        for wname, wd in gate.WINDOW_DEFS.items():
            frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
            baseline_windows[wname] = run_window(frame, bcol, tcol, K_ENTRY, None, start=wd["start"], end=wd["end"])
        report["results"][f"{name}:{side}:baseline_no_stop"] = baseline_windows
        baseline_sum = sum(w["total_return"] for w in baseline_windows.values())
        log(f"{name}:{side} baseline (no regime_stop): sum(total_return)={baseline_sum*100:.1f}%")

        for theta_exit in THETA_EXIT_GRID:
            for persist in PERSIST_GRID:
                key = f"{name}:{side}:theta{theta_exit}:persist{persist}"
                windows_out = {}
                for wname, wd in gate.WINDOW_DEFS.items():
                    frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                    windows_out[wname] = run_window(frame, bcol, tcol, K_ENTRY, theta_exit,
                                                     start=wd["start"], end=wd["end"], regime_persist_bars=persist)
                report["results"][key] = windows_out
                total_return_sum = sum(w["total_return"] for w in windows_out.values())
                total_regime_stops = sum(w["n_regime_stop"] for w in windows_out.values())
                mean_worst_mae = np.nanmean([w["worst_mae"] for w in windows_out.values()])
                delta_vs_baseline = (total_return_sum - baseline_sum) * 100
                log(f"{key}: sum(total_return)={total_return_sum*100:.1f}%  "
                    f"(delta vs no-stop={delta_vs_baseline:+.2f}pp)  regime_stops={total_regime_stops}  "
                    f"mean(worst_mae)={mean_worst_mae*100:.2f}%")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
