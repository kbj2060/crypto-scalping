#!/usr/bin/env python3
"""Sweep total_notional_cap / per-asset shares for the v4 prealloc portfolio cap,
plus a duration-gate (ou_halflife) stress test, reusing
replay_portfolio_concurrent_3asset_native_20260712._replay_concurrent.

Builds the validation/oos worlds ONCE and replays many configs against them
in-process (native._build_world is the expensive step; the bar-by-bar replay
itself is cheap), rather than invoking the CLI script once per config.

Research/diagnostic only. No trading_bot.py wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_concurrent_3asset_native_20260712 as v4  # noqa: E402
import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sweep_portfolio_concurrent_3asset_v4_20260712"
Q1_CUTOFF = pd.Timestamp("2026-04-01")

REFERENCE_SHARES = {"eth": 0.5, "btc": 0.3, "sol": 0.2}
TOTAL_CAP_GRID = [None, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ALT_SHARE_GRID = {
    "eth50_btc30_sol20": {"eth": 0.5, "btc": 0.3, "sol": 0.2},
    "eth40_btc35_sol25": {"eth": 0.4, "btc": 0.35, "sol": 0.25},
    "equal_third": {"eth": 1 / 3, "btc": 1 / 3, "sol": 1 / 3},
    "eth60_btc25_sol15": {"eth": 0.6, "btc": 0.25, "sol": 0.15},
}
ETH_MULTIPLIER_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def _run_one(
    world_val: dict, world_oos: dict, device, *,
    total_cap: float | None, shares: dict[str, float], duration_gate_on: bool = True,
    multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    orig_thresholds = dict(native.DURATION_THRESHOLDS)
    if not duration_gate_on:
        native.DURATION_THRESHOLDS = {k: -999.0 for k in orig_thresholds}
    try:
        cap_mode = "prealloc" if total_cap is not None else "scale"
        m_val, _, _, d_val = v4._replay_concurrent(
            world_val, device=device, total_notional_cap=total_cap, cap_mode=cap_mode, asset_shares=shares, asset_notional_multipliers=multipliers
        )
        m_oos, ledger_oos, _, d_oos = v4._replay_concurrent(
            world_oos, device=device, total_notional_cap=total_cap, cap_mode=cap_mode, asset_shares=shares, asset_notional_multipliers=multipliers
        )
        m_q1, _, _, d_q1 = v4._replay_concurrent(
            world_oos, device=device, total_notional_cap=total_cap, cap_mode=cap_mode, asset_shares=shares, asset_notional_multipliers=multipliers, entry_cutoff=Q1_CUTOFF
        )
        return {
            "validation": {"metrics": m_val, "diag": d_val},
            "oos_extended": {"metrics": m_oos, "diag": d_oos},
            "oos_frozen_q1_2026": {"metrics": m_q1, "diag": d_q1},
        }
    finally:
        native.DURATION_THRESHOLDS = orig_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-gate", choices=("on", "off"), default="on",
                        help="Apply to the total_notional_cap and asset_shares sweeps (the dedicated duration_gate_stress section always runs both regardless of this flag).")
    parser.add_argument("--only", choices=("all", "total_cap", "shares", "duration_gate", "eth_multiplier"), default="all",
                        help="Restrict to one sweep section (world-building is the slow part; skip sections you don't need).")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    sweep_gate_on = args.duration_gate == "on"
    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR if sweep_gate_on else OUT_DIR.parent / f"{OUT_DIR.name}_gate_off")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = native.eth_retest.DEVICE
    print("stage=build_world split=validation", flush=True)
    world_val = native._build_world("validation", device)
    print("stage=build_world split=oos", flush=True)
    world_oos = native._build_world("oos", device)

    report: dict[str, Any] = {"duration_gate_on_for_main_sweeps": sweep_gate_on, "total_cap_sweep": {}, "share_sweep": {}, "duration_gate_stress": {}, "eth_multiplier_sweep": {}}

    if args.only in ("all", "total_cap"):
        print(f"stage=sweep axis=total_notional_cap duration_gate={'on' if sweep_gate_on else 'off'}", flush=True)
        for total_cap in TOTAL_CAP_GRID:
            key = "uncapped" if total_cap is None else f"total_{total_cap:.1f}"
            print(f"  config={key}", flush=True)
            report["total_cap_sweep"][key] = {
                "total_notional_cap": total_cap,
                "shares": REFERENCE_SHARES if total_cap is not None else None,
                "results": _run_one(world_val, world_oos, device, total_cap=total_cap, shares=REFERENCE_SHARES, duration_gate_on=sweep_gate_on),
            }

    if args.only in ("all", "shares"):
        print(f"stage=sweep axis=asset_shares (total_notional_cap=3.0) duration_gate={'on' if sweep_gate_on else 'off'}", flush=True)
        for name, shares in ALT_SHARE_GRID.items():
            print(f"  config={name}", flush=True)
            report["share_sweep"][name] = {
                "total_notional_cap": 3.0,
                "shares": shares,
                "results": _run_one(world_val, world_oos, device, total_cap=3.0, shares=shares, duration_gate_on=sweep_gate_on),
            }

    if args.only in ("all", "duration_gate"):
        print("stage=duration_gate_stress_test (reference config: total=3.0, eth50/btc30/sol20)", flush=True)
        for gate_on in (True, False):
            key = "gate_on" if gate_on else "gate_off"
            print(f"  config={key}", flush=True)
            report["duration_gate_stress"][key] = {
                "duration_gate_on": gate_on,
                "results": _run_one(world_val, world_oos, device, total_cap=3.0, shares=REFERENCE_SHARES, duration_gate_on=gate_on),
            }

    if args.only in ("all", "eth_multiplier"):
        print("stage=sweep axis=eth_notional_multiplier (uncapped, duration_gate=off, btc/sol multiplier=1.0)", flush=True)
        for mult in ETH_MULTIPLIER_GRID:
            key = f"eth_mult_{mult:.2f}"
            print(f"  config={key}", flush=True)
            report["eth_multiplier_sweep"][key] = {
                "eth_notional_multiplier": mult,
                "results": _run_one(
                    world_val, world_oos, device, total_cap=None, shares=REFERENCE_SHARES, duration_gate_on=False,
                    multipliers={"eth": mult, "btc": 1.0, "sol": 1.0},
                ),
            }

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=native._json_default) + "\n", encoding="utf-8"
    )

    # Compact console summary
    if report["total_cap_sweep"]:
        print("\n=== total_notional_cap sweep (eth50/btc30/sol20), oos_extended ===", flush=True)
        for key, cfg in report["total_cap_sweep"].items():
            m = cfg["results"]["oos_extended"]["metrics"]["portfolio"]
            print(f"{key:12s} pnl={m['pnl']:8.2f}% mdd={m['mdd']:8.2f}% mtm_mdd={m['mark_to_market_mdd']:8.2f}% trades={m['trades']}", flush=True)

    if report["share_sweep"]:
        print("\n=== asset_shares sweep (total_notional_cap=3.0), oos_extended ===", flush=True)
        for key, cfg in report["share_sweep"].items():
            m = cfg["results"]["oos_extended"]["metrics"]["portfolio"]
            print(f"{key:20s} pnl={m['pnl']:8.2f}% mdd={m['mdd']:8.2f}% mtm_mdd={m['mark_to_market_mdd']:8.2f}% trades={m['trades']}", flush=True)

    if report["eth_multiplier_sweep"]:
        print("\n=== eth_notional_multiplier sweep (uncapped, gate=off), all splits ===", flush=True)
        for key, cfg in report["eth_multiplier_sweep"].items():
            for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
                m = cfg["results"][split]["metrics"]["portfolio"]
                print(f"{key:14s} {split:20s} pnl={m['pnl']:8.2f}% mdd={m['mdd']:8.2f}% mtm_mdd={m['mark_to_market_mdd']:8.2f}% trades={m['trades']}", flush=True)

    if report["duration_gate_stress"]:
        print("\n=== duration-gate stress test (total=3.0, eth50/btc30/sol20) ===", flush=True)
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            m_on = report["duration_gate_stress"]["gate_on"]["results"][split]["metrics"]["portfolio"]
            m_off = report["duration_gate_stress"]["gate_off"]["results"][split]["metrics"]["portfolio"]
            print(
                f"{split:20s} gate_on: pnl={m_on['pnl']:8.2f}% mdd={m_on['mdd']:8.2f}% trades={m_on['trades']:3d}  |  "
                f"gate_off: pnl={m_off['pnl']:8.2f}% mdd={m_off['mdd']:8.2f}% trades={m_off['trades']:3d}",
                flush=True,
            )

    print(f"\nreport={out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
