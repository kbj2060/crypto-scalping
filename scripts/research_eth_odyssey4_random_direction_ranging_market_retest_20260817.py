#!/usr/bin/env python3
"""RESEARCH ONLY -- ranging-market re-verification (2nd round) of research_eth_odyssey4_random_
direction_risk_management_ablation_20260817.py / ..._exit_reason_distribution_20260817.py (both
imported here, never duplicated).

The original ablation's 3 judged windows (VAL, OOS-Q1, OOS-Q2) were ALL persistent ETH downtrends
(always_short beat always_long by 40-124pp), a confound flagged in that experiment's own
limitations. A first ranging retest (2025-05-12..07-07, this script's earlier revision) removed the
confound (spread collapsed to +4.5pp) and found the sign flipped: real_g0 underperformed random by
-1.01 sigma. User asked to re-verify with ANOTHER ranging window. Two candidates were found by a
systematic scan (drift/range ratio over all 56-day windows, daily closes) excluding any days
already covered by a prior test -- see CANDIDATES below for the exact rationale of each. Both are
run so the re-verification isn't a single cherry-picked window.

Only new code vs the first retest: CANDIDATES is now a list (2 windows) instead of one hardcoded
range, and load_ranging_window/run_one_candidate take oof/split/base_csv/wide24_csv as parameters
so the SAME function handles a 2025-OOF-train window and a 2026-OOS window. Everything downstream
(build_ablation_components, run_real_g0_arm, run_arm_with_reasons, side selectors, N=5 SAME seeds
as every prior test for direct cross-regime comparability) is imported unmodified.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. No live files touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_odyssey4_random_direction_risk_management_ablation_20260817 as abl  # noqa: E402
import research_eth_odyssey4_random_direction_exit_reason_distribution_20260817 as reasons_mod  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_random_direction_ranging_market_retest_20260817"

# Two candidates, both found by a systematic scan (drift/range ratio) over daily closes, excluding
# any days already covered by a previously-tested window:
#   1) 2025-03-10..2025-05-05: fully independent of every prior window (2025q1/q2/q3/val/oos_q1/
#      oos_q2 and the first ranging retest 2025-05-12..07-07 never touch these days). Net drift
#      +2.4%, range 41.3% (ratio 0.059). Character: a V-shaped reversal (down leg into 2025-04-07,
#      up leg back out) rather than continuous chop -- still "no persistent single-direction bias",
#      the property the always_long/always_short spread actually tests, but a different mechanism
#      than pure oscillation.
#   2) 2026-02-09..2026-04-06: drift +0.1%, range 27.0% (ratio 0.004, the single best-scoring window
#      found) -- genuinely oscillating (down/up/down/flat), not a single reversal. Overlaps ~7 of 8
#      weeks with the already-tested oos_q1 window's calendar days (oos_q1 = 2026-01-01..03-31,
#      itself a strong downtrend in aggregate) -- included anyway because it answers a genuinely
#      different question (a local chop pocket INSIDE a quarter that trended in aggregate), not to
#      claim full independence from oos_q1's specific trades.
CANDIDATES = [
    {"key": "ranging_2025_03_10_to_05_05", "start": "2025-03-10", "end": "2025-05-05",
     "oof": True, "split": "train", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    {"key": "ranging_2026_02_09_to_04_06", "start": "2026-02-09", "end": "2026-04-06",
     "oof": False, "split": "oos", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_ranging_window(start: str, end: str, *, oof: bool, base_csv, wide24_csv) -> dict[str, Any]:
    """Mirrors load_all_windows()'s per-window construction for one arbitrary date range."""
    frame = sweep.load_frame(start, end, base_csv=base_csv, wide24_csv=wide24_csv)
    frame, n_dropped = gate._drop_route_nan(frame)
    return {"frame": frame, "oof": oof, "tier": "context_ranging_retest",
            "start": start, "end": end, "route_nan_dropped": n_dropped}


def run_one_candidate(cand: dict) -> pd.DataFrame:
    window_key = cand["key"]
    device = abl.DEVICE
    fee, slip = abl.omega._load_fee_slip()

    log(f"=== stage=load_window {window_key} {cand['start']}..{cand['end']} (oof={cand['oof']}) ===")
    w = load_ranging_window(cand["start"], cand["end"], oof=cand["oof"], base_csv=cand["base_csv"], wide24_csv=cand["wide24_csv"])
    log(f"  frame rows={len(w['frame'])} route_nan_dropped={w['route_nan_dropped']}")
    windows = {window_key: w}
    gate.WINDOW_DEFS[window_key] = {"split": cand["split"], "base_csv": cand["base_csv"], "wide24_csv": cand["wide24_csv"]}

    log("=== stage=detector_build (reused, zero new free parameters) ===")
    score_by_base, _robustness, threshold = abl.guard.build_detector()

    # SAME seeds as the original (persistent-downtrend) ablation for direct cross-regime comparison.
    seed_sequence = np.random.SeedSequence(20260817)
    seeds = [int(s) for s in seed_sequence.generate_state(abl.N_SEEDS)]
    log(f"  seeds (identical to prior test): {seeds}")

    results: list[dict[str, Any]] = []
    log("  arm=real_g0")
    results.append(reasons_mod.run_real_g0_arm(window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip))
    log("  arm=always_long")
    results.append(reasons_mod.run_arm_with_reasons(
        "always_long", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
        side_selector=lambda n: abl._side_selector_constant(n, 1),
    ))
    log("  arm=always_short")
    results.append(reasons_mod.run_arm_with_reasons(
        "always_short", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
        side_selector=lambda n: abl._side_selector_constant(n, -1),
    ))
    for seed in seeds:
        log(f"  arm=random seed={seed}")
        results.append(reasons_mod.run_arm_with_reasons(
            f"random_seed{seed}", window_key, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
            side_selector=lambda n, _seed=seed: abl._side_selector_random(n, _seed),
        ))

    log(f"\n=== {window_key} summary (with_gate) ===")
    rows = []
    for r in results:
        wg = r["with_gate"]
        rows.append({"arm": r["arm"], "pnl": wg["pnl"], "mdd": wg["mdd"], "trades": wg["trades"], "wr": wg["wr"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / f"summary_{window_key}.csv", index=False)

    log(f"\n=== {window_key} exit-reason distribution (KEPT trades) ===")
    reason_rows = []
    for r in results:
        kept = r["reason_breakdown"]["kept"]
        total = sum(kept.values())
        row = {"arm": r["arm"], "n_kept_trades": total}
        for reason in ("take_profit", "stop_loss", "exit_head", "trailing_stop"):
            n = kept.get(reason, 0)
            row[f"{reason}_n"] = n
            row[f"{reason}_pct"] = round(100.0 * n / total, 1) if total else 0.0
        reason_rows.append(row)
    rdf = pd.DataFrame(reason_rows)
    print(rdf.to_string(index=False))
    rdf.to_csv(OUT_DIR / f"exit_reason_distribution_{window_key}.csv", index=False)

    random_rows = df[df["arm"].str.startswith("random_seed")]
    always_long = df[df["arm"] == "always_long"].iloc[0]
    always_short = df[df["arm"] == "always_short"].iloc[0]
    real_g0 = df[df["arm"] == "real_g0"].iloc[0]
    spread = always_short["pnl"] - always_long["pnl"]
    gap_sigma = (real_g0["pnl"] - random_rows["pnl"].mean()) / random_rows["pnl"].std()
    log(f"\n[{window_key}] always_long vs always_short spread (trend-confound check): {spread:+.2f}pp "
        f"(prior downtrend windows: VAL +72.9pp / OOS-Q1 +123.7pp / OOS-Q2 +75.9pp; first ranging retest +4.52pp)")
    log(f"[{window_key}] real_g0 pnl={real_g0['pnl']:+.2f}%  random mean={random_rows['pnl'].mean():+.2f}% "
        f"std={random_rows['pnl'].std():.2f}%  gap/std={gap_sigma:.2f}sigma")
    log(f"[{window_key}] random per-seed pnl: {random_rows['pnl'].tolist()}")
    return {"window_key": window_key, "spread_pp": float(spread), "real_g0_pnl": float(real_g0["pnl"]),
            "random_mean": float(random_rows["pnl"].mean()), "random_std": float(random_rows["pnl"].std()),
            "gap_sigma": float(gap_sigma)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_one_candidate(cand) for cand in CANDIDATES]
    log("\n\n=== cross-candidate summary ===")
    for s in summaries:
        log(f"  {s['window_key']}: spread={s['spread_pp']:+.2f}pp  real_g0={s['real_g0_pnl']:+.2f}%  "
            f"random={s['random_mean']:+.2f}%+-{s['random_std']:.2f}%  gap={s['gap_sigma']:+.2f}sigma")
    pd.DataFrame(summaries).to_csv(OUT_DIR / "cross_candidate_summary.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'cross_candidate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
