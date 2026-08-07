"""Does a zig075-ONLY live router beat the combined (h48qual>zig075 priority) router?

User question (2026-07-31): given zig075 is the majority PnL contributor and h48qual mostly gets
crowded out anyway (see project-eth-omega461-slot-occupancy-trade-count-20260728), would removing
h48qual from the greedy single-slot router free up more slots for zig075 and improve overall
performance? Not directly tested before -- prior work only tested RETUNING h48qual's TP/SL
(rejected, project-eth-omega461-tpsl-floor-portfolio-check-20260728), not REMOVING it entirely.

Reuses scripts/replay_omega4_6_1_greedy_router_20260706.py's exact `greedy_replay`/
`prepare_component` (the genuine single-shared-slot router that produced the live-realistic
numbers), fresh-recomputed on the same 2026-01-01..06-30 window (the frozen baseline is known not
to reproduce from current data -- see project-eth-omega461-tpsl-floor-portfolio-check-20260728),
comparing combined (both components, current priority) vs zig075-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

omega = router.omega


def run(components: dict, ext_frame, fee, slip) -> dict:
    _, ledger = router.greedy_replay(ext_frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=router.retest.DEVICE)
    active = ledger.copy()
    returns = active["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    no_gate = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(active)),
               "wr": float((returns > 0).mean()) if len(returns) else 0.0}

    market = ext_frame[["timestamp", "ou_halflife"]]
    active["entry_timestamp_dt"] = router.pd.to_datetime(active["entry_timestamp"])
    active = active.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = active["ou_halflife"] <= router.DURATION_THRESHOLD
    gated_returns = np.where(hit, 0.0, active["trade_return"])
    curve_g = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns)])
    peak_g = np.maximum.accumulate(curve_g)
    dd_g = curve_g / np.maximum(peak_g, 1e-12) - 1.0
    n_active_after_gate = int((~hit).sum())
    with_gate = {"pnl": float((curve_g[-1] - 1.0) * 100.0), "mdd": float(dd_g.min() * 100.0),
                 "trades": n_active_after_gate, "wr": float((gated_returns[~hit] > 0).mean()) if n_active_after_gate else 0.0,
                 "skipped": int(hit.sum())}
    return {"no_gate": no_gate, "with_gate": with_gate,
            "source_component_counts": active["source_component"].value_counts().to_dict()}


def main() -> None:
    device = router.retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = omega._load_fee_slip()

    import pandas as pd
    import tempfile

    all_components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = router.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        # prediction CSVs were regenerated later and now extend past ext_frame's 06-30 cutoff
        # (known data-drift issue, see project-eth-omega461-tpsl-floor-portfolio-check-20260728) --
        # truncate to the overlapping timestamp range so prepare_component's exact-match check passes.
        aligned = pred[pred["timestamp"].isin(set(ext_frame["timestamp"]))].reset_index(drop=True)
        assert aligned["timestamp"].equals(ext_frame["timestamp"].reset_index(drop=True)), f"{name}: still misaligned after truncation"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            aligned.to_csv(tf.name, index=False)
            aligned_path = Path(tf.name)
        all_components[name] = router.prepare_component(ext_frame, aligned_path, cfg, device)

    results = {}
    results["combined_h48qual_gt_zig075"] = run(all_components, ext_frame, fee, slip)
    results["zig075_only"] = run({"zig075": all_components["zig075"]}, ext_frame, fee, slip)
    results["h48qual_only"] = run({"h48qual": all_components["h48qual"]}, ext_frame, fee, slip)

    print(json.dumps(results, indent=2, default=str))
    with open(ROOT / "data/research/eth_zig075_only_vs_combined_20260731.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
