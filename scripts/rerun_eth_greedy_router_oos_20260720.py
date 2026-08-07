"""Re-run replay_omega4_6_1_greedy_router_20260706.py's exact logic for the Jan-Jun 2026 OOS
window, truncating the (since-07-13-session) extended oos_predictions_*.csv files back to
2026-06-30 so they match the router script's own hardcoded ext_frame window. This is a pure
re-run of the unmodified original logic (prepare_component/greedy_replay imported, not copied),
just with a safe truncation of a now-longer prediction file to the length it originally expected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402


def main() -> int:
    device = retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = router.omega._load_fee_slip()
    end_ts = ext_frame["timestamp"].iloc[-1]

    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = router.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        truncated = pred[pred["timestamp"] <= end_ts].reset_index(drop=True)
        tmp_csv = Path(f"/tmp/{name}_oos_predictions_truncated_20260630.csv")
        truncated.to_csv(tmp_csv, index=False)
        components[name] = router.prepare_component(ext_frame, tmp_csv, cfg, device)
        print(f"{name}: prepared, nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)

    _, ledger = router.greedy_replay(ext_frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    active = ledger.copy()
    returns = active["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    no_gate = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(active)),
               "wr": float((returns > 0).mean()) if len(returns) else 0.0}
    print("\n=== Greedy live-realistic router (no duration gate), OOS Jan-Jun 2026 ===", flush=True)
    print(no_gate, flush=True)
    print("source_component counts:", active["source_component"].value_counts().to_dict(), flush=True)

    market = ext_frame[["timestamp", "ou_halflife"]]
    active["entry_timestamp_dt"] = pd.to_datetime(active["entry_timestamp"])
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
    print("\n=== Greedy live-realistic router + duration gate, OOS Jan-Jun 2026 ===", flush=True)
    print(with_gate, flush=True)

    out_dir = ROOT / "tmp/causal_regen_20260516/eth_greedy_router_rerun_20260720"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_dir / "oos_ledger.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps({"no_gate": no_gate, "with_gate": with_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
