#!/usr/bin/env python3
"""RESEARCH ONLY -- TP/SL floor sizing redesign, portfolio-level check (caveat (a) and (b) from
research_eth_omega461_tpsl_floor_sweep_20260728.py's findings).

That sweep found min_tp=0.12/min_sl=0.025 is the only VAL+OOS-confirmed win for h48qual
(standalone replay, 2026-01-01..03-31 OOS). Two things were left unchecked:
(a) the mixed config (h48qual retuned, zig075 UNCHANGED at live 0.075/0.040) at the
    PORTFOLIO/combined-router level, not per-component isolation.
(b) a second/adjacent OOS window for robustness.

This script reuses replay_omega4_6_1_greedy_router_20260706.py's exact greedy_replay/
prepare_component machinery unmodified (the same single-shared-position-slot, h48qual>zig075
priority router used to produce the frozen live-baseline numbers: +145.34%/-10.13%MDD/24 trades
over 2026-01-01..06-30), only overriding h48qual's min_tp/min_sl in the COMPONENTS dict before
calling prepare_component. Because that window already extends past the 2026-03-31 OOS boundary
used in the standalone sweep, it doubles as an adjacent-window robustness check (Apr-Jun 2026 is
unpeeked by the standalone sweep) and a portfolio-level check in one pass.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Research artifact only -- no promotion-gate claim.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260728/tpsl_floor_portfolio_check"
H48QUAL_MIN_TP, H48QUAL_MIN_SL = 0.12, 0.025

# Frozen live-baseline reference (tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/
# greedy_router_result.json, built 2026-07-06, unchanged min_tp/min_sl for both components).
LIVE_BASELINE = {
    "no_gate": {"pnl": 138.19338965711995, "mdd": -14.154462813803049, "trades": 32, "wr": 0.5},
    "with_gate": {"pnl": 145.3353677513158, "mdd": -10.134492720083554, "trades": 24,
                  "wr": 0.5416666666666666, "skipped": 8},
}


def _truncated_pred_csv(name: str, cfg: dict, frame: pd.DataFrame) -> Path:
    # The frozen oos_predictions_*.csv files were later silently extended past 2026-06-30 (now
    # run to 07-12) by an unrelated later regen -- prepare_component requires an EXACT timestamp
    # match against `frame`, so truncate a scratch copy to frame's range. The overlapping portion
    # is byte-identical to the original (verified: rows <= frame's max timestamp match frame's
    # timestamps 1:1), so this is a pure length-alignment fix, not a data change.
    src = router.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
    pred = pd.read_csv(src)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    sub = pred[pred["timestamp"] <= frame["timestamp"].max()].reset_index(drop=True)
    if not sub["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"truncated pred timestamps still mismatch frame for {name}")
    out = OUT_DIR / f"_truncated_{name}_{cfg['q_tag']}.csv"
    sub.to_csv(out, index=False)
    return out


def run(components_cfg: dict) -> dict:
    device = retest.DEVICE
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = router.omega._load_fee_slip()

    components = {}
    for name, cfg in components_cfg.items():
        pred_csv = _truncated_pred_csv(name, cfg, ext_frame)
        components[name] = router.prepare_component(ext_frame, pred_csv, cfg, device)
        print(f"{name}: prepared min_tp={cfg['min_tp']} min_sl={cfg['min_sl']} "
              f"nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)

    _, ledger = router.greedy_replay(ext_frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    active = ledger.copy()
    returns = active["trade_return"].to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    no_gate = {"pnl": float((curve[-1] - 1.0) * 100.0), "mdd": float(dd.min() * 100.0), "trades": int(len(active)),
               "wr": float((returns > 0).mean()) if len(returns) else 0.0}
    src_counts = active["source_component"].value_counts().to_dict() if len(active) else {}

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
    return {"no_gate": no_gate, "with_gate": with_gate, "source_component_counts": src_counts, "ledger": ledger}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=sanity_reproduce_live_baseline", flush=True)
    baseline_cfg = copy.deepcopy(retest.COMPONENTS)
    baseline_result = run(baseline_cfg)
    print(baseline_result["with_gate"], flush=True)
    ref = LIVE_BASELINE["with_gate"]
    got = baseline_result["with_gate"]
    stale_reference_drifted = not (abs(got["pnl"] - ref["pnl"]) <= 0.01 and abs(got["mdd"] - ref["mdd"]) <= 0.01 and got["trades"] == ref["trades"])
    if stale_reference_drifted:
        print(f"stage=sanity_reproduce_live_baseline_DRIFTED (underlying data files changed since "
              f"2026-07-06): frozen={ref} fresh_recompute={got} -- falling back to a same-day "
              f"apples-to-apples comparison (fresh unmodified baseline vs fresh mixed config, both "
              f"computed from today's data) instead of the stale frozen reference.", flush=True)
    else:
        print("stage=sanity_reproduce_live_baseline_PASSED", flush=True)

    print("stage=mixed_config_replay (h48qual min_tp=%s min_sl=%s, zig075 unchanged)" % (H48QUAL_MIN_TP, H48QUAL_MIN_SL), flush=True)
    mixed_cfg = copy.deepcopy(retest.COMPONENTS)
    mixed_cfg["h48qual"]["min_tp"] = H48QUAL_MIN_TP
    mixed_cfg["h48qual"]["min_sl"] = H48QUAL_MIN_SL
    mixed_result = run(mixed_cfg)
    print(mixed_result["with_gate"], flush=True)

    confirmed_vs_stale_frozen = bool(mixed_result["with_gate"]["pnl"] > ref["pnl"] and mixed_result["with_gate"]["mdd"] >= ref["mdd"] - 1e-9)
    confirmed_vs_fresh_baseline = bool(mixed_result["with_gate"]["pnl"] > got["pnl"] and mixed_result["with_gate"]["mdd"] >= got["mdd"] - 1e-9)
    print(f"stage=done confirmed_vs_stale_frozen={confirmed_vs_stale_frozen} confirmed_vs_fresh_baseline={confirmed_vs_fresh_baseline}", flush=True)

    mixed_result["ledger"].to_csv(OUT_DIR / "mixed_config_ledger.csv", index=False)
    baseline_result["ledger"].to_csv(OUT_DIR / "baseline_ledger.csv", index=False)
    summary = {
        "live_baseline_frozen_20260706": LIVE_BASELINE,
        "stale_reference_drifted": stale_reference_drifted,
        "fresh_baseline_today": baseline_result["with_gate"],
        "mixed_config": {"h48qual_min_tp": H48QUAL_MIN_TP, "h48qual_min_sl": H48QUAL_MIN_SL,
                          "no_gate": mixed_result["no_gate"], "with_gate": mixed_result["with_gate"],
                          "source_component_counts": mixed_result["source_component_counts"]},
        "confirmed_vs_stale_frozen_baseline": confirmed_vs_stale_frozen,
        "confirmed_vs_fresh_same_day_baseline": confirmed_vs_fresh_baseline,
        "window": "2026-01-01..2026-06-30 (extends past the 03-31 standalone-sweep OOS boundary into unpeeked Apr-Jun)",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
