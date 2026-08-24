#!/usr/bin/env python3
"""RESEARCH ONLY -- diagnostic (not a promotion candidate). Follow-up to
research_eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817.py: that 5-level geometric
ladder (1x/0.5x/0.25x/0.125x/0.0625x of the current 7.5%/4.0% floor, ratio preserved at 1.875)
showed exit_head engagement barely moves at ANY scale while PnL/MDD degrade sharply in most
windows -- user's reaction: the ladder went too far too fast; test a single specific, more modest
point instead: min_tp=0.03 (3%), min_sl=0.015 (1.5%) -- note this pair has ratio 2.0 (matching
tp_mult:sl_mult=12:6=2.0 exactly), slightly different from the current floor's ratio (1.875), per
the user's explicit numbers. tp_mult/sl_mult/atr_window/max_tp/max_sl and exit_head unchanged, same
as the parent sweep. Reuses every helper from the parent script unmodified (imported, not copied).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_odyssey4_tpsl_floor_shrink_exit_head_engagement_20260817 as parent_mod  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import pandas as pd  # noqa: E402

MIN_TP, MIN_SL = 0.03, 0.015
OUT_DIR = parent_mod.OUT_DIR.parent / "eth_odyssey4_tpsl_floor_3pct_1p5pct_check_20260817"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = parent_mod.DEVICE
    fee, slip = omega._load_fee_slip()

    log(f"=== stage=load_windows === testing min_tp={MIN_TP * 100:.2f}% min_sl={MIN_SL * 100:.2f}% "
        f"(ratio={MIN_TP / MIN_SL:.3f}, vs current 7.5%/4.0% ratio={0.075 / 0.040:.3f})")
    windows = dict(gate.load_all_windows())

    rows = []
    for window_key in parent_mod.WINDOW_KEYS:
        w = windows[window_key]
        oof = bool(w["oof"])
        split = gate.WINDOW_DEFS[window_key]["split"]
        q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in sweep.COMPONENTS}
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)

        prep = parent_mod.portfolio._prepare_component_val if oof else parent_mod.greedy.prepare_component
        cfgs = {}
        for name in ("h48qual", "zig075"):
            # BUG FIX 2026-08-17: see identical fix + note in the parent floor-shrink script's
            # _floor_cfg -- must use NEW_H48QUAL_BUNDLE for h48qual or this silently reverts to
            # the pre-liveATR original bundle.
            bundle_override = parent_mod.portfolio.NEW_H48QUAL_BUNDLE if name == "h48qual" else None
            cfg = parent_mod.portfolio._component_cfg(name, bundle_override=bundle_override)
            cfg["min_tp"], cfg["min_sl"] = MIN_TP, MIN_SL
            cfgs[name] = cfg
        components = {name: prep(aligned_frame, aligned_paths[name], cfgs[name], device) for name in ("h48qual", "zig075")}

        diag, ledger = parent_mod.greedy.greedy_replay(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        with_gate = parent_mod.mfe_width._duration_gated(ledger, aligned_frame, parent_mod.greedy.DURATION_THRESHOLD)
        reason_bd = parent_mod.reasons_mod._reason_breakdown(ledger, aligned_frame, parent_mod.greedy.DURATION_THRESHOLD)
        bound_frac = parent_mod._floor_bound_frac(aligned_frame, 192, 12.0, MIN_TP)

        row = {"window": window_key, "min_tp": MIN_TP, "min_sl": MIN_SL, "floor_bound_frac": round(bound_frac, 4),
               "pnl": with_gate["pnl"], "mdd": with_gate["mdd"], "trades": with_gate["trades"],
               "n_trades_raw": reason_bd["n_trades"], "reason_shares": parent_mod._reason_shares(reason_bd)}
        rows.append(row)
        log(f"  {window_key}: floor_bound={bound_frac * 100:5.2f}%  pnl={row['pnl']:+7.2f}%  mdd={row['mdd']:+7.2f}%  "
            f"trades={row['trades']:3d}  reasons={row['reason_shares']}")

    log("\n=== FINAL: min_tp=3.0% / min_sl=1.5% vs current(7.5%/4.0%) reference ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "floor_3pct_1p5pct_summary.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'floor_3pct_1p5pct_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
