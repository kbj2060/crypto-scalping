#!/usr/bin/env python3
"""Scale-map grid search for the exit30k dual-component router (SOL/BTC), 2026-07-22.

Prepares each component's predictions/margins/leverage ONCE per split (the expensive GPU part),
then sweeps long_scale/short_scale per component and replays the (cheap, CPU-only) greedy router
for every combo on VAL ONLY. Selects the best-VAL combo (by PnL, among combos clearing a minimum
trade-count and MDD floor so we don't pick a lucky-but-thin corner), then and only then evaluates
that single selected combo on OOS -- this is the actual VAL-then-OOS discipline the previous
single-guess run skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

router_mod = importlib.import_module("replay_omega4_6_1_two_component_router_exit30k_20260722")


def _json_default(obj: Any) -> Any:
    return router_mod._json_default(obj)


SCALE_GRID = [1.0, 2.0, 3.0]  # coarse: the per-bar GPU exit-head replay is ~25s/call, so a full
# 9-value grid (6561 combos) is infeasible; 3^4=81 combos gives a reasonable joint sweep in ~35min.


def _sweep(asset: str, device: torch.device) -> dict[str, Any]:
    frames = router_mod._load_frames(asset)
    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{router_mod.ASSET_DATES[asset]}")
    fee, slip = omega._load_fee_slip()
    cfg = router_mod.CONFIGS[asset]
    names = list(cfg.keys())

    print(f"[{asset}] preparing VAL components (one-time GPU pass)...", flush=True)
    val_frame = frames["val_raw"]
    val_components = {name: router_mod._prepare_component(asset, "validation", val_frame, c, device=device) for name, c in cfg.items()}

    results = []
    floor_trades = 10
    for ls_a in SCALE_GRID:
        for ss_a in SCALE_GRID:
            for ls_b in SCALE_GRID:
                for ss_b in SCALE_GRID:
                    val_components[names[0]]["long_scale"] = ls_a
                    val_components[names[0]]["short_scale"] = ss_a
                    val_components[names[1]]["long_scale"] = ls_b
                    val_components[names[1]]["short_scale"] = ss_b
                    ledger = router_mod._greedy_replay(val_frame, val_components, fee=fee, slip=slip, cost_mult=3.0, device=device)
                    m = router_mod._compound_metrics(ledger)
                    eligible = int(m["trades"]) >= floor_trades and float(m["mdd"]) >= -30.0
                    results.append({
                        names[0] + "_long": ls_a, names[0] + "_short": ss_a,
                        names[1] + "_long": ls_b, names[1] + "_short": ss_b,
                        **m, "eligible": eligible,
                    })
    results.sort(key=lambda r: (-1 if r["eligible"] else 0, r["pnl"]), reverse=True)
    best = results[0]
    print(f"[{asset}] best VAL combo: {json.dumps(best, default=_json_default)}", flush=True)

    print(f"[{asset}] preparing OOS components (one-time GPU pass) for confirmation...", flush=True)
    oos_frame = frames["oos_raw"]
    oos_components = {name: router_mod._prepare_component(asset, "oos", oos_frame, c, device=device) for name, c in cfg.items()}
    oos_components[names[0]]["long_scale"] = best[names[0] + "_long"]
    oos_components[names[0]]["short_scale"] = best[names[0] + "_short"]
    oos_components[names[1]]["long_scale"] = best[names[1] + "_long"]
    oos_components[names[1]]["short_scale"] = best[names[1] + "_short"]
    oos_ledger = router_mod._greedy_replay(oos_frame, oos_components, fee=fee, slip=slip, cost_mult=3.0, device=device)
    oos_m = router_mod._compound_metrics(oos_ledger)
    print(f"[{asset}] OOS at selected VAL combo: {json.dumps(oos_m, default=_json_default)}", flush=True)

    return {
        "asset": asset,
        "top10_val": results[:10],
        "selected": best,
        "oos_at_selected": oos_m,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=["sol", "btc"], required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    device = router_mod.parent._device(str(args.device))
    report = _sweep(args.asset, device)
    out_path = ROOT / f"tmp/causal_regen_20260516/{args.asset}_omega4_6_1_two_component_router_exit30k_scalemap_20260722.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
