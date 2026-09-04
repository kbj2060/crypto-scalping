#!/usr/bin/env python3
"""A4 cross-symbol-exposure-cap confirmation (docs/eth_cross_symbol_exposure_cap_design_20260831.md):
run the CURRENT_BASELINE config (--duration-gate off --eth-notional-multiplier 1.5) with and
WITHOUT the v4 `prealloc` portfolio cap, on data extended through ~2026-08-31, and report
validation / oos_extended / fresh_window(entry_floor=2026-07-01) for both.

This directly answers the standing caveat left open by
docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md ("combining the ETH
multiplier with the v4 prealloc cap design has not been tested") on data that postdates the
2026-01-01..06-30 window every earlier v1-v4/duration-gate/ETH-multiplier design choice in that
chain was selected on (see that doc + gate_off_cap_sweep doc's "heavily peeked window" caveat).

Two configs, both duration_gate=off, eth_notional_multiplier=1.5 (btc/sol=1.0):
  - config A: cap_mode="prealloc", total_notional_cap=3.0, eth/btc/sol shares 0.5/0.3/0.2.
    total_notional_cap=3.0 is NOT a new sweep -- it is reused as-is from the already-swept grid in
    docs/model_contracts/portfolio_concurrent_3asset_v4_prealloc_20260712.md (whose whole results
    table is reported at this cap value) and re-surfaced as the worked example in
    docs/model_contracts/portfolio_concurrent_3asset_gate_off_cap_sweep_20260712.md's own
    recommendation section. This repo's stated policy is that a parameter never swept at a given
    stage must not be carried forward un-swept into the next one -- 3.0 is the one value in that
    prior grid that was already anointed as the reference point by both docs, so re-running the
    grid here would not be "sweeping a new parameter", it would be re-litigating an old one.
  - config B: cap_mode="scale", total_notional_cap=None (uncapped) -- this is byte-for-byte the
    existing CURRENT_BASELINE config already established as this line's comparison point.

Reuses (unmodified) replay_portfolio_concurrent_3asset_native_20260712's `native`/`_replay_concurrent`
plumbing, exactly like scripts/replay_portfolio_fresh_window_20260713.py did -- including that
script's monkeypatch of native.eth_retest.load_frame_current's hardcoded ("2026-01-01","2026-06-30")
literal (not exposed via CLI) and its `_replay_concurrent_entry_floor` copy (adds a lower-bound
entry_floor alongside the upstream entry_cutoff upper bound, needed to isolate the fresh window
without re-simulating from scratch). NEW_END is computed dynamically from the actual extended data
on disk (not hardcoded) since the exact extension end date depends on what was reachable at
pipeline run time.

No trading_bot.py / portfolio_risk.py changes. Purely a research replay -- not a live candidate.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_concurrent_3asset_native_20260712 as concurrent  # noqa: E402

native = concurrent.native
eth_retest = native.eth_retest
ASSETS = concurrent.ASSETS
_price_move = concurrent._price_move
_committed_margin = concurrent._committed_margin
_committed_notional = concurrent._committed_notional
_committed_same_direction_notional = concurrent._committed_same_direction_notional
_asset_trade_aggregate = concurrent._asset_trade_aggregate

OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_prealloc_eth15x_fresh_confirmation_20260831"
FRESH_CUTOFF = pd.Timestamp("2026-07-01")

ASSET_SHARES = {"eth": 0.5, "btc": 0.3, "sol": 0.2}
ASSET_NOTIONAL_MULT = {"eth": 1.5, "btc": 1.0, "sol": 1.0}
TOTAL_NOTIONAL_CAP = 3.0  # reused as-is from the already-swept v4_prealloc / gate_off_cap_sweep grid

CONFIGS: dict[str, dict[str, Any]] = {
    "config_A_prealloc_cap3": {"cap_mode": "prealloc", "total_notional_cap": TOTAL_NOTIONAL_CAP},
    "config_B_uncapped_current_baseline": {"cap_mode": "scale", "total_notional_cap": None},
}


def _replay_concurrent_entry_floor(
    world: dict[str, Any],
    *,
    device: Any,
    margin_cap: float = concurrent.MARGIN_CAP,
    total_notional_cap: float | None = None,
    same_direction_notional_cap: float | None = None,
    cap_mode: str = "scale",
    min_notional: float = concurrent.MIN_NOTIONAL,
    asset_shares: dict[str, float] | None = None,
    asset_notional_multipliers: dict[str, float] | None = None,
    enabled_assets: tuple[str, ...] = None,
    entry_floor: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Copy of concurrent._replay_concurrent with entry_cutoff (upper bound) replaced by
    entry_floor (lower bound: only allow entries at/after the floor) -- identical to
    scripts/replay_portfolio_fresh_window_20260713.py's copy of the same name, reproduced here so
    this script has no import-time dependency on that one-off 07-13 script (which is itself frozen
    to duration_gate=off/eth1.5x/cap_mode=scale only, no prealloc/total_notional_cap knobs)."""
    enabled_assets = enabled_assets or ASSETS
    cash = 1.0
    mtm_peak = 1.0
    mtm_mdd = 0.0
    realized_peak = 1.0
    realized_mdd = 0.0
    positions: dict[str, native.Position | None] = {a: None for a in ASSETS}
    rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    candidate_events = {a: 0 for a in ASSETS}
    margin_capped_skips = {a: 0 for a in ASSETS}
    notional_capped_skips = {a: 0 for a in ASSETS}
    same_direction_capped_skips = {a: 0 for a in ASSETS}
    notional_scaled_events = {a: 0 for a in ASSETS}
    notional_skipped_below_floor = {a: 0 for a in ASSETS}
    scale_ratios: list[float] = []
    bars_by_concurrency_count = {0: 0, 1: 0, 2: 0, 3: 0}
    pair_overlap_bars = {"eth_sol": 0, "eth_btc": 0, "sol_btc": 0}
    max_concurrent = 0

    for ts in world["timestamps"]:
        for asset in ASSETS:
            pos = positions[asset]
            if pos is not None:
                new_pos, cash, closed, _mark = native._try_close(world, pos, ts, cash, device)
                positions[asset] = new_pos
                if closed is not None:
                    rows.append(closed)
                    realized_peak = max(realized_peak, cash)
                    realized_mdd = min(realized_mdd, cash / max(realized_peak, 1e-12) - 1.0)

        n_open = sum(p is not None for p in positions.values())
        max_concurrent = max(max_concurrent, n_open)
        bars_by_concurrency_count[n_open] = bars_by_concurrency_count.get(n_open, 0) + 1
        if positions["eth"] is not None and positions["sol"] is not None:
            pair_overlap_bars["eth_sol"] += 1
        if positions["eth"] is not None and positions["btc"] is not None:
            pair_overlap_bars["eth_btc"] += 1
        if positions["sol"] is not None and positions["btc"] is not None:
            pair_overlap_bars["sol_btc"] += 1

        moves = {a: (_price_move(world, positions[a], ts) if positions[a] is not None else None) for a in ASSETS}
        mtm_equity = cash + sum(
            moves[a] * positions[a].notional * positions[a].entry_equity
            for a in ASSETS
            if positions[a] is not None and moves[a] is not None
        )
        mtm_peak = max(mtm_peak, mtm_equity)
        mtm_mdd = min(mtm_mdd, mtm_equity / max(mtm_peak, 1e-12) - 1.0)

        timeline_rows.append(
            {
                "timestamp": str(ts),
                "n_open": n_open,
                "eth_move": moves["eth"],
                "sol_move": moves["sol"],
                "btc_move": moves["btc"],
                "cash": cash,
                "mtm_equity": mtm_equity,
            }
        )

        if entry_floor is not None and ts < entry_floor:
            continue
        for asset in ASSETS:
            if asset not in enabled_assets or positions[asset] is not None:
                continue
            c = native._candidate_for_asset(world, asset, ts)
            if c is None:
                continue
            mult = (asset_notional_multipliers or {}).get(asset, 1.0)
            if mult != 1.0:
                new_notional = c.notional * mult
                c = dataclasses.replace(c, notional=new_notional, leverage=new_notional / max(c.margin, 1e-12))
            candidate_events[asset] += 1
            committed_margin = _committed_margin(positions)
            if committed_margin + c.margin > margin_cap + 1e-9:
                margin_capped_skips[asset] += 1
                continue

            if cap_mode == "reject":
                if total_notional_cap is not None:
                    committed_notional = _committed_notional(positions)
                    if committed_notional + c.notional > total_notional_cap + 1e-9:
                        notional_capped_skips[asset] += 1
                        continue
                if same_direction_notional_cap is not None:
                    committed_same_dir = _committed_same_direction_notional(positions, c.side)
                    if committed_same_dir + c.notional > same_direction_notional_cap + 1e-9:
                        same_direction_capped_skips[asset] += 1
                        continue
            elif cap_mode == "prealloc":
                notional_final = c.notional
                if total_notional_cap is not None:
                    budget = total_notional_cap * (asset_shares or {}).get(asset, 0.0)
                    notional_final = min(notional_final, max(0.0, budget))
                if notional_final < min_notional - 1e-9:
                    notional_skipped_below_floor[asset] += 1
                    continue
                if notional_final < c.notional - 1e-9:
                    notional_scaled_events[asset] += 1
                    scale_ratios.append(notional_final / max(c.notional, 1e-12))
                    c = dataclasses.replace(c, notional=notional_final, leverage=notional_final / max(c.margin, 1e-12))
            else:
                notional_final = c.notional
                if total_notional_cap is not None:
                    remaining = total_notional_cap - _committed_notional(positions)
                    notional_final = min(notional_final, max(0.0, remaining))
                if same_direction_notional_cap is not None:
                    remaining_same_dir = same_direction_notional_cap - _committed_same_direction_notional(positions, c.side)
                    notional_final = min(notional_final, max(0.0, remaining_same_dir))
                if notional_final < min_notional - 1e-9:
                    notional_skipped_below_floor[asset] += 1
                    continue
                if notional_final < c.notional - 1e-9:
                    notional_scaled_events[asset] += 1
                    scale_ratios.append(notional_final / max(c.notional, 1e-12))
                    c = dataclasses.replace(c, notional=notional_final, leverage=notional_final / max(c.margin, 1e-12))

            pos, cash = native._open_position(world, c, cash)
            positions[asset] = pos
            realized_peak = max(realized_peak, cash)
            realized_mdd = min(realized_mdd, cash / max(realized_peak, 1e-12) - 1.0)

    for asset in ASSETS:
        pos = positions[asset]
        if pos is not None:
            cash, row = native._force_close(world, pos, cash)
            rows.append(row)
            positions[asset] = None
            realized_peak = max(realized_peak, cash)
            realized_mdd = min(realized_mdd, cash / max(realized_peak, 1e-12) - 1.0)

    ledger = pd.DataFrame(rows)
    timeline = pd.DataFrame(timeline_rows)
    wins = int((ledger["trade_return"] > 0).sum()) if not ledger.empty else 0
    n_trades = int(len(ledger))
    portfolio_metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(realized_mdd * 100.0),
        "trades": n_trades,
        "wr": float(wins / n_trades) if n_trades else 0.0,
        "mark_to_market_mdd": float(mtm_mdd * 100.0),
    }
    per_asset_metrics: dict[str, Any] = {asset: _asset_trade_aggregate(ledger, asset) for asset in ASSETS}
    n_bars = max(len(world["timestamps"]), 1)
    diagnostics = {
        "max_concurrent_positions": int(max_concurrent),
        "bars_by_concurrency_count": {str(k): int(v) for k, v in bars_by_concurrency_count.items()},
        "pct_bars_2plus_open": float(100.0 * sum(v for k, v in bars_by_concurrency_count.items() if k >= 2) / n_bars),
        "pct_bars_3_open": float(100.0 * bars_by_concurrency_count.get(3, 0) / n_bars),
        "pair_overlap_bars": pair_overlap_bars,
        "candidate_events": candidate_events,
        "margin_capped_skips": margin_capped_skips,
        "notional_capped_skips": notional_capped_skips,
        "same_direction_capped_skips": same_direction_capped_skips,
        "notional_scaled_events": notional_scaled_events,
        "notional_skipped_below_floor": notional_skipped_below_floor,
        "scale_ratio_mean": float(np.mean(scale_ratios)) if scale_ratios else None,
        "scale_ratio_min": float(np.min(scale_ratios)) if scale_ratios else None,
        "scale_ratio_count": len(scale_ratios),
        "combined_mtm_mdd": float(mtm_mdd * 100.0),
        "final_cash": float(cash),
    }
    return {"portfolio": portfolio_metrics, "per_asset": per_asset_metrics}, ledger, timeline, diagnostics


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _compute_new_end() -> str:
    """Date-only upper bound that must exceed the last actual bar of all three assets (so "<="
    in eth_retest.load_frame_current keeps everything) -- computed from what's actually on disk
    rather than hardcoded, since the exact reachable extension date depends on data availability
    at pipeline run time."""
    ends = []
    for path, col_idx in [
        (ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv", 0),
        (ROOT / "data/splits/year_oos/sol_features_2026.csv", 0),
        (ROOT / "data/splits/year_oos/btc_features_2026.csv", 0),
    ]:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 4096)
            f.seek(size - chunk)
            tail = f.read()
        last_line = [ln for ln in tail.strip().splitlines() if ln.strip()][-1]
        ts_str = last_line.split(",")[0]
        ends.append(pd.Timestamp(ts_str))
    max_end = max(ends)
    new_end = (max_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"per-asset last bars: {ends} -> NEW_END={new_end}", flush=True)
    return new_end


def _run_one_config(world: dict[str, Any], device: Any, *, cap_mode: str, total_notional_cap: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name, entry_floor in (
        ("oos_extended", None),
        ("fresh_window_2026_07_01_onward", FRESH_CUTOFF),
    ):
        print(f"  stage=replay split={split_name} cap_mode={cap_mode} total_notional_cap={total_notional_cap}", flush=True)
        metrics, ledger, timeline, diag = _replay_concurrent_entry_floor(
            world, device=device, cap_mode=cap_mode, total_notional_cap=total_notional_cap,
            asset_shares=ASSET_SHARES, asset_notional_multipliers=ASSET_NOTIONAL_MULT,
            enabled_assets=("eth", "sol", "btc"), entry_floor=entry_floor,
        )
        result[split_name] = {"metrics": metrics, "diagnostics": diag}
        result[split_name]["_ledger"] = ledger
        result[split_name]["_timeline"] = timeline
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_end = _compute_new_end()

    _orig_load_frame_current = eth_retest.load_frame_current

    def _patched_load_frame_current(start: str, end: str) -> pd.DataFrame:  # noqa: ARG001
        return _orig_load_frame_current(start, new_end)

    eth_retest.load_frame_current = _patched_load_frame_current
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}  # --duration-gate off

        print("stage=build_world split=validation", flush=True)
        world_val = native._build_world("validation", device)
        print(f"validation world timestamp range: {world_val['timestamps'][0]} .. {world_val['timestamps'][-1]}, n={len(world_val['timestamps'])}", flush=True)

        print("stage=build_world split=oos", flush=True)
        world_oos = native._build_world("oos", device)
        print(f"oos world timestamp range: {world_oos['timestamps'][0]} .. {world_oos['timestamps'][-1]}, n={len(world_oos['timestamps'])}", flush=True)

        all_results: dict[str, Any] = {}
        for config_name, cfg in CONFIGS.items():
            print(f"\n=== {config_name}: cap_mode={cfg['cap_mode']} total_notional_cap={cfg['total_notional_cap']} ===", flush=True)
            print(f"  stage=replay split=validation config={config_name}", flush=True)
            val_metrics, val_ledger, val_timeline, val_diag = _replay_concurrent_entry_floor(
                world_val, device=device, cap_mode=cfg["cap_mode"], total_notional_cap=cfg["total_notional_cap"],
                asset_shares=ASSET_SHARES, asset_notional_multipliers=ASSET_NOTIONAL_MULT,
                enabled_assets=("eth", "sol", "btc"), entry_floor=None,
            )
            oos_results = _run_one_config(world_oos, device, cap_mode=cfg["cap_mode"], total_notional_cap=cfg["total_notional_cap"])

            splits = {"validation": {"metrics": val_metrics, "diagnostics": val_diag, "_ledger": val_ledger, "_timeline": val_timeline}}
            splits.update(oos_results)

            for split_name, payload in splits.items():
                ledger = payload.pop("_ledger")
                timeline = payload.pop("_timeline")
                ledger.to_csv(OUT_DIR / f"{config_name}__{split_name}__ledger.csv", index=False)
                timeline.to_csv(OUT_DIR / f"{config_name}__{split_name}__timeline.csv", index=False)

            all_results[config_name] = {
                "cap_mode": cfg["cap_mode"],
                "total_notional_cap": cfg["total_notional_cap"],
                "asset_shares": ASSET_SHARES if cfg["cap_mode"] == "prealloc" else None,
                "results": {k: v["metrics"] for k, v in splits.items()},
                "diagnostics": {k: v["diagnostics"] for k, v in splits.items()},
            }
    finally:
        eth_retest.load_frame_current = _orig_load_frame_current

    report = {
        "method": "portfolio_prealloc_eth15x_fresh_confirmation_20260831",
        "base_config": "duration_gate=off, eth_notional_multiplier=1.5, btc/sol_notional_multiplier=1.0 (CURRENT_BASELINE knobs)",
        "fresh_cutoff": str(FRESH_CUTOFF),
        "data_extended_through": new_end,
        "configs": all_results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
