#!/usr/bin/env python3
"""Re-run the FROZEN CURRENT_BASELINE 3-asset portfolio config
(docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md:
--duration-gate off --eth-notional-multiplier 1.5) with entry_cutoff=2026-07-01,
i.e. restricted to entries in the genuinely fresh window that was NOT used to
select any of v1/v2/v3/v4/duration-gate-off/eth-multiplier design choices
(all of which repeatedly looked at 2026-01-01..06-30 OOS data).

Data has been extended through ~2026-07-12 for ETH/SOL/BTC (raw klines,
funding, metrics, features, regime3 wide24 overlay, and all three assets'
frozen-bundle oos_predictions re-scored -- no retraining) specifically to
make this confirmation possible.

Monkeypatches native.eth_retest.load_frame_current's hardcoded ("2026-01-01",
"2026-06-30") call inside _eth_components to extend through the new data,
since that literal is not exposed via CLI in the underlying replay scripts.
Reuses _replay_concurrent/_build_world unmodified (no changes to the shared
v1-v4/CURRENT_BASELINE scripts).
"""
from __future__ import annotations

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

import dataclasses  # noqa: E402

import replay_portfolio_concurrent_3asset_native_20260712 as concurrent  # noqa: E402

native = concurrent.native
eth_retest = native.eth_retest
ASSETS = concurrent.ASSETS
_price_move = concurrent._price_move
_committed_margin = concurrent._committed_margin
_committed_notional = concurrent._committed_notional
_committed_same_direction_notional = concurrent._committed_same_direction_notional
_asset_trade_aggregate = concurrent._asset_trade_aggregate


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
    """Copy of concurrent._replay_concurrent with entry_cutoff (upper bound: only
    allow entries strictly before the cutoff, used for the oos_frozen_q1_2026
    sub-split) replaced by entry_floor (lower bound: only allow entries at or
    after the floor) -- needed to isolate the genuinely fresh 2026-07-01+
    window without re-simulating it as if no prior positions ever existed
    (there are none carried in anyway since no entries occur before the floor,
    but bar-by-bar close-pass/mark-to-market bookkeeping for the whole window
    is preserved identically to the original function, just gated on entries).
    """
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

FRESH_CUTOFF = pd.Timestamp("2026-07-01")
NEW_END = "2026-07-13"  # date-only bound must exceed the last actual bar (07-12) so "<=" keeps it
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_fresh_window_20260713"


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _orig_load_frame_current = eth_retest.load_frame_current

    def _patched_load_frame_current(start: str, end: str) -> pd.DataFrame:  # noqa: ARG001
        return _orig_load_frame_current(start, NEW_END)

    eth_retest.load_frame_current = _patched_load_frame_current
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}  # --duration-gate off

        print("stage=build_world split=oos", flush=True)
        world = native._build_world("oos", device)
        print(f"world oos timestamp range: {world['timestamps'][0]} .. {world['timestamps'][-1]}, n={len(world['timestamps'])}", flush=True)

        asset_notional_multipliers = {"eth": 1.5, "btc": 1.0, "sol": 1.0}  # CURRENT_BASELINE

        print("stage=replay full_extended_oos (for comparison)", flush=True)
        full_metrics, full_ledger, full_timeline, full_diag = concurrent._replay_concurrent(
            world, device=device, cap_mode="scale", asset_shares={"eth": 0.5, "btc": 0.3, "sol": 0.2},
            asset_notional_multipliers=asset_notional_multipliers, enabled_assets=("eth", "sol", "btc"),
        )

        print(f"stage=replay fresh_window entry_floor={FRESH_CUTOFF}", flush=True)
        fresh_metrics, fresh_ledger, fresh_timeline, fresh_diag = _replay_concurrent_entry_floor(
            world, device=device, cap_mode="scale", asset_shares={"eth": 0.5, "btc": 0.3, "sol": 0.2},
            asset_notional_multipliers=asset_notional_multipliers, enabled_assets=("eth", "sol", "btc"),
            entry_floor=FRESH_CUTOFF,
        )
    finally:
        eth_retest.load_frame_current = _orig_load_frame_current

    full_ledger.to_csv(OUT_DIR / "full_extended_oos_ledger.csv", index=False)
    fresh_ledger.to_csv(OUT_DIR / "fresh_window_ledger.csv", index=False)
    fresh_timeline.to_csv(OUT_DIR / "fresh_window_concurrency_timeline.csv", index=False)

    report = {
        "method": "portfolio_fresh_window_confirmation_20260713",
        "config": "duration_gate=off, eth_notional_multiplier=1.5, btc/sol=1.0, cap_mode=scale, uncapped (matches CURRENT_BASELINE)",
        "fresh_cutoff": str(FRESH_CUTOFF),
        "data_extended_through": NEW_END,
        "full_extended_oos_2026_01_01_to_new_end": full_metrics,
        "fresh_window_2026_07_01_onward": fresh_metrics,
        "fresh_window_diag": fresh_diag,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
