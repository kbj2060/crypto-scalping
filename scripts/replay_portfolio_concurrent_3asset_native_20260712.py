#!/usr/bin/env python3
"""Native bar-by-bar TRUE concurrent 3-asset (ETH/SOL/BTC) portfolio replay.

Unlike every prior portfolio_*.py script (which enforces a single shared
position slot across all three assets, so cross-asset overlap was
structurally impossible), this script tracks one independent position slot
per asset and lets ETH/SOL/BTC be open at the same time. It measures how
often that happens and what combined mark-to-market drawdown looks like, as
a precursor to deciding real portfolio-level caps. It is a research/
diagnostic step only: no learned routing, no trading_bot.py wiring.

Reuses (unmodified) from replay_portfolio_rl_gate_2action_native_20260708.py:
_build_world, _candidate_for_asset, Candidate, Position, _open_position,
_try_close, _force_close, _compound_metrics.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_concurrent_3asset_native_20260712"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md"

ASSETS = ("eth", "sol", "btc")
MARGIN_CAP = 1.0
MIN_NOTIONAL = 0.05


def _price_move(world: dict[str, Any], pos: "native.Position", ts: pd.Timestamp) -> float | None:
    """Read-only price-move for one open position at ts. Duplicates the
    price-move math in native._try_close without mutating cash or
    evaluating exit conditions."""
    aw = world[pos.candidate.asset]
    i = aw["ts_to_i"].get(ts)
    if i is None:
        return None
    arrays = aw["arrays"]
    _, slip = aw["fee_slip"]
    slip_eff = slip * native.COST_MULT
    close_px = arrays["close"][i] * (1 - slip_eff if pos.side > 0 else 1 + slip_eff)
    if pos.side > 0:
        return (close_px - pos.entry_price) / max(pos.entry_price, 1e-12)
    return (pos.entry_price - close_px) / max(pos.entry_price, 1e-12)


def _committed_margin(positions: dict[str, "native.Position | None"]) -> float:
    return float(sum(p.margin for p in positions.values() if p is not None))


def _committed_notional(positions: dict[str, "native.Position | None"]) -> float:
    return float(sum(p.notional for p in positions.values() if p is not None))


def _committed_same_direction_notional(positions: dict[str, "native.Position | None"], side: int) -> float:
    return float(sum(p.notional for p in positions.values() if p is not None and p.side == side))


def _asset_trade_aggregate(ledger: pd.DataFrame, asset: str) -> dict[str, Any]:
    """Simple (non-compounding) per-asset aggregate from the shared ledger.

    Does NOT attempt a compounded PnL/MDD: each trade's `trade_return` is a
    fraction of the shared cash pool at that trade's own entry time, which is
    itself contaminated by other assets' realized gains/losses in between
    (the pool is shared). Chain-compounding these (as native._compound_metrics
    does) silently double-counts realized gains whenever trades from
    different assets overlap in time -- see the portfolio-level fix in
    _replay_concurrent for the same bug applied to the aggregate ledger. For
    an isolated dedicated-capital replay of a single asset, run this script
    with --assets <asset> instead.
    """
    sub = ledger[ledger["asset"] == asset] if not ledger.empty else ledger
    if sub.empty:
        return {"trades": 0, "wr": 0.0, "mean_trade_return": 0.0, "sum_trade_return": 0.0}
    rets = sub["trade_return"].to_numpy(dtype=float)
    return {
        "trades": int(len(sub)),
        "wr": float((rets > 0).mean()),
        "mean_trade_return": float(rets.mean()),
        "sum_trade_return": float(rets.sum()),
    }


def _replay_concurrent(
    world: dict[str, Any],
    *,
    device: torch.device,
    margin_cap: float = MARGIN_CAP,
    total_notional_cap: float | None = None,
    same_direction_notional_cap: float | None = None,
    cap_mode: str = "scale",
    min_notional: float = MIN_NOTIONAL,
    asset_shares: dict[str, float] | None = None,
    asset_notional_multipliers: dict[str, float] | None = None,
    enabled_assets: tuple[str, ...] = ASSETS,
    entry_cutoff: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """cap_mode:

    - "reject": v2 behavior. Skip the candidate entirely if opening it would
      breach total_notional_cap or same_direction_notional_cap. Known issue
      (see docs/model_contracts/portfolio_concurrent_3asset_v2_cap_comparison_20260712.md):
      a rejected candidate is NOT deferred -- the asset stays flat and later
      picks up a different, independent signal, so this can substitute a good
      trade for a worse one and occasionally make both PnL and MDD worse.
    - "scale" (v3, default): never reject for these two caps. Instead shrink
      the candidate's notional down to whatever budget remains (holding
      margin_fraction fixed and reducing leverage = notional/margin, mirroring
      how native._candidate_for_asset already caps to NOTIONAL_CAP). Preserves
      entry timing/side, avoiding the reject-mode substitution effect. Only
      skipped if the capped notional would fall below `min_notional` (a dust
      floor to avoid fee-eating near-zero positions).
    - "prealloc" (v4): fixes the v3 finding that the fixed eth/sol/btc checking
      order starves whichever asset is checked last for a *shared* budget.
      Each asset instead gets its own fixed, non-competing notional budget
      = total_notional_cap * asset_shares[asset], checked independently of
      what any other asset is doing (no cross-asset lookup at all). Same
      shrink-not-reject behavior and min_notional floor as "scale".
      `same_direction_notional_cap` is not used in this mode: worst-case
      same-direction stacking is already structurally bounded by the sum of
      all three per-asset shares (<= total_notional_cap when all three are
      simultaneously at their own max, same direction), so a separate
      same-direction budget would just be a redundant, arbitrary extra cap
      on top of an already-fixed allocation.

    The trivial committed-margin sanity cap (`margin_cap`) is always
    reject-mode regardless of `cap_mode` -- it has never triggered in practice
    (margin_fraction per sleeve is small) and is kept only as a sanity net.

    `asset_notional_multipliers`: applied unconditionally, before any cap_mode
    logic, holding margin_fraction fixed and rescaling leverage = (notional *
    multiplier) / margin -- same convention as every other notional adjustment
    in this file. Composes with any cap_mode including "scale" with no
    total_notional_cap set (i.e. an otherwise-uncapped run), so it can express
    "keep everything uncapped, just deliberately size one asset up/down" as
    its own axis independent of the portfolio caps above. Not clamped back to
    native.NOTIONAL_CAP -- a multiplier > 1 can push a trade's notional above
    what the risk model itself calibrated; this is a deliberate stress/what-if
    knob, not a risk-neutral operation.
    """
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
        # 1) close pass, for every currently-open asset
        for asset in ASSETS:
            pos = positions[asset]
            if pos is not None:
                new_pos, cash, closed, _mark = native._try_close(world, pos, ts, cash, device)
                positions[asset] = new_pos
                if closed is not None:
                    rows.append(closed)
                    realized_peak = max(realized_peak, cash)
                    realized_mdd = min(realized_mdd, cash / max(realized_peak, 1e-12) - 1.0)

        # 2) mark-to-market / concurrency diagnostics (post-close state)
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

        # 3) open pass, flat + enabled assets only
        if entry_cutoff is not None and ts >= entry_cutoff:
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

    # force-close any remaining open positions at end of data
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


def _write_doc(report: dict[str, Any], doc_path: Path) -> None:
    lines = [
        "# Portfolio Concurrent 3-Asset Native Replay - 2026-07-12",
        "",
        "Status: `research_diagnostic_not_live_wired`.",
        "",
        "First TRUE concurrent bar-by-bar replay across ETH/SOL/BTC: each asset has its own",
        "independent position slot (unlike every prior portfolio_*.py script, which enforces a",
        "single shared slot across all three assets and so structurally prevented cross-asset",
        "overlap rather than measuring it to be zero). Native fresh-forward, no saved trade ledger",
        "or saved exit timestamps used as replay input.",
        "",
        f"Concurrency model: `{report['concurrency_model']}`.",
        f"Entry equity convention: `{report['entry_equity_convention']}`.",
        f"MTM equity formula: `{report['mtm_equity_formula']}`.",
        f"Committed margin cap: `{report['committed_margin_cap']}`.",
        f"Total notional cap: `{report['total_notional_cap']}` (mode=`{report['cap_mode']}`).",
        f"Same-direction notional cap: `{report['same_direction_notional_cap']}` (ignored in prealloc mode).",
    ] + (
        [f"Per-asset pre-allocated shares: `{report['asset_shares']}`."]
        if report["cap_mode"] == "prealloc" else []
    ) + [
        "",
        "## Portfolio results",
        "",
        "| split | PnL | MDD (realized) | MTM MDD | trades | WR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        m = report["results"][split]["portfolio"]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} |")

    lines += [
        "",
        "## Per-asset trade aggregates (from the same combined, shared-cash ledger)",
        "",
        "Not a dedicated-capital PnL/MDD -- each trade's return is a fraction of the shared pool",
        "at that trade's own entry time, which is itself affected by other assets' realized",
        "gains/losses in between. For an isolated per-asset baseline, see the separate",
        "`--assets <asset>` solo runs.",
        "",
    ]
    lines += ["| split | asset | trades | WR | mean trade return | sum trade return |", "|---|---|---:|---:|---:|---:|"]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        for asset in ASSETS:
            m = report["results"][split]["per_asset"][asset]
            lines.append(
                f"| {split} | {asset} | {m['trades']} | {m['wr']:.2%} | {m['mean_trade_return']:.4f} | {m['sum_trade_return']:.4f} |"
            )

    lines += ["", "## Concurrency diagnostics", ""]
    lines += ["| split | max concurrent | % bars 2+ open | % bars 3 open | eth&sol bars | eth&btc bars | sol&btc bars | combined MTM MDD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        d = report["diagnostics"][split]
        lines.append(
            f"| {split} | {d['max_concurrent_positions']} | {d['pct_bars_2plus_open']:.2f}% | {d['pct_bars_3_open']:.2f}% | "
            f"{d['pair_overlap_bars']['eth_sol']} | {d['pair_overlap_bars']['eth_btc']} | {d['pair_overlap_bars']['sol_btc']} | "
            f"{d['combined_mtm_mdd']:.2f}% |"
        )

    if report["cap_mode"] == "reject":
        lines += [
            "",
            "## Cap-triggered skips (reject mode)",
            "",
            "| split | cap | eth | sol | btc |",
            "|---|---|---:|---:|---:|",
        ]
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            d = report["diagnostics"][split]
            for cap_name, field in (
                ("margin<=%.2f" % report["committed_margin_cap"], "margin_capped_skips"),
                ("total_notional", "notional_capped_skips"),
                ("same_direction_notional", "same_direction_capped_skips"),
            ):
                s = d[field]
                lines.append(f"| {split} | {cap_name} | {s['eth']} | {s['sol']} | {s['btc']} |")
    else:
        lines += [
            "",
            "## Cap-triggered scaling (scale mode, min_notional=%.3f)" % report["min_notional"],
            "",
            "Preserves entry timing/side; only shrinks notional to fit remaining budget. Skipped",
            "entirely only if the capped notional would fall below `min_notional`.",
            "",
            "| split | margin skips (eth/sol/btc) | scaled events (eth/sol/btc) | skipped below floor (eth/sol/btc) | mean scale ratio | min scale ratio |",
            "|---|---|---|---|---:|---:|",
        ]
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            d = report["diagnostics"][split]
            ms, se, sf = d["margin_capped_skips"], d["notional_scaled_events"], d["notional_skipped_below_floor"]
            mean_r = f"{d['scale_ratio_mean']:.3f}" if d["scale_ratio_mean"] is not None else "n/a"
            min_r = f"{d['scale_ratio_min']:.3f}" if d["scale_ratio_min"] is not None else "n/a"
            lines.append(
                f"| {split} | {ms['eth']}/{ms['sol']}/{ms['btc']} | {se['eth']}/{se['sol']}/{se['btc']} | "
                f"{sf['eth']}/{sf['sol']}/{sf['btc']} | {mean_r} | {min_r} |"
            )

    lines += [
        "",
        "Replay flags:",
        "",
        f"- `fresh_forward_bar_by_bar={str(report['fresh_forward_bar_by_bar']).lower()}`",
        f"- `trade_ledgers_used_as_input={str(report['trade_ledgers_used_as_input']).lower()}`",
        f"- `saved_parent_exit_timestamps_used={str(report['saved_parent_exit_timestamps_used']).lower()}`",
        f"- `future_rows_used_for_entry={str(report['future_rows_used_for_entry']).lower()}`",
        "",
        "## Caveats",
        "",
        "- Committed-margin cap is a trivial sanity check only; the notional cap(s) above are the",
        "  real portfolio-level risk control.",
        "- New positions size off current *realized* cash only (ignore other sleeves' unrealized",
        "  PnL) -- a conservative, explicit modeling choice, not the only valid one.",
        "- Asset processing order is fixed `eth, sol, btc` each bar; in `reject`/`scale` modes this",
        "  order determines who claims a *shared* budget first (see the v3 doc's SOL-starvation",
        "  finding). In `prealloc` mode each asset has its own fixed budget, so processing order no",
        "  longer matters for capping (it still determines close-pass and margin-cap-check order).",
        "- Not a promotion artifact. No live wiring.",
        "",
    ]
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="eth,sol,btc", help="Comma-separated subset of assets allowed to open positions (world still loads all three).")
    parser.add_argument("--margin-cap", type=float, default=MARGIN_CAP)
    parser.add_argument("--total-notional-cap", type=float, default=None, help="Max sum(notional) across all concurrently open positions. Disabled if unset.")
    parser.add_argument("--same-direction-notional-cap", type=float, default=None, help="Max sum(notional) across concurrently open positions sharing the same side. Disabled if unset.")
    parser.add_argument("--cap-mode", choices=("reject", "scale", "prealloc"), default="scale",
                        help="reject=v2 hard-reject (known path-dependence issue); scale=v3 shrink notional to fit shared budget; prealloc=v4 fixed per-asset budget, order-independent (default: scale).")
    parser.add_argument("--min-notional", type=float, default=MIN_NOTIONAL, help="scale/prealloc modes only: skip instead of opening a dust position below this notional.")
    parser.add_argument("--eth-share", type=float, default=0.5, help="prealloc mode: ETH's fraction of total_notional_cap.")
    parser.add_argument("--btc-share", type=float, default=0.3, help="prealloc mode: BTC's fraction of total_notional_cap.")
    parser.add_argument("--sol-share", type=float, default=0.2, help="prealloc mode: SOL's fraction of total_notional_cap.")
    parser.add_argument("--eth-notional-multiplier", type=float, default=1.0, help="Unconditional notional multiplier for ETH, applied before any cap logic (composes with uncapped runs).")
    parser.add_argument("--btc-notional-multiplier", type=float, default=1.0, help="Unconditional notional multiplier for BTC.")
    parser.add_argument("--sol-notional-multiplier", type=float, default=1.0, help="Unconditional notional multiplier for SOL.")
    parser.add_argument("--duration-gate", choices=("on", "off"), default="on",
                        help="off disables native.DURATION_THRESHOLDS (see docs/model_contracts/portfolio_concurrent_3asset_v4_sweep_duration_gate_20260712.md: gate-off strictly dominated gate-on in this session's testing).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    enabled_assets = tuple(a.strip() for a in args.assets.split(",") if a.strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_shares = {"eth": args.eth_share, "btc": args.btc_share, "sol": args.sol_share}
    share_sum = sum(raw_shares.values()) or 1.0
    asset_shares = {k: v / share_sum for k, v in raw_shares.items()}
    asset_notional_multipliers = {
        "eth": args.eth_notional_multiplier,
        "btc": args.btc_notional_multiplier,
        "sol": args.sol_notional_multiplier,
    }
    if args.duration_gate == "off":
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}

    device = native.eth_retest.DEVICE
    results: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    timelines: dict[str, pd.DataFrame] = {}

    for split in ("validation", "oos"):
        print(f"stage=build_world split={split}", flush=True)
        world = native._build_world(split, device)
        print(f"stage=replay split={split}", flush=True)
        metrics, ledger, timeline, diag = _replay_concurrent(
            world,
            device=device,
            margin_cap=args.margin_cap,
            total_notional_cap=args.total_notional_cap,
            same_direction_notional_cap=args.same_direction_notional_cap,
            cap_mode=args.cap_mode,
            min_notional=args.min_notional,
            asset_shares=asset_shares,
            asset_notional_multipliers=asset_notional_multipliers,
            enabled_assets=enabled_assets,
        )
        key = "validation" if split == "validation" else "oos_extended"
        results[key] = metrics
        diagnostics[key] = diag
        ledgers[key] = ledger
        timelines[key] = timeline
        if split == "oos":
            # Re-simulate with an entry cutoff rather than post-hoc filtering the combined
            # ledger: filtering an already-mixed ledger by entry_timestamp would reintroduce
            # the same cross-asset compounding double-count this script exists to avoid,
            # since a kept Q1 trade can still overlap in time with an excluded post-Q1 trade.
            print("stage=replay split=oos_frozen_q1_2026", flush=True)
            q1_metrics, q1_ledger, q1_timeline, q1_diag = _replay_concurrent(
                world,
                device=device,
                margin_cap=args.margin_cap,
                total_notional_cap=args.total_notional_cap,
                same_direction_notional_cap=args.same_direction_notional_cap,
                cap_mode=args.cap_mode,
                min_notional=args.min_notional,
                asset_shares=asset_shares,
                asset_notional_multipliers=asset_notional_multipliers,
                enabled_assets=enabled_assets,
                entry_cutoff=pd.Timestamp("2026-04-01"),
            )
            results["oos_frozen_q1_2026"] = q1_metrics
            diagnostics["oos_frozen_q1_2026"] = q1_diag
            ledgers["oos_frozen_q1_2026"] = q1_ledger
            timelines["oos_frozen_q1_2026"] = q1_timeline

    for split, ledger in ledgers.items():
        ledger.to_csv(out_dir / f"{split}_ledger.csv", index=False)
    for split, timeline in timelines.items():
        timeline.to_csv(out_dir / f"{split}_concurrency_timeline.csv", index=False)

    report = {
        "method": "portfolio_concurrent_3asset_native_bar_by_bar_replay",
        "concurrency_model": "independent_open_positions_per_asset_shared_cash_pool",
        "entry_equity_convention": "new_position_sized_off_realized_cash_only_ignores_other_sleeves_unrealized_pnl",
        "mtm_equity_formula": "cash + sum(move_i * notional_i * pos_i.entry_equity for open i)",
        "committed_margin_cap": float(args.margin_cap),
        "total_notional_cap": args.total_notional_cap,
        "same_direction_notional_cap": args.same_direction_notional_cap,
        "cap_mode": args.cap_mode,
        "min_notional": float(args.min_notional),
        "asset_shares": asset_shares,
        "asset_notional_multipliers": asset_notional_multipliers,
        "duration_gate": args.duration_gate,
        "asset_processing_order": list(ASSETS),
        "enabled_assets": list(enabled_assets),
        "results": results,
        "diagnostics": diagnostics,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=native._json_default) + "\n", encoding="utf-8"
    )
    if tuple(enabled_assets) == ASSETS:
        _write_doc(report, out_dir / "report.md")
        if out_dir == OUT_DIR:
            _write_doc(report, DOC_PATH)
    print(json.dumps({"report": str(out_dir / "report.json"), "results": results}, ensure_ascii=False, indent=2, default=native._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
