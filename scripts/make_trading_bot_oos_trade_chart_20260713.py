#!/usr/bin/env python3
"""Build a trading_bot.py-aligned fresh-forward OOS trade chart.

This runs the current Omega4.6.1 concurrent ETH/SOL/BTC logic bar by bar over
the requested OOS window. It does not read saved trade ledgers or saved exit
timestamps as replay input; the ledger written by this script is produced by
the replay loop itself.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_concurrent_3asset_native_20260712 as concurrent  # noqa: E402
import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as eth_greedy  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as eth_retest  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/trading_bot_oos_trade_chart_20260713"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    raise TypeError(type(obj).__name__)


def _concat_frame_and_predictions(
    *,
    warmup_frame: pd.DataFrame,
    warmup_pred: pd.DataFrame,
    oos_frame: pd.DataFrame,
    oos_pred: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.concat([warmup_frame, oos_frame], ignore_index=True)
    pred = pd.concat([warmup_pred, oos_pred], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    pred = pred.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any() or pred["timestamp"].duplicated().any():
        raise RuntimeError(f"{label}: duplicate warm-up/OOS timestamps")
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"{label}: parent prediction timestamps do not exactly match warm-up/OOS frame")
    return frame, pred


def _patch_eth_oos_loader(warmup_start: str | None, oos_start: str, oos_end: str) -> None:
    original = native._eth_components

    def _eth_components(split: str, device):
        if split != "oos":
            return original(split, device)
        oos_frame = eth_retest.load_frame_current(oos_start, oos_end)
        warmup_frame = None
        if warmup_start is not None:
            val_frame = native.eth_valmod.load_val_frame()
            val_frame["timestamp"] = pd.to_datetime(val_frame["timestamp"])
            warmup_frame = val_frame[
                (val_frame["timestamp"] >= pd.Timestamp(warmup_start))
                & (val_frame["timestamp"] < pd.Timestamp(oos_start))
            ].reset_index(drop=True)
        components = {}
        for name, cfg in eth_retest.COMPONENTS.items():
            pred_csv = eth_greedy.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
            if warmup_frame is None:
                component_frame = oos_frame
                component_pred = pd.read_csv(pred_csv)
            else:
                warmup_pred = pd.read_csv(native.eth_valmod.VAL_PRED[name]).rename(
                    columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pd.read_csv(native.eth_valmod.VAL_PRED[name], nrows=0).columns}
                )
                warmup_pred["timestamp"] = pd.to_datetime(warmup_pred["timestamp"])
                component_warmup = warmup_frame[warmup_frame["timestamp"].isin(warmup_pred["timestamp"])].reset_index(drop=True)
                warmup_pred = warmup_pred[warmup_pred["timestamp"].isin(component_warmup["timestamp"])].reset_index(drop=True)
                oos_pred = pd.read_csv(pred_csv)
                oos_pred["timestamp"] = pd.to_datetime(oos_pred["timestamp"])
                oos_pred = oos_pred[oos_pred["timestamp"].isin(oos_frame["timestamp"])].reset_index(drop=True)
                component_frame, component_pred = _concat_frame_and_predictions(
                    warmup_frame=component_warmup,
                    warmup_pred=warmup_pred,
                    oos_frame=oos_frame,
                    oos_pred=oos_pred,
                    label=f"eth/{name}",
                )
            temp_pred = ROOT / "tmp" / "causal_regen_20260516" / f"_warmup_eth_{name}_predictions.csv"
            temp_pred.parent.mkdir(parents=True, exist_ok=True)
            component_pred.to_csv(temp_pred, index=False)
            components[name] = eth_greedy.prepare_component(component_frame, temp_pred, cfg, device)
            components[name]["sidecar"] = eth_greedy.sidecar
            components[name]["long_scale"] = eth_greedy.SCALE_MAP[f"{name}_L"]
            components[name]["short_scale"] = eth_greedy.SCALE_MAP[f"{name}_S"]
            if warmup_frame is not None:
                frame = component_frame
        if warmup_frame is None:
            frame = oos_frame
        fee, slip = eth_greedy.omega._load_fee_slip()
        return frame, components, (float(fee), float(slip))

    native._eth_components = _eth_components


def _patch_sol_btc_oos_loader(warmup_start: str | None, oos_start: str, oos_end: str) -> None:
    if warmup_start is None:
        return
    original = native._asset_component

    def _asset_component(asset: str, split: str, device):
        if split != "oos":
            return original(asset, split, device)
        frames = native.asset_router._load_frames(asset)
        warmup_frame = frames["val_raw"].copy()
        warmup_frame["timestamp"] = pd.to_datetime(warmup_frame["timestamp"])
        warmup_frame = warmup_frame[
            (warmup_frame["timestamp"] >= pd.Timestamp(warmup_start))
            & (warmup_frame["timestamp"] < pd.Timestamp(oos_start))
        ].reset_index(drop=True)
        oos_frame = frames["oos_raw"].copy()
        oos_frame["timestamp"] = pd.to_datetime(oos_frame["timestamp"])
        oos_frame = oos_frame[
            (oos_frame["timestamp"] >= pd.Timestamp(oos_start))
            & (oos_frame["timestamp"] <= pd.Timestamp(oos_end))
        ].reset_index(drop=True)
        cfg = native.SOL_CFG if asset == "sol" else native.BTC_CFG
        parent_dir = ROOT / cfg["parent_dir"]
        warmup_pred = pd.read_csv(parent_dir / f"validation_predictions_{cfg['tag']}.csv").rename(
            columns=lambda column: column.replace("_expertdq_oof_", "_expertdq_")
        )
        warmup_pred["timestamp"] = pd.to_datetime(warmup_pred["timestamp"])
        warmup_pred = warmup_pred[
            warmup_pred["timestamp"].isin(warmup_frame["timestamp"])
        ].reset_index(drop=True)
        oos_pred = pd.read_csv(parent_dir / f"oos_predictions_{cfg['tag']}.csv")
        oos_pred["timestamp"] = pd.to_datetime(oos_pred["timestamp"])
        oos_pred = oos_pred[oos_pred["timestamp"].isin(oos_frame["timestamp"])].reset_index(drop=True)
        frame, pred = _concat_frame_and_predictions(
            warmup_frame=warmup_frame,
            warmup_pred=warmup_pred,
            oos_frame=oos_frame,
            oos_pred=oos_pred,
            label=asset,
        )
        sidecar = importlib.import_module(
            f"train_eval_omega4_2_risk_sidecar_{asset}_{native.asset_router.ASSET_DATES[asset]}"
        )
        original_loader = sidecar._load_precomputed_prediction

        def _load_precomputed_prediction(_parent_dir, requested_split, requested_tag, expected_frame):
            if requested_split != "oos" or requested_tag != cfg["tag"]:
                return original_loader(_parent_dir, requested_split, requested_tag, expected_frame)
            expected_ts = pd.to_datetime(expected_frame["timestamp"]).reset_index(drop=True)
            if not expected_ts.equals(pred["timestamp"]):
                raise RuntimeError(f"{asset}: patched warm-up prediction/frame timestamp mismatch")
            return pred.copy()

        sidecar._load_precomputed_prediction = _load_precomputed_prediction
        try:
            component = native.asset_router._prepare_component(asset, "oos", frame, cfg, device=device)
        finally:
            sidecar._load_precomputed_prediction = original_loader
        omega = importlib.import_module(
            f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{native.asset_router.ASSET_DATES[asset]}"
        )
        fee, slip = omega._load_fee_slip()
        component_name = "zig075" if asset == "sol" else "h48qual"
        return frame, {component_name: component}, (float(fee), float(slip))

    native._asset_component = _asset_component


def _patch_entry_start(oos_start: str) -> None:
    original = native._candidate_for_asset
    start = pd.Timestamp(oos_start)

    def _candidate_for_asset(world: dict[str, Any], asset: str, ts: pd.Timestamp):
        if pd.Timestamp(ts) < start:
            return None
        return original(world, asset, ts)

    native._candidate_for_asset = _candidate_for_asset


def _series_for_asset(world: dict[str, Any], asset: str) -> pd.DataFrame:
    frame = world[asset]["frame"][["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _draw_chart(
    *,
    world: dict[str, Any],
    ledger: pd.DataFrame,
    timeline: pd.DataFrame,
    metrics: dict[str, Any],
    chart_path: Path,
    oos_start: str,
) -> None:
    timeline = timeline.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
    timeline = timeline[timeline["timestamp"] >= pd.Timestamp(oos_start)].reset_index(drop=True)
    timeline["equity_pct"] = (timeline["mtm_equity"].astype(float) - 1.0) * 100.0
    peak = timeline["mtm_equity"].astype(float).cummax()
    timeline["dd_pct"] = (timeline["mtm_equity"].astype(float) / peak.clip(lower=1e-12) - 1.0) * 100.0

    prices = {
        asset: _series_for_asset(world, asset).query("timestamp >= @oos_start")
        for asset in concurrent.ASSETS
    }
    common_start = min(df["timestamp"].min() for df in prices.values())
    prices_norm = {}
    for asset, df in prices.items():
        base = float(df.loc[df["timestamp"] >= common_start, "close"].iloc[0])
        tmp = df.copy()
        tmp["norm"] = tmp["close"].astype(float) / base * 100.0
        prices_norm[asset] = tmp

    ledger = ledger.copy()
    if not ledger.empty:
        ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
        ledger["exit_timestamp"] = pd.to_datetime(ledger["exit_timestamp"])
        ledger["side_label"] = ledger["side"].map({1: "LONG", -1: "SHORT"}).fillna("UNKNOWN")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.25, 1.0]},
    )
    ax_eq, ax_px, ax_dd = axes

    ax_eq.plot(timeline["timestamp"], timeline["equity_pct"], color="#1f77b4", linewidth=1.8)
    ax_eq.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.55)
    ax_eq.set_ylabel("Portfolio MTM PnL (%)")
    ax_eq.set_title(
        "trading_bot.py Omega4.6.1 fresh-forward OOS | "
        f"PnL {metrics['portfolio']['pnl']:.2f}% | "
        f"realized MDD {metrics['portfolio']['mdd']:.2f}% | "
        f"MTM MDD {metrics['portfolio']['mark_to_market_mdd']:.2f}% | "
        f"trades {metrics['portfolio']['trades']}"
    )
    if not ledger.empty:
        ymin, ymax = ax_eq.get_ylim()
        y_marker = ymin + (ymax - ymin) * 0.08
        for asset, color in {"eth": "#2ca02c", "sol": "#9467bd", "btc": "#ff7f0e"}.items():
            sub = ledger[ledger["asset"] == asset]
            if sub.empty:
                continue
            ax_eq.scatter(sub["entry_timestamp"], [y_marker] * len(sub), s=16, marker="^", color=color, alpha=0.75, label=f"{asset} entry")
            ax_eq.scatter(sub["exit_timestamp"], [y_marker] * len(sub), s=16, marker="x", color=color, alpha=0.75, label=f"{asset} exit")
    ax_eq.legend(ncol=3, fontsize=8, loc="upper left")
    ax_eq.grid(True, alpha=0.25)

    for asset, color in {"eth": "#2ca02c", "sol": "#9467bd", "btc": "#ff7f0e"}.items():
        df = prices_norm[asset]
        ax_px.plot(df["timestamp"], df["norm"], linewidth=1.0, alpha=0.9, color=color, label=f"{asset.upper()} close, Jan1=100")
    ax_px.set_ylabel("Normalized close")
    ax_px.legend(ncol=3, fontsize=8, loc="upper left")
    ax_px.grid(True, alpha=0.25)

    ax_dd.fill_between(timeline["timestamp"], timeline["dd_pct"], 0.0, color="#d62728", alpha=0.28)
    ax_dd.plot(timeline["timestamp"], timeline["dd_pct"], color="#d62728", linewidth=1.0)
    ax_dd.set_ylabel("MTM DD (%)")
    ax_dd.set_xlabel("UTC timestamp")
    ax_dd.grid(True, alpha=0.25)

    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-start", default="2026-01-01")
    parser.add_argument("--oos-end", default="2026-07-12 23:59:59")
    parser.add_argument("--warmup-start", default="2025-12-01")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--duration-gate", choices=("on", "off"), default="off")
    parser.add_argument("--eth-notional-multiplier", type=float, default=1.5)
    parser.add_argument("--btc-notional-multiplier", type=float, default=1.0)
    parser.add_argument("--sol-notional-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_eth_oos_loader(args.warmup_start, args.oos_start, args.oos_end)
    _patch_sol_btc_oos_loader(args.warmup_start, args.oos_start, args.oos_end)
    _patch_entry_start(args.oos_start)
    if args.duration_gate == "off":
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}

    device = native.eth_retest.DEVICE
    world = native._build_world("oos", device)
    metrics, ledger, timeline, diagnostics = concurrent._replay_concurrent(
        world,
        device=device,
        asset_notional_multipliers={
            "eth": float(args.eth_notional_multiplier),
            "btc": float(args.btc_notional_multiplier),
            "sol": float(args.sol_notional_multiplier),
        },
        enabled_assets=concurrent.ASSETS,
    )

    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
    timeline = timeline[timeline["timestamp"] >= pd.Timestamp(args.oos_start)].reset_index(drop=True)

    ledger_path = out_dir / "oos_20260101_20260712_fresh_forward_ledger.csv"
    timeline_path = out_dir / "oos_20260101_20260712_timeline.csv"
    report_path = out_dir / "report.json"
    chart_path = out_dir / "trading_bot_oos_trade_chart_20260101_20260712.png"

    ledger.to_csv(ledger_path, index=False)
    timeline.to_csv(timeline_path, index=False)
    _draw_chart(
        world=world,
        ledger=ledger,
        timeline=timeline,
        metrics=metrics,
        chart_path=chart_path,
        oos_start=args.oos_start,
    )

    report = {
        "method": "trading_bot_py_omega461_concurrent_3asset_fresh_forward_chart",
        "source_logic": "trading_bot.py Omega4.6.1 ETH + SOL/BTC shadow asset path; concurrent replay implementation reused from replay_portfolio_concurrent_3asset_native_20260712.py",
        "window": [args.oos_start, args.oos_end],
        "warmup_window": [args.warmup_start, args.oos_start],
        "warmup_bar_by_bar": args.warmup_start is not None,
        "entry_decisions_blocked_during_warmup": args.warmup_start is not None,
        "actual_common_window": [
            str(timeline["timestamp"].min()) if not timeline.empty else "",
            str(timeline["timestamp"].max()) if not timeline.empty else "",
        ],
        "duration_gate": args.duration_gate,
        "asset_notional_multipliers": {
            "eth": float(args.eth_notional_multiplier),
            "btc": float(args.btc_notional_multiplier),
            "sol": float(args.sol_notional_multiplier),
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "results": metrics,
        "diagnostics": diagnostics,
        "artifacts": {
            "ledger": str(ledger_path),
            "timeline": str(timeline_path),
            "chart": str(chart_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "chart": str(chart_path), "results": metrics}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
