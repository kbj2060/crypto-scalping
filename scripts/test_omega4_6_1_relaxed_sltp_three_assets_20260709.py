#!/usr/bin/env python3
"""Fresh-forward replay of relaxed Omega4.6.1 SLTP floors across ETH/SOL/BTC.

This is a research-only sensitivity test. It reuses the frozen parent/risk/exit
artifacts and existing greedy live-style replay implementations, overriding only
ATR safety TP/SL floors and the exit-head threshold for each variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_relaxed_sltp_three_assets_20260709"
ETH_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"

VARIANTS = [
    {"name": "baseline", "min_tp": 0.075, "min_sl": 0.040, "exit_threshold": 0.95},
    {"name": "relax_025_012_x090", "min_tp": 0.025, "min_sl": 0.012, "exit_threshold": 0.90},
]


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


@contextmanager
def _sltp_floor_override(min_tp: float, min_sl: float):
    original = atr_eval._apply_atr_safety_sltp

    def wrapped(decisions: pd.DataFrame, frame: pd.DataFrame, **kwargs: Any):
        kwargs["min_tp"] = float(min_tp)
        kwargs["min_sl"] = float(min_sl)
        return original(decisions, frame, **kwargs)

    atr_eval._apply_atr_safety_sltp = wrapped
    try:
        yield
    finally:
        atr_eval._apply_atr_safety_sltp = original


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
    }


def _ledger_stats(ledger: pd.DataFrame) -> dict[str, Any]:
    return {
        "metrics": _compound_metrics(ledger),
        "reason_counts": ledger["reason"].value_counts().to_dict() if not ledger.empty else {},
        "source_counts": ledger["source_component"].value_counts().to_dict() if not ledger.empty else {},
    }


def _attach_duration(ledger: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        out = ledger.copy()
        out["ou_halflife"] = []
        return out
    out = ledger.copy()
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"])
    market = frame[["timestamp", "ou_halflife"]].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    return out.merge(
        market.rename(columns={"timestamp": "entry_timestamp"}),
        on="entry_timestamp",
        how="left",
        validate="one_to_one",
    )


def _set_exit_threshold(components: dict[str, Any], exit_threshold: float) -> dict[str, Any]:
    for comp in components.values():
        comp["exit_threshold"] = float(exit_threshold)
    return components


def _prepare_eth_val_component(name: str, cfg: dict[str, Any], frame: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    import replay_omega4_6_1_greedy_router_20260706 as eth_replay
    import replay_omega4_6_1_greedy_val_20260706 as eth_val

    pred = pd.read_csv(eth_val.VAL_PRED[name])
    pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred = pred[(pred["timestamp"] >= eth_val.START) & (pred["timestamp"] <= eth_val.END)].reset_index(drop=True)
    pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
    tmp_pred = OUT_DIR / f"_eth_val_{name}_aligned.csv"
    pred.to_csv(tmp_pred, index=False)
    return eth_replay.prepare_component(frame, tmp_pred, cfg, device)


def _eth_val_frame() -> pd.DataFrame:
    import replay_omega4_6_1_greedy_val_20260706 as eth_val

    frame = eth_val.load_val_frame()
    keep = None
    for pred_path in eth_val.VAL_PRED.values():
        pred = pd.read_csv(pred_path, usecols=["timestamp"])
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[(pred["timestamp"] >= eth_val.START) & (pred["timestamp"] <= eth_val.END)]
        pred_ts = set(pred["timestamp"])
        keep = pred_ts if keep is None else keep.intersection(pred_ts)
    if keep is None:
        raise RuntimeError("ETH validation prediction timestamps unavailable")
    return frame[frame["timestamp"].isin(keep)].reset_index(drop=True)


def _run_eth_variant(variant: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    import replay_omega4_6_1_greedy_router_20260706 as eth_replay
    import replay_omega4_6_1_greedy_val_20260706 as eth_val
    import retest_omega4_6_1_extended_oos_20260706 as eth_retest
    import train_eval_omega1_2_tabm_diffusion_risk_20260603 as eth_omega

    fee, slip = eth_omega._load_fee_slip()
    result: dict[str, Any] = {}
    with _sltp_floor_override(variant["min_tp"], variant["min_sl"]):
        val_frame = _eth_val_frame()
        val_components = {
            name: _prepare_eth_val_component(name, cfg, val_frame, device)
            for name, cfg in eth_retest.COMPONENTS.items()
        }
        _set_exit_threshold(val_components, variant["exit_threshold"])
        _, val_ledger = eth_replay.greedy_replay(
            val_frame,
            val_components,
            fee=fee,
            slip=slip,
            cost_mult=eth_retest.COST_MULT,
            device=device,
        )
        val_d = _attach_duration(val_ledger, val_frame)
        val_gated = val_d.loc[val_d["ou_halflife"] > eth_replay.DURATION_THRESHOLD].reset_index(drop=True)
        result["validation"] = {
            "no_duration_gate": _ledger_stats(val_ledger),
            "with_duration_gate": {**_ledger_stats(val_gated), "skipped": int(len(val_d) - len(val_gated))},
            "duration_threshold": float(eth_replay.DURATION_THRESHOLD),
        }

        oos_frame = eth_retest.load_frame_current("2026-01-01", "2026-06-30")
        oos_components = {}
        for name, cfg in eth_retest.COMPONENTS.items():
            pred_csv = ETH_OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
            oos_components[name] = eth_replay.prepare_component(oos_frame, pred_csv, cfg, device)
        _set_exit_threshold(oos_components, variant["exit_threshold"])
        _, oos_ledger = eth_replay.greedy_replay(
            oos_frame,
            oos_components,
            fee=fee,
            slip=slip,
            cost_mult=eth_retest.COST_MULT,
            device=device,
        )
        oos_d = _attach_duration(oos_ledger, oos_frame)
        oos_gated = oos_d.loc[oos_d["ou_halflife"] > eth_replay.DURATION_THRESHOLD].reset_index(drop=True)
        q1 = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
        result["oos"] = {
            "no_duration_gate": _ledger_stats(oos_ledger),
            "with_duration_gate": {**_ledger_stats(oos_gated), "skipped": int(len(oos_d) - len(oos_gated))},
            "with_duration_gate_q1_2026": _ledger_stats(q1),
            "duration_threshold": float(eth_replay.DURATION_THRESHOLD),
        }
    return result


def _run_asset_variant(asset: str, variant: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    import replay_omega4_6_1_two_component_router_assets_20260708 as asset_replay

    date = asset_replay.ASSET_DATES[asset]
    omega = __import__(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{date}")
    fee, slip = omega._load_fee_slip()
    frames = asset_replay._load_frames(asset)
    result: dict[str, Any] = {}
    selected_duration = 0.0
    with _sltp_floor_override(variant["min_tp"], variant["min_sl"]):
        for split, frame_key in (("validation", "val_raw"), ("oos", "oos_raw")):
            frame = frames[frame_key]
            components = {
                name: asset_replay._prepare_component(asset, split, frame, cfg, device=device)
                for name, cfg in asset_replay.CONFIGS[asset].items()
            }
            _set_exit_threshold(components, variant["exit_threshold"])
            ledger = asset_replay._greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=3.0, device=device)
            ledger_d = _attach_duration(ledger, frame)
            if split == "validation":
                duration = asset_replay._duration_search(ledger_d)
                selected_duration = float(duration["selected"]["threshold"])
                result["duration_gate"] = duration
            gated = ledger_d.loc[ledger_d["ou_halflife"] > selected_duration].reset_index(drop=True)
            split_result: dict[str, Any] = {
                "no_duration_gate": _ledger_stats(ledger),
                "with_duration_gate": {**_ledger_stats(gated), "skipped": int(len(ledger_d) - len(gated))},
                "duration_threshold": selected_duration,
            }
            if split == "oos":
                q1 = gated.loc[gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
                split_result["with_duration_gate_q1_2026"] = _ledger_stats(q1)
            result[split] = split_result
    return result


def _flatten_row(asset: str, variant: dict[str, Any], split: str, gate: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "asset": asset,
        "variant": variant["name"],
        "min_tp": variant["min_tp"],
        "min_sl": variant["min_sl"],
        "exit_threshold": variant["exit_threshold"],
        "split": split,
        "gate": gate,
        "pnl": metrics["pnl"],
        "mdd": metrics["mdd"],
        "trades": metrics["trades"],
        "wr": metrics["wr"],
        "reason_counts": json.dumps(payload.get("reason_counts", {}), sort_keys=True),
        "source_counts": json.dumps(payload.get("source_counts", {}), sort_keys=True),
        "skipped": payload.get("skipped", 0),
    }


def _write_outputs(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--assets", nargs="+", choices=["eth", "sol", "btc"], default=["eth", "sol", "btc"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"starting relaxed SLTP replay assets={args.assets} device={device}", flush=True)
    report: dict[str, Any] = {
        "method": "omega4_6_1_relaxed_sltp_floor_sensitivity",
        "variants": VARIANTS,
        "assets": {},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    rows: list[dict[str, Any]] = []

    for asset in args.assets:
        report["assets"][asset] = {}
        for variant in VARIANTS:
            print(f"[{asset}] {variant['name']} start", flush=True)
            if asset == "eth":
                result = _run_eth_variant(variant, device=device)
            else:
                result = _run_asset_variant(asset, variant, device=device)
            report["assets"][asset][variant["name"]] = result
            for split in ("validation", "oos"):
                for gate_key in ("no_duration_gate", "with_duration_gate"):
                    rows.append(_flatten_row(asset, variant, split, gate_key, result[split][gate_key]))
                if split == "oos":
                    rows.append(_flatten_row(asset, variant, split, "with_duration_gate_q1_2026", result[split]["with_duration_gate_q1_2026"]))
            _write_outputs(report, rows)
            print(f"[{asset}] {variant['name']} done", flush=True)

    _write_outputs(report, rows)
    print(f"wrote {OUT_DIR / 'summary.csv'}", flush=True)
    print(f"wrote {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
