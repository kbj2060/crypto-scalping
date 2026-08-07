#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_gated_gross_v2 as dgg  # noqa: E402
from scripts import train_eval_clean_base_deep_gated_gross_v2_safe_cap_buckets as safe  # noqa: E402
from scripts.experiment_cost_firewall_learned_cap_buckets_2026 import (  # noqa: E402
    _compact as _cap_compact,
    _learn_cap_map,
    _thresholds,
    replay_bucket_map,
)
from scripts.experiment_cost_firewall_notional_cap_sweep_2026 import (  # noqa: E402
    CapCandidate,
    replay as replay_static_cap,
)


MODEL_ID = "safe_cap_strict_noleak_walkforward"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/safe_cap_strict_noleak_walkforward"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/safe_cap_strict_noleak_walkforward_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/safe_cap_strict_noleak_walkforward_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_strict_noleak_walkforward_oos_ledger.csv"
DEFAULT_HOLDOUT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_strict_noleak_walkforward_holdout_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/safe_cap_strict_noleak_walkforward_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/safe_cap_strict_noleak_walkforward_contract.md"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _score(metrics: dict[str, Any]) -> float:
    sel = metrics["selection_cost1"]
    c2 = metrics["selection_cost2"]
    c3 = metrics["selection_cost3"]
    score = float(sel["pnl"]) + 0.10 * float(c2["pnl"]) + 0.06 * float(c3["pnl"])
    score -= 7.0 * max(0.0, abs(float(sel["mdd"])) - 25.0)
    score -= 120.0 * int(sel.get("liquidations", 0) or 0)
    score -= 250.0 * int(sel.get("ruin_events", 0) or 0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    return float(score)


def _selection_blockers(row: pd.Series, static_selection: dict[str, Any], args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    if int(row.get("selection_liquidations", 0) or 0) > 0:
        blockers.append("selection_liquidation")
    if int(row.get("selection_ruin_events", 0) or 0) > 0:
        blockers.append("selection_ruin")
    if float(row.get("selection_max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blockers.append("selection_margin_fraction_gt_1")
    if float(row.get("selection_cost2_pnl", -1e9)) <= 0.0:
        blockers.append("selection_cost2_not_survived")
    if float(row.get("selection_cost3_pnl", -1e9)) <= 0.0:
        blockers.append("selection_cost3_not_survived")
    if float(row.get("selection_pnl", -1e9)) <= float(static_selection.get("pnl", -1e9)):
        blockers.append("selection_pnl_not_above_static_cost_firewall")
    if float(row.get("selection_mdd", 0.0)) < float(static_selection.get("mdd", 0.0)) - float(args.max_selection_mdd_worsening):
        blockers.append("selection_mdd_worse_than_allowed")
    return blockers


def _period_survives(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("liquidations", 0) or 0) == 0
        and int(metrics.get("ruin_events", 0) or 0) == 0
        and float(metrics.get("max_margin_fraction", 0.0)) <= 1.0 + 1e-12
        and float(metrics.get("pnl", -1e9)) > 0.0
    )


def _contract(report: dict[str, Any]) -> str:
    sel = report["selected"]
    h1 = sel["holdout_cost1"]
    o1 = sel["oos_cost1"]
    return f"""# Safe Cap Strict No-Leak Walk-Forward

Status: `{report['verdict']}`

## Split Protocol

- Parent train: `{report['data']['parent_train_range']}`
- Cap-map train: `{report['data']['cap_train_range']}`
- Selection: `{report['data']['selection_range']}`
- Untouched holdout: `{report['data']['holdout_range']}`
- Final OOS: `{report['data']['oos_range']}`

## Architecture

```mermaid
flowchart TD
    A["2025 Jan-Aug"] --> B["Train DGG parent"]
    C["2025 Sep"] --> D["Learn safe cap buckets"]
    E["2025 Oct"] --> F["Select DGG/cap candidate"]
    G["2025 Nov-Dec"] --> H["Untouched holdout audit"]
    I["2026 Jan-Feb"] --> J["Final report-only OOS"]
    B --> F
    D --> F
    F --> H
    H --> J
```

## Selected

- Parent DGG config: `{report['selected_parent_config']['name']}`
- Cap candidate: `{sel['candidate']['name']}`
- Scheme: `{sel['candidate'].get('scheme')}`
- Fallback cap: `{sel['candidate'].get('fallback_cap')}`

## Results

- Holdout PnL: `{h1['pnl']:.6f}%`
- Holdout MDD: `{h1['mdd']:.6f}%`
- OOS PnL: `{o1['pnl']:.6f}%`
- OOS MDD: `{o1['mdd']:.6f}%`

## Invariants

- 2025 Nov-Dec holdout is never used for parent config, cap map, or candidate selection.
- 2026 OOS is report-only.
- Fees and slippage are applied on final notional.
- Cap choices cannot exceed exchange leverage cap.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict no-leak walk-forward evaluation for safe learned cap.")
    p.add_argument("--policy", type=Path, default=dgg.v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=dgg.v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=dgg.v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=dgg.v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=dgg.v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--parent-train-end", default="2025-09-01")
    p.add_argument("--cap-train-end", default="2025-10-01")
    p.add_argument("--selection-end", default="2025-11-01")
    p.add_argument("--holdout-end", default=None)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--exchange-leverage-cap", type=float, default=5.0)
    p.add_argument("--cap-choices", default="3.6,4.0,4.5,5.0")
    p.add_argument("--fallback-cap-max", type=float, default=3.6)
    p.add_argument("--min-bucket-trades-floor", type=int, default=10)
    p.add_argument("--min-cost-buffer", type=float, default=0.0035)
    p.add_argument("--max-selection-mdd-worsening", type=float, default=8.0)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--holdout-ledger-csv-out", type=Path, default=DEFAULT_HOLDOUT_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap_choices = [float(x) for x in str(args.cap_choices).split(",") if str(x).strip()]
    if max(cap_choices) > float(args.exchange_leverage_cap) + 1e-12:
        raise SystemExit("cap choices exceed exchange leverage cap")

    train_full = dgg.v1.base._read(args.train_csv)
    parent_train = safe._split_by_date(train_full, None, args.parent_train_end)
    cap_train_raw = safe._split_by_date(train_full, args.parent_train_end, args.cap_train_end)
    selection_raw = safe._split_by_date(train_full, args.cap_train_end, args.selection_end)
    holdout_raw = safe._split_by_date(train_full, args.selection_end, args.holdout_end)
    oos_df = dgg.v1.base._read(args.eval_csv)
    if parent_train.empty or cap_train_raw.empty or selection_raw.empty or holdout_raw.empty or oos_df.empty:
        raise SystemExit("empty chronological split")

    bundle = safe._build_parent_model(args, parent_train)
    selected_parent, parent_grid, parent_selection = safe._select_dgg_config(args, bundle, selection_raw)

    cap_train, cap_train_prices, cap_train_parent = safe._dgg_ledger_for(args, bundle, selected_parent, cap_train_raw)
    selection, selection_prices, selection_parent = safe._dgg_ledger_for(args, bundle, selected_parent, selection_raw)
    holdout, holdout_prices, holdout_parent = safe._dgg_ledger_for(args, bundle, selected_parent, holdout_raw)
    oos, oos_prices, oos_parent = safe._dgg_ledger_for(args, bundle, selected_parent, oos_df)

    thresholds = _thresholds(cap_train)
    baseline_cand = CapCandidate("static_cost_firewall_0p0035_cap3p6", 0.0035, 1.0, 3.6, "base")
    static_baseline = {
        "candidate": {
            "name": baseline_cand.name,
            "scheme": "static",
            "fallback_cap": 3.6,
            "cost_buffer": 0.0035,
            "gate_notional_mode": "base",
        },
        "selection_cost1": replay_static_cap(selection, baseline_cand, prices=selection_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "selection_cost2": replay_static_cap(selection, baseline_cand, prices=selection_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "selection_cost3": replay_static_cap(selection, baseline_cand, prices=selection_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
    }

    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cfg in safe._config_grid(args):
        cap_map, fallback_cap_raw, learn_diag = _learn_cap_map(
            cap_train,
            prices=cap_train_prices,
            cfg=cfg,
            thresholds=thresholds,
            cap_choices=cap_choices,
            fee=float(args.fee),
            slip=float(args.slip),
            exchange_leverage_cap=float(args.exchange_leverage_cap),
        )
        fallback_cap = float(min(float(fallback_cap_raw), float(args.fallback_cap_max)))
        metrics = {
            "selection_cost1": replay_bucket_map(selection, prices=selection_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "selection_cost2": replay_bucket_map(selection, prices=selection_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "selection_cost3": replay_bucket_map(selection, prices=selection_prices, scheme=cfg.scheme, thresholds=thresholds, cap_map=cap_map, fallback_cap=fallback_cap, cost_buffer=cfg.cost_buffer, gate_mode=cfg.gate_notional_mode, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
        score = _score(metrics)
        candidate = {
            **asdict(cfg),
            "cap_map": cap_map,
            "fallback_cap_raw": fallback_cap_raw,
            "fallback_cap": fallback_cap,
            "thresholds": thresholds,
            "learn_diagnostics": learn_diag,
        }
        detailed.append({"candidate": candidate, "score": score, **metrics})
        row: dict[str, Any] = {
            "name": cfg.name,
            "scheme": cfg.scheme,
            "min_bucket_trades": cfg.min_bucket_trades,
            "cost_buffer": cfg.cost_buffer,
            "gate_notional_mode": cfg.gate_notional_mode,
            "fallback_cap": fallback_cap,
            "learned_buckets": len(cap_map),
            "score": score,
        }
        for prefix, data in (
            ("selection", metrics["selection_cost1"]),
            ("selection_cost2", metrics["selection_cost2"]),
            ("selection_cost3", metrics["selection_cost3"]),
        ):
            for key in ("pnl", "mdd", "trades", "blocked", "boosted", "liquidations", "ruin_events", "avg_notional", "max_notional", "max_margin_fraction"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)

    grid = pd.DataFrame(rows)
    blockers = grid.apply(lambda r: _selection_blockers(r, static_baseline["selection_cost1"], args), axis=1)
    grid["selection_eligible"] = blockers.apply(lambda xs: len(xs) == 0)
    grid["selection_blockers"] = blockers.apply(lambda xs: "|".join(xs))
    grid = grid.sort_values("score", ascending=False).reset_index(drop=True)

    eligible = grid[grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    selected_name = str(eligible.iloc[0]["name"]) if not eligible.empty else baseline_cand.name
    selected = next((d for d in detailed if d["candidate"]["name"] == selected_name), static_baseline)

    static_baseline.update(
        {
            "holdout_cost1": replay_static_cap(holdout, baseline_cand, prices=holdout_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "holdout_cost2": replay_static_cap(holdout, baseline_cand, prices=holdout_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "holdout_cost3": replay_static_cap(holdout, baseline_cand, prices=holdout_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost1": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost2": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "oos_cost3": replay_static_cap(oos, baseline_cand, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
    )
    if selected is not static_baseline:
        c = selected["candidate"]
        for prefix, frame, prices in (("holdout", holdout, holdout_prices), ("oos", oos, oos_prices)):
            for mult_name, mult in (("cost1", 1.0), ("cost2", 2.0), ("cost3", 3.0)):
                selected[f"{prefix}_{mult_name}"] = replay_bucket_map(
                    frame,
                    prices=prices,
                    scheme=c["scheme"],
                    thresholds=c["thresholds"],
                    cap_map=c["cap_map"],
                    fallback_cap=c["fallback_cap"],
                    cost_buffer=c["cost_buffer"],
                    gate_mode=c["gate_notional_mode"],
                    fee=args.fee,
                    slip=args.slip,
                    cost_mult=mult,
                    exchange_leverage_cap=args.exchange_leverage_cap,
                )

    if selected is static_baseline:
        replay_static_cap(
            holdout,
            baseline_cand,
            prices=holdout_prices,
            fee=args.fee,
            slip=args.slip,
            cost_mult=1.0,
            exchange_leverage_cap=args.exchange_leverage_cap,
            ledger_out=args.holdout_ledger_csv_out,
        )
        replay_static_cap(
            oos,
            baseline_cand,
            prices=oos_prices,
            fee=args.fee,
            slip=args.slip,
            cost_mult=1.0,
            exchange_leverage_cap=args.exchange_leverage_cap,
            ledger_out=args.ledger_csv_out,
        )
    else:
        c = selected["candidate"]
        replay_bucket_map(
            holdout,
            prices=holdout_prices,
            scheme=c["scheme"],
            thresholds=c["thresholds"],
            cap_map=c["cap_map"],
            fallback_cap=c["fallback_cap"],
            cost_buffer=c["cost_buffer"],
            gate_mode=c["gate_notional_mode"],
            fee=args.fee,
            slip=args.slip,
            cost_mult=1.0,
            exchange_leverage_cap=args.exchange_leverage_cap,
            ledger_out=args.holdout_ledger_csv_out,
        )
        replay_bucket_map(
            oos,
            prices=oos_prices,
            scheme=c["scheme"],
            thresholds=c["thresholds"],
            cap_map=c["cap_map"],
            fallback_cap=c["fallback_cap"],
            cost_buffer=c["cost_buffer"],
            gate_mode=c["gate_notional_mode"],
            fee=args.fee,
            slip=args.slip,
            cost_mult=1.0,
            exchange_leverage_cap=args.exchange_leverage_cap,
            ledger_out=args.ledger_csv_out,
        )

    holdout_ledger = pd.read_csv(args.holdout_ledger_csv_out)
    oos_ledger = pd.read_csv(args.ledger_csv_out)
    blocking: list[str] = []
    warnings: list[str] = []
    if len(holdout_ledger) != len(holdout):
        blocking.append("holdout ledger row count mismatch")
    if len(oos_ledger) != len(oos):
        blocking.append("oos ledger row count mismatch")
    for period_name in ("holdout", "oos"):
        for mult_name in ("cost1", "cost2", "cost3"):
            metrics = selected[f"{period_name}_{mult_name}"]
            if int(metrics.get("liquidations", 0) or 0) > 0:
                blocking.append(f"{period_name}_{mult_name}_liquidation")
            if int(metrics.get("ruin_events", 0) or 0) > 0:
                blocking.append(f"{period_name}_{mult_name}_ruin")
            if float(metrics.get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
                blocking.append(f"{period_name}_{mult_name}_margin_fraction_gt_1")
    if float(selected["holdout_cost2"].get("pnl", 0.0)) <= 0.0:
        warnings.append("holdout_cost2_not_profitable")
    if float(selected["holdout_cost3"].get("pnl", 0.0)) <= 0.0:
        warnings.append("holdout_cost3_not_profitable")
    if float(selected["oos_cost2"].get("pnl", 0.0)) <= 0.0:
        warnings.append("oos_cost2_not_profitable")
    if float(selected["oos_cost3"].get("pnl", 0.0)) <= 0.0:
        warnings.append("oos_cost3_not_profitable")

    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "parent_train_before_cap_train": pd.Timestamp(args.parent_train_end) <= pd.Timestamp(args.cap_train_end),
            "cap_train_before_selection": pd.Timestamp(args.cap_train_end) <= pd.Timestamp(args.selection_end),
            "selection_before_holdout": True,
            "holdout_never_used_for_selection": True,
            "oos_report_only": True,
            "cap_choices_lte_exchange_cap": max(cap_choices) <= float(args.exchange_leverage_cap) + 1e-12,
            "holdout_no_liquidations": int(selected["holdout_cost1"].get("liquidations", 0) or 0) == 0,
            "holdout_no_ruin": int(selected["holdout_cost1"].get("ruin_events", 0) or 0) == 0,
            "oos_no_liquidations": int(selected["oos_cost1"].get("liquidations", 0) or 0) == 0,
            "oos_no_ruin": int(selected["oos_cost1"].get("ruin_events", 0) or 0) == 0,
            "holdout_cost2_profitable": float(selected["holdout_cost2"].get("pnl", 0.0)) > 0.0,
            "holdout_cost3_profitable": float(selected["holdout_cost3"].get("pnl", 0.0)) > 0.0,
            "oos_cost2_profitable": float(selected["oos_cost2"].get("pnl", 0.0)) > 0.0,
            "oos_cost3_profitable": float(selected["oos_cost3"].get("pnl", 0.0)) > 0.0,
        },
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    model_out = args.model_dir / "safe_cap_strict_noleak_walkforward.pkl"
    torch.save({"models": [m.state_dict() for m in bundle["deep_model"].models], "meta": bundle["deep_meta"], "sequence_features": bundle["seq_features"]}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_features": bundle["seq_features"],
            "sequence_scaler": bundle["seq_scaler"],
            "state_model": bundle["state_model"],
            "head_model": bundle["head_model"],
            "head_meta": bundle["head_meta"],
            "deep_meta": bundle["deep_meta"],
            "selected_parent_config": asdict(selected_parent),
            "selected_cap_candidate": selected["candidate"],
            "torch_model": str(torch_out),
        },
        model_out,
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_csv_out, index=False)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    holdout_pass = _period_survives(selected["holdout_cost1"]) and float(selected["holdout_cost2"].get("pnl", 0.0)) > 0.0 and float(selected["holdout_cost3"].get("pnl", 0.0)) > 0.0
    oos_pass = _period_survives(selected["oos_cost1"]) and float(selected["oos_cost2"].get("pnl", 0.0)) > 0.0 and float(selected["oos_cost3"].get("pnl", 0.0)) > 0.0
    verdict = "promote_candidate" if audit["status"] == "pass" and selected is not static_baseline and holdout_pass and oos_pass else "reject_or_requires_iteration"
    report = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "selected_parent_config": asdict(selected_parent),
        "parent_selection": parent_selection,
        "static_baseline": {
            **static_baseline,
            "compact": {
                "selection_cost1": _cap_compact(static_baseline["selection_cost1"]),
                "holdout_cost1": _cap_compact(static_baseline["holdout_cost1"]),
                "holdout_cost2": _cap_compact(static_baseline["holdout_cost2"]),
                "holdout_cost3": _cap_compact(static_baseline["holdout_cost3"]),
                "oos_cost1": _cap_compact(static_baseline["oos_cost1"]),
                "oos_cost2": _cap_compact(static_baseline["oos_cost2"]),
                "oos_cost3": _cap_compact(static_baseline["oos_cost3"]),
            },
        },
        "selected": {
            **selected,
            "compact": {
                "selection_cost1": _cap_compact(selected["selection_cost1"]),
                "selection_cost2": _cap_compact(selected["selection_cost2"]),
                "selection_cost3": _cap_compact(selected["selection_cost3"]),
                "holdout_cost1": _cap_compact(selected["holdout_cost1"]),
                "holdout_cost2": _cap_compact(selected["holdout_cost2"]),
                "holdout_cost3": _cap_compact(selected["holdout_cost3"]),
                "oos_cost1": _cap_compact(selected["oos_cost1"]),
                "oos_cost2": _cap_compact(selected["oos_cost2"]),
                "oos_cost3": _cap_compact(selected["oos_cost3"]),
            },
        },
        "audit_path": str(args.audit_out),
        "audit": audit,
        "data": {
            "parent_train_range": dgg.v1.base._range(parent_train),
            "cap_train_range": dgg.v1.base._range(cap_train_raw),
            "selection_range": dgg.v1.base._range(selection_raw),
            "holdout_range": dgg.v1.base._range(holdout_raw),
            "oos_range": dgg.v1.base._range(oos_df),
            "parent_train_rows": int(len(parent_train)),
            "cap_train_parent_trades": int(len(cap_train)),
            "selection_parent_trades": int(len(selection)),
            "holdout_parent_trades": int(len(holdout)),
            "oos_parent_trades": int(len(oos)),
        },
        "artifacts": {
            "model": str(model_out),
            "torch_model": str(torch_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "holdout_ledger_csv": str(args.holdout_ledger_csv_out),
            "oos_ledger_csv": str(args.ledger_csv_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
        },
        "parent_grid_top10": sorted(parent_grid, key=lambda r: r["parent_selection_score"], reverse=True)[:10],
        "cap_grid_top15": grid.head(15).to_dict(orient="records"),
        "cap_grid_top_eligible": eligible.head(15).to_dict(orient="records"),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_contract(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "selected": selected_name,
                "audit": audit["status"],
                "selection": report["selected"]["compact"]["selection_cost1"],
                "holdout": report["selected"]["compact"]["holdout_cost1"],
                "holdout_cost2": report["selected"]["compact"]["holdout_cost2"],
                "holdout_cost3": report["selected"]["compact"]["holdout_cost3"],
                "oos": report["selected"]["compact"]["oos_cost1"],
                "oos_cost2": report["selected"]["compact"]["oos_cost2"],
                "oos_cost3": report["selected"]["compact"]["oos_cost3"],
                "report": str(args.report_out),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
