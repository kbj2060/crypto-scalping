#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import full_replay_omega4_4_v18_short_aged_profit_overlays_20260625 as v18  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


MODEL_ID = "omega44_v18_borrow_omega3_risk_full_replay_20260626"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RiskSpec:
    variant: str
    risk_mode: str
    tp_sl_mode: str
    fixed_notional: float = 0.0
    fixed_leverage: float = 2.0
    notional_scale: float = 1.0
    notional_cap: float = 0.0
    exit_head: bool = True
    short_partial: bool = False


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


def _specs() -> list[RiskSpec]:
    specs = [
        RiskSpec("baseline_v18", "sidecar", "atr", exit_head=True),
        RiskSpec("baseline_v18_no_exithead", "sidecar", "atr", exit_head=False),
        RiskSpec("sidecar_omega3_equity_barrier_exit", "sidecar", "omega3_equity", exit_head=True),
        RiskSpec("sidecar_omega3_equity_barrier_noexit", "sidecar", "omega3_equity", exit_head=False),
        RiskSpec("sidecar_omega3_equity_barrier_shortpartial", "sidecar", "omega3_equity", exit_head=True, short_partial=True),
    ]
    for notional in (0.55, 0.65, 0.75, 0.81, 0.90, 1.20, 1.58):
        tag = f"{notional:.2f}".replace(".", "p")
        specs.extend(
            [
                RiskSpec(f"fixed{tag}_omega3_equity_barrier_exit", "fixed", "omega3_equity", fixed_notional=notional, exit_head=True),
                RiskSpec(f"fixed{tag}_omega3_equity_barrier_noexit", "fixed", "omega3_equity", fixed_notional=notional, exit_head=False),
                RiskSpec(f"fixed{tag}_omega3_equity_barrier_shortpartial", "fixed", "omega3_equity", fixed_notional=notional, exit_head=True, short_partial=True),
                RiskSpec(f"fixed{tag}_keep_v18_atr_exit", "fixed", "atr", fixed_notional=notional, exit_head=True),
                RiskSpec(f"fixed{tag}_keep_v18_atr_noexit", "fixed", "atr", fixed_notional=notional, exit_head=False),
            ]
        )
    for scale, cap in ((1.15, 0.95), (1.35, 1.05), (1.60, 1.20), (2.00, 1.58)):
        tag = f"s{scale:.2f}_cap{cap:.2f}".replace(".", "p")
        specs.extend(
            [
                RiskSpec(f"scaled_{tag}_omega3_equity_barrier_exit", "scaled", "omega3_equity", notional_scale=scale, notional_cap=cap, exit_head=True),
                RiskSpec(f"scaled_{tag}_omega3_equity_barrier_noexit", "scaled", "omega3_equity", notional_scale=scale, notional_cap=cap, exit_head=False),
                RiskSpec(f"scaled_{tag}_keep_v18_atr_exit", "scaled", "atr", notional_scale=scale, notional_cap=cap, exit_head=True),
            ]
        )
    return specs


def _risk_arrays(spec: RiskSpec, base_margin: np.ndarray, base_leverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_notional = np.asarray(base_margin, dtype=np.float64) * np.asarray(base_leverage, dtype=np.float64)
    if spec.risk_mode == "sidecar":
        return np.asarray(base_margin, dtype=np.float64).copy(), np.asarray(base_leverage, dtype=np.float64).copy()
    if spec.risk_mode == "fixed":
        lev = np.full_like(base_notional, float(spec.fixed_leverage), dtype=np.float64)
        margin = np.full_like(base_notional, float(spec.fixed_notional) / max(float(spec.fixed_leverage), 1.0e-12), dtype=np.float64)
        return margin, lev
    if spec.risk_mode == "scaled":
        notional = base_notional * float(spec.notional_scale)
        if float(spec.notional_cap) > 0.0:
            notional = np.minimum(notional, float(spec.notional_cap))
        lev = np.full_like(base_notional, float(spec.fixed_leverage), dtype=np.float64)
        margin = notional / max(float(spec.fixed_leverage), 1.0e-12)
        return margin, lev
    raise ValueError(spec.risk_mode)


def _apply_tp_sl(spec: RiskSpec, dec: pd.DataFrame, margin: np.ndarray, leverage: np.ndarray) -> pd.DataFrame:
    out = dec.copy()
    notional = np.asarray(margin, dtype=np.float64) * np.asarray(leverage, dtype=np.float64)
    if spec.tp_sl_mode == "atr":
        return out
    if spec.tp_sl_mode == "omega3_equity":
        safe_notional = np.maximum(notional, 1.0e-12)
        # Omega3 aggressive ledger contract: TP 5.2% equity, SL 2.8% equity.
        out["take_profit"] = 0.052 / safe_notional
        out["stop_loss"] = 0.028 / safe_notional
        return out
    raise ValueError(spec.tp_sl_mode)


def _overlay_spec(spec: RiskSpec) -> v18.OverlaySpec:
    if spec.short_partial:
        return v18.OverlaySpec(
            f"{spec.variant}_shortpartial",
            "partial_deleverage",
            -1,
            1152,
            0.035,
            partial_fraction=0.50,
        )
    return v18.OverlaySpec(spec.variant, "none", -1, 0, 0.0)


def _report_for_exit(report: dict[str, Any], spec: RiskSpec) -> dict[str, Any]:
    if spec.exit_head:
        return report
    out = copy.deepcopy(report)
    out["contract"]["exit_threshold"] = 2.0
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(v18.REPORT_PATH.read_text(encoding="utf-8"))
    device = parent._device("cuda")
    payload, extra = v18._prepare_payload(report, device)
    fee, slip = v18.omega._load_fee_slip()
    rows: list[dict[str, Any]] = []
    for spec in _specs():
        row: dict[str, Any] = {
            "variant": spec.variant,
            "risk_mode": spec.risk_mode,
            "tp_sl_mode": spec.tp_sl_mode,
            "fixed_notional": spec.fixed_notional,
            "fixed_leverage": spec.fixed_leverage,
            "notional_scale": spec.notional_scale,
            "notional_cap": spec.notional_cap,
            "exit_head": spec.exit_head,
            "short_partial": spec.short_partial,
        }
        for split, (frame, base_x, dec, base_margin, base_leverage) in payload.items():
            margin, leverage = _risk_arrays(spec, base_margin, base_leverage)
            dec2 = _apply_tp_sl(spec, dec, margin, leverage)
            metrics, ledger = v18._replay_overlay(
                frame,
                base_x,
                dec2,
                extra["loaded"],
                margin,
                leverage,
                _overlay_spec(spec),
                report=_report_for_exit(report, spec),
                fee=fee,
                slip=slip,
                device=device,
            )
            for key, value in metrics.items():
                if key == "exit_reasons":
                    row[f"{split}_exit_reasons"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    row[f"{split}_{key}"] = value
            if spec.variant in {"baseline_v18", "fixed0p81_omega3_equity_barrier_exit", "fixed0p90_omega3_equity_barrier_exit", "fixed1p58_omega3_equity_barrier_exit"}:
                ledger.to_csv(OUT_DIR / f"{split}_{spec.variant}_ledger.csv", index=False)
        rows.append(row)
        print(json.dumps({"variant": spec.variant, "validation_pnl": row["validation_pnl"], "oos_pnl": row["oos_pnl"]}, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df["variant"].eq("baseline_v18")].iloc[0]
    for split in ("validation", "oos"):
        df[f"{split}_delta_vs_v18_pnl"] = df[f"{split}_pnl"] - float(base[f"{split}_pnl"])
        df[f"{split}_delta_vs_v18_mdd"] = df[f"{split}_mdd"] - float(base[f"{split}_mdd"])
        df[f"{split}_delta_vs_v18_log_risk"] = df[f"{split}_log_risk_utility"] - float(base[f"{split}_log_risk_utility"])
    # Conservative validation selector: reward PnL, penalize MDD beyond -16 and huge notional.
    df["validation_selector_score"] = (
        df["validation_pnl"]
        - 2.5 * np.maximum(0.0, -16.0 - df["validation_mdd"])
        - 6.0 * np.maximum(0.0, df["validation_avg_notional"] - 1.0)
    )
    df = df.sort_values(["validation_selector_score", "validation_pnl"], ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "full_replay_borrowed_risk_grid.csv", index=False)
    strict_v18 = df[
        (df["validation_pnl"] > base["validation_pnl"])
        & (df["validation_mdd"] >= base["validation_mdd"])
        & (df["oos_pnl"] > base["oos_pnl"])
        & (df["oos_mdd"] >= base["oos_mdd"])
    ].copy()
    strict_v18.to_csv(OUT_DIR / "strict_beats_v18_grid.csv", index=False)
    report_out = {
        "model_id": MODEL_ID,
        "source_model": "omega4_4_v18_baseline_20260624",
        "borrowed_from": "omega3_aggressive_compensated_scale200_cap090_20260618",
        "borrowed_contract": {
            "fixed_aggressive_notional_reference": 0.81,
            "fixed_leverage_reference": 2.0,
            "tp_equity_return": 0.052,
            "sl_equity_return": 0.028,
            "tp_sl_mapping": "price_move_barrier = equity_return_barrier / notional",
        },
        "baseline": base.to_dict(),
        "selected_by_validation_selector": df.iloc[0].to_dict(),
        "strict_beats_v18_count": int(len(strict_v18)),
        "top15": df.head(15).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "full_replay_borrowed_risk_grid.csv"),
            "strict": str(OUT_DIR / "strict_beats_v18_grid.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": str(df.iloc[0]["variant"]), "strict_beats_v18_count": int(len(strict_v18))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
