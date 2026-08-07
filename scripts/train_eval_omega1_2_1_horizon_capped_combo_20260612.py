#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_capped_runner_router_20260612 as cap  # noqa: E402
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402
import train_eval_omega1_2_1_horizon_router_sweep_20260611 as sweep  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_horizon_capped_combo_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ROT_SPEC = hr.RotationSpec("rot_tp065_sl080_floor50_35", 0.65, 0.80, 0.50, 0.35)
CAP_SPEC = cap.CappedSpec("cap1536_profit2", 1536, 0.02)


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


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, route: str) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, 0)
    row["combo_route"] = route
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, ledger: pd.DataFrame) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0, "route_counts": {}})
        return out
    hold = pd.to_numeric(ledger["hold_bars"], errors="raise")
    out.update({"avg_hold_bars": float(hold.mean()), "median_hold_bars": float(hold.median()), "max_hold_bars": int(hold.max()), "route_counts": ledger["combo_route"].astype(str).value_counts().to_dict()})
    return out


def _simulate_combo(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    rot_model: Any,
    rot_cols: list[str],
    rot_proba: float,
    cap_model: Any,
    cap_cols: list[str],
    cap_proba: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    route = "runner"
    long_entries = 0
    short_entries = 0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            if route == "rotation":
                pos = hr._maybe_rotation_floor(pos, unreal, ROT_SPEC)
            equity_curve.append(cash * (1.0 + base._unreal(arrays, pos, i, slip_eff)))
            reason = base._hit_reason(base._unreal(arrays, pos, i, slip_eff), pos)
            if not reason and route == "capped" and int(i) - int(pos.entry_i) >= CAP_SPEC.cap_bars and unreal >= CAP_SPEC.min_unreal:
                reason = "capped_runner_exit"
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, route))
                route = "runner"
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        if hr._rotation_proba(rot_model, rot_cols, frame, state, dec, int(i)) >= float(rot_proba):
            route = "rotation"
        elif cap._proba(cap_model, cap_cols, frame, state, dec, int(i)) >= float(cap_proba):
            route = "capped"
        else:
            route = "runner"
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, int(i), fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            if route == "rotation":
                pos = hr._apply_rotation(pos, ROT_SPEC)
        else:
            route = "runner"

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", route))

    ledger = pd.DataFrame(rows)
    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, ledger), ledger


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_hold": float(metrics["avg_hold_bars"]),
        f"{prefix}_median_hold": float(metrics["median_hold_bars"]),
        f"{prefix}_max_hold": int(metrics["max_hold_bars"]),
        f"{prefix}_route_counts": metrics["route_counts"],
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()

    rot_labels = sweep._relabel(
        hr._build_counterfactual_labels(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            spec=ROT_SPEC,
            edge=0.0,
        ),
        edge=0.50,
        hold_penalty=0.0005,
    )
    rot_model, rot_cols, rot_diag = hr._fit_router(rot_labels, kind="hgb", seed=260611)
    cap_labels = cap._build_labels(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        spec=CAP_SPEC,
        edge=0.0,
        hold_penalty=0.0005,
    )
    cap_model, cap_cols, cap_diag = cap._fit(cap_labels, kind="et", seed=260611)
    if rot_model is None or cap_model is None:
        raise RuntimeError(f"router fit failed rot={rot_diag} cap={cap_diag}")

    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    variant_id = 0
    for rot_proba in (0.65, 0.675):
        for cap_proba in (0.45, 0.55, 0.65, 0.75):
            row: dict[str, Any] = {"variant_id": int(variant_id), "variant": f"horizon_rot{str(rot_proba).replace('.','')}_cap{str(cap_proba).replace('.','')}", "rot_proba": float(rot_proba), "cap_proba": float(cap_proba)}
            split_ledgers: dict[str, pd.DataFrame] = {}
            for split in ("validation", "oos"):
                payload = data[split]
                metrics, ledger = _simulate_combo(
                    payload["frame"],
                    payload["dec"],
                    payload["state"],
                    fee=float(payload["fee"]),
                    slip=float(payload["slip"]),
                    cost_mult=3.0,
                    rot_model=rot_model,
                    rot_cols=rot_cols,
                    rot_proba=float(rot_proba),
                    cap_model=cap_model,
                    cap_cols=cap_cols,
                    cap_proba=float(cap_proba),
                )
                row.update(_row(split, metrics))
                split_ledgers[split] = ledger
            rows.append(row)
            ledgers_by_id[variant_id] = split_ledgers
            variant_id += 1

    ranking = pd.DataFrame(rows)
    ranking["score"] = ranking["oos_pnl"] + 0.55 * ranking["validation_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["oos_trades"] - 0.020 * ranking["oos_avg_hold"] - 0.008 * ranking["oos_max_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_capped_combo_ranking.csv", index=False)
    for sid in sorted(set(int(x) for x in ranking["variant_id"].head(8).tolist())):
        for split, ledger in ledgers_by_id[sid].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Combine current best horizon-router with capped-runner router to reduce extreme holds.",
        "rot_diag": rot_diag,
        "cap_diag": cap_diag,
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_capped_combo_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(8).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
