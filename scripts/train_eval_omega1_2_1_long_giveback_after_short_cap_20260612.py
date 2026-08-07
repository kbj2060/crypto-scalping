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
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_horizon_long_cap_sweep_20260612 as sidecap  # noqa: E402
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402
import train_eval_omega1_2_1_horizon_router_sweep_20260611 as sweep  # noqa: E402


MODEL_ID = "omega1_2_1_long_giveback_after_short_cap_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ROT_SPEC = sidecap.ROT_SPEC
ROT_PROBA = 0.65
SHORT_CAP_BARS = 2000
SHORT_CAP_MIN_UNREAL = 0.035


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
    row["route"] = route
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, ledger: pd.DataFrame) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0, "route_counts": {}})
        return out
    hold = pd.to_numeric(ledger["hold_bars"], errors="raise")
    out.update(
        {
            "avg_hold_bars": float(hold.mean()),
            "median_hold_bars": float(hold.median()),
            "max_hold_bars": int(hold.max()),
            "route_counts": ledger["route"].astype(str).value_counts().to_dict(),
        }
    )
    return out


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    rot_model: Any,
    rot_cols: list[str],
    long_bars: int,
    min_mfe: float,
    giveback_frac: float,
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
            if not reason and route == "runner" and int(pos.side) == -1:
                if int(i) - int(pos.entry_i) >= SHORT_CAP_BARS and unreal >= SHORT_CAP_MIN_UNREAL:
                    reason = "short_static_profit_cap_exit"
            if not reason and int(pos.side) == 1 and int(long_bars) > 0:
                hold = int(i) - int(pos.entry_i)
                giveback = max(float(pos.mfe) - float(unreal), 0.0)
                if hold >= int(long_bars) and float(pos.mfe) >= float(min_mfe) and giveback >= float(pos.mfe) * float(giveback_frac):
                    reason = "long_giveback_exit"
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
        route = "rotation" if hr._rotation_proba(rot_model, rot_cols, frame, state, dec, int(i)) >= ROT_PROBA else "runner"
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
    labels = sweep._relabel(
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
    feature_cols = hr._feature_cols(labels)
    base._reject_forbidden(feature_cols, "long_giveback_after_short_cap_router")
    rot_model, rot_cols, diag = hr._fit_router(labels, kind="hgb", seed=260611)
    if rot_model is None:
        raise RuntimeError(f"rotation router fit failed: {diag}")
    base._reject_forbidden(rot_cols, "long_giveback_after_short_cap_router_fit")

    configs: list[dict[str, Any]] = [{"variant": "short_cap_only", "long_bars": 0, "min_mfe": 0.0, "giveback_frac": 1.0}]
    for long_bars in (1024, 1536, 2048, 2400, 2800):
        for min_mfe in (0.04, 0.06, 0.08, 0.10, 0.12):
            for giveback_frac in (0.20, 0.30, 0.40, 0.50):
                configs.append(
                    {
                        "variant": f"long_gb_b{long_bars}_mfe{min_mfe:.2f}_gb{giveback_frac:.2f}",
                        "long_bars": int(long_bars),
                        "min_mfe": float(min_mfe),
                        "giveback_frac": float(giveback_frac),
                    }
                )

    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    for candidate_id, cfg in enumerate(configs):
        row: dict[str, Any] = {"candidate_id": int(candidate_id), **cfg}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                rot_model=rot_model,
                rot_cols=rot_cols,
                long_bars=int(cfg["long_bars"]),
                min_mfe=float(cfg["min_mfe"]),
                giveback_frac=float(cfg["giveback_frac"]),
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
        rows.append(row)
        ledgers_by_id[candidate_id] = split_ledgers

    ranking = pd.DataFrame(rows)
    baseline = ranking.loc[ranking["variant"].eq("short_cap_only")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(baseline["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(baseline["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(baseline["oos_max_hold"])
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "long_giveback_after_short_cap_ranking.csv", index=False)

    balanced = ranking[
        (ranking["oos_pnl"] >= float(baseline["oos_pnl"]) * 0.98)
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]) * 0.98)
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) - 1.0)
        & ((ranking["oos_avg_hold"] < float(baseline["oos_avg_hold"])) | (ranking["oos_max_hold"] < int(baseline["oos_max_hold"])))
    ].copy()
    balanced.to_csv(OUT_DIR / "long_giveback_after_short_cap_balanced.csv", index=False)

    for sid in sorted(set([0] + [int(x) for x in ranking["candidate_id"].head(12).tolist()] + [int(x) for x in balanced["candidate_id"].head(12).tolist()])):
        for split, ledger in ledgers_by_id[sid].items():
            ledger.to_csv(OUT_DIR / f"{split}_candidate{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Long giveback exit after current best short-cap horizon-router; targets OOS max hold without static long cap.",
        "forbidden_feature_audit": "pass",
        "rot_diag": diag,
        "baseline_short_cap_only": baseline.to_dict(),
        "balanced_count": int(len(balanced)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "balanced": balanced.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "long_giveback_after_short_cap_ranking.csv"),
            "balanced": str(OUT_DIR / "long_giveback_after_short_cap_balanced.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "balanced_count": int(len(balanced)), "top": ranking.head(10).to_dict(orient="records"), "balanced": balanced.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
