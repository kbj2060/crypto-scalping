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


MODEL_ID = "omega1_2_1_side_compression_sweep_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, profile: str) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, 0)
    row["risk_profile"] = profile
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, ledger: pd.DataFrame) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0})
        return out
    hold = pd.to_numeric(ledger["hold_bars"], errors="raise")
    out.update({"avg_hold_bars": float(hold.mean()), "median_hold_bars": float(hold.median()), "max_hold_bars": int(hold.max())})
    return out


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    long_tp_scale: float,
    long_sl_scale: float,
    short_tp_scale: float,
    short_sl_scale: float,
    profile: str,
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
    long_entries = 0
    short_entries = 0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + base._unreal(arrays, pos, i, slip_eff)))
            reason = base._hit_reason(base._unreal(arrays, pos, i, slip_eff), pos)
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, profile))
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            if pos.side > 0:
                pos.take_profit = max(float(pos.take_profit) * float(long_tp_scale), 1e-8)
                pos.stop_loss = max(float(pos.stop_loss) * float(long_sl_scale), 1e-8)
            else:
                pos.take_profit = max(float(pos.take_profit) * float(short_tp_scale), 1e-8)
                pos.stop_loss = max(float(pos.stop_loss) * float(short_sl_scale), 1e-8)
            pos.floor_unreal = -abs(float(pos.stop_loss))

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", profile))

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
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    configs: list[dict[str, Any]] = [
        {
            "long_tp_scale": 1.0,
            "long_sl_scale": 1.0,
            "short_tp_scale": 1.0,
            "short_sl_scale": 1.0,
            "profile": "baseline",
        }
    ]
    for long_tp in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        for long_sl in (0.90, 1.00, 1.10):
            configs.append(
                {
                    "long_tp_scale": float(long_tp),
                    "long_sl_scale": float(long_sl),
                    "short_tp_scale": 1.0,
                    "short_sl_scale": 1.0,
                    "profile": f"long_tp{long_tp:.2f}_sl{long_sl:.2f}_short_base",
                }
            )
    for short_tp in (0.85, 0.95, 1.05):
        for short_sl in (0.90, 1.00, 1.10):
            configs.append(
                {
                    "long_tp_scale": 1.0,
                    "long_sl_scale": 1.0,
                    "short_tp_scale": float(short_tp),
                    "short_sl_scale": float(short_sl),
                    "profile": f"long_base_short_tp{short_tp:.2f}_sl{short_sl:.2f}",
                }
            )

    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[tuple[str, int], pd.DataFrame] = {}
    for candidate_id, cfg in enumerate(configs):
        row: dict[str, Any] = {"candidate_id": int(candidate_id), **cfg}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate(
                payload["frame"],
                payload["dec"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                long_tp_scale=float(cfg["long_tp_scale"]),
                long_sl_scale=float(cfg["long_sl_scale"]),
                short_tp_scale=float(cfg["short_tp_scale"]),
                short_sl_scale=float(cfg["short_sl_scale"]),
                profile=str(cfg["profile"]),
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
        rows.append(row)
        if candidate_id == 0:
            ledgers_by_id[("validation", candidate_id)] = split_ledgers["validation"]
            ledgers_by_id[("oos", candidate_id)] = split_ledgers["oos"]

    ranking = pd.DataFrame(rows)
    baseline = ranking.loc[ranking["profile"].eq("baseline")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(baseline["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(baseline["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(baseline["oos_max_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.55 * ranking["validation_pnl"] + 0.30 * ranking["oos_mdd"] + 0.35 * ranking["delta_oos_trades"] - 0.018 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "side_compression_ranking.csv", index=False)

    promotable = ranking[
        (ranking["profile"] != "baseline")
        & (ranking["oos_pnl"] >= float(baseline["oos_pnl"]) * 0.95)
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]) * 0.90)
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) - 2.0)
        & (ranking["oos_trades"] >= int(baseline["oos_trades"]))
        & (ranking["oos_avg_hold"] < float(baseline["oos_avg_hold"]))
    ].copy()
    promotable.to_csv(OUT_DIR / "side_compression_promotable.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["candidate_id"].head(20).tolist()] + [int(x) for x in promotable["candidate_id"].head(20).tolist()]))
    cfg_by_id = {int(x["candidate_id"]): x for x in rows}
    for sid in save_ids:
        cfg = cfg_by_id[sid]
        for split in ("validation", "oos"):
            if (split, sid) in ledgers_by_id:
                ledger = ledgers_by_id[(split, sid)]
            else:
                payload = data[split]
                _metrics, ledger = _simulate(
                    payload["frame"],
                    payload["dec"],
                    fee=float(payload["fee"]),
                    slip=float(payload["slip"]),
                    cost_mult=3.0,
                    long_tp_scale=float(cfg["long_tp_scale"]),
                    long_sl_scale=float(cfg["long_sl_scale"]),
                    short_tp_scale=float(cfg["short_tp_scale"]),
                    short_sl_scale=float(cfg["short_sl_scale"]),
                    profile=str(cfg["profile"]),
                )
            ledger.to_csv(OUT_DIR / f"{split}_candidate{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Side-specific TP/SL compression to target OOS long-hold without globally damaging short runner behavior.",
        "baseline": baseline.to_dict(),
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "side_compression_ranking.csv"),
            "promotable": str(OUT_DIR / "side_compression_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records"), "promotable": promotable.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
