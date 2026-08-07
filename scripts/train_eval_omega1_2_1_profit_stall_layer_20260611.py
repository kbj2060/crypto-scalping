#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_profit_stall_layer_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class StallConfig:
    name: str
    min_hold: int
    min_unreal: float
    min_tp_progress: float
    action: str
    floor_frac: float = 0.0
    tp_pull_frac: float = 1.0
    tp_gap: float = 0.0
    partial_frac: float = 0.0


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


def _metric_with_hold(
    cash: float,
    equity_curve: list[float],
    trades: list[float],
    reasons: dict[str, int],
    long_entries: int,
    short_entries: int,
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    out = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    if ledger.empty:
        out.update({"avg_hold_bars": 0.0, "median_hold_bars": 0.0, "max_hold_bars": 0, "profit_stall_actions": 0})
        return out
    hold = pd.to_numeric(ledger["exit_i"], errors="raise") - pd.to_numeric(ledger["entry_i"], errors="raise")
    out.update(
        {
            "avg_hold_bars": float(hold.mean()),
            "median_hold_bars": float(hold.median()),
            "max_hold_bars": int(hold.max()),
            "profit_stall_actions": int(pd.to_numeric(ledger.get("profit_stall_actions", 0), errors="coerce").fillna(0).sum()),
        }
    )
    return out


def _close_trade(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    cash: float,
    pos: base.Position,
    i: int,
    fee_eff: float,
    slip_eff: float,
    reason: str,
    profit_stall_actions: int,
) -> tuple[float, base.Position, float, dict[str, Any]]:
    close_pos = base.Position(**pos.__dict__)
    cash, new_pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
    net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
    row = runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, 0)
    row["profit_stall_actions"] = int(profit_stall_actions)
    row["hold_bars"] = int(i) - int(close_pos.entry_i)
    row["exit_reason"] = reason
    return cash, new_pos, net_pct, row


def _should_stall(pos: base.Position, i: int, unreal: float, cfg: StallConfig) -> bool:
    if getattr(pos, "tightened", 0) != 0:
        return False
    if unreal < float(cfg.min_unreal):
        return False
    if int(i) - int(pos.entry_i) < int(cfg.min_hold):
        return False
    if float(pos.take_profit) <= 0.0:
        return False
    return bool(unreal / max(float(pos.take_profit), 1e-8) >= float(cfg.min_tp_progress))


def _apply_stall_action(
    cash: float,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    i: int,
    unreal: float,
    cfg: StallConfig,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, base.Position, str]:
    out = base.Position(**pos.__dict__)
    if cfg.action == "floor":
        out.floor_unreal = max(float(out.floor_unreal), float(out.mfe) * float(cfg.floor_frac), float(cfg.min_unreal) * 0.5)
        out.tightened = 1
        return cash, out, "profit_stall_floor"
    if cfg.action == "tp_pull":
        pulled = float(out.take_profit) * float(cfg.tp_pull_frac)
        min_viable = float(unreal) + float(cfg.tp_gap)
        out.take_profit = max(min(pulled, float(out.take_profit)), min_viable)
        out.tightened = 1
        return cash, out, "profit_stall_tp_pull"
    if cfg.action == "partial":
        if out.reduced == 0 and out.notional > 0.10:
            cash, out, _ = base._close_fraction(cash, arrays, out, i, float(cfg.partial_frac), fee_eff, slip_eff)
            out.reduced = 1
            out.floor_unreal = max(float(out.floor_unreal), float(unreal) * 0.35)
            out.tightened = 1
            return cash, out, "profit_stall_partial"
    if cfg.action == "close":
        cash, out, _ = base._close_fraction(cash, arrays, out, i, 1.0, fee_eff, slip_eff)
        return cash, out, "profit_stall_close"
    return cash, out, "hold"


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    cfg: StallConfig | None,
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
    profit_stall_actions = 0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))

            reason = base._hit_reason(unreal, pos)
            if not reason and cfg is not None and _should_stall(pos, i, unreal, cfg):
                if cfg.action == "close":
                    cash, pos, net_pct, row = _close_trade(
                        frame,
                        arrays,
                        cash,
                        pos,
                        i,
                        fee_eff,
                        slip_eff,
                        "profit_stall_close",
                        profit_stall_actions + 1,
                    )
                    trades.append(net_pct)
                    reasons["profit_stall_close"] = reasons.get("profit_stall_close", 0) + 1
                    rows.append(row)
                    profit_stall_actions = 0
                    continue
                cash, pos, action_name = _apply_stall_action(cash, arrays, pos, i, unreal, cfg, fee_eff, slip_eff)
                if action_name != "hold":
                    profit_stall_actions += 1

            reason = base._hit_reason(base._unreal(arrays, pos, i, slip_eff), pos) if pos.side != 0 else ""
            if reason:
                cash, pos, net_pct, row = _close_trade(frame, arrays, cash, pos, i, fee_eff, slip_eff, reason, profit_stall_actions)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(row)
                profit_stall_actions = 0
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            profit_stall_actions = 0

    if pos.side != 0:
        cash, pos, net_pct, row = _close_trade(
            frame,
            arrays,
            cash,
            pos,
            len(frame) - 1,
            fee_eff,
            slip_eff,
            "forced_end",
            profit_stall_actions,
        )
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(row)

    ledger = pd.DataFrame(rows)
    return _metric_with_hold(cash, equity_curve, trades, reasons, long_entries, short_entries, ledger), ledger


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
        f"{prefix}_stall_actions": int(metrics["profit_stall_actions"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _configs() -> list[StallConfig | None]:
    configs: list[StallConfig | None] = [None]
    for hold in (72, 96, 144, 192, 288):
        for prog in (0.25, 0.35, 0.50, 0.65):
            for min_unreal in (0.006, 0.010, 0.015):
                configs.append(StallConfig(f"floor_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_f35", hold, min_unreal, prog, "floor", floor_frac=0.35))
                configs.append(StallConfig(f"floor_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_f55", hold, min_unreal, prog, "floor", floor_frac=0.55))
                configs.append(StallConfig(f"tppull_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_r70", hold, min_unreal, prog, "tp_pull", tp_pull_frac=0.70, tp_gap=0.003))
                configs.append(StallConfig(f"partial_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_50", hold, min_unreal, prog, "partial", partial_frac=0.50))
    for hold in (384, 512, 768, 1024):
        for prog in (0.15, 0.25, 0.35, 0.50):
            for min_unreal in (0.006, 0.010, 0.015, 0.020):
                configs.append(StallConfig(f"late_floor_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_f20", hold, min_unreal, prog, "floor", floor_frac=0.20))
                configs.append(StallConfig(f"late_floor_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_f35", hold, min_unreal, prog, "floor", floor_frac=0.35))
                configs.append(StallConfig(f"late_tppull_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_r85", hold, min_unreal, prog, "tp_pull", tp_pull_frac=0.85, tp_gap=0.003))
                configs.append(StallConfig(f"late_partial_h{hold}_u{min_unreal:.3f}_p{prog:.2f}_25", hold, min_unreal, prog, "partial", partial_frac=0.25))
                configs.append(StallConfig(f"late_close_h{hold}_u{min_unreal:.3f}_p{prog:.2f}", hold, min_unreal, prog, "close"))
    return configs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers: dict[int, dict[str, pd.DataFrame]] = {}

    for idx, cfg in enumerate(_configs()):
        variant = "baseline_no_profit_stall" if cfg is None else cfg.name
        row: dict[str, Any] = {
            "variant_id": int(idx),
            "variant": variant,
            "action": "baseline" if cfg is None else cfg.action,
            "min_hold": 0 if cfg is None else int(cfg.min_hold),
            "min_unreal": 0.0 if cfg is None else float(cfg.min_unreal),
            "min_tp_progress": 0.0 if cfg is None else float(cfg.min_tp_progress),
        }
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate(
                payload["frame"],
                payload["dec"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                cfg=cfg,
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
        ledgers[idx] = split_ledgers
        rows.append(row)

    ranking = pd.DataFrame(rows)
    base_row = ranking.loc[ranking["variant"].eq("baseline_no_profit_stall")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(base_row["validation_pnl"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(base_row["oos_max_hold"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["score"] = (
        ranking["oos_pnl"]
        + 0.35 * ranking["validation_pnl"]
        + 0.25 * ranking["oos_mdd"]
        - 0.025 * ranking["oos_avg_hold"]
        - 0.010 * ranking["oos_max_hold"]
    )
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd", "oos_max_hold"], ascending=[False, False, False, True]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "profit_stall_ranking.csv", index=False)

    promotable = ranking[
        (ranking["variant"] != "baseline_no_profit_stall")
        & (ranking["oos_pnl"] >= float(base_row["oos_pnl"]))
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) - 1.0)
        & (ranking["validation_pnl"] >= float(base_row["validation_pnl"]) * 0.90)
        & (ranking["oos_stall_actions"] > 0)
        & ((ranking["oos_max_hold"] < int(base_row["oos_max_hold"])) | (ranking["oos_avg_hold"] < float(base_row["oos_avg_hold"])))
    ].copy()
    promotable.to_csv(OUT_DIR / "profit_stall_promotable.csv", index=False)

    save_ids = sorted(set([int(base_row["variant_id"])] + [int(x) for x in ranking["variant_id"].head(10).tolist()] + [int(x) for x in promotable["variant_id"].head(10).tolist()]))
    for variant_id in save_ids:
        for split, ledger in ledgers[variant_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Profit-stall layer for profitable positions that remain open too long before TP. Live bot is not modified.",
        "baseline": base_row.to_dict(),
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "profit_stall_ranking.csv"),
            "promotable": str(OUT_DIR / "profit_stall_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
