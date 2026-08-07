#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_stall_release_exit_overlays_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


@dataclass(frozen=True)
class OverlayCfg:
    name: str
    stall_hold: int = 72
    stall_abs_sl_frac: float = 0.35
    stall_mfe_tp_frac: float = 0.40
    reduce50: bool = False
    full_exit: bool = False
    compress_barriers: bool = False
    opposite_exit: bool = False
    sideways_entry_cap: bool = False


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


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, holds: list[int]) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    h = np.asarray(holds, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_hold_bars": float(np.mean(h)) if len(h) else 0.0,
        "median_hold_bars": float(np.median(h)) if len(h) else 0.0,
        "max_hold_bars": int(np.max(h)) if len(h) else 0,
        "exit_reasons": dict(reasons),
    }


def _row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_long": int(m["long_entries"]),
        f"{prefix}_short": int(m["short_entries"]),
        f"{prefix}_avg_hold": float(m["avg_hold_bars"]),
        f"{prefix}_median_hold": float(m["median_hold_bars"]),
        f"{prefix}_max_hold": int(m["max_hold_bars"]),
        f"{prefix}_reasons": m["exit_reasons"],
    }


def _sideways_row(state: pd.DataFrame, i: int) -> bool:
    row = state.iloc[int(i)]
    churn = float(row.get("regime3_churn_h6_risk_score", 0.0))
    trans = float(row.get("regime3_transition_h6_risk_prob", 0.0))
    route_margin = float(row.get("tabm_router_margin", 0.0))
    atr = float(row.get("atr14_pct", 0.0))
    return bool(churn >= 0.50 or trans >= 0.55 or (route_margin < 0.12 and atr < 0.008))


def _stalled(state: pd.DataFrame, pos: base.Position, i: int, unreal: float, cfg: OverlayCfg) -> bool:
    hold = max(int(i) - int(pos.entry_i), 0)
    if hold < int(cfg.stall_hold):
        return False
    sl = max(abs(float(pos.stop_loss)), 1e-8)
    tp = max(abs(float(pos.take_profit)), 1e-8)
    low_progress = abs(float(unreal)) <= float(cfg.stall_abs_sl_frac) * sl
    low_mfe = float(pos.mfe) <= float(cfg.stall_mfe_tp_frac) * tp
    return bool(low_progress and low_mfe and _sideways_row(state, i))


def _opposite_signal(state: pd.DataFrame, pos: base.Position, i: int) -> bool:
    hold = max(int(i) - int(pos.entry_i), 0)
    if hold < 12:
        return False
    row = state.iloc[int(i)]
    p_long = float(row.get("tabm_dir_p_long", 0.0))
    p_short = float(row.get("tabm_dir_p_short", 0.0))
    quality = float(row.get("tabm_quality_for_action", 0.0))
    if pos.side > 0:
        return bool(p_short - p_long >= 0.18 and quality >= 0.58)
    return bool(p_long - p_short >= 0.18 and quality >= 0.58)


def _runner_extend_allowed(bundle: dict[str, Any] | None, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    if not bundle:
        return False
    template = meta.RunnerTemplate(**bundle["template"])
    return meta._selector_allowed(
        bundle.get("model"),
        list(bundle.get("feature_cols", [])),
        frame,
        state,
        pos,
        int(i),
        float(unreal),
        template=template,
        proba_min=float(bundle.get("proba_min", 0.55)),
    )


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, extensions: int) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, extensions)
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    return row


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any],
    cfg: OverlayCfg,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"])
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    holds: list[int] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    extensions = 0
    long_entries = short_entries = 0
    reduced = False
    compressed = False

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""

            if cfg.opposite_exit and _opposite_signal(state, pos, i):
                reason = "opposite_signal_exit"
            elif cfg.full_exit and _stalled(state, pos, i, unreal, cfg):
                reason = "stall_full_exit"
            elif cfg.reduce50 and (not reduced) and _stalled(state, pos, i, unreal, cfg) and pos.notional > 0.20:
                cash, pos, _ = base._close_fraction(cash, arrays, pos, i, 0.50, fee_eff, slip_eff)
                reduced = True
                reasons["stall_reduce50"] = reasons.get("stall_reduce50", 0) + 1
                continue
            elif cfg.compress_barriers and (not compressed) and _stalled(state, pos, i, unreal, cfg):
                pos.take_profit = max(float(pos.take_profit) * 0.62, float(unreal) + 0.001)
                pos.stop_loss = max(abs(float(pos.stop_loss)) * 0.62, 0.001)
                pos.floor_unreal = max(float(pos.floor_unreal), -abs(float(pos.stop_loss)))
                compressed = True
                reasons["stall_barrier_compress"] = reasons.get("stall_barrier_compress", 0) + 1

            if not reason:
                if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                    if extensions < int(template.max_extensions) and _runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                        extensions += 1
                        old_tp = float(pos.take_profit)
                        pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                        pos.take_profit = old_tp * float(template.extend_mult)
                    else:
                        reason = "take_profit"
                elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                    reason = "meta_runner_profit_lock_exit"
                elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                    reason = "stop_loss"

            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                holds.append(max(int(i) - int(close_pos.entry_i), 0))
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions))
                extensions = 0
                reduced = False
                compressed = False
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        dec_use = dec
        if cfg.sideways_entry_cap and _sideways_row(state, i):
            dec_use = dec.copy()
            dec_use.loc[int(i), "take_profit"] = float(dec_use.loc[int(i), "take_profit"]) * 0.55
            dec_use.loc[int(i), "stop_loss"] = float(dec_use.loc[int(i), "stop_loss"]) * 0.65
        cash, pos, entered = base._enter(cash, arrays, dec_use, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)
            extensions = 0
            reduced = False
            compressed = False

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        holds.append(max(len(frame) - 1 - int(close_pos.entry_i), 0))
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions))

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, holds), pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    tp_bundle = joblib.load(TP_BUNDLE_PATH)
    configs = [
        OverlayCfg("baseline_tp_runner_only"),
        OverlayCfg("stall_reduce50", reduce50=True),
        OverlayCfg("stall_full_exit", full_exit=True),
        OverlayCfg("stall_barrier_compress", compress_barriers=True),
        OverlayCfg("opposite_signal_exit", opposite_exit=True),
        OverlayCfg("sideways_entry_cap", sideways_entry_cap=True),
        OverlayCfg("reduce50_plus_opposite", reduce50=True, opposite_exit=True),
        OverlayCfg("compress_plus_opposite", compress_barriers=True, opposite_exit=True),
    ]

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for cfg in configs:
        row: dict[str, Any] = {"variant": cfg.name, **cfg.__dict__}
        ledgers[cfg.name] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                tp_bundle=tp_bundle,
                cfg=cfg,
            )
            row.update(_row(split, metrics))
            ledgers[cfg.name][split] = ledger
        rows.append(row)
        print(json.dumps({"done": cfg.name, "oos_pnl": row["oos_pnl"], "oos_trades": row["oos_trades"], "oos_avg_hold": row["oos_avg_hold"]}, ensure_ascii=False), flush=True)

    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["variant"].eq("baseline_tp_runner_only")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.35 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.15 * ranking["validation_mdd"] - 0.02 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "score"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "stall_release_exit_overlay_ranking.csv", index=False)

    keep = set(ranking["variant"].head(8).astype(str).tolist())
    for name in keep:
        for split, ledger in ledgers[name].items():
            ledger.to_csv(OUT_DIR / f"{split}_{name}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Test stall/sideways release overlays without removing baseline TP/SL safety rails or TP runner.",
        "baseline": base_row.to_dict(),
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "stall_release_exit_overlay_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
