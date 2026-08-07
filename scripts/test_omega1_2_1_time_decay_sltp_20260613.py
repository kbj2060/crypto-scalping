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


MODEL_ID = "omega1_2_1_time_decay_sltp_20260613"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


@dataclass(frozen=True)
class DecaySpec:
    name: str
    # Each tuple is (min_hold_bars, tp_multiplier, sl_multiplier).
    schedule: tuple[tuple[int, float, float], ...]
    runner: bool = True
    breakeven_after_bars: int = 0
    breakeven_min_mfe: float = 0.0
    breakeven_floor: float = 0.0


SPECS = (
    DecaySpec("baseline_wide_runner", ((0, 1.0, 1.0),), True),
    DecaySpec(
        "decay_1h_mild_runner",
        (
            (0, 1.0, 1.0),
            (12, 0.85, 0.85),
            (24, 0.70, 0.70),
            (48, 0.55, 0.55),
            (96, 0.40, 0.40),
        ),
        True,
    ),
    DecaySpec(
        "decay_1h_fast_runner",
        (
            (0, 1.0, 1.0),
            (12, 0.75, 0.75),
            (24, 0.55, 0.55),
            (48, 0.38, 0.38),
            (96, 0.25, 0.25),
        ),
        True,
    ),
    DecaySpec(
        "decay_1h_mild_no_runner",
        (
            (0, 1.0, 1.0),
            (12, 0.85, 0.85),
            (24, 0.70, 0.70),
            (48, 0.55, 0.55),
            (96, 0.40, 0.40),
        ),
        False,
    ),
    DecaySpec(
        "decay_2h_breakeven_runner",
        (
            (0, 1.0, 1.0),
            (24, 0.82, 0.82),
            (48, 0.62, 0.62),
            (96, 0.42, 0.42),
        ),
        True,
        breakeven_after_bars=48,
        breakeven_min_mfe=0.004,
        breakeven_floor=0.0005,
    ),
    DecaySpec(
        "decay_4h_profit_lock_runner",
        (
            (0, 1.0, 1.0),
            (48, 0.75, 0.75),
            (96, 0.55, 0.55),
            (144, 0.38, 0.38),
        ),
        True,
        breakeven_after_bars=96,
        breakeven_min_mfe=0.006,
        breakeven_floor=0.0010,
    ),
)


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


def _decay_mult(spec: DecaySpec, hold_bars: int) -> tuple[float, float]:
    tp_mult, sl_mult = 1.0, 1.0
    for min_bars, next_tp, next_sl in spec.schedule:
        if int(hold_bars) >= int(min_bars):
            tp_mult, sl_mult = float(next_tp), float(next_sl)
    return tp_mult, sl_mult


def _effective_thresholds(pos: base.Position, entry_tp: float, entry_sl: float, tp_mult: float, sl_mult: float) -> tuple[float, float]:
    # With multiplier 1.0, preserve the live TP-runner semantics exactly.
    # Only elapsed-time compression is allowed to cap an extended TP/SL.
    tp_now = float(pos.take_profit)
    sl_now = abs(float(pos.stop_loss))
    if tp_mult < 0.999999 and entry_tp > 0.0:
        tp_now = min(tp_now, float(entry_tp) * float(tp_mult))
    if sl_mult < 0.999999 and entry_sl > 0.0:
        sl_now = min(sl_now, abs(float(entry_sl)) * float(sl_mult))
    return tp_now, sl_now


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


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, extensions: int, spec: DecaySpec) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, extensions)
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    row["time_decay_spec"] = spec.name
    return row


def _simulate(payload: dict[str, Any], *, spec: DecaySpec, tp_bundle: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    template = meta.RunnerTemplate(**tp_bundle["template"])
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    holds: list[int] = []
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    pos = base.Position()
    extensions = 0
    long_entries = 0
    short_entries = 0
    entry_tp = 0.0
    entry_sl = 0.0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))

            hold_bars = max(int(i) - int(pos.entry_i), 0)
            tp_mult, sl_mult = _decay_mult(spec, hold_bars)
            effective_tp, effective_sl = _effective_thresholds(pos, entry_tp, entry_sl, tp_mult, sl_mult)
            dynamic_floor = float(pos.floor_unreal)
            if spec.breakeven_after_bars and hold_bars >= int(spec.breakeven_after_bars) and float(pos.mfe) >= float(spec.breakeven_min_mfe):
                dynamic_floor = max(dynamic_floor, float(spec.breakeven_floor))

            reason = ""
            if effective_tp > 0.0 and unreal >= effective_tp:
                if spec.runner and extensions < int(template.max_extensions) and _runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "time_decay_take_profit" if effective_tp < float(pos.take_profit) else "take_profit"
            elif dynamic_floor > -effective_sl and unreal <= dynamic_floor:
                reason = "time_decay_profit_lock_exit" if dynamic_floor > float(pos.floor_unreal) else "meta_runner_profit_lock_exit"
            elif effective_sl > 0.0 and unreal <= -effective_sl:
                reason = "time_decay_stop_loss" if effective_sl < abs(float(pos.stop_loss)) else "stop_loss"

            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                holds.append(hold_bars)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions, spec))
                extensions = 0
                entry_tp = 0.0
                entry_sl = 0.0
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            extensions = 0
            entry_tp = float(pos.take_profit)
            entry_sl = abs(float(pos.stop_loss))

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        hold_bars = max(len(frame) - 1 - int(close_pos.entry_i), 0)
        holds.append(hold_bars)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions, spec))

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, holds), pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    tp_bundle = joblib.load(TP_BUNDLE_PATH)
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in SPECS:
        out: dict[str, Any] = {
            "candidate": spec.name,
            "schedule": list(spec.schedule),
            "runner": bool(spec.runner),
            "breakeven_after_bars": int(spec.breakeven_after_bars),
            "breakeven_min_mfe": float(spec.breakeven_min_mfe),
            "breakeven_floor": float(spec.breakeven_floor),
        }
        ledgers[spec.name] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate(data[split], spec=spec, tp_bundle=tp_bundle)
            out.update(_row(split, metrics))
            ledgers[spec.name][split] = ledger
        rows.append(out)
        print(json.dumps({"done": spec.name, "oos_pnl": out["oos_pnl"], "oos_mdd": out["oos_mdd"], "oos_wr": out["oos_wr"], "oos_trades": out["oos_trades"], "oos_avg_hold": out["oos_avg_hold"]}, ensure_ascii=False), flush=True)

    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["candidate"].eq("baseline_wide_runner")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.35 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.15 * ranking["validation_mdd"] - 0.02 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "score"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "time_decay_sltp_ranking.csv", index=False)
    for candidate in ranking["candidate"].astype(str).tolist():
        for split, ledger in ledgers[candidate].items():
            ledger.to_csv(OUT_DIR / f"{split}_{candidate}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Time-decay TP/SL test. Entry signal is unchanged; open-position TP/SL thresholds are narrowed by elapsed hold time.",
        "order_semantics": {
            "take_profit": "Backtest treats TP as a threshold. Live equivalent is reduce-only limit re-priced toward market over time.",
            "stop_loss": "Backtest treats SL as a threshold. Live equivalent should be stop-market or monitored market close, not a passive limit.",
        },
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "time_decay_sltp_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
