#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/omega1_layer12_action_model_family_compare_20260531_fast"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_zigzag_swing_execution_20260531"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _read_decisions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "p_long", "p_short", "p_cash", "confidence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required decision columns: {missing}")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _join(frame: pd.DataFrame, dec: pd.DataFrame, tag: str) -> pd.DataFrame:
    before = len(dec)
    out = dec.merge(frame[["timestamp", "open", "high", "low", "close"]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag} row count changed on join: {before} -> {len(out)}")
    missing = out[["open", "high", "low", "close"]].isna().any(axis=1)
    if missing.any():
        raise RuntimeError(f"{tag} decision timestamps missing from frame: {int(missing.sum())}")
    return out.reset_index(drop=True)


def _days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    return float(max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0))


def _signals(dec: pd.DataFrame, entry_threshold: float, side_margin: float) -> np.ndarray:
    lp = pd.to_numeric(dec["p_long"], errors="raise").to_numpy(dtype=np.float64)
    sp = pd.to_numeric(dec["p_short"], errors="raise").to_numpy(dtype=np.float64)
    long_ok = (lp >= float(entry_threshold)) & ((lp - sp) >= float(side_margin))
    short_ok = (sp >= float(entry_threshold)) & ((sp - lp) >= float(side_margin))
    return np.where(long_ok, 1, np.where(short_ok, -1, 0)).astype(np.int8)


def _bt_swing(
    frame: pd.DataFrame,
    *,
    entry_threshold: float,
    side_margin: float,
    cash_exit_bars: int,
    trail_arm: float,
    trail_gap: float,
    hard_sl: float,
    max_hold_bars: int,
    fee: float,
    slip: float,
    exposure: float,
    allow_flip: bool,
) -> dict[str, Any]:
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    sig = _signals(frame, entry_threshold, side_margin)
    cash = 1.0
    peak_eq = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    hold = 0
    cash_count = 0
    best_px = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    exits: dict[str, int] = {}

    def mark(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * float(exposure))

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, hold, cash_count, best_px, cash, long_entries, short_entries
        fill_i = min(i + 1, len(open_px) - 1)
        side = int(new_side)
        entry = open_px[fill_i] * (1.0 + float(slip) if side > 0 else 1.0 - float(slip))
        entry_equity = cash
        cash -= cash * float(fee) * float(exposure)
        hold = 0
        cash_count = 0
        best_px = entry
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, entry_equity, hold, cash_count, best_px, cash, trades, wins
        if side == 0:
            return
        if fill_px is None:
            fill_i = min(i + 1, len(open_px) - 1)
            fill_px = open_px[fill_i] * (1.0 - float(slip) if side > 0 else 1.0 + float(slip))
        before_fee = cash
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        cash = cash * (1.0 + raw * float(exposure))
        cash -= before_fee * float(fee) * float(exposure)
        pnl = cash / max(entry_equity, 1e-12) - 1.0
        wins += int(pnl > 0.0)
        trades += 1
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        cash_count = 0
        best_px = 0.0

    for i in range(len(frame) - 2):
        desired = int(sig[i])
        if side != 0:
            hold += 1
            if desired == 0:
                cash_count += 1
            else:
                cash_count = 0

            if side > 0:
                best_px = max(best_px, float(high[i]))
                hard_px = entry * (1.0 - float(hard_sl))
                if float(low[i]) <= hard_px:
                    exit_pos(i, "hard_sl", hard_px * (1.0 - float(slip)))
                else:
                    runup = best_px / max(entry, 1e-12) - 1.0
                    trail_px = best_px * (1.0 - float(trail_gap))
                    if runup >= float(trail_arm) and float(low[i]) <= trail_px:
                        exit_pos(i, "trailing", trail_px * (1.0 - float(slip)))
            else:
                best_px = min(best_px, float(low[i]))
                hard_px = entry * (1.0 + float(hard_sl))
                if float(high[i]) >= hard_px:
                    exit_pos(i, "hard_sl", hard_px * (1.0 + float(slip)))
                else:
                    runup = entry / max(best_px, 1e-12) - 1.0
                    trail_px = best_px * (1.0 + float(trail_gap))
                    if runup >= float(trail_arm) and float(high[i]) >= trail_px:
                        exit_pos(i, "trailing", trail_px * (1.0 + float(slip)))

            if side != 0:
                if desired == -side and desired != 0:
                    exit_pos(i, "opposite")
                    if allow_flip:
                        enter(i, desired)
                elif cash_count >= int(cash_exit_bars):
                    exit_pos(i, "cash_decay")
                elif hold >= int(max_hold_bars):
                    exit_pos(i, "max_hold")

        if side == 0 and desired != 0:
            enter(i, desired)

        eq = mark(i)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, eq / max(peak_eq, 1e-12) - 1.0)

    if side != 0:
        exit_pos(len(frame) - 2, "end")

    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
        "config": {
            "entry_threshold": float(entry_threshold),
            "side_margin": float(side_margin),
            "cash_exit_bars": int(cash_exit_bars),
            "trail_arm": float(trail_arm),
            "trail_gap": float(trail_gap),
            "hard_sl": float(hard_sl),
            "max_hold_bars": int(max_hold_bars),
            "allow_flip": bool(allow_flip),
        },
    }


def _costs(frame: pd.DataFrame, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for mult in (1, 2, 3):
        out[f"cost{mult}"] = _bt_swing(
            frame,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
            exposure=float(args.exposure),
            **cfg,
        )
    return out


def _calmar(cost: dict[str, Any]) -> float:
    c3 = cost["cost3"]
    if int(c3["trades"]) < 20:
        return -1e9
    return float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))


def main() -> int:
    p = argparse.ArgumentParser(description="Test ZigZag-aligned swing execution on Omega1 action/confidence outputs.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--slip", type=float, default=0.00015)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--fast", action="store_true", help="Run a compact first-pass grid.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f25 = _read_frame(args.split_dir / "training_features_2025.csv")
    f26 = _read_frame(args.split_dir / "training_features_2026_rebuilt.csv")
    dval = _read_decisions(args.model_dir / "validation_decisions.csv")
    doos = _read_decisions(args.model_dir / "oos_2026_decisions.csv")
    val = _join(f25, dval, "validation")
    oos = _join(f26, doos, "oos_2026")

    if args.fast:
        entry_thresholds = [0.56, 0.60, 0.64]
        side_margins = [0.00, 0.04]
        cash_exit_bars_grid = [3, 6]
        trail_arms = [0.010, 0.015]
        trail_gaps = [0.006, 0.010]
        hard_sls = [0.018, 0.024]
        max_hold_bars_grid = [96, 144]
        allow_flips = [False]
    else:
        entry_thresholds = [0.52, 0.56, 0.60, 0.64, 0.68]
        side_margins = [0.00, 0.04, 0.08]
        cash_exit_bars_grid = [3, 6, 12]
        trail_arms = [0.010, 0.015, 0.020]
        trail_gaps = [0.005, 0.008, 0.012]
        hard_sls = [0.012, 0.018, 0.024]
        max_hold_bars_grid = [96, 144, 288]
        allow_flips = [False, True]

    grid = []
    for entry_threshold, side_margin, cash_exit_bars, trail_arm, trail_gap, hard_sl, max_hold_bars, allow_flip in product(
        entry_thresholds,
        side_margins,
        cash_exit_bars_grid,
        trail_arms,
        trail_gaps,
        hard_sls,
        max_hold_bars_grid,
        allow_flips,
    ):
        cfg = {
            "entry_threshold": entry_threshold,
            "side_margin": side_margin,
            "cash_exit_bars": cash_exit_bars,
            "trail_arm": trail_arm,
            "trail_gap": trail_gap,
            "hard_sl": hard_sl,
            "max_hold_bars": max_hold_bars,
            "allow_flip": allow_flip,
        }
        cost = _costs(val, cfg, args)
        score = _calmar(cost)
        grid.append({"score": score, "validation": cost, "config": cfg})
    grid.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = grid[0]
    oos_cost = _costs(oos, selected["config"], args)
    summary = {
        "model_id": "omega1_zigzag_swing_execution_20260531",
        "design": "Causal swing execution for ZigZag action models: probability entry, model-side hold, opposite/cash exits, trailing stop, wide hard stop.",
        "leakage_note": "Does not use future ZigZag labels for exits; uses only model probabilities and current OHLC path during backtest.",
        "selection": "config selected on validation Cost3 Calmar, OOS 2026 evaluated once",
        "selected_config": selected["config"],
        "validation": selected["validation"],
        "oos_2026": oos_cost,
        "top10": grid[:10],
        "artifacts": {
            "out_dir": str(args.out_dir),
            "summary": str(args.out_dir / "summary.json"),
            "source_decisions": str(args.model_dir),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    pd.DataFrame(
        [
            {
                **r["config"],
                "score": r["score"],
                "val_cost1_pnl": r["validation"]["cost1"]["pnl"],
                "val_cost2_pnl": r["validation"]["cost2"]["pnl"],
                "val_cost3_pnl": r["validation"]["cost3"]["pnl"],
                "val_cost3_mdd": r["validation"]["cost3"]["mdd"],
                "val_cost3_trades": r["validation"]["cost3"]["trades"],
                "val_cost3_wr": r["validation"]["cost3"]["wr"],
            }
            for r in grid
        ]
    ).to_csv(args.out_dir / "grid.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
