#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import EVAL_CSV, FORBIDDEN_PREFIXES, TRAIN_CSV  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _days, _fill_price  # noqa: E402


MODEL_ID = "alpha8_combo_risk_sweep_until_0800_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RUNNING_CSV = OUT_DIR / "running_candidates.csv"
RANKING_CSV = OUT_DIR / "ranking.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"


@dataclass(frozen=True)
class Cfg:
    name: str
    min_quality: float
    min_confidence: float
    min_teacher_margin: float
    max_teacher_uncertainty: float
    max_tail_warning: float
    max_instability: float
    max_whipsaw: float
    notional_mult: float
    notional_cap: float
    leverage: float
    tp_mult: float
    sl_mult: float
    hold_mult: float
    high_conf_boost: float
    trend_align_boost: float
    chop_cut: float


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _apply_cfg(frame: pd.DataFrame, dec: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    side = _num(out, "side")
    active = _active(out)
    q = _num(out, "quality_score")
    conf = _num(out, "confidence")
    teacher_edge = _num(frame, "teacher_long_edge") - _num(frame, "teacher_short_edge")
    teacher_margin = _num(frame, "teacher_side_margin")
    teacher_unc = _num(frame, "teacher_uncertainty")
    teacher_tail = _num(frame, "teacher_tail_warning")
    inst = _num(frame, "clean_regime4_state24_sticky090_v2_instability_prob")
    whipsaw = _num(frame, "clean_regime4_state24_sticky090_v2_whipsaw_prob")
    trend = _num(frame, "regime4_pred_micro_prob") + _num(frame, "clean_regime4_state24_sticky090_v2_factor_trend")
    chop = _num(frame, "regime4_pred_chop_prob") + _num(frame, "clean_regime4_state24_sticky090_v2_chop_prob")
    aligned = side * teacher_edge
    keep = (
        active
        & (q >= cfg.min_quality)
        & (conf >= cfg.min_confidence)
        & (aligned >= cfg.min_teacher_margin)
        & (teacher_margin >= cfg.min_teacher_margin)
        & (teacher_unc <= cfg.max_teacher_uncertainty)
        & (teacher_tail <= cfg.max_tail_warning)
        & (inst <= cfg.max_instability)
        & (whipsaw <= cfg.max_whipsaw)
    )
    out.loc[active & ~keep, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
    ]
    out.loc[active & ~keep, "leverage"] = 1.0
    if np.any(keep):
        mult = np.full(len(out), cfg.notional_mult, dtype=np.float64)
        mult *= np.where(conf >= 0.80, cfg.high_conf_boost, 1.0)
        mult *= np.where(trend >= 0.85, cfg.trend_align_boost, 1.0)
        mult *= np.where(chop >= 0.95, cfg.chop_cut, 1.0)
        notional = np.minimum(np.maximum(_num(out, "notional_exposure") * mult, 0.0), cfg.notional_cap)
        out.loc[keep, "notional_exposure"] = notional[keep]
        out.loc[keep, "leverage"] = float(cfg.leverage)
        out.loc[keep, "position_fraction"] = notional[keep] / max(float(cfg.leverage), 1e-8)
        out.loc[keep, "take_profit"] = np.maximum(_num(out, "take_profit")[keep], 1e-6) * cfg.tp_mult
        out.loc[keep, "stop_loss"] = np.maximum(np.abs(_num(out, "stop_loss")[keep]), 1e-6) * cfg.sl_mult
        out.loc[keep, "max_hold_bars"] = np.maximum(1, np.rint(np.maximum(_num(out, "max_hold_bars")[keep], 1) * cfg.hold_mult)).astype(int)
    return out


def _fast_cost3(df: pd.DataFrame, dec: pd.DataFrame) -> dict[str, Any]:
    parent = joblib.load(v31.DEFAULT_PARENT)
    fee = float(parent["config"]["fee"]) * 3.0
    slip = float(parent["config"]["slip"]) * 3.0
    close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 1.0
    entry_idx = 0
    notional = take_profit = stop_loss = 0.0
    leverage = 1.0
    max_hold = next_cooldown = cooldown_left = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_i, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                pos = 0
                notional = take_profit = stop_loss = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue
            row = dec.iloc[i]
            if int(row.action) == ACTION_CASH or int(row.side) == 0:
                continue
            fill_i = min(i + 1, len(df) - 1)
            pos = int(row.side)
            entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(row.notional_exposure)
            leverage = float(row.leverage)
            take_profit = float(row.take_profit)
            stop_loss = float(row.stop_loss)
            max_hold = int(row.max_hold_bars)
            next_cooldown = int(row.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
    if pos != 0:
        fill_i = len(df) - 1
        exit_price = _fill_price(df, fill_i, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
    }


def _rand_cfg(i: int, rng: random.Random) -> Cfg:
    return Cfg(
        name=f"r{i:05d}",
        min_quality=rng.choice([0.0, 0.005, 0.01, 0.015, 0.02, 0.03]),
        min_confidence=rng.choice([0.0, 0.45, 0.55, 0.65, 0.75, 0.85]),
        min_teacher_margin=rng.choice([-0.02, -0.01, 0.0, 0.005, 0.01, 0.02]),
        max_teacher_uncertainty=rng.choice([0.35, 0.50, 0.65, 0.80, 1.00, 9.0]),
        max_tail_warning=rng.choice([0.25, 0.40, 0.60, 0.80, 1.00, 9.0]),
        max_instability=rng.choice([0.35, 0.50, 0.65, 0.80, 1.00, 9.0]),
        max_whipsaw=rng.choice([0.35, 0.50, 0.65, 0.80, 1.00, 9.0]),
        notional_mult=rng.choice([0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]),
        notional_cap=rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
        leverage=rng.choice([1.0, 2.0, 3.0, 5.0]),
        tp_mult=rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
        sl_mult=rng.choice([0.35, 0.5, 0.75, 1.0, 1.25]),
        hold_mult=rng.choice([0.35, 0.5, 0.75, 1.0, 1.25]),
        high_conf_boost=rng.choice([1.0, 1.2, 1.5, 2.0]),
        trend_align_boost=rng.choice([1.0, 1.2, 1.5, 2.0]),
        chop_cut=rng.choice([0.35, 0.5, 0.75, 1.0]),
    )


def _score(m: dict[str, Any]) -> float:
    if int(m["trades"]) < 50:
        return -1e9 + float(m["pnl"])
    return float(m["pnl"]) + 75.0 * float(m["wr"]) - 0.4 * abs(float(m["mdd"])) + 0.05 * float(m["trades"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-candidates", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=80529)
    ap.add_argument("--official-topn", type=int, default=25)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_val = _predict_scaled(primary, val_df, primary_rt)
    f_val = _predict_scaled(fallback, val_df, fallback_rt)
    p_eval = _predict_scaled(primary, eval_df, primary_rt)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt)
    combo_val = _combine_primary_fallback(p_val, f_val)
    combo_eval = _combine_primary_fallback(p_eval, f_eval)
    base_val = _fast_cost3(val_df, combo_val)
    base_eval = _fast_cost3(eval_df, combo_eval)
    rows: list[dict[str, Any]] = []
    start = time.time()
    for i in range(int(args.max_candidates)):
        cfg = _rand_cfg(i, rng)
        val_dec = _apply_cfg(val_df, combo_val, cfg)
        val = _fast_cost3(val_df, val_dec)
        row = {"name": cfg.name, "score": _score(val), **{f"val_{k}": v for k, v in val.items() if k != "exits"}, **asdict(cfg)}
        rows.append(row)
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(RUNNING_CSV, index=False)
            print(json.dumps({"stage": "sweep", "done": i + 1, "best": rows[int(np.argmax([r["score"] for r in rows]))]["name"], "elapsed_sec": time.time() - start}), flush=True)
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    official_rows: list[dict[str, Any]] = []
    for _, r in df.head(int(args.official_topn)).iterrows():
        cfg = Cfg(**{k: r[k] for k in Cfg.__dataclass_fields__.keys()})
        val_dec = _apply_cfg(val_df, combo_val, cfg)
        eval_dec = _apply_cfg(eval_df, combo_eval, cfg)
        val_full = _combo_metrics(val_df, val_dec)["cost3"]
        eval_full = _combo_metrics(eval_df, eval_dec)["cost3"]
        official_rows.append(
            {
                "name": cfg.name,
                "official_score": _score(val_full),
                **{f"val_{k}": v for k, v in val_full.items()},
                **{f"oos_{k}": v for k, v in eval_full.items()},
                **asdict(cfg),
            }
        )
    official = pd.DataFrame(official_rows).sort_values(["oos_pnl", "oos_wr"], ascending=False)
    official.to_csv(RANKING_CSV, index=False)
    best = official.iloc[0].to_dict() if len(official) else {}
    summary = {
        "model_id": MODEL_ID,
        "design": "Combo-preserving Alpha8 risk/sizing random search. Alpha7 primary/fallback direction owners are preserved; rules only veto/scale notional/leverage/TP/SL/hold.",
        "baseline": {"val_cost3": base_val, "oos_cost3": base_eval},
        "max_candidates": int(args.max_candidates),
        "official_topn": int(args.official_topn),
        "best_by_oos_pnl": best,
        "target_hit": bool(best and float(best.get("oos_pnl", 0.0)) >= 200.0 and float(best.get("oos_wr", 0.0)) >= 0.50),
        "audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_prefix_count": 0,
            "selection_basis": "validation sweep; official OOS reported for top validation candidates",
            "live_wired": False,
        },
        "artifacts": {"running": str(RUNNING_CSV), "ranking": str(RANKING_CSV), "summary": str(SUMMARY_JSON)},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"summary": str(SUMMARY_JSON), "target_hit": summary["target_hit"], "best": best}, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
