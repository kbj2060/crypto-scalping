#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _read  # noqa: E402
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    COMBO_SUMMARY,
    EVAL_CSV,
    OLD_CLEAN_PREFIX,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    TRAIN_CSV,
    _candidate_specs,
    _load_best_scale_runtime,
    _predict_scaled,
)


DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_idea_pack_20260526"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _signal_outcome(frame: pd.DataFrame, dec: pd.DataFrame, i: int, *, fee: float, slip: float) -> dict[str, Any]:
    row = dec.iloc[int(i)]
    action = int(pd.to_numeric(row["action"], errors="coerce"))
    side = int(pd.to_numeric(row["side"], errors="coerce"))
    if action == 0 or side == 0 or i + 1 >= len(frame):
        return {
            "active": 0,
            "side": 0,
            "confidence": float(pd.to_numeric(row["confidence"], errors="coerce")),
            "quality_score": float(pd.to_numeric(row["quality_score"], errors="coerce")),
            "reward": 0.0,
            "raw_return": 0.0,
            "exit_reason": "cash",
            "hold_bars_realized": 0,
            "mfe": 0.0,
            "mae": 0.0,
            "giveback": 0.0,
            "entry_good_exit_bad": 0,
            "entry_bad": 0,
            "tp_hit": 0,
            "sl_hit": 0,
            "max_hold_exit": 0,
        }

    notional = float(np.clip(pd.to_numeric(row["notional_exposure"], errors="coerce"), 0.01, 2.75))
    tp = float(max(pd.to_numeric(row["take_profit"], errors="coerce"), 1e-4))
    sl = float(max(pd.to_numeric(row["stop_loss"], errors="coerce"), 1e-4))
    max_hold = int(np.clip(pd.to_numeric(row["max_hold_bars"], errors="coerce"), 1, 96))
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    entry_i = min(i + 1, len(frame) - 1)
    entry = _fill_price(frame, entry_i, side, slip, entry=True)
    end_i = min(entry_i + max_hold, len(frame) - 1)

    raw = None
    exit_reason = "end"
    hold_realized = 0
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = float(high[j] / max(entry, 1e-12) - 1.0)
            adverse = float(low[j] / max(entry, 1e-12) - 1.0)
            mark = float(close[j] / max(entry, 1e-12) - 1.0)
        else:
            favorable = float(entry / max(low[j], 1e-12) - 1.0)
            adverse = float(entry / max(high[j], 1e-12) - 1.0)
            mark = float(entry / max(close[j], 1e-12) - 1.0)
        mfe = max(mfe, favorable)
        mae = max(mae, max(0.0, -adverse))
        hold_realized = j - entry_i
        if adverse <= -sl:
            raw = -sl
            exit_reason = "sl"
            break
        if favorable >= tp:
            raw = tp
            exit_reason = "tp"
            break
        final_mark = mark
    if raw is None:
        exit_px = _fill_price(frame, end_i, side, slip, entry=False)
        raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
        exit_reason = "max_hold" if end_i < len(frame) - 1 else "end"
        hold_realized = end_i - entry_i
    giveback = max(0.0, mfe - max(raw, 0.0))
    reward = (1.0 + raw * notional)
    reward = reward - fee * notional
    reward = reward - fee * notional
    reward = reward - 1.0
    entry_good_exit_bad = int(mfe > max(tp * 0.35, 0.0020) and raw < 0.0)
    entry_bad = int(mfe < max(tp * 0.10, 0.0008) and raw < 0.0)
    return {
        "active": 1,
        "side": side,
        "confidence": float(pd.to_numeric(row["confidence"], errors="coerce")),
        "quality_score": float(pd.to_numeric(row["quality_score"], errors="coerce")),
        "notional_exposure": notional,
        "take_profit": tp,
        "stop_loss": sl,
        "max_hold_bars": max_hold,
        "reward": float(reward),
        "raw_return": float(raw),
        "exit_reason": exit_reason,
        "hold_bars_realized": int(hold_realized),
        "mfe": float(mfe),
        "mae": float(mae),
        "giveback": float(giveback),
        "entry_good_exit_bad": entry_good_exit_bad,
        "entry_bad": entry_bad,
        "tp_hit": int(exit_reason == "tp"),
        "sl_hit": int(exit_reason == "sl"),
        "max_hold_exit": int(exit_reason == "max_hold"),
    }


def _bin_calibration(series: pd.Series, rewards: pd.Series, *, bins: int, label: str) -> pd.DataFrame:
    valid = series.notna() & rewards.notna()
    if int(valid.sum()) == 0:
        return pd.DataFrame()
    q = pd.qcut(series[valid], q=min(bins, int(valid.sum())), duplicates="drop")
    grouped = pd.DataFrame({"bin": q.astype(str), "score": series[valid], "reward": rewards[valid]})
    out = (
        grouped.groupby("bin", dropna=False)
        .agg(
            rows=("reward", "size"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            reward_mean=("reward", "mean"),
            reward_sum=("reward", "sum"),
            win_rate=("reward", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    out.insert(0, "metric", label)
    return out


def _overlap_matrix(names: list[str], ledgers: dict[str, pd.DataFrame], mask: np.ndarray) -> pd.DataFrame:
    rows = []
    actives = {name: ledgers[name].loc[mask, f"{name}_active"].to_numpy(dtype=np.int64) for name in names}
    sides = {name: ledgers[name].loc[mask, f"{name}_side"].to_numpy(dtype=np.int64) for name in names}
    for a in names:
        for b in names:
            both = (actives[a] == 1) & (actives[b] == 1)
            same = both & (np.sign(sides[a]) == np.sign(sides[b]))
            opposite = both & (np.sign(sides[a]) == -np.sign(sides[b]))
            rows.append(
                {
                    "left": a,
                    "right": b,
                    "both_active_rows": int(both.sum()),
                    "same_side_rows": int(same.sum()),
                    "opposite_side_rows": int(opposite.sum()),
                }
            )
    return pd.DataFrame(rows)


def _candidate_partition_summary(
    frame: pd.DataFrame,
    primary_active: np.ndarray,
    fallback_active: np.ndarray,
    fallback_rewards: np.ndarray,
) -> pd.DataFrame:
    primary_cash = ~primary_active
    parts = {
        "primary_active": primary_active,
        "primary_cash_fallback_active": primary_cash & fallback_active,
        "primary_cash_fallback_cash": primary_cash & (~fallback_active),
    }
    rows = []
    for name, mask in parts.items():
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "partition": name,
                "rows": int(mask.sum()),
                "rows_pct": float(mask.mean()),
                "signal_count": int((fallback_active & mask).sum()),
                "signal_reward_sum_pct": float(fallback_rewards[mask].sum() * 100.0),
                "signal_reward_mean_pct": float(fallback_rewards[mask].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _top_cases(cash_ledger: pd.DataFrame, *, ascending: bool, limit: int) -> pd.DataFrame:
    cols = ["timestamp", "primary_quality", "primary_confidence", "winner_candidate", "winner_reward_pct"]
    extra = [c for c in cash_ledger.columns if c.endswith("_active") or c.endswith("_reward")]
    return cash_ledger.sort_values("winner_reward_pct", ascending=ascending).loc[:, cols + extra].head(limit)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate an idea pack for Alpha7 fallback action model design.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--calibration-bins", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=100)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    cutoff = pd.Timestamp("2025-10-01")
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)

    primary_parent = joblib.load(PRIMARY_PARENT)
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)

    candidate_specs = []
    val_candidate_decs = []
    eval_candidate_decs = []
    for spec in _candidate_specs():
        if not spec.parent.exists():
            continue
        parent = joblib.load(spec.parent)
        if any(str(c).startswith(OLD_CLEAN_PREFIX) for c in parent.get("feature_cols", [])):
            continue
        rt = _load_best_scale_runtime(spec.summary)
        candidate_specs.append(spec)
        val_candidate_decs.append(_predict_scaled(parent, val_df, rt))
        eval_candidate_decs.append(_predict_scaled(parent, eval_df, rt))
    if not candidate_specs:
        raise RuntimeError("no valid fallback candidates")

    cost_summary = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    fee = 0.0004
    slip = 0.00015

    datasets = {
        "val_2025_q4": (val_df, primary_val, val_candidate_decs),
        "oos_2026": (eval_df, primary_eval, eval_candidate_decs),
    }
    dataset_reports: dict[str, Any] = {}
    for ds_name, (frame, primary_dec, candidate_decs) in datasets.items():
        base = pd.DataFrame({"timestamp": frame["timestamp"].astype(str)})
        base["primary_active"] = _active(primary_dec).astype(np.int64)
        base["primary_quality"] = pd.to_numeric(primary_dec["quality_score"], errors="coerce").fillna(0.0)
        base["primary_confidence"] = pd.to_numeric(primary_dec["confidence"], errors="coerce").fillna(0.0)
        for spec, dec in zip(candidate_specs, candidate_decs):
            rows = [_signal_outcome(frame, dec, i, fee=fee, slip=slip) for i in range(len(frame))]
            cand = pd.DataFrame(rows).add_prefix(f"{spec.name}_")
            base = pd.concat([base, cand], axis=1)

        primary_cash = base["primary_active"].to_numpy(dtype=np.int64) == 0
        reward_cols = [f"{spec.name}_reward" for spec in candidate_specs]
        rewards = base[reward_cols].to_numpy(dtype=np.float64)
        best_idx = np.argmax(rewards, axis=1)
        best_reward = rewards[np.arange(len(base)), best_idx]
        base["winner_candidate"] = np.where(
            best_reward > 0.0,
            np.asarray([candidate_specs[i].name for i in best_idx], dtype=object),
            "cash",
        )
        base["winner_reward_pct"] = best_reward * 100.0

        partition = _candidate_partition_summary(
            frame,
            base["primary_active"].to_numpy(dtype=bool),
            base[f"{candidate_specs[0].name}_active"].to_numpy(dtype=bool),
            base[f"{candidate_specs[0].name}_reward"].to_numpy(dtype=np.float64),
        )
        overlap = _overlap_matrix([spec.name for spec in candidate_specs], {spec.name: base for spec in candidate_specs}, primary_cash)

        calibration_rows = []
        for spec in candidate_specs:
            active_mask = primary_cash & (base[f"{spec.name}_active"].to_numpy(dtype=np.int64) == 1)
            if int(active_mask.sum()) == 0:
                continue
            cal_q = _bin_calibration(
                base.loc[active_mask, f"{spec.name}_quality_score"],
                base.loc[active_mask, f"{spec.name}_reward"],
                bins=int(args.calibration_bins),
                label=f"{spec.name}:quality",
            )
            cal_c = _bin_calibration(
                base.loc[active_mask, f"{spec.name}_confidence"],
                base.loc[active_mask, f"{spec.name}_reward"],
                bins=int(args.calibration_bins),
                label=f"{spec.name}:confidence",
            )
            calibration_rows.extend([cal_q, cal_c])
        calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()

        entry_exit_rows = []
        for spec in candidate_specs:
            active_mask = primary_cash & (base[f"{spec.name}_active"].to_numpy(dtype=np.int64) == 1)
            if int(active_mask.sum()) == 0:
                continue
            df = base.loc[active_mask, [
                f"{spec.name}_reward",
                f"{spec.name}_mfe",
                f"{spec.name}_mae",
                f"{spec.name}_giveback",
                f"{spec.name}_entry_good_exit_bad",
                f"{spec.name}_entry_bad",
                f"{spec.name}_tp_hit",
                f"{spec.name}_sl_hit",
                f"{spec.name}_max_hold_exit",
                f"{spec.name}_hold_bars_realized",
            ]].copy()
            entry_exit_rows.append(
                {
                    "candidate": spec.name,
                    "signals": int(len(df)),
                    "reward_sum_pct": float(df[f"{spec.name}_reward"].sum() * 100.0),
                    "reward_mean_pct": float(df[f"{spec.name}_reward"].mean() * 100.0),
                    "win_rate": float((df[f"{spec.name}_reward"] > 0.0).mean()),
                    "entry_good_exit_bad_rate": float(df[f"{spec.name}_entry_good_exit_bad"].mean()),
                    "entry_bad_rate": float(df[f"{spec.name}_entry_bad"].mean()),
                    "tp_hit_rate": float(df[f"{spec.name}_tp_hit"].mean()),
                    "sl_hit_rate": float(df[f"{spec.name}_sl_hit"].mean()),
                    "max_hold_exit_rate": float(df[f"{spec.name}_max_hold_exit"].mean()),
                    "avg_mfe_pct": float(df[f"{spec.name}_mfe"].mean() * 100.0),
                    "avg_mae_pct": float(df[f"{spec.name}_mae"].mean() * 100.0),
                    "avg_giveback_pct": float(df[f"{spec.name}_giveback"].mean() * 100.0),
                    "hold_bars_median": float(df[f"{spec.name}_hold_bars_realized"].median()),
                }
            )
        entry_exit = pd.DataFrame(entry_exit_rows).sort_values("reward_sum_pct", ascending=False)

        cash_cols = [
            "timestamp",
            "primary_quality",
            "primary_confidence",
            "winner_candidate",
            "winner_reward_pct",
        ]
        for spec in candidate_specs:
            cash_cols.extend(
                [
                    f"{spec.name}_active",
                    f"{spec.name}_side",
                    f"{spec.name}_quality_score",
                    f"{spec.name}_confidence",
                    f"{spec.name}_reward",
                    f"{spec.name}_exit_reason",
                    f"{spec.name}_mfe",
                    f"{spec.name}_mae",
                    f"{spec.name}_giveback",
                ]
            )
        cash_ledger = base.loc[primary_cash, cash_cols].copy()
        cash_ledger_path = args.out_dir / f"{ds_name}_primary_cash_signal_ledger.csv"
        cash_ledger.to_csv(cash_ledger_path, index=False)
        overlap_path = args.out_dir / f"{ds_name}_candidate_overlap.csv"
        overlap.to_csv(overlap_path, index=False)
        partition_path = args.out_dir / f"{ds_name}_primary_cash_partition.csv"
        partition.to_csv(partition_path, index=False)
        calibration_path = args.out_dir / f"{ds_name}_calibration.csv"
        calibration.to_csv(calibration_path, index=False)
        entry_exit_path = args.out_dir / f"{ds_name}_entry_exit_breakdown.csv"
        entry_exit.to_csv(entry_exit_path, index=False)
        best_cases_path = args.out_dir / f"{ds_name}_top_best_cash_cases.csv"
        worst_cases_path = args.out_dir / f"{ds_name}_top_worst_cash_cases.csv"
        _top_cases(cash_ledger, ascending=False, limit=int(args.top_k)).to_csv(best_cases_path, index=False)
        _top_cases(cash_ledger, ascending=True, limit=int(args.top_k)).to_csv(worst_cases_path, index=False)

        dataset_reports[ds_name] = {
            "rows": int(len(frame)),
            "days": float(_days(frame)),
            "primary_cash_rows": int(primary_cash.sum()),
            "primary_cash_ratio": float(primary_cash.mean()),
            "winner_distribution": cash_ledger["winner_candidate"].value_counts().to_dict(),
            "partition_csv": str(partition_path),
            "overlap_csv": str(overlap_path),
            "calibration_csv": str(calibration_path),
            "entry_exit_csv": str(entry_exit_path),
            "cash_ledger_csv": str(cash_ledger_path),
            "best_cases_csv": str(best_cases_path),
            "worst_cases_csv": str(worst_cases_path),
            "entry_exit_summary": entry_exit.to_dict(orient="records"),
        }

    report = {
        "model_id": "alpha7_fallback_idea_pack_20260526",
        "design": "Primary/fallback idea pack for creative fallback action model design. The report focuses on primary-cash regions, candidate overlap/conflict, quality-confidence calibration, and entry-vs-exit responsibility.",
        "baseline_combo_summary": cost_summary.get("selected_metrics"),
        "candidate_names": [spec.name for spec in candidate_specs],
        "datasets": dataset_reports,
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
