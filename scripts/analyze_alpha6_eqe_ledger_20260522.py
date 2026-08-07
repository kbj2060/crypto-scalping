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

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    _days,
    _fill_price,
    _json_default,
    _label_frame,
    _numeric_matrix,
    _read_feature_frame,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    CONTEXT_COLS,
    TARGET_BUCKET_TO_HORIZON,
    _exit_close_prob,
    _exit_state_vec,
    _threshold_for_bucket,
)


def _parse_threshold(value: Any) -> float | tuple[float, ...]:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if "," in raw:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    return float(raw)


def _x_val_for_bundle(feature_csv: Path, label_dir: Path, bundle: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    features = list(bundle["feature_cols"])
    frame, present, _ = _read_feature_frame(feature_csv, features, CONTEXT_COLS)
    merged = frame.merge(_label_frame(label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    val = merged[merged["dataset_split"].astype(str).str.lower().ne("train")].copy().reset_index(drop=True)
    pipe = bundle["pipeline"]
    x_val = pipe.transform(_numeric_matrix(val, present))
    return val, x_val


def _backtest_with_ledger(
    frame: pd.DataFrame,
    x_val: np.ndarray,
    dec: pd.DataFrame,
    exit_model: Any,
    *,
    entry_threshold: float,
    exit_threshold: float | tuple[float, ...],
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    regime_drift: bool,
    capture_ratio: bool,
    expected_return_by_bucket: dict[int, float],
    guard_max_target_hold: bool = False,
    guard_adverse_atr: float = 0.0,
    guard_giveback_ratio: float = 0.0,
    guard_min_mfe: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    if "atr14_pct" in frame.columns:
        atr_pct = pd.to_numeric(frame["atr14_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.003).to_numpy(dtype=np.float64)
    else:
        atr_pct = np.full(len(frame), 0.003, dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = 0.0
    exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, notional: float, horizon: int, bucket: int) -> None:
        nonlocal side, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_idx = int(i)
        exposure = float(np.clip(notional, 0.01, 2.0))
        target_horizon = int(np.clip(horizon, 2, state_horizon))
        target_bucket = int(np.clip(bucket, 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        after = cash
        trades += 1
        wins += int(after > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        ledger.append(
            {
                "trade_id": trades,
                "entry_idx": int(entry_idx),
                "exit_idx": int(i),
                "entry_timestamp": str(frame.iloc[min(entry_idx + 1, len(frame) - 1)]["timestamp"]),
                "exit_timestamp": str(frame.iloc[fill_i]["timestamp"]),
                "side": "long" if side > 0 else "short",
                "entry_price": float(entry),
                "exit_price": float(fill_px),
                "raw_return": float(raw),
                "exposure": float(exposure),
                "pnl_pct_equity": float((after / max(entry_equity, 1e-12) - 1.0) * 100.0),
                "cash_after": float(after),
                "hold_bars": int(hold),
                "target_horizon": int(target_horizon),
                "target_bucket": int(target_bucket),
                "mae_pct_equity": float(mae * 100.0),
                "mfe_pct_equity": float(mfe * 100.0),
                "giveback_pct_equity": float(max(0.0, mfe - max(raw * exposure, 0.0)) * 100.0),
                "exit_reason": reason,
                "win": int(after > entry_equity),
            }
        )
        side = 0
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(len(frame) - 2):
        row = dec.iloc[i]
        desired = int(row.action) if float(row.quality_score) >= float(entry_threshold) else 0
        closed_this_bar = False
        if side != 0:
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if hold >= int(min_exit_hold):
                current_atr = max(float(atr_pct[i]), 1e-9)
                giveback = max(0.0, mfe - max(raw * exposure, 0.0))
                giveback_ratio = giveback / max(mfe, 1e-9)
                adverse_atr = max(0.0, -raw) / current_atr
                if guard_max_target_hold and hold >= int(target_horizon):
                    exit_pos(i, "guard_target_hold")
                    closed_this_bar = True
                elif float(guard_adverse_atr) > 0.0 and adverse_atr >= float(guard_adverse_atr):
                    exit_pos(i, "guard_adverse_atr")
                    closed_this_bar = True
                elif (
                    float(guard_giveback_ratio) > 0.0
                    and mfe >= float(guard_min_mfe)
                    and giveback_ratio >= float(guard_giveback_ratio)
                ):
                    exit_pos(i, "guard_giveback")
                    closed_this_bar = True
                if closed_this_bar:
                    eq = equity(i)
                    peak = max(peak, eq)
                    mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
                    continue
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=regime_drift,
                    capture_ratio=capture_ratio,
                    expected_return=float(expected_return_by_bucket.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(exit_model, x_val[i], state)
                if close_prob >= _threshold_for_bucket(exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
                    closed_this_bar = True
                elif exit_on_flip and desired != 0 and ((desired == 1 and side < 0) or (desired == 2 and side > 0)):
                    exit_pos(i, "model_flip")
                    closed_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0 and not closed_this_bar:
            enter(
                i,
                1 if desired == 1 else -1,
                float(row.notional),
                int(getattr(row, "target_horizon", state_horizon)),
                int(getattr(row, "target_bucket", 4)),
            )
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "exit_model_closes": int(exit_model_closes),
        "exits": exits,
    }
    return metrics, pd.DataFrame(ledger)


def _bootstrap_sum_ci(values: np.ndarray, *, n_boot: int = 2000, ci: float = 0.90, seed: int = 42) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(int(n_boot), values.size), replace=True).sum(axis=1)
    lo = (1.0 - float(ci)) / 2.0 * 100.0
    hi = (1.0 + float(ci)) / 2.0 * 100.0
    return {"lower": float(np.percentile(samples, lo)), "upper": float(np.percentile(samples, hi)), "mean": float(np.mean(samples))}


def _ledger_diagnostics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {}
    gaps = ledger["entry_idx"].to_numpy()[1:] - ledger["exit_idx"].to_numpy()[:-1]
    mfe = ledger["mfe_pct_equity"].to_numpy(dtype=np.float64)
    giveback = ledger["giveback_pct_equity"].to_numpy(dtype=np.float64)
    giveback_ratio = np.divide(giveback, np.maximum(mfe, 1e-12))
    return {
        "micro_hold_le_3": int((ledger["hold_bars"] <= 3).sum()),
        "micro_hold_le_3_rate": float((ledger["hold_bars"] <= 3).mean()),
        "hold_bars_median": float(ledger["hold_bars"].median()),
        "hold_bars_p90": float(ledger["hold_bars"].quantile(0.90)),
        "same_or_next_bar_reentry": int((gaps <= 1).sum()) if gaps.size else 0,
        "same_or_next_bar_reentry_rate": float((gaps <= 1).mean()) if gaps.size else 0.0,
        "avg_mae_pct_equity": float(ledger["mae_pct_equity"].mean()),
        "avg_mfe_pct_equity": float(ledger["mfe_pct_equity"].mean()),
        "avg_giveback_pct_equity": float(ledger["giveback_pct_equity"].mean()),
        "giveback_ratio_median": float(np.median(giveback_ratio[np.isfinite(giveback_ratio)])) if np.isfinite(giveback_ratio).any() else 0.0,
        "exit_reasons": ledger["exit_reason"].value_counts().to_dict(),
        "side_counts": ledger["side"].value_counts().to_dict(),
        "bucket_counts": ledger["target_bucket"].value_counts().sort_index().to_dict(),
    }


def analyze_run(run_dir: Path, *, feature_csv: Path, label_dir: Path, n_boot: int, costs: tuple[int, ...]) -> dict[str, Any]:
    summary_path = next(run_dir.glob("*_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    variant = str(summary["variant"])
    prefix = run_dir / variant
    bundle = joblib.load(f"{prefix}_bundle.joblib")
    pred = pd.read_csv(f"{prefix}_val_predictions.csv", parse_dates=["timestamp"])
    frame, x_val = _x_val_for_bundle(feature_csv, label_dir, bundle)
    best = summary["best"]
    params = summary["params"]
    cfg_src = summary.get("config") or summary.get("label_meta") or {}
    metrics_by_cost: dict[str, Any] = {}
    ledger_by_cost: dict[str, pd.DataFrame] = {}
    for mult in costs:
        metrics, ledger = _backtest_with_ledger(
            frame,
            x_val,
            pred,
            bundle["exit_model"],
            entry_threshold=float(best["entry_threshold"]),
            exit_threshold=_parse_threshold(best["exit_threshold"]),
            fee=float(cfg_src.get("fee", 0.0004)) * mult,
            slip=float(cfg_src.get("slip", 0.00015)) * mult,
            min_exit_hold=int(params.get("min_exit_hold", 2)),
            state_horizon=int(cfg_src.get("max_train_horizon_bars", 96)),
            exit_on_flip=bool(params.get("exit_on_flip", False)),
            regime_drift=bool(params.get("enable_regime_drift_state", False)),
            capture_ratio=bool(params.get("enable_capture_ratio_state", False)),
            expected_return_by_bucket={int(k): float(v) for k, v in bundle.get("expected_return_by_bucket", {}).items()},
            guard_max_target_hold=bool(params.get("guard_max_target_hold", False)),
            guard_adverse_atr=float(params.get("guard_adverse_atr", 0.0)),
            guard_giveback_ratio=float(params.get("guard_giveback_ratio", 0.0)),
            guard_min_mfe=float(params.get("guard_min_mfe", 0.0)),
        )
        ledger_by_cost[f"cost{mult}"] = ledger
        metrics_by_cost[f"cost{mult}"] = metrics
        metrics_by_cost[f"cost{mult}"]["bootstrap_pnl_ci_90"] = _bootstrap_sum_ci(
            ledger["pnl_pct_equity"].to_numpy(dtype=np.float64) if not ledger.empty else np.asarray([]),
            n_boot=n_boot,
        )
        metrics_by_cost[f"cost{mult}"]["ledger_diagnostics"] = _ledger_diagnostics(ledger)
        ledger.to_csv(f"{prefix}_cost{mult}_reconstructed_ledger.csv", index=False)
    report = {
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "variant": variant,
        "selected_thresholds": {
            "entry_threshold": float(best["entry_threshold"]),
            "exit_threshold": best["exit_threshold"],
            "exit_threshold_type": best.get("exit_threshold_type"),
        },
        "calmar_success_reference": {
            "primary": "cost3.calmar",
            "min_trades_stat_filter": 30,
            "trade_count_is_metric_not_objective": True,
        },
        "metrics": metrics_by_cost,
    }
    Path(f"{prefix}_ledger_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--costs", default="3", help="Comma-separated cost multipliers to reconstruct. Default is primary Cost3 only.")
    ap.add_argument("--out", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_eqe_ledger_analysis_20260522.json")
    args = ap.parse_args()
    costs = tuple(int(x.strip()) for x in str(args.costs).split(",") if x.strip())
    reports = [analyze_run(p, feature_csv=args.feature_csv, label_dir=args.label_dir, n_boot=int(args.n_boot), costs=costs) for p in args.run_dirs]
    rows = []
    for report in reports:
        c3 = report["metrics"]["cost3"]
        diag = c3["ledger_diagnostics"]
        ci = c3["bootstrap_pnl_ci_90"]
        rows.append(
            {
                "run_dir": report["run_dir"],
                "cost3_pnl": c3["pnl"],
                "cost3_mdd": c3["mdd"],
                "cost3_calmar": c3["calmar"],
                "cost3_trades": c3["trades"],
                "cost3_wr": c3["wr"],
                "cost3_ci90_lower": ci["lower"],
                "cost3_ci90_upper": ci["upper"],
                "micro_hold_le_3_rate": diag.get("micro_hold_le_3_rate"),
                "same_or_next_reentry_rate": diag.get("same_or_next_bar_reentry_rate"),
                "avg_giveback_pct_equity": diag.get("avg_giveback_pct_equity"),
            }
        )
    out = {"reports": reports, "ranking": sorted(rows, key=lambda x: x["cost3_calmar"], reverse=True)}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(out["ranking"], ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
