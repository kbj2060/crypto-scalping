#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
    DEFAULT_SPEC_DIR,
    _days,
    _feature_matrix,
    _fill_price,
    _json_default,
    _label_frame,
    _read_feature_frame,
    _read_spec,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    CONTEXT_COLS,
    EQEConfig,
    _apply_label_preset,
    _backtest,
    _build_entry_labels,
    _build_exit_dataset,
    _entry_threshold_grid,
    _estimate_expected_return_by_bucket,
    _exit_close_prob,
    _exit_state_vec,
    _fit_entry_models,
    _fit_exit_model,
    _frame_value,
    _parse_bucket_thresholds,
    _parse_cost_multipliers,
    _predict_entry,
    _score,
    _threshold_for_bucket,
)


DEFAULT_2026_FEATURE_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519/rl_training_2026_direction_router.csv"
DEFAULT_PRIMARY_DECISIONS = ROOT / "tmp/causal_regen_20260516/live_alpha5_fallback_audit_20260525/live_runtime_decisions_2026.csv"
DEFAULT_2026_ENRICH_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_THRESHOLD_ROOT = ROOT / "tmp/causal_regen_20260516/alpha6_target_mode_abc_gpu_rapid_20260523"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_5m_cash_sleeve_20260525"
DEFAULT_VARIANTS = (
    "current_quality",
    "density_balanced",
    "regime_conditional",
    "perturbation_robust",
    "adverse_conformal",
    "sam_conformal",
    "high_precision_robust",
    "turnover_balanced_robust",
    "scalp_short_horizon",
    "short_horizon_robust",
)
THRESHOLD_DIR_BY_PRESET = {
    "current_quality": "current_quality",
    "density_balanced": "density_balanced",
    "regime_conditional": "regime_conditional",
    "perturbation_robust": "perturbation_robust",
    "adverse_conformal": "adverse_conformal",
    "sam_conformal": "sam_conformal",
    "high_precision_robust": "high_precision_robust",
    "turnover_balanced_robust": "turnover_balanced_robust",
    "scalp_short_horizon": "scalp_short_horizon_hreg",
    "short_horizon_robust": "short_horizon_robust_hreg",
}


def _primary_active_mask(oos_frame: pd.DataFrame, decisions_path: Path) -> np.ndarray:
    dec = pd.read_csv(decisions_path, parse_dates=["timestamp"])
    merged = oos_frame[["timestamp"]].merge(dec[["timestamp", "action"]], on="timestamp", how="left")
    if merged["action"].isna().any():
        miss = int(merged["action"].isna().sum())
        raise ValueError(f"missing primary decisions for {miss} OOS timestamps")
    return merged["action"].to_numpy(dtype=np.int64) != 0


def _load_reused_thresholds(root: Path, preset: str) -> tuple[float, float | tuple[float, ...]]:
    subdir = THRESHOLD_DIR_BY_PRESET.get(preset)
    if subdir is None:
        raise KeyError(f"no threshold-dir mapping for preset: {preset}")
    summary_path = root / subdir / "current_tail111_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    obj = json.loads(summary_path.read_text())
    best = obj.get("best") or {}
    if not isinstance(best, dict):
        raise ValueError(f"invalid best section in {summary_path}")
    entry = float(best["entry_threshold"])
    exit_raw = best["exit_threshold"]
    if isinstance(exit_raw, str) and "," in exit_raw:
        return entry, tuple(float(v) for v in exit_raw.split(","))
    return entry, float(exit_raw)


def _enrich_oos_frame(
    frame: pd.DataFrame,
    *,
    required_features: list[str],
    enrich_csv: Path,
) -> pd.DataFrame:
    enrich = pd.read_csv(enrich_csv, parse_dates=["timestamp"], low_memory=False)
    merged = frame.merge(enrich, on="timestamp", how="left", suffixes=("", "__enrich"))
    out = frame.copy()
    for feature in required_features:
        if feature in out.columns:
            continue
        source_col = None
        if feature.startswith("clean_regime4_state24_sticky090_v2_"):
            suffix = feature.removeprefix("clean_regime4_state24_sticky090_v2_")
            cand = f"clean_regime4_2024_unsup_v1_{suffix}"
            if cand in merged.columns:
                source_col = cand
        elif feature in merged.columns:
            source_col = feature
        if source_col is not None:
            out[feature] = merged[source_col]
    for feature in ["atr14_pct"]:
        if feature not in out.columns and feature in merged.columns:
            out[feature] = merged[feature]
    if "atr14_pct" not in out.columns:
        high = pd.to_numeric(out["high"], errors="coerce").astype(np.float64)
        low = pd.to_numeric(out["low"], errors="coerce").astype(np.float64)
        close = pd.to_numeric(out["close"], errors="coerce").astype(np.float64)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        out["atr14_pct"] = (atr14 / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.003)
    return out


def _backtest_cash_sleeve(
    frame: pd.DataFrame,
    x_oos: np.ndarray,
    dec: pd.DataFrame,
    exit_model: Any,
    *,
    primary_active: np.ndarray,
    entry_threshold: float,
    exit_threshold: float | tuple[float, ...],
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    regime_drift: bool = False,
    capture_ratio: bool = False,
    expected_return_by_bucket: dict[int, float] | None = None,
    guard_max_target_hold: bool = False,
    guard_adverse_atr: float = 0.0,
    guard_giveback_ratio: float = 0.0,
    guard_min_mfe: float = 0.0,
    entry_pullback_atr: float = 0.0,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = 0.0
    mfe = 0.0
    exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    expected_return_by_bucket = expected_return_by_bucket or {k: 0.01 for k in range(5)}
    trades = wins = long_entries = short_entries = exit_model_closes = missed_entries = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, notional: float, horizon: int, bucket: int) -> None:
        nonlocal side, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries, missed_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_idx = int(i)
        exposure = float(np.clip(notional, 0.01, 2.0))
        target_horizon = int(np.clip(horizon, 2, state_horizon))
        target_bucket = int(np.clip(bucket, 0, 4))
        if float(entry_pullback_atr) > 0.0:
            pullback = float(entry_pullback_atr) * max(float(atr[fill_i]), 0.0)
            if side > 0:
                limit_px = float(open_px[fill_i]) * (1.0 - pullback)
                if float(low[fill_i]) > limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 + slip)
            else:
                limit_px = float(open_px[fill_i]) * (1.0 + pullback)
                if float(high[fill_i]) < limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 - slip)
        else:
            entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = 0.0
        mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        mae = 0.0
        mfe = 0.0
        exposure = 0.0
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
            if bool(primary_active[i]):
                exit_pos(i, "primary_preempt")
                closed_this_bar = True
            elif hold >= int(min_exit_hold):
                current_atr = max(_frame_value(frame, "atr14_pct", i, 0.003), 1e-9)
                giveback = max(0.0, mfe - max(raw * exposure, 0.0))
                giveback_ratio = giveback / max(mfe, 1e-9)
                adverse_atr = max(0.0, -raw) / current_atr
                if guard_max_target_hold and hold >= int(target_horizon):
                    exit_pos(i, "guard_target_hold")
                    closed_this_bar = True
                elif float(guard_adverse_atr) > 0.0 and adverse_atr >= float(guard_adverse_atr):
                    exit_pos(i, "guard_adverse_atr")
                    closed_this_bar = True
                elif float(guard_giveback_ratio) > 0.0 and mfe >= float(guard_min_mfe) and giveback_ratio >= float(guard_giveback_ratio):
                    exit_pos(i, "guard_giveback")
                    closed_this_bar = True
                if not closed_this_bar:
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
                    close_prob = _exit_close_prob(exit_model, x_oos[i], state)
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
        if side == 0 and desired != 0 and not closed_this_bar and not bool(primary_active[i]):
            enter(
                i,
                1 if desired == 1 else -1,
                float(row.notional),
                int(getattr(row, "target_horizon", state_horizon)),
                int(getattr(row, "target_bucket", 4)),
            )
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
        "missed_entries": int(missed_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "exit_model_closes": int(exit_model_closes),
        "exits": exits,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-evaluate Alpha6 EQE presets with 5m candidate generation and Alpha7 cash-sleeve gating.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--feature-csv-2026", type=Path, default=DEFAULT_2026_FEATURE_CSV)
    ap.add_argument("--feature-csv-2026-enrich", type=Path, default=DEFAULT_2026_ENRICH_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--primary-decisions", type=Path, default=DEFAULT_PRIMARY_DECISIONS)
    ap.add_argument("--reuse-thresholds-root", type=Path, default=DEFAULT_THRESHOLD_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--presets", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--iterations", type=int, default=650)
    ap.add_argument("--learning-rate", type=float, default=0.055)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2-leaf-reg", type=float, default=5.0)
    ap.add_argument("--exit-iterations", type=int, default=500)
    ap.add_argument("--exit-learning-rate", type=float, default=0.045)
    ap.add_argument("--exit-depth", type=int, default=5)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=1)
    ap.add_argument("--entry-thresholds", type=int, default=50)
    ap.add_argument("--eval-costs", default="1,2,3")
    ap.add_argument("--exit-threshold-grid", default="0.35,0.45,0.55,0.65,0.75")
    ap.add_argument("--exit-bucket-threshold-grid", default="")
    ap.add_argument("--fixed-notional", type=float, default=0.25)
    ap.add_argument("--exit-step", type=int, default=16)
    ap.add_argument("--exit-max-trades", type=int, default=9000)
    ap.add_argument("--exit-cost-mult", type=float, default=3.0)
    ap.add_argument("--exit-weight-scale", type=float, default=80.0)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--exit-on-flip", action="store_true")
    ap.add_argument("--guard-max-target-hold", action="store_true")
    ap.add_argument("--guard-adverse-atr", type=float, default=0.0)
    ap.add_argument("--guard-giveback-ratio", type=float, default=0.0)
    ap.add_argument("--guard-min-mfe", type=float, default=0.0)
    ap.add_argument("--entry-pullback-atr", type=float, default=0.0)
    ap.add_argument("--target-head-mode", choices=["bucket5", "horizon_reg", "fixed"], default="bucket5")
    ap.add_argument("--fixed-target-horizon", type=int, default=0)
    ap.add_argument("--max-target-horizon", type=int, default=96)
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--session-topk", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--no-pca", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0

    feat_2025, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), CONTEXT_COLS)
    frame_2025 = feat_2025.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame_2025[frame_2025["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame_2025[frame_2025["dataset_split"].astype(str).str.lower().ne("train")].copy()
    x_train_all, x_val, model_features, pipe = _feature_matrix(
        train,
        val,
        present,
        use_pca=use_pca,
        pca_components=int(spec.get("extra_pca_components") or 0),
    )

    feat_2026, present_2026, missing_2026 = _read_feature_frame(args.feature_csv_2026, list(spec["features"]), CONTEXT_COLS)
    if missing_2026:
        feat_2026 = _enrich_oos_frame(feat_2026, required_features=list(spec["features"]), enrich_csv=args.feature_csv_2026_enrich)
        present_2026 = [c for c in spec["features"] if c in feat_2026.columns]
        missing_2026 = [c for c in spec["features"] if c not in feat_2026.columns]
    missing_in_2026 = sorted(set(present) - set(present_2026))
    if missing_in_2026:
        raise ValueError(f"2026 feature contract missing columns: {missing_in_2026[:20]}")
    x_oos = pipe.transform(feat_2026[present].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan))
    primary_active = _primary_active_mask(feat_2026, args.primary_decisions)

    presets = [p.strip() for p in str(args.presets).split(",") if p.strip()]
    ranking_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for idx, preset in enumerate(presets):
        print(json.dumps({"stage": "preset_start", "preset": preset, "index": idx + 1, "total": len(presets)}, ensure_ascii=False), flush=True)
        cfg = _apply_label_preset(replace(EQEConfig(), fixed_notional=float(args.fixed_notional)), preset)
        valid, y, label_meta = _build_entry_labels(
            train,
            cfg,
            stride_bars=int(args.stride_bars),
            batch_size=int(args.batch_size),
            adaptive_sampling=False,
            event_quantile=0.85,
            max_extra=12000,
            label_preset=preset,
            session_topk=int(args.session_topk),
        )
        print(json.dumps({"stage": "labels_built", "preset": preset, "candidates": int(label_meta["candidates"]), "stride_bars": int(label_meta["stride_bars"])}, ensure_ascii=False), flush=True)
        x_entry = x_train_all[valid]
        entry_models = _fit_entry_models(x_entry, y, args)
        print(json.dumps({"stage": "entry_fit_done", "preset": preset}, ensure_ascii=False), flush=True)
        train_dec = _predict_entry(entry_models, x_train_all, cfg)
        expected_return_by_bucket = _estimate_expected_return_by_bucket(train, valid, y, cfg)
        x_exit, y_exit, w_exit, exit_meta = _build_exit_dataset(
            train,
            x_train_all,
            valid,
            y,
            train_dec,
            cfg,
            max_samples=int(args.exit_max_trades),
            step=int(args.exit_step),
            cost_mult=float(args.exit_cost_mult),
            weight_scale=float(args.exit_weight_scale),
            regime_drift=False,
            capture_ratio=False,
            adaptive_sampling=False,
            expected_return_by_bucket=expected_return_by_bucket,
            target_head_mode=str(args.target_head_mode),
        )
        print(json.dumps({"stage": "exit_dataset_built", "preset": preset, "samples": int(exit_meta["samples"])}, ensure_ascii=False), flush=True)
        exit_model = _fit_exit_model(x_exit, y_exit, w_exit, args)
        print(json.dumps({"stage": "exit_fit_done", "preset": preset}, ensure_ascii=False), flush=True)
        val_dec = _predict_entry(entry_models, x_val, cfg)
        oos_dec = _predict_entry(entry_models, x_oos, cfg)
        exit_thresholds: list[float | tuple[float, ...]] = [float(x.strip()) for x in str(args.exit_threshold_grid).split(",") if x.strip()]
        if str(args.exit_bucket_threshold_grid).strip():
            exit_thresholds.extend(
                _parse_bucket_thresholds(x.strip())
                for x in str(args.exit_bucket_threshold_grid).split(";")
                if x.strip()
            )
        best: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        eval_costs = _parse_cost_multipliers(str(args.eval_costs))
        reused_eth, reused_xth = _load_reused_thresholds(args.reuse_thresholds_root, preset)
        bt = {
            f"cost{m}": _backtest(
                val,
                x_val,
                val_dec,
                exit_model,
                entry_threshold=float(reused_eth),
                exit_threshold=reused_xth,
                fee=cfg.fee * m,
                slip=cfg.slip * m,
                min_exit_hold=int(args.min_exit_hold),
                state_horizon=int(cfg.max_train_horizon_bars),
                exit_on_flip=bool(args.exit_on_flip),
                expected_return_by_bucket=expected_return_by_bucket,
                guard_max_target_hold=bool(args.guard_max_target_hold),
                guard_adverse_atr=float(args.guard_adverse_atr),
                guard_giveback_ratio=float(args.guard_giveback_ratio),
                guard_min_mfe=float(args.guard_min_mfe),
                entry_pullback_atr=float(args.entry_pullback_atr),
            )
            for m in eval_costs
        }
        primary_bt = bt.get("cost3") or bt[f"cost{eval_costs[-1]}"]
        score = _score(bt.get("cost1", primary_bt), bt.get("cost2", primary_bt), primary_bt)
        row = {
            "entry_threshold": float(reused_eth),
            "exit_threshold": ",".join(f"{v:.6g}" for v in reused_xth) if isinstance(reused_xth, tuple) else float(reused_xth),
            "exit_threshold_type": "bucket" if isinstance(reused_xth, tuple) else "scalar",
            "score": float(score),
            "val_cost3_pnl": float(primary_bt["pnl"]),
            "val_cost3_mdd": float(primary_bt["mdd"]),
            "val_cost3_trades": int(primary_bt["trades"]),
        }
        rows.append(row)
        best = {"summary": row, "val_backtest": bt}
        print(json.dumps({"stage": "threshold_selection_done", "preset": preset, "entry_threshold": best["summary"]["entry_threshold"], "exit_threshold": best["summary"]["exit_threshold"], "selection_score": best["summary"]["score"]}, ensure_ascii=False), flush=True)
        exit_threshold = best["summary"]["exit_threshold"]
        if best["summary"]["exit_threshold_type"] == "bucket":
            selected_xth: float | tuple[float, ...] = tuple(float(v) for v in str(exit_threshold).split(","))
        else:
            selected_xth = float(exit_threshold)
        selected_eth = float(best["summary"]["entry_threshold"])
        oos_standalone = {
            f"cost{m}": _backtest(
                feat_2026,
                x_oos,
                oos_dec,
                exit_model,
                entry_threshold=selected_eth,
                exit_threshold=selected_xth,
                fee=cfg.fee * m,
                slip=cfg.slip * m,
                min_exit_hold=int(args.min_exit_hold),
                state_horizon=int(cfg.max_train_horizon_bars),
                exit_on_flip=bool(args.exit_on_flip),
                expected_return_by_bucket=expected_return_by_bucket,
                guard_max_target_hold=bool(args.guard_max_target_hold),
                guard_adverse_atr=float(args.guard_adverse_atr),
                guard_giveback_ratio=float(args.guard_giveback_ratio),
                guard_min_mfe=float(args.guard_min_mfe),
                entry_pullback_atr=float(args.entry_pullback_atr),
            )
            for m in eval_costs
        }
        oos_cash = {
            f"cost{m}": _backtest_cash_sleeve(
                feat_2026,
                x_oos,
                oos_dec,
                exit_model,
                primary_active=primary_active,
                entry_threshold=selected_eth,
                exit_threshold=selected_xth,
                fee=cfg.fee * m,
                slip=cfg.slip * m,
                min_exit_hold=int(args.min_exit_hold),
                state_horizon=int(cfg.max_train_horizon_bars),
                exit_on_flip=bool(args.exit_on_flip),
                expected_return_by_bucket=expected_return_by_bucket,
                guard_max_target_hold=bool(args.guard_max_target_hold),
                guard_adverse_atr=float(args.guard_adverse_atr),
                guard_giveback_ratio=float(args.guard_giveback_ratio),
                guard_min_mfe=float(args.guard_min_mfe),
                entry_pullback_atr=float(args.entry_pullback_atr),
            )
            for m in eval_costs
        }
        variant_out = args.out_dir / preset
        variant_out.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_id": "alpha6_5m_cash_sleeve_20260525",
            "variant": args.variant,
            "preset": preset,
            "config": asdict(cfg),
            "feature_cols": present,
            "model_features": model_features,
            "missing_features_train": missing,
            "missing_features_2026": missing_2026,
            "use_pca": use_pca,
            "pipeline": pipe,
            "entry_models": entry_models,
            "exit_model": exit_model,
            "exit_meta": exit_meta,
            "expected_return_by_bucket": expected_return_by_bucket,
            "selected_entry_threshold": selected_eth,
            "selected_exit_threshold": selected_xth,
        }
        joblib.dump(artifact, variant_out / f"{preset}_bundle.joblib")
        summary = {
            "model_id": "alpha6_5m_cash_sleeve_20260525",
            "variant": args.variant,
            "preset": preset,
            "design": "Alpha6 EQE relabeled on 5m candidate generation (stride=1) and re-evaluated standalone plus Alpha7 strict cash-sleeve overlay.",
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "oos_rows": int(len(feat_2026)),
            "label_meta": label_meta,
            "entry_label_distribution": entry_models["label_distribution"],
            "exit_meta": exit_meta,
            "raw_feature_count": int(len(present)),
            "missing_features_train": missing,
            "missing_features_2026": missing_2026,
            "model_feature_count": int(len(model_features)),
            "use_pca": bool(use_pca),
            "threshold_selection": best["summary"],
            "threshold_grid_top10": sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:10],
            "oos_standalone": oos_standalone,
            "oos_cash_sleeve": oos_cash,
            "primary_cash_ratio_oos": float((~primary_active).mean()),
        }
        summary_path = variant_out / f"{preset}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        manifest_rows.append({"preset": preset, "summary": str(summary_path)})
        ranking_rows.append(
            {
                "preset": preset,
                "selection_score": float(best["summary"]["score"]),
                "standalone_cost3_pnl": float(oos_standalone["cost3"]["pnl"]),
                "standalone_cost3_mdd": float(oos_standalone["cost3"]["mdd"]),
                "standalone_cost3_trades": int(oos_standalone["cost3"]["trades"]),
                "cash_sleeve_cost3_pnl": float(oos_cash["cost3"]["pnl"]),
                "cash_sleeve_cost3_mdd": float(oos_cash["cost3"]["mdd"]),
                "cash_sleeve_cost3_trades": int(oos_cash["cost3"]["trades"]),
                "primary_cash_ratio_oos": float((~primary_active).mean()),
                "summary": str(summary_path),
            }
        )
        print(
            json.dumps(
                {
                    "preset": preset,
                    "standalone_cost3_pnl": oos_standalone["cost3"]["pnl"],
                    "cash_sleeve_cost3_pnl": oos_cash["cost3"]["pnl"],
                    "cash_sleeve_cost3_trades": oos_cash["cost3"]["trades"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(["cash_sleeve_cost3_pnl", "standalone_cost3_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(args.out_dir / "ranking.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
