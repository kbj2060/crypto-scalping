#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_zigzag_action_labels_v2_20260604 as zigzag_v2  # noqa: E402


DEFAULT_OUT_ROOT = ROOT / "tmp/causal_regen_20260516/zigzag_multithreshold_horizon_20260624"
PRICE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _apply_max_bars(labels: pd.DataFrame, *, max_bars: int) -> pd.DataFrame:
    out = labels.copy()
    active = pd.to_numeric(out["zigzag_action"], errors="raise").to_numpy(dtype=np.int64) != 0
    wave_bars = pd.to_numeric(out["zigzag_wave_bars"], errors="raise").to_numpy(dtype=np.float64)
    masked = active & (wave_bars > float(max_bars))
    if bool(masked.any()):
        out.loc[masked, "zigzag_action"] = 0
        out.loc[masked, "zigzag_action_name"] = "CASH"
        out.loc[masked, "zigzag_quality_gate"] = 0
        for col in (
            "zigzag_path_return",
            "zigzag_path_mae",
            "zigzag_path_mfe",
            "zigzag_path_calmar",
            "zigzag_path_edge",
            "zigzag_soft_long",
            "zigzag_soft_short",
        ):
            if col in out.columns:
                out.loc[masked, col] = 0.0
        if "zigzag_soft_cash" in out.columns:
            out.loc[masked, "zigzag_soft_cash"] = 1.0
    out["zigzag_horizon_cap_bars"] = int(max_bars)
    out["zigzag_horizon_cap_masked"] = masked.astype(np.int8)
    return out


def _prefixed(labels: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "timestamp",
        "zigzag_action",
        "zigzag_raw_wave_action",
        "zigzag_segment_id",
        "zigzag_wave_return",
        "zigzag_wave_bars",
        "zigzag_transition_buffer",
        "zigzag_quality_gate",
        "zigzag_phase",
        "zigzag_path_return",
        "zigzag_path_mae",
        "zigzag_path_mfe",
        "zigzag_path_calmar",
        "zigzag_path_edge",
        "zigzag_soft_cash",
        "zigzag_soft_long",
        "zigzag_soft_short",
        "zigzag_horizon_cap_bars",
        "zigzag_horizon_cap_masked",
    ]
    missing = sorted(set(keep) - set(labels.columns))
    if missing:
        raise RuntimeError(f"missing zigzag stack columns for {prefix}: {missing}")
    out = labels[keep].copy()
    return out.rename(columns={c: f"{prefix}_{c}" for c in keep if c != "timestamp"})


def _build_one(
    frame: pd.DataFrame,
    *,
    threshold: float,
    max_bars: int,
    min_wave_bars: int,
    transition_buffer: int,
    mae_penalty: float,
    softmax_temperature: float,
    min_risk_floor: float,
    min_edge_pct: float,
    min_calmar: float,
    min_mfe_efficiency: float,
    min_phase: float,
    max_phase: float,
) -> pd.DataFrame:
    labels = zigzag_v2.build_zigzag_action_labels(
        frame,
        min_reversal_pct=float(threshold),
        max_reversal_pct=float(threshold),
        min_wave_bars=int(min_wave_bars),
        transition_buffer=int(transition_buffer),
        atr_window=14,
        atr_multiplier=0.0,
        mae_penalty=float(mae_penalty),
        softmax_temperature=float(softmax_temperature),
        min_risk_floor=float(min_risk_floor),
        min_edge_pct=float(min_edge_pct),
        min_calmar=float(min_calmar),
        min_mfe_efficiency=float(min_mfe_efficiency),
        min_phase=float(min_phase),
        max_phase=float(max_phase),
    )
    return _apply_max_bars(labels, max_bars=int(max_bars))


def _standard_export(base: pd.DataFrame, medium: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    out = medium.copy()
    extra_cols = [c for c in stack.columns if c != "timestamp"]
    for col in extra_cols:
        if col not in out.columns:
            out[col] = stack[col].to_numpy()
    return out


def _quality_export(medium: pd.DataFrame, stack: pd.DataFrame, *, min_score: float) -> pd.DataFrame:
    out = medium.copy()
    med = pd.to_numeric(stack["medium_zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    short = pd.to_numeric(stack["short_zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    long = pd.to_numeric(stack["long_zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    active = med != 0
    short_same = active & (short == med)
    long_same = active & (long == med)
    score = np.zeros(len(stack), dtype=np.float64)
    score[active] = (1.0 + short_same[active].astype(np.float64) + long_same[active].astype(np.float64)) / 3.0
    quality_action = np.where((med != 0) & (score >= float(min_score)), med, 0).astype(np.int8)
    out["zigzag_action"] = quality_action
    out["zigzag_action_name"] = pd.Series(quality_action).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["zigzag_multithreshold_consensus_score"] = score.astype(np.float32)
    out["zigzag_short_medium_aligned"] = short_same.astype(np.int8)
    out["zigzag_medium_long_aligned"] = long_same.astype(np.int8)
    out["zigzag_quality_min_consensus_score"] = float(min_score)
    out["zigzag_soft_cash"] = np.where(quality_action == 0, 1.0, np.maximum(0.0, 1.0 - score)).astype(np.float32)
    out["zigzag_soft_long"] = np.where(quality_action == 1, score, 0.0).astype(np.float32)
    out["zigzag_soft_short"] = np.where(quality_action == 2, score, 0.0).astype(np.float32)
    for col in [c for c in stack.columns if c != "timestamp"]:
        if col not in out.columns:
            out[col] = stack[col].to_numpy()
    return out


def _summary(labels: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    active = y != 0
    out: dict[str, Any] = {
        "rows": int(len(labels)),
        "counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "active_ratio": float(active.mean()) if len(active) else 0.0,
    }
    if "zigzag_wave_bars" in labels.columns:
        bars = pd.to_numeric(labels.loc[active, "zigzag_wave_bars"], errors="coerce").dropna().astype(float)
        out["active_wave_bars"] = {
            "mean": float(bars.mean()) if len(bars) else 0.0,
            "p50": float(bars.quantile(0.50)) if len(bars) else 0.0,
            "p95": float(bars.quantile(0.95)) if len(bars) else 0.0,
            "max": float(bars.max()) if len(bars) else 0.0,
        }
    if "zigzag_multithreshold_consensus_score" in labels.columns:
        score = pd.to_numeric(labels["zigzag_multithreshold_consensus_score"], errors="coerce").fillna(0.0)
        out["consensus_score"] = {
            "mean": float(score.mean()),
            "p50": float(score.quantile(0.50)),
            "p90": float(score.quantile(0.90)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--short-threshold", type=float, default=0.003)
    ap.add_argument("--medium-threshold", type=float, default=0.008)
    ap.add_argument("--long-threshold", type=float, default=0.015)
    ap.add_argument("--short-max-bars", type=int, default=24)
    ap.add_argument("--medium-max-bars", type=int, default=48)
    ap.add_argument("--long-max-bars", type=int, default=96)
    ap.add_argument("--quality-min-score", type=float, default=0.667)
    ap.add_argument("--min-wave-bars", type=int, default=3)
    ap.add_argument("--transition-buffer", type=int, default=2)
    ap.add_argument("--mae-penalty", type=float, default=1.35)
    ap.add_argument("--softmax-temperature", type=float, default=1.75)
    ap.add_argument("--min-risk-floor", type=float, default=0.0010)
    ap.add_argument("--min-edge-pct", type=float, default=0.0015)
    ap.add_argument("--min-calmar", type=float, default=0.25)
    ap.add_argument("--min-mfe-efficiency", type=float, default=0.45)
    ap.add_argument("--min-phase", type=float, default=0.04)
    ap.add_argument("--max-phase", type=float, default=0.82)
    args = ap.parse_args()

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    stack_dir = out_root / "stack"
    direction_dir = out_root / "direction_medium"
    quality_dir = out_root / "quality_consensus"
    for d in (stack_dir, direction_dir, quality_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = {
        "short": {"threshold": float(args.short_threshold), "max_bars": int(args.short_max_bars)},
        "medium": {"threshold": float(args.medium_threshold), "max_bars": int(args.medium_max_bars)},
        "long": {"threshold": float(args.long_threshold), "max_bars": int(args.long_max_bars)},
    }
    audit: dict[str, Any] = {
        "type": "zigzag_multithreshold_time_constrained_stack",
        "params": {
            **cfg,
            "quality_min_score": float(args.quality_min_score),
            "min_wave_bars": int(args.min_wave_bars),
            "transition_buffer": int(args.transition_buffer),
            "mae_penalty": float(args.mae_penalty),
            "softmax_temperature": float(args.softmax_temperature),
            "min_risk_floor": float(args.min_risk_floor),
            "min_edge_pct": float(args.min_edge_pct),
            "min_calmar": float(args.min_calmar),
            "min_mfe_efficiency": float(args.min_mfe_efficiency),
            "min_phase": float(args.min_phase),
            "max_phase": float(args.max_phase),
        },
        "contract": {
            "direction_label_dir": str(direction_dir),
            "quality_label_dir": str(quality_dir),
            "direction_target": "medium threshold time-constrained zigzag_action",
            "quality_target": "medium action retained only when short/medium/long consensus_score >= quality_min_score",
            "thresholds_are_fixed_pct": True,
            "atr_multiplier": 0.0,
            "uses_future_only_for_offline_labeling": True,
        },
        "artifacts": {},
        "summaries": {},
    }

    for year, path in PRICE_FILES.items():
        frame = zigzag_v2._read_frame(path, expected_year=int(year))
        short = _build_one(
            frame,
            threshold=float(args.short_threshold),
            max_bars=int(args.short_max_bars),
            min_wave_bars=int(args.min_wave_bars),
            transition_buffer=int(args.transition_buffer),
            mae_penalty=float(args.mae_penalty),
            softmax_temperature=float(args.softmax_temperature),
            min_risk_floor=float(args.min_risk_floor),
            min_edge_pct=float(args.min_edge_pct),
            min_calmar=float(args.min_calmar),
            min_mfe_efficiency=float(args.min_mfe_efficiency),
            min_phase=float(args.min_phase),
            max_phase=float(args.max_phase),
        )
        medium = _build_one(
            frame,
            threshold=float(args.medium_threshold),
            max_bars=int(args.medium_max_bars),
            min_wave_bars=int(args.min_wave_bars),
            transition_buffer=int(args.transition_buffer),
            mae_penalty=float(args.mae_penalty),
            softmax_temperature=float(args.softmax_temperature),
            min_risk_floor=float(args.min_risk_floor),
            min_edge_pct=float(args.min_edge_pct),
            min_calmar=float(args.min_calmar),
            min_mfe_efficiency=float(args.min_mfe_efficiency),
            min_phase=float(args.min_phase),
            max_phase=float(args.max_phase),
        )
        long = _build_one(
            frame,
            threshold=float(args.long_threshold),
            max_bars=int(args.long_max_bars),
            min_wave_bars=int(args.min_wave_bars),
            transition_buffer=int(args.transition_buffer),
            mae_penalty=float(args.mae_penalty),
            softmax_temperature=float(args.softmax_temperature),
            min_risk_floor=float(args.min_risk_floor),
            min_edge_pct=float(args.min_edge_pct),
            min_calmar=float(args.min_calmar),
            min_mfe_efficiency=float(args.min_mfe_efficiency),
            min_phase=float(args.min_phase),
            max_phase=float(args.max_phase),
        )
        stack = frame[["timestamp", "open", "high", "low", "close"]].copy()
        for part in (_prefixed(short, "short"), _prefixed(medium, "medium"), _prefixed(long, "long")):
            stack = stack.merge(part, on="timestamp", how="inner", validate="one_to_one")
        direction = _standard_export(frame, medium, stack)
        quality = _quality_export(medium, stack, min_score=float(args.quality_min_score))

        stack_path = stack_dir / f"zigzag_action_labels_{year}.csv"
        direction_path = direction_dir / f"zigzag_action_labels_{year}.csv"
        quality_path = quality_dir / f"zigzag_action_labels_{year}.csv"
        stack.to_csv(stack_path, index=False)
        direction.to_csv(direction_path, index=False)
        quality.to_csv(quality_path, index=False)
        audit["artifacts"][str(year)] = {
            "stack": str(stack_path),
            "direction": str(direction_path),
            "quality": str(quality_path),
        }
        audit["summaries"][str(year)] = {
            "short": _summary(short),
            "medium_direction": _summary(direction),
            "long": _summary(long),
            "quality_consensus": _summary(quality),
        }

    for d, role in ((direction_dir, "direction_medium"), (quality_dir, "quality_consensus")):
        role_audit = dict(audit)
        role_audit["role"] = role
        role_audit["artifacts"] = {
            str(year): str(d / f"zigzag_action_labels_{year}.csv") for year in PRICE_FILES
        }
        path = d / "zigzag_action_label_audit.json"
        role_audit["artifacts"]["audit"] = str(path)
        path.write_text(json.dumps(role_audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    audit_path = out_root / "stack_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(audit_path), "direction_label_dir": str(direction_dir), "quality_label_dir": str(quality_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
