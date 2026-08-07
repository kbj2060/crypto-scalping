#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_dir3_cycle_20260531"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_cycle_20260531"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_cycle_20260531"

OUTPUT_COLS = [
    "dir3_cycle_h6_fl_prob",
    "dir3_cycle_h6_up_prob",
    "dir3_cycle_h6_dn_prob",
    "dir3_cycle_h6_confidence",
    "dir3_cycle_h6_side_edge",
    "dir3_cycle_h6_trade_prob",
    "dir3_cycle_h6_group_support",
]

REGIME3_CURRENT_FEATURES = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]


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


def _split_file(split_dir: Path, year: int) -> Path:
    return split_dir / ("training_features_2026_rebuilt.csv" if int(year) == 2026 else f"training_features_{int(year)}.csv")


def _read_base(split_dir: Path, year: int) -> pd.DataFrame:
    path = _split_file(split_dir, year)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], source: str, *, allow_tail_drop: bool = False) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    miss = merged[cols].isna().any(axis=1).to_numpy()
    if miss.any():
        idx = np.flatnonzero(miss)
        tail_only = np.array_equal(idx, np.arange(len(merged) - idx.size, len(merged)))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - idx.size].reset_index(drop=True)
        missing = {c: int(merged[c].isna().sum()) for c in cols if int(merged[c].isna().sum())}
        raise RuntimeError(f"{source} exact join missing values: {missing}")
    return merged


def _add_label(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{int(year)}.csv"
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    return _exact_join(frame, labels, ["zigzag_action"], f"ZigZag labels {year}")


def _add_regime3_current(frame: pd.DataFrame, regime_dir: Path, year: int) -> pd.DataFrame:
    name = "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv" if int(year) == 2026 else f"training_features_{int(year)}_regime3_current_sensitive_hmm_wide24.csv"
    side = pd.read_csv(regime_dir / name, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CURRENT_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{regime_dir / name} missing {missing}")
    return _exact_join(frame, side, REGIME3_CURRENT_FEATURES, f"Regime3 current {year}", allow_tail_drop=True)


def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["_hour"] = ts.dt.hour.astype("int16")
    out["_funding_phase"] = (ts.dt.hour % 8).astype("int16")
    out["_weekday"] = ts.dt.dayofweek.astype("int16")
    probs = out[
        [
            "regime3_current_sensitive_wide24_bull_prob",
            "regime3_current_sensitive_wide24_bear_prob",
            "regime3_current_sensitive_wide24_chop_prob",
        ]
    ].to_numpy(float)
    out["_regime3_current_id"] = probs.argmax(axis=1).astype("int16")
    vol_src = "atr_pct_rank_288" if "atr_pct_rank_288" in out.columns else "volatility_z"
    if vol_src in out.columns:
        vol = pd.to_numeric(out[vol_src], errors="coerce").replace([np.inf, -np.inf], np.nan)
        bins = pd.qcut(vol.rank(method="first"), 4, labels=False, duplicates="drop")
        out["_vol_bin"] = pd.Series(bins, index=out.index).fillna(0).astype("int16")
    else:
        out["_vol_bin"] = np.int16(0)
    return out


def _counts(train: pd.DataFrame, keys: list[str], alpha: float) -> tuple[dict[tuple[Any, ...], np.ndarray], np.ndarray]:
    y = pd.to_numeric(train["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    global_counts = np.bincount(y, minlength=3).astype(float) + float(alpha)
    table: dict[tuple[Any, ...], np.ndarray] = {}
    for key, grp in train.groupby(keys, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        counts = np.bincount(grp["zigzag_action"].astype(int).to_numpy(), minlength=3).astype(float) + float(alpha)
        table[key] = counts
    return table, global_counts


def _score(train: pd.DataFrame, target: pd.DataFrame, keys: list[str], *, alpha: float, min_support: int) -> pd.DataFrame:
    table, global_counts = _counts(train, keys, alpha)
    backoff_keys = keys[:1]
    backoff, _ = _counts(train, backoff_keys, alpha)
    probs = np.zeros((len(target), 3), dtype=float)
    support = np.zeros(len(target), dtype=float)
    for i, row in enumerate(target[keys].itertuples(index=False, name=None)):
        counts = table.get(tuple(row))
        if counts is None or counts.sum() - 3.0 * alpha < min_support:
            bkey = tuple(row[:1])
            counts = backoff.get(bkey, global_counts)
        support[i] = max(0.0, counts.sum() - 3.0 * alpha)
        probs[i] = counts / counts.sum()
    out = pd.DataFrame(
        {
            "timestamp": target["timestamp"].to_numpy(),
            "dir3_cycle_h6_fl_prob": probs[:, 0],
            "dir3_cycle_h6_up_prob": probs[:, 1],
            "dir3_cycle_h6_dn_prob": probs[:, 2],
            "dir3_cycle_h6_confidence": probs.max(axis=1),
            "dir3_cycle_h6_side_edge": probs[:, 1] - probs[:, 2],
            "dir3_cycle_h6_trade_prob": probs[:, 1] + probs[:, 2],
            "dir3_cycle_h6_group_support": support,
        }
    )
    if out[OUTPUT_COLS].isna().any().any():
        raise RuntimeError("dir3 cycle output contains NaN")
    return out


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    df = scored.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    y = df["zigzag_action"].astype(int).to_numpy()
    proba = df[["dir3_cycle_h6_fl_prob", "dir3_cycle_h6_up_prob", "dir3_cycle_h6_dn_prob"]].to_numpy(float)
    pred = proba.argmax(axis=1)
    trade_mask = pred != 0
    trade_count = int(trade_mask.sum())
    return {
        "rows": int(len(df)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "proxy_trades": trade_count,
        "proxy_long_trades": int((pred == 1).sum()),
        "proxy_short_trades": int((pred == 2).sum()),
        "proxy_trade_rate": float(trade_count / len(df)) if len(df) else None,
        "proxy_wr": float((pred[trade_mask] == y[trade_mask]).mean()) if trade_count else None,
        "mean_confidence": float(proba.max(axis=1).mean()),
        "mean_trade_prob": float((proba[:, 1] + proba[:, 2]).mean()),
        "ovr_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    frames: dict[int, pd.DataFrame] = {}
    for year in [2024, 2025, 2026]:
        frame = _read_base(args.split_dir, year)
        frame = _add_label(frame, args.label_dir, year)
        frame = _add_regime3_current(frame, args.regime3_current_dir, year)
        frames[year] = _decorate(frame)

    variants = {
        "time_funding": ["_hour", "_funding_phase"],
        "time_funding_weekday": ["_hour", "_funding_phase", "_weekday"],
        "time_regime": ["_hour", "_regime3_current_id"],
        "time_regime_vol": ["_hour", "_regime3_current_id", "_vol_bin"],
        "funding_regime_vol": ["_funding_phase", "_regime3_current_id", "_vol_bin"],
    }
    grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_scored: tuple[pd.DataFrame, pd.DataFrame] | None = None
    for name, keys in variants.items():
        for alpha in [5.0, 20.0, 80.0]:
            for min_support in [30, 100, 300]:
                scored_2025 = _score(frames[2024], frames[2025], keys, alpha=alpha, min_support=min_support)
                scored_2026 = _score(frames[2024], frames[2026], keys, alpha=alpha, min_support=min_support)
                m25 = _metrics(scored_2025, frames[2025])
                m26 = _metrics(scored_2026, frames[2026])
                row = {
                    "variant": name,
                    "keys": keys,
                    "alpha": float(alpha),
                    "min_support": int(min_support),
                    "selection_score_2025": float(m25["balanced_accuracy"] + 0.25 * m25["ovr_auc"]),
                    "metrics_2025": m25,
                    "metrics_2026": m26,
                }
                grid.append(row)
                if best is None or row["selection_score_2025"] > best["selection_score_2025"]:
                    best = row
                    best_scored = (scored_2025, scored_2026)

    if best is None or best_scored is None:
        raise RuntimeError("no dir3 cycle candidate was produced")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_2025 = args.out_dir / f"training_features_2025_{MODEL_ID}.csv"
    out_2026 = args.out_dir / f"training_features_2026_rebuilt_{MODEL_ID}.csv"
    best_scored[0].to_csv(out_2025, index=False)
    best_scored[1].to_csv(out_2026, index=False)
    grid_sorted = sorted(grid, key=lambda r: r["selection_score_2025"], reverse=True)
    audit = {
        "model_id": MODEL_ID,
        "role": "Omega1 third-stage cycle/session direction context",
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "selected": best,
        "top20": grid_sorted[:20],
        "artifacts": {"features_2025": str(out_2025), "features_2026": str(out_2026)},
        "outputs": OUTPUT_COLS,
        "contract": {
            "allowed_inputs": "timestamp-derived cycle keys plus current Regime3 exact timestamp join",
            "forbidden_inputs": ["teacher_*", "a5dir_*", "Regime4", "regime3_pred_*", "label/target/future/PnL/action_score"],
        },
    }
    audit_path = args.report_dir / "dir3_cycle_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "variant": r["variant"],
                "alpha": r["alpha"],
                "min_support": r["min_support"],
                "selection_score_2025": r["selection_score_2025"],
                "bacc_2025": r["metrics_2025"]["balanced_accuracy"],
                "auc_2025": r["metrics_2025"]["ovr_auc"],
                "bacc_2026": r["metrics_2026"]["balanced_accuracy"],
                "auc_2026": r["metrics_2026"]["ovr_auc"],
                "proxy_wr_2026": r["metrics_2026"]["proxy_wr"],
                "proxy_trades_2026": r["metrics_2026"]["proxy_trades"],
            }
            for r in grid_sorted
        ]
    ).to_csv(args.report_dir / "dir3_cycle_grid.csv", index=False)
    print(json.dumps({"audit": str(audit_path), "selected": best, "artifacts": audit["artifacts"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
