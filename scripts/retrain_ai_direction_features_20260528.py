#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "ai_direction_retrain_20260528"
DEFAULT_OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}_v2_noleak"

FORBIDDEN_TOKENS = (
    "label",
    "target",
    "future",
    "fwd_",
    "realized",
    "pnl",
    "exit_reason",
    "dir_label",
    "dir_edge",
    "dir_long",
    "dir_short",
    "dir_valid",
)
FORBIDDEN_PREFIXES = (
    "ai_",
    "teacher_",
    "m7_",
    "pred_patchtst",
    "conf_patchtst",
    "patchtst_",
    "tide_",
    "timesnet_",
    "dlinear_",
)
RAW_LEVEL_COLS = {
    "open",
    "high",
    "low",
    "close",
    "close_btc",
    "volume",
    "quote_volume",
    "volume_btc",
    "quote_volume_btc",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
}
ID_COLS = {"timestamp", "symbol", "open_time", "close_time", "ignore"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild direction labels and retrain causal AI direction feature models.")
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--train-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--prefix", default="ai_dir_v2")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--min-edge", type=float, default=0.0012)
    p.add_argument("--atr-mult", type=float, default=0.22)
    p.add_argument("--cost", type=float, default=0.00055)
    p.add_argument("--mae-penalty", type=float, default=0.55)
    p.add_argument("--direction-margin", type=float, default=0.00035)
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--random-seed", type=int, default=20260528)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


def _read_year(path: Path, *, limit: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if limit > 0:
        frame = frame.tail(int(limit)).reset_index(drop=True)
    if "timestamp" not in frame.columns:
        raise KeyError(f"{path} missing timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    required = {"open", "high", "low", "close", "quote_volume", "taker_buy_quote"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path} missing required market columns: {missing}")
    return frame


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work = FeatureEngineer(keep_only_active=False)._create_directional_alpha_features(work)
    return work.replace([np.inf, -np.inf], np.nan)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _future_rolling_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(f"unknown mode: {mode}")


def _make_direction_labels(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    close = _num(frame, "close").clip(lower=1e-12)
    high = _num(frame, "high", close)
    low = _num(frame, "low", close)

    prev_close = close.shift(1).fillna(close)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=3).mean() / close).replace([np.inf, -np.inf], np.nan).fillna(0.001)
    edge_floor = np.maximum(float(args.min_edge), np.maximum(float(args.cost) * 2.0, atr_pct.to_numpy() * float(args.atr_mult)))

    fut_high = _future_rolling_extreme(high, int(args.horizon), "max")
    fut_low = _future_rolling_extreme(low, int(args.horizon), "min")
    fut_close = close.shift(-int(args.horizon))

    long_mfe = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    long_mae = (1.0 - fut_low / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    short_mfe = (1.0 - fut_low / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    short_mae = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vertical_ret = (fut_close / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    long_score = long_mfe - float(args.mae_penalty) * long_mae - float(args.cost)
    short_score = short_mfe - float(args.mae_penalty) * short_mae - float(args.cost)
    margin = float(args.direction_margin)

    label = np.zeros(len(frame), dtype=np.int64)
    long_ok = (long_score.to_numpy() - short_score.to_numpy() > margin) & (long_score.to_numpy() > edge_floor)
    short_ok = (short_score.to_numpy() - long_score.to_numpy() > margin) & (short_score.to_numpy() > edge_floor)
    label[short_ok] = 1
    label[long_ok] = 2

    # The final horizon rows cannot be causally labeled; keep them out of training/metrics.
    valid = np.ones(len(frame), dtype=bool)
    if int(args.horizon) > 0:
        valid[-int(args.horizon) :] = False

    out = frame.copy()
    out["dir_label"] = label
    out["dir_valid"] = valid.astype(np.int8)
    out["dir_long_score"] = long_score.to_numpy(dtype=np.float64)
    out["dir_short_score"] = short_score.to_numpy(dtype=np.float64)
    out["dir_edge_score"] = (long_score - short_score).to_numpy(dtype=np.float64)
    out[f"fwd_ret_{int(args.horizon)}"] = vertical_ret.to_numpy(dtype=np.float64)
    return out


def _allowed_feature(col: str, s: pd.Series) -> bool:
    if col in ID_COLS or col in RAW_LEVEL_COLS:
        return False
    lc = col.lower()
    if lc.startswith("dir_"):
        return False
    if any(lc.startswith(p) for p in FORBIDDEN_PREFIXES):
        return False
    if any(tok in lc for tok in FORBIDDEN_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(s)


def _feature_cols(train: pd.DataFrame, score_a: pd.DataFrame, score_b: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in train.columns:
        if c not in score_a.columns or c not in score_b.columns:
            continue
        if not _allowed_feature(str(c), train[c]):
            continue
        x = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if float(x.notna().mean()) < 0.95:
            continue
        if float(x.fillna(0.0).std(ddof=0)) <= 1e-10:
            continue
        cols.append(str(c))
    return sorted(cols)


def _fit_fill_values(train: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for c in cols:
        x = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        v = float(x.median()) if x.notna().any() else 0.0
        values[c] = v if math.isfinite(v) else 0.0
    return values


def _matrix(frame: pd.DataFrame, cols: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    data = {}
    for c in cols:
        data[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill_values[c])
    return pd.DataFrame(data, index=frame.index)


def _class_weights(y: np.ndarray) -> list[float]:
    counts = np.bincount(y.astype(int), minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    inv = np.sqrt(counts.sum() / (3.0 * counts))
    return [float(x) for x in inv]


def _train_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    fill_values: dict[str, float],
    args: argparse.Namespace,
    model_path: Path,
) -> CatBoostClassifier:
    train_valid = train[train["dir_valid"] > 0].reset_index(drop=True)
    split = int(len(train_valid) * 0.82)
    fit_df = train_valid.iloc[:split].reset_index(drop=True)
    hold_df = train_valid.iloc[split:].reset_index(drop=True)
    x_fit = _matrix(fit_df, cols, fill_values)
    y_fit = fit_df["dir_label"].to_numpy(dtype=np.int64)
    x_hold = _matrix(hold_df, cols, fill_values)
    y_hold = hold_df["dir_label"].to_numpy(dtype=np.int64)

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(args.random_seed),
        "class_weights": _class_weights(y_fit),
        "allow_writing_files": False,
        "verbose": 100,
        "task_type": str(args.task_type),
    }
    model = CatBoostClassifier(**params)
    try:
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True, early_stopping_rounds=80)
    except Exception as exc:
        if str(args.task_type) != "GPU":
            raise
        print(f"[WARN] GPU CatBoost failed, retrying CPU: {exc}")
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True, early_stopping_rounds=80)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return model


def _score_frame(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    cols: list[str],
    fill_values: dict[str, float],
    prefix: str,
) -> pd.DataFrame:
    x = _matrix(frame, cols, fill_values)
    prob = model.predict_proba(x)
    if prob.shape[1] != 3:
        raise RuntimeError(f"Unexpected probability output shape: {prob.shape}")
    eps = 1e-12
    entropy = -(prob * np.log(np.clip(prob, eps, 1.0))).sum(axis=1) / math.log(3.0)
    out = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_p_down": prob[:, 1],
            f"{prefix}_p_flat": prob[:, 0],
            f"{prefix}_p_up": prob[:, 2],
            f"{prefix}_edge": prob[:, 2] - prob[:, 1],
            f"{prefix}_entropy": entropy,
        }
    )
    for c in ("dir_label", "dir_valid", "dir_edge_score"):
        if c in frame.columns:
            out[c] = frame[c].to_numpy()
    for c in frame.columns:
        if c.startswith("fwd_ret_"):
            out[c] = frame[c].to_numpy()
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 30:
        return float("nan")
    return float(x[m].rank().corr(y[m].rank()))


def _metrics(scored: pd.DataFrame, prefix: str, name: str, horizon: int) -> dict[str, Any]:
    valid = scored[scored.get("dir_valid", 1) > 0].copy()
    y = valid["dir_label"].to_numpy(dtype=np.int64)
    edge = valid[f"{prefix}_edge"].to_numpy(dtype=np.float64)
    pred = np.argmax(
        valid[[f"{prefix}_p_flat", f"{prefix}_p_down", f"{prefix}_p_up"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    out: dict[str, Any] = {
        "name": name,
        "rows": int(len(valid)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "edge_ic_fwd_ret": _spearman(pd.Series(edge), valid[f"fwd_ret_{int(horizon)}"]),
        "edge_ic_label_edge": _spearman(pd.Series(edge), valid["dir_edge_score"]),
    }
    try:
        prob = valid[[f"{prefix}_p_flat", f"{prefix}_p_down", f"{prefix}_p_up"]].to_numpy(dtype=np.float64)
        out["ovr_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", labels=[0, 1, 2]))
    except Exception as exc:
        out["ovr_auc_error"] = str(exc)
    nonflat = valid[y != 0]
    if len(nonflat) > 30 and len(np.unique(nonflat["dir_label"])) == 2:
        out["nonflat_updown_auc"] = float(
            roc_auc_score((nonflat["dir_label"].to_numpy(dtype=np.int64) == 2).astype(int), nonflat[f"{prefix}_edge"])
        )
    if "ai_dir_edge" in valid.columns:
        out["legacy_ai_edge_ic_fwd_ret"] = _spearman(valid["ai_dir_edge"], valid[f"fwd_ret_{int(horizon)}"])
    return out


def _run_chain(
    *,
    chain_name: str,
    train_raw: pd.DataFrame,
    score_raw: pd.DataFrame,
    extra_raw: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"[{chain_name}] augmenting frames")
    train = _make_direction_labels(_augment(train_raw), args)
    score = _make_direction_labels(_augment(score_raw), args)
    extra = _make_direction_labels(_augment(extra_raw), args)
    cols = _feature_cols(train, score, extra)
    if not cols:
        raise RuntimeError(f"{chain_name}: no usable feature columns")
    fill_values = _fit_fill_values(train, cols)

    chain_dir = out_dir / chain_name
    model_path = chain_dir / "catboost_ai_direction.cbm"
    print(f"[{chain_name}] training rows={int((train['dir_valid'] > 0).sum())} features={len(cols)}")
    model = _train_model(train, score, cols, fill_values, args, model_path)

    score_out = _score_frame(model, score, cols, fill_values, str(args.prefix))
    extra_out = _score_frame(model, extra, cols, fill_values, str(args.prefix))
    chain_dir.mkdir(parents=True, exist_ok=True)
    score_out.to_csv(chain_dir / "score_primary.csv", index=False)
    extra_out.to_csv(chain_dir / "score_extra.csv", index=False)
    (chain_dir / "feature_cols.json").write_text(json.dumps(cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (chain_dir / "fill_values.json").write_text(json.dumps(fill_values, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "chain": chain_name,
        "model_path": str(model_path),
        "feature_count": int(len(cols)),
        "train_label_counts": {
            str(i): int(v)
            for i, v in enumerate(np.bincount(train.loc[train["dir_valid"] > 0, "dir_label"].to_numpy(dtype=np.int64), minlength=3))
        },
        "primary_metrics": _metrics(score_out, str(args.prefix), "primary", int(args.horizon)),
        "extra_metrics": _metrics(extra_out, str(args.prefix), "extra", int(args.horizon)),
    }
    (chain_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return metrics


def _write_report(out_dir: Path, metrics: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        f"# {MODEL_ID}",
        "",
        "## Diagnosis",
        "",
        "- Existing `ai_dir_*` is produced by the NeuralForecast PatchTST wrapper from a scalar edge forecast and then converted to pseudo probabilities.",
        "- The retrain here builds explicit cost/path-aware direction labels and trains causal CatBoost multiclass direction models.",
        "- Active `data/nf_*` artifacts are not overwritten; generated outputs use the new `ai_dir_v2_*` prefix.",
        "",
        "## Label Contract",
        "",
        f"- horizon bars: `{int(args.horizon)}`",
        f"- edge floor: `max(min_edge={float(args.min_edge)}, 2*cost={float(args.cost) * 2.0}, atr_pct*{float(args.atr_mult)})`",
        f"- score: `MFE - {float(args.mae_penalty)} * MAE - cost`",
        f"- direction margin: `{float(args.direction_margin)}`",
        "",
        "## Results",
        "",
    ]
    for m in metrics:
        lines.append(f"### {m['chain']}")
        lines.append("")
        lines.append(f"- features: `{m['feature_count']}`")
        lines.append(f"- train labels: `{m['train_label_counts']}`")
        for key in ("primary_metrics", "extra_metrics"):
            r = m[key]
            lines.append(
                "- "
                + key
                + f": rows={r.get('rows')} labels={r.get('label_counts')} "
                + f"bal_acc={float(r.get('balanced_accuracy', float('nan'))):.4f} "
                + f"ovr_auc={float(r.get('ovr_auc', float('nan'))):.4f} "
                + f"edge_ic_fwd={float(r.get('edge_ic_fwd_ret', float('nan'))):.4f} "
                + f"edge_ic_label={float(r.get('edge_ic_label_edge', float('nan'))):.4f}"
            )
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: retrain_ai_direction_features_20260528")
        return 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_2024 = _read_year(args.train_2024, limit=int(args.limit))
    score_2025 = _read_year(args.score_2025, limit=int(args.limit))
    train_2025 = _read_year(args.train_2025, limit=int(args.limit))
    score_2026 = _read_year(args.score_2026, limit=int(args.limit))

    metrics = [
        _run_chain(
            chain_name="fit2024_score2025",
            train_raw=train_2024,
            score_raw=score_2025,
            extra_raw=score_2026,
            args=args,
            out_dir=out_dir,
        ),
        _run_chain(
            chain_name="fit2025_score2026",
            train_raw=train_2025,
            score_raw=score_2026,
            extra_raw=score_2025,
            args=args,
            out_dir=out_dir,
        ),
    ]
    summary = {
        "model_id": MODEL_ID,
        "out_dir": str(out_dir),
        "args": vars(args),
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_report(out_dir, metrics, args)
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
