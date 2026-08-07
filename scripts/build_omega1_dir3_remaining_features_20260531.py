#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_dir3_remaining_20260531"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_BASE_META_DIR = ROOT / "tmp/causal_regen_20260516/omega1_hgb_teacher_m7zigzag_20260531"
DEFAULT_OUT_ROOT = ROOT / "data/ensemble/supervised"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_remaining_20260531"

CLASS_COUNT = 3
SEQ_LEN = 72

CHART_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "sum_open_interest_value",
    "last_funding_rate",
    "log_return",
    "volatility_z",
    "rsi",
    "wick_ratio",
    "btc_ret_1",
    "btc_ret_3",
    "cvd_12",
    "cvd_slope_12",
]

PATCH_COLS = [
    "log_return",
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "garman_klass_vol",
    "realized_vol_ratio",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "amihud_illiquidity_z",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "chop_index",
    "cvp_poc_dist",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "cvp_regime",
    "breakout_strength",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "funding_price_divergence",
    "regime_trending",
    "ofi_acceleration",
    "kalman_velocity",
    "ofti",
    "kel",
    "funding_abs",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_slope_12",
    "cvd_slope_48",
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_6",
    "eth_btc_ret_spread_12",
    "compression_score",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "funding_oi_divergence",
    "oi_up_price_down",
    "oi_up_price_up",
    "upper_wick_z",
    "lower_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "garch_vol_z",
    "jump_z",
    "evt_excess_z",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
    "regime_persistence",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]

FORBIDDEN_PREFIXES = ("teacher_", "a5dir_", "clean_regime_", "clean_regime4_", "regime4_pred_", "regime3_pred_", "dir3_")
FORBIDDEN_SUBSTRINGS = ("label", "target", "future", "pnl", "action_score")

OUTPUTS = {
    "chartcnn": [
        "dir3_chartcnn_h6_fl_prob",
        "dir3_chartcnn_h6_up_prob",
        "dir3_chartcnn_h6_dn_prob",
        "dir3_chartcnn_h6_confidence",
        "dir3_chartcnn_h6_side_edge",
        "dir3_chartcnn_h6_trade_prob",
    ],
    "patch": [
        "dir3_patch_h6_fl_prob",
        "dir3_patch_h6_up_prob",
        "dir3_patch_h6_dn_prob",
        "dir3_patch_h6_confidence",
        "dir3_patch_h6_side_edge",
        "dir3_patch_h6_trade_prob",
    ],
    "duet": [
        "dir3_duet_h6_fl_prob",
        "dir3_duet_h6_up_prob",
        "dir3_duet_h6_dn_prob",
        "dir3_duet_h6_confidence",
        "dir3_duet_h6_side_edge",
        "dir3_duet_h6_trade_prob",
    ],
}


class ChartCNN(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 48, kernel_size=5, padding=2),
            torch.nn.SiLU(),
            torch.nn.BatchNorm1d(48),
            torch.nn.Conv1d(48, 64, kernel_size=5, padding=2),
            torch.nn.SiLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Conv1d(64, 96, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.head = torch.nn.Linear(96, CLASS_COUNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)
        return self.head(z)


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


def _exact_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], source: str) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    missing = {c: int(merged[c].isna().sum()) for c in cols if int(merged[c].isna().sum())}
    if missing:
        raise RuntimeError(f"{source} exact join missing values: {missing}")
    return merged


def _add_label(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{int(year)}.csv"
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    return _exact_join(frame, labels, ["zigzag_action"], f"ZigZag labels {year}")


def _safe_numeric_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        lower = col.lower()
        if col == "timestamp" or col.startswith(FORBIDDEN_PREFIXES):
            continue
        if any(token in lower for token in FORBIDDEN_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _check_cols(cols: list[str]) -> None:
    bad: list[str] = []
    for col in cols:
        lower = col.lower()
        if col.startswith(FORBIDDEN_PREFIXES) or any(token in lower for token in FORBIDDEN_SUBSTRINGS):
            bad.append(col)
    if bad:
        raise ValueError(f"forbidden inputs selected: {bad[:40]}")


def _sequence_array(frame: pd.DataFrame, cols: list[str], scaler: RobustScaler | None = None) -> tuple[np.ndarray, RobustScaler]:
    raw = frame[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)
        raw = scaler.fit_transform(raw).astype(np.float32)
    else:
        raw = scaler.transform(raw).astype(np.float32)
    n = len(raw) - SEQ_LEN + 1
    if n <= 0:
        raise RuntimeError("not enough rows for sequence")
    seq = np.lib.stride_tricks.sliding_window_view(raw, SEQ_LEN, axis=0)
    seq = seq.copy()
    return seq, scaler


def _train_chartcnn(train: pd.DataFrame, cols: list[str], *, seed: int, epochs: int, batch_size: int) -> dict[str, Any]:
    _check_cols(cols)
    seq, scaler = _sequence_array(train, cols, None)
    y = train["zigzag_action"].astype(int).to_numpy()[SEQ_LEN - 1 :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    model = ChartCNN(len(cols)).to(device)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    x_t = torch.tensor(seq, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    n = len(y)
    rng = np.random.default_rng(int(seed))
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = x_t[idx].to(device, non_blocking=True)
            yb = y_t[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    return {"model": model, "scaler": scaler, "cols": cols, "device": str(device)}


def _score_chartcnn(frame: pd.DataFrame, artifact: dict[str, Any], batch_size: int) -> pd.DataFrame:
    seq, _ = _sequence_array(frame, artifact["cols"], artifact["scaler"])
    model: ChartCNN = artifact["model"]
    device = next(model.parameters()).device
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq), int(batch_size)):
            xb = torch.tensor(seq[start : start + int(batch_size)], dtype=torch.float32, device=device)
            p = torch.softmax(model(xb), dim=1).cpu().numpy()
            probs.append(p)
    proba = np.vstack(probs)
    out = pd.DataFrame({"timestamp": frame["timestamp"].iloc[SEQ_LEN - 1 :].to_numpy()})
    out["dir3_chartcnn_h6_fl_prob"] = proba[:, 0]
    out["dir3_chartcnn_h6_up_prob"] = proba[:, 1]
    out["dir3_chartcnn_h6_dn_prob"] = proba[:, 2]
    out["dir3_chartcnn_h6_confidence"] = proba.max(axis=1)
    out["dir3_chartcnn_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out["dir3_chartcnn_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


def _patch_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    _check_cols(cols)
    out = pd.DataFrame({"timestamp": frame["timestamp"]})
    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        out[f"{col}__last"] = s.astype("float32")
        for win in [6, 12, 24, 72]:
            roll = s.rolling(win, min_periods=max(2, win // 3))
            mean = roll.mean()
            out[f"{col}__m{win}"] = mean.astype("float32")
            out[f"{col}__d{win}"] = (s - mean).astype("float32")
        for win in [12, 72]:
            out[f"{col}__std{win}"] = s.rolling(win, min_periods=max(2, win // 3)).std().fillna(0.0).astype("float32")
    out = out.iloc[SEQ_LEN - 1 :].reset_index(drop=True)
    return out


def _fit_hgb(train_x: pd.DataFrame, train_y: np.ndarray, *, seed: int, max_iter: int = 220) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=int(max_iter),
                    learning_rate=0.035,
                    max_leaf_nodes=31,
                    l2_regularization=0.08,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=25,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(train_x, train_y, hgb__sample_weight=compute_sample_weight(class_weight="balanced", y=train_y))
    return model


def _append_proba(prefix: str, timestamps: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": timestamps.to_numpy()})
    out[f"dir3_{prefix}_h6_fl_prob"] = proba[:, 0]
    out[f"dir3_{prefix}_h6_up_prob"] = proba[:, 1]
    out[f"dir3_{prefix}_h6_dn_prob"] = proba[:, 2]
    out[f"dir3_{prefix}_h6_confidence"] = proba.max(axis=1)
    out[f"dir3_{prefix}_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out[f"dir3_{prefix}_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


def _score_model(prefix: str, model: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    input_x = x.drop(columns=["timestamp"], errors="ignore")
    proba_raw = model.predict_proba(input_x)
    classes = list(model.named_steps["hgb"].classes_)
    proba = np.zeros((len(x), 3), dtype=float)
    for i, cls in enumerate(classes):
        proba[:, int(cls)] = proba_raw[:, i]
    return _append_proba(prefix, x["timestamp"], proba)


def _duet_features(train: pd.DataFrame, frames: dict[int, pd.DataFrame], cols: list[str], *, threshold: float = 0.72) -> tuple[dict[int, pd.DataFrame], list[list[str]]]:
    _check_cols(cols)
    sample = train[cols].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(sample)
    scaled = RobustScaler(quantile_range=(5, 95), unit_variance=True).fit_transform(imputed)
    corr = np.corrcoef(scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = np.clip(1.0 - np.abs(corr), 0.0, 2.0)
    try:
        clusterer = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1.0 - threshold)
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=1.0 - threshold)
    labels = clusterer.fit_predict(dist)
    clusters: list[list[str]] = []
    for lab in sorted(set(labels.tolist())):
        members = [cols[i] for i, v in enumerate(labels) if v == lab]
        clusters.append(members)
    outputs: dict[int, pd.DataFrame] = {}
    for year, frame in frames.items():
        out = pd.DataFrame({"timestamp": frame["timestamp"]})
        for i, members in enumerate(clusters):
            vals = frame[members].replace([np.inf, -np.inf], np.nan).astype(float)
            out[f"duet_cluster_{i:03d}_mean"] = vals.mean(axis=1).astype("float32")
            if len(members) > 1:
                out[f"duet_cluster_{i:03d}_std"] = vals.std(axis=1).fillna(0.0).astype("float32")
        outputs[year] = out
    return outputs, clusters


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    df = scored.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    y = df["zigzag_action"].astype(int).to_numpy()
    proba = df[cols].to_numpy(float)
    pred = proba.argmax(axis=1)
    trade = pred != 0
    return {
        "rows": int(len(df)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ovr_auc": float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2])),
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "proxy_trades": int(trade.sum()),
        "proxy_long_trades": int((pred == 1).sum()),
        "proxy_short_trades": int((pred == 2).sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((pred[trade] == y[trade]).mean()) if trade.any() else None,
    }


def _write_feature_set(out_root: Path, name: str, scored_2025: pd.DataFrame, scored_2026: pd.DataFrame) -> dict[str, str]:
    out_dir = out_root / f"omega1_dir3_{name}_20260531"
    out_dir.mkdir(parents=True, exist_ok=True)
    p25 = out_dir / f"training_features_2025_omega1_dir3_{name}_20260531.csv"
    p26 = out_dir / f"training_features_2026_rebuilt_omega1_dir3_{name}_20260531.csv"
    scored_2025.to_csv(p25, index=False)
    scored_2026.to_csv(p26, index=False)
    return {"features_2025": str(p25), "features_2026": str(p26)}


def _load_meta_base(year: int, base_dir: Path, feature_dirs: dict[str, Path]) -> pd.DataFrame:
    name = "trade_candidates_2025_alpha6_current_tail111_exact.csv" if year == 2025 else "trade_candidates_2026_alpha6_current_tail111_exact.csv"
    frame = pd.read_csv(base_dir / name, parse_dates=["timestamp"], low_memory=False)
    for prefix, path in feature_dirs.items():
        fname = f"training_features_2025_omega1_dir3_{prefix}_20260531.csv" if year == 2025 else f"training_features_2026_rebuilt_omega1_dir3_{prefix}_20260531.csv"
        side = pd.read_csv(path / fname, parse_dates=["timestamp"])
        frame = frame.merge(side, on="timestamp", how="left", validate="one_to_one")
    return frame


def _meta_probe(base_dir: Path, feature_dirs: dict[str, Path], report_dir: Path) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.build_hgb_teacher_features_20260531 import OMEGA1_CORE_INPUTS

    train = _load_meta_base(2025, base_dir, feature_dirs)
    oos = _load_meta_base(2026, base_dir, feature_dirs)
    all_dir3_cols = [c for c in train.columns if c.startswith("dir3_")]
    miss = train[all_dir3_cols].isna().any(axis=1) | oos.iloc[:0].empty
    train = train.loc[~train[all_dir3_cols].isna().any(axis=1)].reset_index(drop=True)
    oos = oos.loc[~oos[all_dir3_cols].isna().any(axis=1)].reset_index(drop=True)
    base_cols = [c for c in OMEGA1_CORE_INPUTS if c in train.columns and c in oos.columns]
    groups = {
        "core_only": base_cols,
        "core_plus_chartcnn": base_cols + [c for c in all_dir3_cols if c.startswith("dir3_chartcnn_")],
        "core_plus_patch": base_cols + [c for c in all_dir3_cols if c.startswith("dir3_patch_")],
        "core_plus_duet": base_cols + [c for c in all_dir3_cols if c.startswith("dir3_duet_")],
        "core_plus_all_remaining": base_cols + all_dir3_cols,
    }
    y = train["zigzag_action"].astype(int).to_numpy()
    yo = oos["zigzag_action"].astype(int).to_numpy()

    def fit_eval(cols: list[str]) -> dict[str, Any]:
        model = _fit_hgb(train[cols], y, seed=20260531, max_iter=260)

        def met(df: pd.DataFrame, yy: np.ndarray) -> dict[str, Any]:
            raw = model.predict_proba(df[cols])
            classes = list(model.named_steps["hgb"].classes_)
            proba = np.zeros((len(df), 3), dtype=float)
            for i, cls in enumerate(classes):
                proba[:, int(cls)] = raw[:, i]
            pred = proba.argmax(axis=1)
            trade = pred != 0
            return {
                "rows": int(len(df)),
                "bacc": float(balanced_accuracy_score(yy, pred)),
                "macro_f1": float(f1_score(yy, pred, average="macro")),
                "ovr_auc": float(roc_auc_score(yy, proba, multi_class="ovr", labels=[0, 1, 2])),
                "proxy_trades": int(trade.sum()),
                "proxy_long_trades": int((pred == 1).sum()),
                "proxy_short_trades": int((pred == 2).sum()),
                "proxy_trade_rate": float(trade.mean()),
                "proxy_wr": float((pred[trade] == yy[trade]).mean()) if trade.any() else None,
            }

        return {"feature_count": len(cols), "train": met(train, y), "oos": met(oos, yo)}

    out = {name: fit_eval(cols) for name, cols in groups.items()}
    out["dir3_cols"] = all_dir3_cols
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "remaining_meta_probe_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--base-meta-dir", type=Path, default=DEFAULT_BASE_META_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--chart-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    frames = {year: _add_label(_read_base(args.split_dir, year), args.label_dir, year) for year in [2024, 2025, 2026]}
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # ChartCNN
    chart_cols = [c for c in CHART_COLS if c in frames[2024].columns]
    chart_artifact = _train_chartcnn(frames[2024], chart_cols, seed=args.seed, epochs=args.chart_epochs, batch_size=args.batch_size)
    chart_2025 = _score_chartcnn(frames[2025], chart_artifact, args.batch_size)
    chart_2026 = _score_chartcnn(frames[2026], chart_artifact, args.batch_size)
    chart_paths = _write_feature_set(args.out_root, "chartcnn", chart_2025, chart_2026)
    torch.save({"model_state": chart_artifact["model"].state_dict(), "cols": chart_cols}, args.out_root / "omega1_dir3_chartcnn_20260531" / "chartcnn.pt")

    # Patch HGB
    patch_cols = [c for c in PATCH_COLS if c in frames[2024].columns]
    patch_x = {year: _patch_features(frames[year], patch_cols) for year in [2024, 2025, 2026]}
    patch_train = patch_x[2024].merge(frames[2024][["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    patch_model = _fit_hgb(patch_train.drop(columns=["timestamp", "zigzag_action"]), patch_train["zigzag_action"].astype(int).to_numpy(), seed=args.seed)
    patch_2025 = _score_model("patch", patch_model, patch_x[2025])
    patch_2026 = _score_model("patch", patch_model, patch_x[2026])
    patch_paths = _write_feature_set(args.out_root, "patch", patch_2025, patch_2026)
    joblib.dump({"model": patch_model, "cols": list(patch_x[2024].drop(columns=["timestamp"]).columns), "source_cols": patch_cols}, args.out_root / "omega1_dir3_patch_20260531" / "patch_hgb.joblib")

    # DUET-style cluster HGB.
    duet_cols = _safe_numeric_cols(frames[2024])
    duet_cols = [c for c in duet_cols if c != "zigzag_action"][:140]
    duet_x, clusters = _duet_features(frames[2024], frames, duet_cols)
    duet_train = duet_x[2024].merge(frames[2024][["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    duet_model = _fit_hgb(duet_train.drop(columns=["timestamp", "zigzag_action"]), duet_train["zigzag_action"].astype(int).to_numpy(), seed=args.seed)
    duet_2025 = _score_model("duet", duet_model, duet_x[2025])
    duet_2026 = _score_model("duet", duet_model, duet_x[2026])
    duet_paths = _write_feature_set(args.out_root, "duet", duet_2025, duet_2026)
    joblib.dump({"model": duet_model, "clusters": clusters, "source_cols": duet_cols}, args.out_root / "omega1_dir3_duet_20260531" / "duet_hgb.joblib")

    audit = {
        "model_id": MODEL_ID,
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "standalone": {
            "chartcnn": {
                "features": chart_paths,
                "metrics_2025": _metrics(chart_2025, frames[2025], OUTPUTS["chartcnn"][:3]),
                "metrics_2026": _metrics(chart_2026, frames[2026], OUTPUTS["chartcnn"][:3]),
                "input_cols": chart_cols,
            },
            "patch": {
                "features": patch_paths,
                "metrics_2025": _metrics(patch_2025, frames[2025], OUTPUTS["patch"][:3]),
                "metrics_2026": _metrics(patch_2026, frames[2026], OUTPUTS["patch"][:3]),
                "source_cols": patch_cols,
                "feature_count": int(len(patch_x[2024].columns) - 1),
            },
            "duet": {
                "features": duet_paths,
                "metrics_2025": _metrics(duet_2025, frames[2025], OUTPUTS["duet"][:3]),
                "metrics_2026": _metrics(duet_2026, frames[2026], OUTPUTS["duet"][:3]),
                "source_col_count": int(len(duet_cols)),
                "cluster_count": int(len(clusters)),
            },
        },
        "contract": {
            "forbidden_inputs": ["teacher_*", "a5dir_*", "Regime4", "regime3_pred_*", "label/target/future/PnL/action_score", "same-level dir3 outputs"],
            "notes": ["ChartCNN uses current/past 72-row sequence only.", "Patch uses rolling windows ending at timestamp t.", "DUET uses correlation clusters fitted on 2024 only."],
        },
    }
    (args.report_dir / "dir3_remaining_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    feature_dirs = {
        "chartcnn": args.out_root / "omega1_dir3_chartcnn_20260531",
        "patch": args.out_root / "omega1_dir3_patch_20260531",
        "duet": args.out_root / "omega1_dir3_duet_20260531",
    }
    audit["combined_meta_probe"] = _meta_probe(args.base_meta_dir, feature_dirs, args.report_dir)
    (args.report_dir / "dir3_remaining_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
