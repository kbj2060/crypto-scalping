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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_dir3_finpaper_20260531"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT_ROOT = ROOT / "data/ensemble/supervised"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_finpaper_20260531"

SEQ_LEN = 72
N_CLASS = 3

SEQ_COLS = [
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

FORBIDDEN_PREFIXES = (
    "teacher_",
    "a5dir_",
    "clean_regime_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
    "dir3_",
)
FORBIDDEN_SUBSTRINGS = ("label", "target", "future", "pnl", "action_score")

OUTPUTS = {
    "vsnlstm": [
        "dir3_vsnlstm_h6_fl_prob",
        "dir3_vsnlstm_h6_up_prob",
        "dir3_vsnlstm_h6_dn_prob",
        "dir3_vsnlstm_h6_confidence",
        "dir3_vsnlstm_h6_side_edge",
        "dir3_vsnlstm_h6_trade_prob",
    ],
    "lpatchtst": [
        "dir3_lpatchtst_h6_fl_prob",
        "dir3_lpatchtst_h6_up_prob",
        "dir3_lpatchtst_h6_dn_prob",
        "dir3_lpatchtst_h6_confidence",
        "dir3_lpatchtst_h6_side_edge",
        "dir3_lpatchtst_h6_trade_prob",
    ],
    "xtrend": [
        "dir3_xtrend_h6_fl_prob",
        "dir3_xtrend_h6_up_prob",
        "dir3_xtrend_h6_dn_prob",
        "dir3_xtrend_h6_confidence",
        "dir3_xtrend_h6_side_edge",
        "dir3_xtrend_h6_trade_prob",
    ],
}


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


def _check_cols(cols: list[str]) -> None:
    bad: list[str] = []
    for col in cols:
        lower = col.lower()
        if col.startswith(FORBIDDEN_PREFIXES) or any(token in lower for token in FORBIDDEN_SUBSTRINGS):
            bad.append(col)
    if bad:
        raise ValueError(f"forbidden inputs selected: {bad[:40]}")


def _seq_matrix(frame: pd.DataFrame, cols: list[str], scaler: RobustScaler | None = None) -> tuple[np.ndarray, RobustScaler]:
    _check_cols(cols)
    raw = frame[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)
        raw = scaler.fit_transform(raw).astype(np.float32)
    else:
        raw = scaler.transform(raw).astype(np.float32)
    if len(raw) < SEQ_LEN:
        raise RuntimeError("not enough rows for sequence")
    seq = np.lib.stride_tricks.sliding_window_view(raw, SEQ_LEN, axis=0)
    seq = np.transpose(seq, (0, 2, 1)).copy()
    return seq, scaler


class VsnLstm(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int = 96) -> None:
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, n_features),
        )
        self.proj = torch.nn.Linear(n_features, hidden)
        self.lstm = torch.nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, N_CLASS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=-1) * x.shape[-1]
        z = self.proj(x * weights)
        out, _ = self.lstm(z)
        return self.head(out[:, -1])


class LightweightPatchTST(torch.nn.Module):
    def __init__(self, n_features: int, patch_len: int = 6, d_model: int = 96, n_heads: int = 4) -> None:
        super().__init__()
        if SEQ_LEN % patch_len != 0:
            raise ValueError("SEQ_LEN must be divisible by patch_len")
        self.patch_len = int(patch_len)
        self.n_patches = SEQ_LEN // self.patch_len
        self.proj = torch.nn.Linear(n_features * self.patch_len, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 3,
            dropout=0.08,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d_model), torch.nn.Linear(d_model, N_CLASS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f = x.shape
        z = x.reshape(b, self.n_patches, self.patch_len * f)
        z = self.proj(z) + self.pos
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def _train_torch_model(
    model: torch.nn.Module,
    seq: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    model = model.to(device)
    counts = np.bincount(y, minlength=N_CLASS).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-3)
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
    return model


def _score_torch(prefix: str, frame: pd.DataFrame, artifact: dict[str, Any], batch_size: int) -> pd.DataFrame:
    seq, _ = _seq_matrix(frame, artifact["cols"], artifact["scaler"])
    model: torch.nn.Module = artifact["model"]
    device = next(model.parameters()).device
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq), int(batch_size)):
            xb = torch.tensor(seq[start : start + int(batch_size)], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    proba = np.vstack(probs)
    return _append_proba(prefix, frame["timestamp"].iloc[SEQ_LEN - 1 :], proba)


def _window_summary(seq: np.ndarray) -> np.ndarray:
    last = seq[:, -1, :]
    mean = seq.mean(axis=1)
    early = seq[:, : SEQ_LEN // 3, :].mean(axis=1)
    late = seq[:, -SEQ_LEN // 3 :, :].mean(axis=1)
    std = seq.std(axis=1)
    return np.concatenate([last, mean, late - early, std], axis=1).astype(np.float32)


def _fit_xtrend(train: pd.DataFrame, cols: list[str], *, seed: int, max_context: int, n_components: int, k: int) -> dict[str, Any]:
    seq, scaler = _seq_matrix(train, cols, None)
    y = train["zigzag_action"].astype(int).to_numpy()[SEQ_LEN - 1 :]
    summary = _window_summary(seq)
    imputer = SimpleImputer(strategy="median")
    scaled = RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)
    x = scaled.fit_transform(imputer.fit_transform(summary)).astype(np.float32)
    n_comp = int(min(n_components, x.shape[1], max(2, len(x) - 1)))
    pca = PCA(n_components=n_comp, random_state=int(seed))
    z = pca.fit_transform(x).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    keep: list[int] = []
    per_class = max(1, int(max_context) // N_CLASS)
    for cls in range(N_CLASS):
        idx = np.flatnonzero(y == cls)
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        keep.extend(idx.tolist())
    keep = np.asarray(sorted(keep), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=int(min(k, len(keep))), metric="cosine", algorithm="brute")
    nn.fit(z[keep])
    return {
        "cols": cols,
        "scaler": scaler,
        "imputer": imputer,
        "summary_scaler": scaled,
        "pca": pca,
        "nn": nn,
        "context_z": z[keep],
        "context_y": y[keep],
        "k": int(min(k, len(keep))),
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
    }


def _score_xtrend(frame: pd.DataFrame, artifact: dict[str, Any], *, temperature: float) -> pd.DataFrame:
    seq, _ = _seq_matrix(frame, artifact["cols"], artifact["scaler"])
    summary = _window_summary(seq)
    x = artifact["summary_scaler"].transform(artifact["imputer"].transform(summary)).astype(np.float32)
    z = artifact["pca"].transform(x).astype(np.float32)
    dist, ind = artifact["nn"].kneighbors(z, n_neighbors=artifact["k"], return_distance=True)
    sim = 1.0 - dist
    w = np.exp(sim / max(float(temperature), 1e-6))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    labels = artifact["context_y"][ind]
    proba = np.zeros((len(z), N_CLASS), dtype=np.float32)
    for cls in range(N_CLASS):
        proba[:, cls] = (w * (labels == cls)).sum(axis=1)
    proba = np.clip(proba, 1e-6, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return _append_proba("xtrend", frame["timestamp"].iloc[SEQ_LEN - 1 :], proba)


def _append_proba(prefix: str, timestamps: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": timestamps.to_numpy()})
    out[f"dir3_{prefix}_h6_fl_prob"] = proba[:, 0]
    out[f"dir3_{prefix}_h6_up_prob"] = proba[:, 1]
    out[f"dir3_{prefix}_h6_dn_prob"] = proba[:, 2]
    out[f"dir3_{prefix}_h6_confidence"] = proba.max(axis=1)
    out[f"dir3_{prefix}_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out[f"dir3_{prefix}_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


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
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=N_CLASS))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=N_CLASS))},
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--xtrend-context", type=int, default=24000)
    parser.add_argument("--xtrend-k", type=int, default=96)
    args = parser.parse_args()

    frames = {year: _add_label(_read_base(args.split_dir, year), args.label_dir, year) for year in [2024, 2025, 2026]}
    cols = [c for c in SEQ_COLS if c in frames[2024].columns]
    if len(cols) < 16:
        raise RuntimeError(f"too few sequence columns found: {len(cols)}")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    train_seq, seq_scaler = _seq_matrix(frames[2024], cols, None)
    train_y = frames[2024]["zigzag_action"].astype(int).to_numpy()[SEQ_LEN - 1 :]

    vsn = _train_torch_model(
        VsnLstm(len(cols)),
        train_seq,
        train_y,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=7e-4,
    )
    vsn_art = {"model": vsn, "scaler": seq_scaler, "cols": cols}
    vsn_2025 = _score_torch("vsnlstm", frames[2025], vsn_art, args.batch_size)
    vsn_2026 = _score_torch("vsnlstm", frames[2026], vsn_art, args.batch_size)
    vsn_paths = _write_feature_set(args.out_root, "vsnlstm", vsn_2025, vsn_2026)
    torch.save({"model_state": vsn.state_dict(), "cols": cols}, args.out_root / "omega1_dir3_vsnlstm_20260531" / "vsnlstm.pt")

    patch = _train_torch_model(
        LightweightPatchTST(len(cols)),
        train_seq,
        train_y,
        seed=args.seed + 7,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=6e-4,
    )
    patch_art = {"model": patch, "scaler": seq_scaler, "cols": cols}
    patch_2025 = _score_torch("lpatchtst", frames[2025], patch_art, args.batch_size)
    patch_2026 = _score_torch("lpatchtst", frames[2026], patch_art, args.batch_size)
    patch_paths = _write_feature_set(args.out_root, "lpatchtst", patch_2025, patch_2026)
    torch.save({"model_state": patch.state_dict(), "cols": cols}, args.out_root / "omega1_dir3_lpatchtst_20260531" / "lpatchtst.pt")

    xtrend_art = _fit_xtrend(frames[2024], cols, seed=args.seed + 13, max_context=args.xtrend_context, n_components=48, k=args.xtrend_k)
    xt_2025 = _score_xtrend(frames[2025], xtrend_art, temperature=0.12)
    xt_2026 = _score_xtrend(frames[2026], xtrend_art, temperature=0.12)
    xt_paths = _write_feature_set(args.out_root, "xtrend", xt_2025, xt_2026)
    joblib.dump(xtrend_art, args.out_root / "omega1_dir3_xtrend_20260531" / "xtrend_context.joblib")

    audit = {
        "model_id": MODEL_ID,
        "source_papers": {
            "fintsb": "benchmark protocol inspiration: standardized OOS label-probe metrics plus trade count and proxy WR",
            "oxford_financial_benchmark": "VSN+LSTM and PatchTST-style sequence representations",
            "xtrend": "context-set similarity transfer for regime-adaptive trend signals",
        },
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "input_cols": cols,
        "contract": {
            "forbidden_inputs": ["teacher_*", "a5dir_*", "Regime4", "regime3_pred_*", "label/target/future/PnL/action_score", "same-level dir3 outputs"],
            "notes": [
                "VSN-LSTM and lightweight PatchTST use only current/past 72-row sequences.",
                "X-Trend-style context memory is fitted on 2024 only and scores 2025/2026 by nearest context attention.",
            ],
        },
        "standalone": {
            "vsnlstm": {
                "features": vsn_paths,
                "metrics_2025": _metrics(vsn_2025, frames[2025], OUTPUTS["vsnlstm"][:3]),
                "metrics_2026": _metrics(vsn_2026, frames[2026], OUTPUTS["vsnlstm"][:3]),
            },
            "lpatchtst": {
                "features": patch_paths,
                "metrics_2025": _metrics(patch_2025, frames[2025], OUTPUTS["lpatchtst"][:3]),
                "metrics_2026": _metrics(patch_2026, frames[2026], OUTPUTS["lpatchtst"][:3]),
            },
            "xtrend": {
                "features": xt_paths,
                "metrics_2025": _metrics(xt_2025, frames[2025], OUTPUTS["xtrend"][:3]),
                "metrics_2026": _metrics(xt_2026, frames[2026], OUTPUTS["xtrend"][:3]),
                "pca_explained_variance_sum": xtrend_art["pca_explained_variance_sum"],
                "context_size": int(len(xtrend_art["context_y"])),
                "k": int(xtrend_art["k"]),
            },
        },
    }
    (args.report_dir / "dir3_finpaper_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
