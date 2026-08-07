#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mamba_ssm import Mamba
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_mamba_teacher_20260531"
DEFAULT_CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
DEFAULT_REGIME3_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_REGIME3_CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
DEFAULT_CHRONOS_DIR = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"
DEFAULT_M7_ZIGZAG_DIR = ROOT / "data/splits/year_oos"
DEFAULT_ZIGZAG_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_mamba_teacher_current_chronos_20260531"

OMEGA1_SECONDARY_INPUTS = [
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_zscore",
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
    "cvp_regime",
    "regime_trending",
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]

REGIME3_RISK_FEATURES = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]

REGIME3_CURRENT_FEATURES = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

REGIME3_CMAMBA_FEATURES = [
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
]

CHRONOS_FEATURES = [
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
]

M7_FEATURES = [
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]

TEACHER_MAMBA_OUTPUTS = [
    "teacher_mamba_p_cash",
    "teacher_mamba_p_long",
    "teacher_mamba_p_short",
    "teacher_mamba_confidence",
    "teacher_mamba_side_edge",
    "teacher_mamba_uncertainty",
    "teacher_mamba_risk_veto_score",
]

FORBIDDEN_PREFIXES = (
    "teacher_",
    "a5dir_",
    "clean_regime_",
    "clean_regime4_",
    "regime4_pred_",
    "regime3_pred_",
    "zigzag_",
    "ai_dir_",
    "patchtst_",
    "dlinear_",
    "m7_trend_xgb_",
    "m7_mtl_",
    "m7_quant_",
    "m7_prob_",
    "m7_gmm_",
    "m7_iso_",
    "m7_vae_",
    "m7_hdb_",
)
FORBIDDEN_NAMES = {
    "timestamp",
    "tp_sl_action_score",
    "m7_action",
    "m7_size",
    "m7_confidence",
    "m7_composite_score",
    "m7_tail_risk",
    "pred_patchtst",
    "conf_patchtst",
    "pred_mdjd",
    "conf_mdjd",
}
FORBIDDEN_TOKENS = ("label", "target", "future", "pnl")
TARGET_ALIAS_TOKENS = ("label", "target", "future", "pnl", "action_score")


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 72
    d_model: int = 96
    layers: int = 2
    dropout: float = 0.10
    batch_size: int = 512
    epochs: int = 6
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-5
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-4
    val_fraction: float = 0.18
    seed: int = 20260531


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, seq_len: int):
        self.x = x
        self.y = y
        self.indices = indices.astype(np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.indices[idx])
        start = end - self.seq_len + 1
        return torch.from_numpy(self.x[start : end + 1]), torch.tensor(int(self.y[end]), dtype=torch.long)


class Omega1MambaTeacher(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(int(layers))])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        for block in self.blocks:
            z = z + block(self.norm(z))
        last = self.drop(self.norm(z[:, -1, :]))
        return self.head(last)


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


def _read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
    source: str,
    *,
    allow_tail_drop: bool = False,
    allow_edge_drop: bool = False,
) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    missing = {col: int(merged[col].isna().sum()) for col in cols if int(merged[col].isna().sum()) > 0}
    if missing:
        miss_any = merged[cols].isna().any(axis=1).to_numpy()
        miss_idx = np.flatnonzero(miss_any)
        tail_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(len(merged) - miss_idx.size, len(merged)))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        head_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(0, miss_idx.size))
        if allow_edge_drop and head_only:
            return merged.iloc[miss_idx.size :].reset_index(drop=True)
        if allow_edge_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        raise RuntimeError(f"{source} exact timestamp join has missing values: {missing}")
    return merged


def _add_regime3_risk(frame: pd.DataFrame, *, year: int, regime3_dir: Path) -> pd.DataFrame:
    name = "training_features_2025_regime3_stability_risk_h6.csv" if int(year) == 2025 else "training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
    path = regime3_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_RISK_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 risk columns: {missing}")
    return _exact_join(frame, side, REGIME3_RISK_FEATURES, f"Regime3 h6 risk {year}", allow_tail_drop=True)


def _add_regime3_current(frame: pd.DataFrame, *, year: int, regime3_current_dir: Path) -> pd.DataFrame:
    name = (
        "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
    )
    path = regime3_current_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CURRENT_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 current columns: {missing}")
    return _exact_join(frame, side, REGIME3_CURRENT_FEATURES, f"Regime3 current {year}", allow_tail_drop=True)


def _add_regime3_cmamba(frame: pd.DataFrame, *, year: int, regime3_cmamba_dir: Path) -> pd.DataFrame:
    name = (
        "training_features_2025_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
        if int(year) == 2025
        else "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
    )
    path = regime3_cmamba_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CMAMBA_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{path} missing required Regime3 CryptoMamba columns: {missing}")
    return _exact_join(frame, side, REGIME3_CMAMBA_FEATURES, f"Regime3 CryptoMamba h6 {year}", allow_edge_drop=True)


def _chronos_series(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "q10", "q50", "q90", "width"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path} missing required Chronos columns: {missing}")
    q50 = pd.to_numeric(raw["q50"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q90 = pd.to_numeric(raw["q90"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    width = pd.to_numeric(raw["width"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    out = pd.DataFrame({"timestamp": raw["timestamp"]})
    out[f"chronos_{prefix}_width"] = width.astype("float32")
    out[f"chronos_{prefix}_large_move_score"] = (width * (1.0 + q50.abs())).astype("float32")
    out[f"chronos_{prefix}_upside_band_ewm3"] = q90.clip(lower=0.0).ewm(span=3, adjust=False, min_periods=1).mean().astype("float32")
    out[f"chronos_{prefix}_width_ewm6"] = width.ewm(span=6, adjust=False, min_periods=1).mean().astype("float32")
    return out


def _add_chronos(frame: pd.DataFrame, *, year: int, chronos_dir: Path) -> pd.DataFrame:
    split = "val2025" if int(year) == 2025 else "oos2026"
    atr = _chronos_series(chronos_dir / f"atr14_pct_{split}_chronos.csv", "atr14")
    rv = _chronos_series(chronos_dir / f"realized_vol_24_{split}_chronos.csv", "realized_vol24")
    side = atr.merge(rv, on="timestamp", how="inner", validate="one_to_one")
    missing = sorted(set(CHRONOS_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"Chronos derived feature set missing required columns: {missing}")
    return _exact_join(frame, side, CHRONOS_FEATURES, f"Chronos uncertainty {year}", allow_tail_drop=True)


def _add_m7_zigzag(frame: pd.DataFrame, *, year: int, m7_zigzag_dir: Path) -> pd.DataFrame:
    path = m7_zigzag_dir / f"rl_training_{int(year)}_m7_zigzag_direction.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    m7 = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(M7_FEATURES) - set(m7.columns))
    if missing:
        raise ValueError(f"{path} missing required M7 teacher columns: {missing}")
    overlap = [col for col in M7_FEATURES if col in frame.columns]
    if overlap:
        check = frame[["timestamp", *overlap]].merge(
            m7[["timestamp", *overlap]],
            on="timestamp",
            how="left",
            suffixes=("_existing", "_source"),
            validate="one_to_one",
        )
        missing_overlap = {
            col: int(check[f"{col}_source"].isna().sum())
            for col in overlap
            if int(check[f"{col}_source"].isna().sum()) > 0
        }
        if missing_overlap:
            raise RuntimeError(f"M7 ZigZag direction {year} overlap source missing values: {missing_overlap}")
        mismatched: list[str] = []
        for col in overlap:
            existing = pd.to_numeric(check[f"{col}_existing"], errors="coerce")
            source = pd.to_numeric(check[f"{col}_source"], errors="coerce")
            valid = existing.notna() & source.notna()
            if int(valid.sum()) != len(check) or float((existing[valid] - source[valid]).abs().max()) > 1e-10:
                mismatched.append(col)
        if mismatched:
            raise RuntimeError(f"M7 ZigZag direction {year} overlap columns differ from source: {mismatched}")
    join_cols = [col for col in M7_FEATURES if col not in frame.columns]
    if not join_cols:
        return frame
    return _exact_join(frame, m7, join_cols, f"M7 ZigZag direction {year}", allow_tail_drop=False)


def _add_zigzag_label(frame: pd.DataFrame, *, year: int, zigzag_label_dir: Path) -> pd.DataFrame:
    path = zigzag_label_dir / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["zigzag_action"]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing required ZigZag label columns: {missing}")
    return _exact_join(frame, labels, required, f"ZigZag action labels {year}", allow_tail_drop=False)


def _build_frame(
    path: Path,
    *,
    year: int,
    regime3_dir: Path,
    regime3_current_dir: Path,
    regime3_cmamba_dir: Path,
    chronos_dir: Path,
    m7_zigzag_dir: Path,
    zigzag_label_dir: Path,
) -> pd.DataFrame:
    frame = _read_candidates(path)
    frame = _add_regime3_risk(frame, year=year, regime3_dir=regime3_dir)
    frame = _add_regime3_current(frame, year=year, regime3_current_dir=regime3_current_dir)
    frame = _add_regime3_cmamba(frame, year=year, regime3_cmamba_dir=regime3_cmamba_dir)
    frame = _add_chronos(frame, year=year, chronos_dir=chronos_dir)
    frame = _add_m7_zigzag(frame, year=year, m7_zigzag_dir=m7_zigzag_dir)
    frame = _add_zigzag_label(frame, year=year, zigzag_label_dir=zigzag_label_dir)
    return frame


def _is_forbidden(col: str) -> bool:
    if col in REGIME3_CMAMBA_FEATURES:
        return False
    if col in FORBIDDEN_NAMES:
        return True
    if col.startswith(FORBIDDEN_PREFIXES):
        return True
    lower = col.lower()
    return any(token in lower for token in FORBIDDEN_TOKENS)


def _feature_columns(train: pd.DataFrame, oos: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing_secondary = [col for col in OMEGA1_SECONDARY_INPUTS if col not in train.columns or col not in oos.columns]
    if missing_secondary:
        raise ValueError(f"Omega1 secondary inputs missing: {missing_secondary}")
    base_cols: list[str] = []
    for col in train.columns:
        if col not in oos.columns or col in OMEGA1_SECONDARY_INPUTS or _is_forbidden(col):
            continue
        if col.startswith(("m7_", "ai_", "tide_", "chronos_", "regime3_current_", "regime3_")):
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(oos[col]):
            base_cols.append(col)
    if not base_cols:
        raise ValueError("no base input features selected")
    return OMEGA1_SECONDARY_INPUTS + base_cols, base_cols


def _assert_no_target_aliases(frame: pd.DataFrame, feature_cols: list[str]) -> None:
    target_like_cols = [
        col
        for col in frame.columns
        if col not in feature_cols
        and any(token in col.lower() for token in TARGET_ALIAS_TOKENS)
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    violations: list[str] = []
    for feat in feature_cols:
        if not pd.api.types.is_numeric_dtype(frame[feat]):
            continue
        fx = pd.to_numeric(frame[feat], errors="coerce")
        for target_col in target_like_cols:
            ty = pd.to_numeric(frame[target_col], errors="coerce")
            valid = fx.notna() & ty.notna()
            if int(valid.sum()) == 0:
                continue
            if float((fx[valid] - ty[valid]).abs().max()) <= 1e-12:
                violations.append(f"{feat} == {target_col}")
    if violations:
        raise RuntimeError(f"selected feature aliases target-like column(s): {violations[:20]}")


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if "zigzag_action" not in frame.columns:
        raise ValueError("missing active Omega1 label column: zigzag_action")
    y = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise ValueError(f"invalid zigzag_action class values: {invalid}")
    return y


def _standardize(train: pd.DataFrame, oos: pd.DataFrame, cols: list[str], fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tr = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    oo = oos[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fit = tr.iloc[np.asarray(fit_idx, dtype=np.int64)]
    med = fit.median(axis=0).fillna(0.0)
    q25 = fit.quantile(0.25, axis=0).fillna(0.0)
    q75 = fit.quantile(0.75, axis=0).fillna(0.0)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(1.0)
    x_train = ((tr.fillna(med) - med) / scale).clip(-12.0, 12.0).to_numpy(dtype=np.float32)
    x_oos = ((oo.fillna(med) - med) / scale).clip(-12.0, 12.0).to_numpy(dtype=np.float32)
    norm = {"median": med.to_dict(), "iqr": scale.to_dict(), "clip": 12.0}
    return x_train, x_oos, norm


def _valid_indices(n: int, seq_len: int) -> np.ndarray:
    if n < seq_len + 10:
        raise ValueError(f"too few rows for seq_len={seq_len}: {n}")
    return np.arange(seq_len - 1, n, dtype=np.int64)


def _class_weights(y: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y[indices], minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate_logits(y: np.ndarray, indices: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    yt = y[indices].astype(np.int64)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(yt, minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(yt, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, indices: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = SeqDataset(x, np.zeros(len(x), dtype=np.int64), indices, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    outs: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(device, non_blocking=True))
        outs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _append_outputs(frame: pd.DataFrame, indices: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for col in TEACHER_MAMBA_OUTPUTS:
        out[col] = np.nan
    full = np.zeros((len(frame), 3), dtype=np.float32)
    full[indices] = proba.astype(np.float32)
    out.loc[indices, "teacher_mamba_p_cash"] = full[indices, 0]
    out.loc[indices, "teacher_mamba_p_long"] = full[indices, 1]
    out.loc[indices, "teacher_mamba_p_short"] = full[indices, 2]
    conf = np.max(full[indices], axis=1)
    out.loc[indices, "teacher_mamba_confidence"] = conf
    out.loc[indices, "teacher_mamba_side_edge"] = full[indices, 1] - full[indices, 2]
    out.loc[indices, "teacher_mamba_uncertainty"] = 1.0 - conf
    out.loc[indices, "teacher_mamba_risk_veto_score"] = np.clip(full[indices, 0] + (1.0 - conf), 0.0, 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--regime3-dir", type=Path, default=DEFAULT_REGIME3_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--m7-zigzag-dir", type=Path, default=DEFAULT_M7_ZIGZAG_DIR)
    parser.add_argument("--zigzag-label-dir", type=Path, default=DEFAULT_ZIGZAG_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--no-lr-scheduler", action="store_true")
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Omega1 Mamba teacher requires CUDA because mamba_ssm kernels are GPU-oriented in this environment.")
    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = TrainConfig(
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        lr_factor=float(args.lr_factor),
        lr_patience=int(args.lr_patience),
        lr_min=float(args.lr_min),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        seed=int(args.seed),
    )
    train_name = "trade_candidates_2025_alpha6_current_tail111_exact.csv"
    oos_name = "trade_candidates_2026_alpha6_current_tail111_exact.csv"
    train = _build_frame(
        args.candidate_dir / train_name,
        year=2025,
        regime3_dir=args.regime3_dir,
        regime3_current_dir=args.regime3_current_dir,
        regime3_cmamba_dir=args.regime3_cmamba_dir,
        chronos_dir=args.chronos_dir,
        m7_zigzag_dir=args.m7_zigzag_dir,
        zigzag_label_dir=args.zigzag_label_dir,
    )
    oos = _build_frame(
        args.candidate_dir / oos_name,
        year=2026,
        regime3_dir=args.regime3_dir,
        regime3_current_dir=args.regime3_current_dir,
        regime3_cmamba_dir=args.regime3_cmamba_dir,
        chronos_dir=args.chronos_dir,
        m7_zigzag_dir=args.m7_zigzag_dir,
        zigzag_label_dir=args.zigzag_label_dir,
    )

    feature_cols, base_cols = _feature_columns(train, oos)
    _assert_no_target_aliases(train, feature_cols)
    _assert_no_target_aliases(oos, feature_cols)
    y_train = _labels(train)
    y_oos = _labels(oos)

    all_idx = _valid_indices(len(train), cfg.seq_len)
    split_at = int(round(len(all_idx) * (1.0 - cfg.val_fraction)))
    train_idx = all_idx[:split_at]
    val_idx = all_idx[split_at:]
    oos_idx = _valid_indices(len(oos), cfg.seq_len)
    x_train, x_oos, norm = _standardize(train, oos, feature_cols, train_idx)

    model = Omega1MambaTeacher(x_train.shape[1], cfg.d_model, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if not bool(args.no_lr_scheduler):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=float(cfg.lr_factor),
            patience=max(1, int(cfg.lr_patience)),
            min_lr=float(cfg.lr_min),
            threshold=1e-3,
            threshold_mode="rel",
        )
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_train, train_idx, device), label_smoothing=0.02)
    train_loader = DataLoader(
        SeqDataset(x_train, y_train, train_idx, cfg.seq_len),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    best_epoch = 0
    bad_val_count = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_proba = _predict(model, x_train, val_idx, cfg.seq_len, cfg.batch_size, device)
        val_metrics = _evaluate_logits(y_train, val_idx, val_proba)
        score = float(val_metrics["balanced_accuracy"])
        prev_lr = float(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(score)
        new_lr = float(opt.param_groups[0]["lr"])
        improved = score > (best_val + float(cfg.early_stop_min_delta))
        if improved:
            best_val = score
            best_epoch = int(epoch)
            bad_val_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_val_count += 1
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "lr": new_lr,
            "lr_dropped": bool(new_lr < prev_lr),
            "best_epoch": int(best_epoch),
            "bad_val_count": int(bad_val_count),
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
        if int(cfg.early_stop_patience) > 0 and bad_val_count >= int(cfg.early_stop_patience):
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": int(epoch),
                        "best_epoch": int(best_epoch),
                        "best_val_balanced_accuracy": float(best_val),
                        "bad_val_count": int(bad_val_count),
                        "patience": int(cfg.early_stop_patience),
                    },
                    ensure_ascii=False,
                    default=_json_default,
                ),
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    val_proba = _predict(model, x_train, val_idx, cfg.seq_len, cfg.batch_size, device)
    oos_proba = _predict(model, x_oos, oos_idx, cfg.seq_len, cfg.batch_size, device)
    train_proba = _predict(model, x_train, all_idx, cfg.seq_len, cfg.batch_size, device)
    train_metrics = _evaluate_logits(y_train, all_idx, train_proba)
    val_metrics = _evaluate_logits(y_train, val_idx, val_proba)
    oos_metrics = _evaluate_logits(y_oos, oos_idx, oos_proba)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = _append_outputs(train, all_idx, train_proba)
    oos_out = _append_outputs(oos, oos_idx, oos_proba)
    train_path = args.out_dir / train_name
    oos_path = args.out_dir / oos_name
    train_out.to_csv(train_path, index=False)
    oos_out.to_csv(oos_path, index=False)
    model_path = args.out_dir / "omega1_mamba_teacher.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "base_cols": base_cols,
            "norm": norm,
            "best_epoch": int(best_epoch),
            "best_val_balanced_accuracy": float(best_val),
        },
        model_path,
    )

    audit = {
        "model_id": MODEL_ID,
        "config": asdict(cfg),
        "candidate_dir": str(args.candidate_dir),
        "m7_zigzag_dir": str(args.m7_zigzag_dir),
        "zigzag_label_dir": str(args.zigzag_label_dir),
        "out_dir": str(args.out_dir),
        "label_source": "zigzag_action",
        "feature_count": int(len(feature_cols)),
        "secondary_feature_count": int(len(OMEGA1_SECONDARY_INPUTS)),
        "base_feature_count": int(len(base_cols)),
        "feature_cols": feature_cols,
        "secondary_feature_cols": OMEGA1_SECONDARY_INPUTS,
        "base_feature_cols": base_cols,
        "outputs": TEACHER_MAMBA_OUTPUTS,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_val),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "oos_label_probe_metrics": oos_metrics,
        "contract": {
            "seq_len": int(cfg.seq_len),
            "teacher_feedback_inputs_forbidden": True,
            "regime4_forbidden": True,
            "regime3_pred_forbidden": True,
            "chronos_exact_join": True,
            "regime3_exact_join": True,
            "m7_zigzag_exact_join": True,
            "zigzag_label_exact_join": True,
            "target_alias_guard": True,
            "normalization_fit_scope": "train_idx_only",
            "lr_scheduler": "ReduceLROnPlateau" if scheduler is not None else "OFF",
            "early_stop_patience": int(cfg.early_stop_patience),
            "base_feature_policy": "numeric_current_context_after_explicit_forbidden_filters",
        },
        "artifacts": {"train_csv": str(train_path), "oos_csv": str(oos_path), "model": str(model_path), "audit": str(args.out_dir / "omega1_mamba_teacher_audit.json")},
    }
    forbidden_selected = [c for c in feature_cols if _is_forbidden(c)]
    if forbidden_selected:
        raise RuntimeError(f"forbidden columns selected: {forbidden_selected[:20]}")
    (args.out_dir / "omega1_mamba_teacher_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
