#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    ACTION_DIM,
    ACTION_FALLBACK,
    ACTION_PRIMARY,
    ACTION_SKIP,
    DECISION_COLS,
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    SOURCE_COLS,
    TRAIN_CSV,
    _apply_norm,
    _audit_frame_contract,
    _build_counterfactual_dataset,
    _compose_decisions,
    _directional_features,
    _fit_norm,
    _policy_action,
    _safe_num,
    _state_frame,
    _train_dsac_offline,
    _usage,
)


MODEL_ID = "alpha8_mamba_lgbm_dsac_hybrid_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

REGIME_COLS = [
    "clean_regime4_state24_sticky090_v2_bull_prob",
    "clean_regime4_state24_sticky090_v2_bear_prob",
    "clean_regime4_state24_sticky090_v2_chop_prob",
    "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    "clean_regime4_state24_sticky090_v2_confidence",
    "clean_regime4_state24_sticky090_v2_entropy",
    "regime4_pred_bull_prob",
    "regime4_pred_bear_prob",
    "regime4_pred_chop_prob",
    "regime4_pred_whipsaw_prob",
    "regime4_pred_confidence",
]

SEQUENCE_COLS = [
    "logret_1",
    "price_momentum_3b",
    "price_momentum_6b",
    "price_momentum_12b",
    "price_momentum_24b",
    "ema_cross_signal",
    "linear_slope_12b",
    "linear_slope_24b",
    "higher_high_12b",
    "lower_low_12b",
    "range_atr_proxy",
    "volume_momentum_12b",
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "smart_money_flow",
    "funding_price_divergence",
    "hurst_48",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "breakout_strength",
    "rsi",
    "tp_sl_action_score",
    "ai_dir_edge",
    "ai_flow_pressure",
    "m7_expected_ret",
    "m7_q50",
    "m7_quality_pred",
    *REGIME_COLS,
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_no_forbidden(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _context_frame(df: pd.DataFrame) -> pd.DataFrame:
    _audit_frame_contract(df, name="alpha8_context")
    _assert_no_forbidden(df, name="alpha8_context")
    parts = [_directional_features(df)]
    parts.append(pd.DataFrame({c: _safe_num(df, c) for c in SOURCE_COLS}, index=df.index))
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    missing = [c for c in SEQUENCE_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f"alpha8 context missing sequence columns: {missing}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_robust_norm(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    arr = df[cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": cols, "median": med.tolist(), "scale": scale.tolist()}


def _apply_robust_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"alpha8 normalizer missing columns: {missing}")
    arr = df[cols].to_numpy(dtype=np.float64)
    med = np.asarray(norm["median"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    z = (arr - med) / scale
    return np.tanh(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _rolling_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2D context array, got shape={arr.shape}")
    pad = np.repeat(arr[:1], int(seq_len) - 1, axis=0)
    padded = np.concatenate([pad, arr], axis=0)
    out = np.lib.stride_tricks.sliding_window_view(padded, int(seq_len), axis=0)
    return np.swapaxes(out, 1, 2).copy().astype(np.float32)


def _direction_labels(df: pd.DataFrame, *, horizon: int, barrier: float) -> np.ndarray:
    close = _safe_num(df, "close").replace(0.0, np.nan).ffill().bfill().fillna(1.0)
    fwd = (close.shift(-int(horizon)) / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.zeros(len(df), dtype=np.int64)
    y[fwd.to_numpy(dtype=np.float64) > float(barrier)] = 1
    y[fwd.to_numpy(dtype=np.float64) < -float(barrier)] = 2
    return y


class Alpha8MambaEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, emb_dim: int, n_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.emb = nn.Sequential(nn.Linear(d_model, emb_dim), nn.SiLU())
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.mamba(h)
        last = self.norm(h[:, -1, :])
        emb = self.emb(last)
        return self.head(emb), emb


@dataclass
class MambaArtifacts:
    model: Alpha8MambaEncoder
    train_diag: dict[str, Any]


def _train_mamba(
    seq: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    d_model: int,
    emb_dim: int,
) -> MambaArtifacts:
    model = Alpha8MambaEncoder(seq.shape[-1], d_model=d_model, emb_dim=emb_dim).to(device)
    x = torch.from_numpy(seq)
    y = torch.from_numpy(labels.astype(np.int64))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(weights.mean(), 1e-12)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    last = {"loss": 0.0, "acc": 0.0}
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        correct = 0
        total = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(float(loss.item()))
            pred = torch.argmax(logits.detach(), dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        last = {"epoch": int(epoch), "loss": float(np.mean(losses)), "acc": float(correct / max(total, 1))}
        print(json.dumps({"stage": "mamba_train", **last}, ensure_ascii=False), flush=True)
    return MambaArtifacts(model=model.cpu(), train_diag=last)


def _mamba_predict(
    model: Alpha8MambaEncoder,
    seq: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    probs: list[np.ndarray] = []
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq), int(batch_size)):
            xb = torch.from_numpy(seq[start : start + int(batch_size)]).to(device)
            logits, emb = model(xb)
            probs.append(F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32))
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(probs), np.concatenate(embs)


def _lightgbm_features(ctx: pd.DataFrame, mamba_probs: np.ndarray, mamba_emb: np.ndarray) -> pd.DataFrame:
    out = ctx[SEQUENCE_COLS].copy()
    for i, name in enumerate(["hold", "long", "short"]):
        out[f"mamba_p_{name}"] = mamba_probs[:, i]
        out[f"mamba_p_{name}_delta1"] = pd.Series(mamba_probs[:, i]).diff().fillna(0.0).to_numpy()
        out[f"mamba_p_{name}_mean3"] = pd.Series(mamba_probs[:, i]).rolling(3, min_periods=1).mean().to_numpy()
    for i in range(mamba_emb.shape[1]):
        out[f"mamba_emb_{i:02d}"] = mamba_emb[:, i]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=290529,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def _alpha8_prob_frame(probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "alpha8_p_hold": probs[:, 0],
            "alpha8_p_long": probs[:, 1],
            "alpha8_p_short": probs[:, 2],
        }
    )
    out["alpha8_dir_edge"] = out["alpha8_p_long"] - out["alpha8_p_short"]
    out["alpha8_confidence"] = np.maximum(out["alpha8_p_long"], out["alpha8_p_short"]) - out["alpha8_p_hold"]
    out["alpha8_direction_abs"] = np.abs(out["alpha8_dir_edge"])
    for col in ["alpha8_p_hold", "alpha8_p_long", "alpha8_p_short", "alpha8_dir_edge", "alpha8_confidence"]:
        out[f"{col}_delta1"] = out[col].diff().fillna(0.0)
        out[f"{col}_mean3"] = out[col].rolling(3, min_periods=1).mean()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _alpha8_state(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    alpha8_probs: np.ndarray,
    mamba_probs: np.ndarray,
    mamba_emb: np.ndarray,
) -> pd.DataFrame:
    base = _state_frame(frame, primary, fallback)
    alpha = _alpha8_prob_frame(alpha8_probs)
    mprob = pd.DataFrame(
        {
            "mamba_p_hold": mamba_probs[:, 0],
            "mamba_p_long": mamba_probs[:, 1],
            "mamba_p_short": mamba_probs[:, 2],
        }
    )
    emb_cols = min(16, mamba_emb.shape[1])
    memb = pd.DataFrame({f"mamba_state_emb_{i:02d}": mamba_emb[:, i] for i in range(emb_cols)})
    out = pd.concat([base.reset_index(drop=True), alpha, mprob, memb], axis=1)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate alpha8 state columns: {dup[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _pick(grid: pd.DataFrame, split: str, variant: str, cost: str) -> dict[str, Any]:
    row = grid[(grid["split"].eq(split)) & (grid["variant"].eq(variant)) & (grid["cost"].eq(cost))]
    return {} if row.empty else row.iloc[0].to_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mamba-epochs", type=int, default=3)
    p.add_argument("--mamba-batch-size", type=int, default=512)
    p.add_argument("--mamba-d-model", type=int, default=96)
    p.add_argument("--mamba-emb-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--label-horizon", type=int, default=12)
    p.add_argument("--label-barrier", type=float, default=0.0025)
    p.add_argument("--dsac-steps", type=int, default=2500)
    p.add_argument("--dsac-batch-size", type=int, default=768)
    p.add_argument("--device", choices=["auto", "cuda"], default="auto")
    args = p.parse_args()

    _seed_everything(290529)
    if not torch.cuda.is_available():
        raise RuntimeError("Alpha8 Mamba requires CUDA; mamba_ssm kernels are not CPU-compatible in this environment.")
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_no_forbidden(train_all, name="train_all")
    _assert_no_forbidden(eval_df, name="eval")
    _audit_frame_contract(train_all, name="train_all")
    _audit_frame_contract(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    ctx_train = _context_frame(train_df)
    ctx_val = _context_frame(val_df)
    ctx_eval = _context_frame(eval_df)
    ctx_norm = _fit_robust_norm(ctx_train, SEQUENCE_COLS)
    seq_train = _rolling_sequences(_apply_robust_norm(ctx_train, ctx_norm), args.seq_len)
    seq_val = _rolling_sequences(_apply_robust_norm(ctx_val, ctx_norm), args.seq_len)
    seq_eval = _rolling_sequences(_apply_robust_norm(ctx_eval, ctx_norm), args.seq_len)

    y_train = _direction_labels(train_df, horizon=args.label_horizon, barrier=args.label_barrier)
    y_val = _direction_labels(val_df, horizon=args.label_horizon, barrier=args.label_barrier)
    y_eval = _direction_labels(eval_df, horizon=args.label_horizon, barrier=args.label_barrier)

    print(
        json.dumps(
            {
                "stage": "alpha8_start",
                "device": str(device),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(eval_df),
                "seq_shape": list(seq_train.shape),
                "label_counts_train": np.bincount(y_train, minlength=3).astype(int).tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    mamba_art = _train_mamba(
        seq_train,
        y_train,
        device=device,
        epochs=args.mamba_epochs,
        batch_size=args.mamba_batch_size,
        d_model=args.mamba_d_model,
        emb_dim=args.mamba_emb_dim,
    )
    m_train_prob, m_train_emb = _mamba_predict(mamba_art.model, seq_train, device=device, batch_size=args.mamba_batch_size)
    m_val_prob, m_val_emb = _mamba_predict(mamba_art.model, seq_val, device=device, batch_size=args.mamba_batch_size)
    m_eval_prob, m_eval_emb = _mamba_predict(mamba_art.model, seq_eval, device=device, batch_size=args.mamba_batch_size)

    x_lgb_train = _lightgbm_features(ctx_train, m_train_prob, m_train_emb)
    x_lgb_val = _lightgbm_features(ctx_val, m_val_prob, m_val_emb)
    x_lgb_eval = _lightgbm_features(ctx_eval, m_eval_prob, m_eval_emb)
    lgbm = _fit_lgbm(x_lgb_train, y_train)
    a8_train_prob = lgbm.predict_proba(x_lgb_train)
    a8_val_prob = lgbm.predict_proba(x_lgb_val)
    a8_eval_prob = lgbm.predict_proba(x_lgb_eval)

    primary_parent = joblib.load(baseline.primary_parent)
    fallback_parent = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    f_train = _predict_scaled(fallback_parent, train_df, fallback_rt)
    f_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    f_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    s_train_df = _alpha8_state(train_df, p_train, f_train, a8_train_prob, m_train_prob, m_train_emb)
    s_val_df = _alpha8_state(val_df, p_val, f_val, a8_val_prob, m_val_prob, m_val_emb)
    s_eval_df = _alpha8_state(eval_df, p_eval, f_eval, a8_eval_prob, m_eval_prob, m_eval_emb)
    state_norm = _fit_norm(s_train_df)
    x_train = _apply_norm(s_train_df, state_norm)
    x_val = _apply_norm(s_val_df, state_norm)
    x_eval = _apply_norm(s_eval_df, state_norm)

    fee = 0.0005
    slip = 0.0002
    data, data_diag = _build_counterfactual_dataset(train_df, x_train, p_train, f_train, fee=fee, slip=slip)
    print(
        json.dumps(
            {
                "stage": "dsac_start",
                "state_dim": int(x_train.shape[1]),
                "samples": int(len(data.states)),
                "dataset_diagnostics": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    trained = _train_dsac_offline(
        data,
        state_dim=int(x_train.shape[1]),
        action_dim=ACTION_DIM,
        device=device,
        steps=int(args.dsac_steps),
        batch_size=int(args.dsac_batch_size),
    )
    actor: nn.Module = trained["actor"]

    act_train = _policy_action(actor, x_train, device=device)
    act_val = _policy_action(actor, x_val, device=device)
    act_eval = _policy_action(actor, x_eval, device=device)
    a8_train_dec = _compose_decisions(p_train, f_train, act_train)
    a8_val_dec = _compose_decisions(p_val, f_val, act_val)
    a8_eval_dec = _compose_decisions(p_eval, f_eval, act_eval)
    base_train = _combine_primary_fallback(p_train, f_train)
    base_val = _combine_primary_fallback(p_val, f_val)
    base_eval = _combine_primary_fallback(p_eval, f_eval)

    rows: list[dict[str, Any]] = []
    for split, df, base_dec, alpha8_dec in [
        ("train", train_df, base_train, a8_train_dec),
        ("val", val_df, base_val, a8_val_dec),
        ("oos", eval_df, base_eval, a8_eval_dec),
    ]:
        for name, dec in [("baseline_combo", base_dec), ("alpha8_mamba_lgbm_dsac", alpha8_dec)]:
            for cost, vals in _combo_metrics(df, dec).items():
                rows.append({"split": split, "variant": name, "cost": cost, **vals})
    grid = pd.DataFrame(rows)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)

    torch.save(
        {
            "model_id": MODEL_ID,
            "mamba_state_dict": mamba_art.model.state_dict(),
            "dsac_actor_state_dict": actor.state_dict(),
            "state_dim": int(x_train.shape[1]),
            "action_dim": ACTION_DIM,
            "state_columns": list(state_norm["columns"]),
            "state_normalizer": state_norm,
            "context_normalizer": ctx_norm,
            "sequence_cols": SEQUENCE_COLS,
            "label_horizon": int(args.label_horizon),
            "label_barrier": float(args.label_barrier),
        },
        OUT_DIR / "alpha8_mamba_dsac.pt",
    )
    joblib.dump(lgbm, OUT_DIR / "alpha8_directional_lgbm.pkl")
    (OUT_DIR / "state_columns.json").write_text(json.dumps(list(state_norm["columns"]), indent=2) + "\n")

    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 hierarchical hybrid: sticky-v2/future regime context -> CUDA Mamba sequence encoder -> LightGBM directional alpha probabilities -> discrete SAC execution router over skip/primary/fallback.",
        "live_wired": False,
        "baseline_model_id": baseline.model_id,
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "train_csv": str(TRAIN_CSV),
        "eval_csv": str(EVAL_CSV),
        "mamba": {
            "seq_len": int(args.seq_len),
            "d_model": int(args.mamba_d_model),
            "embedding_dim": int(args.mamba_emb_dim),
            "epochs": int(args.mamba_epochs),
            "train_diag": mamba_art.train_diag,
        },
        "lightgbm": {
            "feature_count": int(x_lgb_train.shape[1]),
            "label_horizon": int(args.label_horizon),
            "label_barrier": float(args.label_barrier),
            "train_label_counts": np.bincount(y_train, minlength=3).astype(int).tolist(),
            "val_label_counts": np.bincount(y_val, minlength=3).astype(int).tolist(),
            "oos_label_counts": np.bincount(y_eval, minlength=3).astype(int).tolist(),
        },
        "dsac": {
            "state_dim": int(x_train.shape[1]),
            "action_dim": ACTION_DIM,
            "steps": int(args.dsac_steps),
            "batch_size": int(args.dsac_batch_size),
            "dataset_diagnostics": data_diag,
            "train_diag": trained["train_diag"],
            "action_usage": {
                "train": _usage(act_train),
                "val": _usage(act_val),
                "oos": _usage(act_eval),
            },
        },
        "cost3": {
            "train_baseline": _pick(grid, "train", "baseline_combo", "cost3"),
            "train_alpha8": _pick(grid, "train", "alpha8_mamba_lgbm_dsac", "cost3"),
            "val_baseline": _pick(grid, "val", "baseline_combo", "cost3"),
            "val_alpha8": _pick(grid, "val", "alpha8_mamba_lgbm_dsac", "cost3"),
            "oos_baseline": _pick(grid, "oos", "baseline_combo", "cost3"),
            "oos_alpha8": _pick(grid, "oos", "alpha8_mamba_lgbm_dsac", "cost3"),
        },
        "artifacts": {
            "grid": str(grid_path),
            "torch": str(OUT_DIR / "alpha8_mamba_dsac.pt"),
            "lightgbm": str(OUT_DIR / "alpha8_directional_lgbm.pkl"),
            "state_columns": str(OUT_DIR / "state_columns.json"),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "cost3": summary["cost3"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
