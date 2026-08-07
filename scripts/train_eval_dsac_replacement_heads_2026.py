#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FEATURE_COLS, prepare_features, predict_policy_frame  # noqa: E402
from ensemble.train_rl_dsac_agent import GaussianActor  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit, _quality_scaled_decisions  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    _days,
    backtest_no_limit_exit,
    collect_exit_samples,
)


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_v4_balanced_h144.pkl"
DEFAULT_EXIT_BUNDLE = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_no_limit_exit_governor_fast.pkl"
DEFAULT_SELECTION = ROOT / "data/ensemble/reports/hf_no_limit_exit_final_selection_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_CKPT_DIR = ROOT / "data/ensemble/ckpt/dsac_replacement_heads"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_replacement_heads_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _load_selected(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    sel = obj.get("selected_balanced") or {}
    return dict(sel["entry_config"]), dict(sel["risk_config"]), dict(sel["exit_config"])


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return z.astype(np.float32), mean, std


def _actor_predict(actor: GaussianActor, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    actor.eval()
    x = (np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(x), int(batch)):
            xb = torch.from_numpy(x[s : s + int(batch)]).to(device)
            out = actor.deterministic(xb).squeeze(-1).detach().cpu().numpy()
            outs.append(out.astype(np.float32))
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def _train_actor_head(
    x: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None,
    state_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> tuple[GaussianActor, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(int(seed))
    xz, mean, std = _standardize(x)
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    w = np.ones_like(y, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    w = np.clip(w, 0.05, np.quantile(w, 0.98) if len(w) > 10 else 10.0)
    w = w / max(float(np.mean(w)), 1e-8)
    ds = TensorDataset(torch.from_numpy(xz), torch.from_numpy(y), torch.from_numpy(w))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    actor = GaussianActor(state_dim=int(state_dim), hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(int(epochs)):
        actor.train()
        total = 0.0
        n = 0
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            pred = actor.deterministic(xb).squeeze(-1)
            loss = (F.smooth_l1_loss(pred, yb, reduction="none") * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
    meta = {
        "samples": int(len(xz)),
        "state_dim": int(state_dim),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "final_loss": float(losses[-1]) if losses else None,
        "target_mean": float(np.mean(y)) if len(y) else 0.0,
        "target_std": float(np.std(y)) if len(y) else 0.0,
    }
    return actor, mean, std, meta


def _save_actor(path: Path, actor: GaussianActor, mean: np.ndarray, std: np.ndarray, feature_cols: list[str], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "state_dim": int(len(feature_cols)),
            "feature_cols": list(feature_cols),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "meta": dict(meta),
        },
        path,
    )


def _future_labels(df: pd.DataFrame, *, horizon: int, fee: float, slip: float, edge_scale: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    n = len(df)
    y = np.zeros(n, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    cost = 2.0 * float(fee + slip)
    for i in range(0, max(0, n - int(horizon) - 2)):
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(horizon)]
        if fut.size == 0:
            continue
        long_edge = float(np.max(fut / base - 1.0)) - cost
        short_edge = float(np.max(base / np.maximum(fut, 1e-12) - 1.0)) - cost
        best = max(long_edge, short_edge, 0.0)
        quality[i] = best
        if best <= 0.0015:
            y[i] = 0.0
        elif long_edge >= short_edge:
            y[i] = min(1.0, best / float(edge_scale))
        else:
            y[i] = -min(1.0, best / float(edge_scale))
    meta = {
        "horizon": int(horizon),
        "fee": float(fee),
        "slip": float(slip),
        "long_labels": int((y > 0.05).sum()),
        "short_labels": int((y < -0.05).sum()),
        "cash_labels": int((np.abs(y) <= 0.05).sum()),
        "quality_mean": float(np.mean(quality)),
        "quality_p95": float(np.quantile(quality, 0.95)),
    }
    return y, quality, meta


def _exposure_labels(df: pd.DataFrame, decisions: pd.DataFrame, *, horizon: int, fee: float, slip: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    sides = decisions["side"].astype(int).to_numpy()
    actions = decisions["action"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    n = len(df)
    mult = np.ones(n, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    cost = 2.0 * float(fee + slip)
    for i in range(0, max(0, n - int(horizon) - 2)):
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0 or float(notionals[i]) <= 0.0:
            mult[i] = 0.0
            continue
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(horizon)]
        side_ret = fut / base - 1.0 if int(sides[i]) > 0 else base / np.maximum(fut, 1e-12) - 1.0
        best = float(np.max(side_ret)) - cost
        worst = float(np.min(side_ret))
        quality[i] = best
        if best <= 0.0 or worst <= -0.018:
            mult[i] = 0.0
        elif best < 0.006 or worst <= -0.010:
            mult[i] = 0.5
        elif best > 0.018 and worst > -0.008:
            mult[i] = 1.25
        else:
            mult[i] = 1.0
    target = (mult / 1.25) * 2.0 - 1.0
    meta = {
        "horizon": int(horizon),
        "block": int((mult <= 0.01).sum()),
        "reduce": int(((mult > 0.01) & (mult < 0.75)).sum()),
        "keep": int(((mult >= 0.75) & (mult <= 1.05)).sum()),
        "increase": int((mult > 1.05).sum()),
    }
    return target.astype(np.float32), quality, meta


def _decision_from_entry_raw(base_dec: pd.DataFrame, raw: np.ndarray, *, threshold: float) -> pd.DataFrame:
    out = base_dec.copy()
    active = np.abs(raw) >= float(threshold)
    side = np.where(raw > float(threshold), 1, np.where(raw < -float(threshold), -1, 0))
    train_active_notional = pd.to_numeric(base_dec.loc[base_dec["action"].astype(int) != ACTION_CASH, "notional_exposure"], errors="coerce").dropna()
    train_active_lev = pd.to_numeric(base_dec.loc[base_dec["action"].astype(int) != ACTION_CASH, "leverage"], errors="coerce").dropna()
    default_notional = float(train_active_notional.median()) if len(train_active_notional) else 0.55
    default_lev = float(train_active_lev.median()) if len(train_active_lev) else 1.5
    out.loc[:, "side"] = side.astype(int)
    out.loc[:, "action"] = np.where(side > 0, ACTION_LONG, np.where(side < 0, ACTION_SHORT, ACTION_CASH)).astype(int)
    original_notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    scaled_default = default_notional * np.clip(np.abs(raw), 0.25, 1.0)
    out.loc[:, "notional_exposure"] = np.where(active, np.where(original_notional > 0.0, original_notional, scaled_default), 0.0)
    out.loc[:, "leverage"] = np.where(active, pd.to_numeric(out["leverage"], errors="coerce").fillna(default_lev), 1.0)
    out.loc[:, "position_fraction"] = out["notional_exposure"] / np.maximum(pd.to_numeric(out["leverage"], errors="coerce").fillna(default_lev), 1e-12)
    return out


def _decision_from_entry_filter(base_dec: pd.DataFrame, raw: np.ndarray, *, threshold: float) -> pd.DataFrame:
    out = base_dec.copy()
    base_side = out["side"].astype(int).to_numpy()
    base_active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (base_side != 0)
    model_side = np.where(raw > float(threshold), 1, np.where(raw < -float(threshold), -1, 0))
    block = base_active & (model_side == -base_side)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


def _decision_from_exposure(base_dec: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    out = base_dec.copy()
    mult = np.clip(((raw + 1.0) / 2.0) * 1.25, 0.0, 1.25)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = notional * np.where(active, mult, 0.0)
    block = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "position_fraction"] = notional / np.maximum(pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0), 1e-12)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


class TorchExitProbaModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, actor: GaussianActor, mean: np.ndarray, std: np.ndarray, device: str):
        self.actor = actor
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.device = str(device)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        raw = _actor_predict(self.actor, arr, self.mean, self.std, self.device)
        p = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - p, p]).astype(np.float64)


def _monthly(eval_df: pd.DataFrame, policy: dict[str, Any], exit_model: Any, entry_cfg: dict[str, Any], risk_cfg: dict[str, Any], exit_cfg: dict[str, Any], precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray], fee: float, slip: float) -> dict[str, Any]:
    if "timestamp" not in eval_df.columns:
        return {}
    out = {}
    for name, mask in (
        ("jan", eval_df["timestamp"] < pd.Timestamp("2026-02-01")),
        ("feb", eval_df["timestamp"] >= pd.Timestamp("2026-02-01")),
    ):
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size == 0:
            continue
        base_feat, decisions, close, fill_px = precomputed
        sub_pre = (
            base_feat.iloc[idx].reset_index(drop=True),
            decisions.iloc[idx].reset_index(drop=True),
            close[idx],
            fill_px[idx],
        )
        bt = backtest_no_limit_exit(
            eval_df.loc[mask].reset_index(drop=True),
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(exit_cfg["exit_threshold"]),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=fee,
            slip=slip,
            precomputed=sub_pre,
        )
        out[name] = _compact(bt)
    return out


def _eval_decision_grid(
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    base_precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    candidates: list[tuple[str, pd.DataFrame]],
    *,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    base_feat, _, close, fill_px = base_precomputed
    rows = []
    for name, dec in candidates:
        pre = (base_feat, dec, close, fill_px)
        bt = backtest_no_limit_exit(
            eval_df,
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(exit_cfg["exit_threshold"]),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=fee,
            slip=slip,
            precomputed=pre,
        )
        rows.append({"name": name, "eval": _compact(bt), "monthly": _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, pre, fee, slip)})
    return sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain DSAC actor heads for four replacement points and backtest each.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--exit-samples", type=int, default=80000)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)

    train_pre = _base_frame(train_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    train_feat, train_base_dec, _, _ = train_pre
    eval_feat, eval_base_dec, eval_close, eval_fill = eval_pre

    x_train = train_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    x_eval = eval_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    entry_y, entry_quality, entry_label_meta = _future_labels(train_df, horizon=int(args.horizon), fee=float(args.fee), slip=float(args.slip), edge_scale=0.025)
    exposure_y, exposure_quality, exposure_label_meta = _exposure_labels(train_df, train_base_dec, horizon=int(args.horizon), fee=float(args.fee), slip=float(args.slip))

    entry_actor, entry_mean, entry_std, entry_meta = _train_actor_head(
        x_train,
        entry_y,
        weights=1.0 + np.minimum(entry_quality / 0.01, 8.0),
        state_dim=x_train.shape[1],
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
    )
    exposure_actor, exposure_mean, exposure_std, exposure_meta = _train_actor_head(
        x_train,
        exposure_y,
        weights=1.0 + np.minimum(np.abs(exposure_quality) / 0.01, 6.0),
        state_dim=x_train.shape[1],
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed) + 1,
        device=device,
    )
    _save_actor(args.ckpt_dir / "entry_actor.pth", entry_actor, entry_mean, entry_std, list(FEATURE_COLS), {"candidate": "entry", **entry_meta, "labels": entry_label_meta})
    _save_actor(args.ckpt_dir / "exposure_actor.pth", exposure_actor, exposure_mean, exposure_std, list(FEATURE_COLS), {"candidate": "exposure", **exposure_meta, "labels": exposure_label_meta})

    entry_raw = _actor_predict(entry_actor, x_eval, entry_mean, entry_std, device)
    exposure_raw = _actor_predict(exposure_actor, x_eval, exposure_mean, exposure_std, device)

    baseline = _eval_decision_grid(
        eval_df,
        policy,
        base_exit_model,
        entry_cfg,
        risk_cfg,
        exit_cfg,
        eval_pre,
        [("baseline_hf_no_limit", eval_base_dec)],
        fee=float(args.fee),
        slip=float(args.slip),
    )[0]

    entry_replace_rows = _eval_decision_grid(
        eval_df,
        policy,
        base_exit_model,
        entry_cfg,
        risk_cfg,
        exit_cfg,
        eval_pre,
        [(f"entry_replace_th{th:.2f}", _decision_from_entry_raw(eval_base_dec, entry_raw, threshold=th)) for th in (0.05, 0.10, 0.15, 0.20, 0.30)],
        fee=float(args.fee),
        slip=float(args.slip),
    )
    entry_filter_rows = _eval_decision_grid(
        eval_df,
        policy,
        base_exit_model,
        entry_cfg,
        risk_cfg,
        exit_cfg,
        eval_pre,
        [(f"entry_filter_th{th:.2f}", _decision_from_entry_filter(eval_base_dec, entry_raw, threshold=th)) for th in (0.05, 0.10, 0.15, 0.20, 0.30)],
        fee=float(args.fee),
        slip=float(args.slip),
    )
    exposure_rows = _eval_decision_grid(
        eval_df,
        policy,
        base_exit_model,
        entry_cfg,
        risk_cfg,
        exit_cfg,
        eval_pre,
        [("exposure_scaler", _decision_from_exposure(eval_base_dec, exposure_raw))],
        fee=float(args.fee),
        slip=float(args.slip),
    )

    x_exit, y_exit, exit_sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=24,
        min_age=3,
        max_age=288,
        age_stride=12,
        future_horizon=int(args.horizon),
        exit_edge=0.0015,
        adverse_gap=0.012,
        max_samples=int(args.exit_samples),
        seed=int(args.seed),
    )
    exit_target = y_exit.astype(np.float32) * 2.0 - 1.0
    x_exit_np = x_exit.reindex(columns=MODEL_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    exit_weights = np.where(y_exit > 0, max(1.0, (len(y_exit) - float(y_exit.sum())) / max(float(y_exit.sum()), 1.0)), 1.0)
    exit_actor, exit_mean, exit_std, exit_meta = _train_actor_head(
        x_exit_np,
        exit_target,
        weights=exit_weights,
        state_dim=x_exit_np.shape[1],
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed) + 2,
        device=device,
    )
    _save_actor(args.ckpt_dir / "exit_actor.pth", exit_actor, exit_mean, exit_std, list(MODEL_COLS), {"candidate": "exit", **exit_meta, "sample_meta": exit_sample_meta})
    exit_model = TorchExitProbaModel(exit_actor, exit_mean, exit_std, device)

    exit_rows = []
    for th in (0.35, 0.45, 0.55, 0.65, 0.75):
        bt = backtest_no_limit_exit(
            eval_df,
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(th),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=float(args.fee),
            slip=float(args.slip),
            precomputed=eval_pre,
        )
        exit_rows.append({"name": f"exit_replace_th{th:.2f}", "eval": _compact(bt), "monthly": _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}, eval_pre, float(args.fee), float(args.slip))})
    exit_rows = sorted(exit_rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)

    full_rows = []
    for entry_row in entry_replace_rows[:2]:
        th = float(str(entry_row["name"]).split("th")[-1])
        dec = _decision_from_entry_raw(eval_base_dec, entry_raw, threshold=th)
        pre = (eval_feat, dec, eval_close, eval_fill)
        for exit_th in (0.45, 0.55, 0.65):
            bt = backtest_no_limit_exit(
                eval_df,
                policy,
                exit_model,
                entry_config=entry_cfg,
                risk_config=risk_cfg,
                exit_threshold=float(exit_th),
                min_exit_age=int(exit_cfg["min_exit_age"]),
                fee=float(args.fee),
                slip=float(args.slip),
                precomputed=pre,
            )
            full_rows.append({"name": f"full_lifecycle_entry{th:.2f}_exit{exit_th:.2f}", "eval": _compact(bt), "monthly": _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, {"exit_threshold": exit_th, "min_exit_age": exit_cfg["min_exit_age"]}, pre, float(args.fee), float(args.slip))})
    full_rows = sorted(full_rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)

    candidates = {
        "entry_replace": entry_replace_rows,
        "entry_filter": entry_filter_rows,
        "exposure_scaler": exposure_rows,
        "exit_replace": exit_rows,
        "full_lifecycle": full_rows,
    }
    best_by_candidate = {k: (v[0] if v else None) for k, v in candidates.items()}
    all_rows = [baseline] + [r for rows in candidates.values() for r in rows]
    ranked = sorted(all_rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    report = {
        "type": "dsac_replacement_heads_2026",
        "note": "DSAC GaussianActor heads are retrained for each replacement point; critic training is not used here so this is an actor-head replacement study, not a full standalone DSAC policy promotion.",
        "policy": str(args.policy),
        "exit_bundle": str(args.exit_bundle),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "ckpt_dir": str(args.ckpt_dir),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "baseline": baseline,
        "label_meta": {
            "entry": entry_label_meta,
            "exposure": exposure_label_meta,
            "exit": exit_sample_meta,
        },
        "train_meta": {
            "entry": entry_meta,
            "exposure": exposure_meta,
            "exit": exit_meta,
        },
        "best_by_candidate": best_by_candidate,
        "ranked": ranked[:30],
        "candidates": candidates,
        "decision": {
            "best_name": ranked[0]["name"] if ranked else None,
            "best_pnl": ranked[0]["eval"]["pnl"] if ranked else None,
            "baseline_pnl": baseline["eval"]["pnl"],
            "delta_vs_baseline": float((ranked[0]["eval"]["pnl"] if ranked else 0.0) - baseline["eval"]["pnl"]),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "baseline": baseline, "best_by_candidate": best_by_candidate, "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
