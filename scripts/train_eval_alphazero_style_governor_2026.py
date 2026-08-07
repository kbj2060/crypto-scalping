#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, FEATURE_COLS  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    backtest_no_limit_exit,
    collect_exit_samples,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)


DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/alphazero_style/az_policy_value_governor.pt"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alphazero_style_governor_2026.json"
ENTRY_ACTIONS = ("block", "half", "keep", "boost")
EXIT_ACTIONS = ("hold", "exit")


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return z.astype(np.float32), mean, std


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 192):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.policy = nn.Linear(int(hidden_dim), int(n_actions))
        self.value = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


@dataclass
class PVBundle:
    net: PolicyValueNet
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]
    actions: tuple[str, ...]


def _train_pv(
    x: np.ndarray,
    pi: np.ndarray,
    value: np.ndarray,
    *,
    n_actions: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> tuple[PolicyValueNet, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(int(seed))
    xz, mean, std = _standardize(x)
    pi = np.asarray(pi, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    ds = TensorDataset(torch.from_numpy(xz), torch.from_numpy(pi), torch.from_numpy(value))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    net = PolicyValueNet(xz.shape[1], int(n_actions), hidden_dim=int(hidden_dim)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(int(epochs)):
        total = 0.0
        n = 0
        net.train()
        for xb, pib, vb in loader:
            xb = xb.to(device)
            pib = pib.to(device)
            vb = vb.to(device)
            logits, pred_v = net(xb)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pib * logp).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(pred_v, vb)
            entropy = -(torch.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
            loss = policy_loss + 0.65 * value_loss - 0.01 * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
    meta = {
        "samples": int(len(xz)),
        "state_dim": int(xz.shape[1]),
        "n_actions": int(n_actions),
        "epochs": int(epochs),
        "final_loss": float(losses[-1]) if losses else None,
        "value_mean": float(np.mean(value)) if len(value) else 0.0,
        "value_std": float(np.std(value)) if len(value) else 0.0,
        "policy_entropy_mean": float(-(pi * np.log(np.maximum(pi, 1e-9))).sum(axis=1).mean()) if len(pi) else 0.0,
    }
    return net, mean, std, meta


def _predict_pv(bundle: PVBundle, x: np.ndarray, device: str, batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    arr = (np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) - bundle.mean) / bundle.std
    probs: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    bundle.net.eval()
    with torch.no_grad():
        for s in range(0, len(arr), int(batch)):
            xb = torch.from_numpy(arr[s : s + int(batch)]).to(device)
            logits, value = bundle.net(xb)
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32))
            vals.append(value.detach().cpu().numpy().astype(np.float32))
    return (np.concatenate(probs) if probs else np.zeros((0, len(bundle.actions)), dtype=np.float32), np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32))


def _rollout_entry_targets(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    x: pd.DataFrame,
    *,
    horizon: int,
    fee: float,
    slip: float,
    temperature: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    idx = np.flatnonzero((actions != ACTION_CASH) & (sides != 0) & (notionals > 0.0))
    idx = idx[idx < len(df) - int(horizon) - 2]
    if len(idx) > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_samples), replace=False))
    rows = x.iloc[idx].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pi = np.zeros((len(idx), len(ENTRY_ACTIONS)), dtype=np.float32)
    value = np.zeros(len(idx), dtype=np.float32)
    scales = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
    cost = 2.0 * float(fee + slip)
    for j, i in enumerate(idx):
        side = int(sides[i])
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(horizon)]
        side_ret = fut / base - 1.0 if side > 0 else base / np.maximum(fut, 1e-12) - 1.0
        vals = []
        for scale in scales:
            n = float(notionals[i]) * float(scale)
            if n <= 0.0:
                vals.append(0.0)
                continue
            path = side_ret * n
            run_min = np.minimum.accumulate(path)
            # Search value: best executable exit minus adverse excursion and turnover cost.
            score_path = path - 0.55 * np.maximum(0.0, -run_min) - cost * n
            vals.append(float(np.max(score_path)))
        vals_np = np.asarray(vals, dtype=np.float64)
        z = vals_np / max(float(temperature), 1e-6)
        z = z - np.max(z)
        p = np.exp(z)
        p = p / max(float(p.sum()), 1e-12)
        pi[j] = p.astype(np.float32)
        value[j] = float(np.tanh(np.max(vals_np) / 0.08))
    meta = {
        "samples": int(len(idx)),
        "horizon": int(horizon),
        "temperature": float(temperature),
        "action_counts": {ENTRY_ACTIONS[i]: int(np.argmax(pi, axis=1).tolist().count(i)) for i in range(len(ENTRY_ACTIONS))} if len(pi) else {},
    }
    return rows, pi, value, meta


def _rollout_exit_targets(
    train_df: pd.DataFrame,
    policy: dict[str, Any],
    entry_cfg: dict[str, Any],
    *,
    fee: float,
    slip: float,
    horizon: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_exit, y_exit, sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(fee),
        slip=float(slip),
        entry_stride=24,
        min_age=3,
        max_age=288,
        age_stride=12,
        future_horizon=int(horizon),
        exit_edge=0.0015,
        adverse_gap=0.012,
        max_samples=int(max_samples),
        seed=int(seed),
    )
    rows = x_exit.reindex(columns=MODEL_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    # collect_exit_samples already encodes search-improved binary exit labels.
    pi = np.zeros((len(y_exit), 2), dtype=np.float32)
    pi[:, 1] = y_exit.astype(np.float32) * 0.85 + 0.075
    pi[:, 0] = 1.0 - pi[:, 1]
    value = np.where(y_exit > 0, 0.35, 0.10).astype(np.float32)
    meta = dict(sample_meta)
    meta["action_counts"] = {"hold": int((y_exit == 0).sum()), "exit": int((y_exit == 1).sum())}
    return rows, pi, value, meta


def _entry_modified_decisions(base_dec: pd.DataFrame, probs: np.ndarray, *, min_keep_prob: float = 0.0) -> pd.DataFrame:
    out = base_dec.copy()
    act = np.argmax(probs, axis=1)
    # Optional probability floor: weak predictions stay at keep to avoid overfitting blocks.
    confidence = probs.max(axis=1)
    keep_idx = ENTRY_ACTIONS.index("keep")
    act = np.where(confidence < float(min_keep_prob), keep_idx, act)
    scales = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(active, notional * scales[act], 0.0)
    block = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "position_fraction"] = notional / np.maximum(pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0), 1e-12)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


class AZExitModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, bundle: PVBundle, device: str):
        self.bundle = bundle
        self.device = str(device)
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in bundle.net.state_dict().items()}
        self._w0 = sd["trunk.0.weight"]
        self._b0 = sd["trunk.0.bias"]
        self._ln0_w = sd["trunk.1.weight"]
        self._ln0_b = sd["trunk.1.bias"]
        self._w1 = sd["trunk.3.weight"]
        self._b1 = sd["trunk.3.bias"]
        self._ln1_w = sd["trunk.4.weight"]
        self._ln1_b = sd["trunk.4.bias"]
        self._pw = sd["policy.weight"]
        self._pb = sd["policy.bias"]

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * weight + bias

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) - self.bundle.mean) / self.bundle.std
        h = arr @ self._w0.T + self._b0
        h = self._silu(self._layer_norm(h, self._ln0_w, self._ln0_b))
        h = h @ self._w1.T + self._b1
        h = self._silu(self._layer_norm(h, self._ln1_w, self._ln1_b))
        logits = h @ self._pw.T + self._pb
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return (ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)).astype(np.float64)


def _monthly(
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    fee: float,
    slip: float,
) -> dict[str, Any]:
    if "timestamp" not in eval_df.columns:
        return {}
    out = {}
    for name, mask in (
        ("jan", eval_df["timestamp"] < pd.Timestamp("2026-02-01")),
        ("feb", eval_df["timestamp"] >= pd.Timestamp("2026-02-01")),
    ):
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if not len(idx):
            continue
        base_feat, decisions, close, fill = precomputed
        sub_pre = (
            base_feat.iloc[idx].reset_index(drop=True),
            decisions.iloc[idx].reset_index(drop=True),
            close[idx],
            fill[idx],
        )
        bt = backtest_no_limit_exit(
            eval_df.loc[mask].reset_index(drop=True),
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(exit_cfg["exit_threshold"]),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=float(fee),
            slip=float(slip),
            precomputed=sub_pre,
        )
        out[name] = _compact(bt)
    return out


def _run_bt(
    name: str,
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    bt = backtest_no_limit_exit(
        eval_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(fee),
        slip=float(slip),
        precomputed=precomputed,
    )
    return {"name": name, "eval": _compact(bt), "monthly": _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, precomputed, fee, slip)}


def _variant_components(
    name: str,
    eval_dec: pd.DataFrame,
    entry_probs: np.ndarray,
    base_exit_model: Any,
    az_exit_model: Any,
    exit_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    dec = eval_dec
    exit_model = base_exit_model
    cfg = exit_cfg
    if name.startswith("az_entry_floor"):
        floor = float(name.split("floor", 1)[1].split("_", 1)[0])
        dec = _entry_modified_decisions(eval_dec, entry_probs, min_keep_prob=floor)
        if "_exit" in name:
            th = float(name.rsplit("_exit", 1)[1])
            exit_model = az_exit_model
            cfg = {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}
    elif name.startswith("az_exit_only"):
        th = float(name.split("az_exit_only", 1)[1])
        exit_model = az_exit_model
        cfg = {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}
    return dec, exit_model, cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero-style policy/value governor for current HF no-limit architecture.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--entry-samples", type=int, default=90000)
    p.add_argument("--exit-samples", type=int, default=90000)
    p.add_argument("--temperature", type=float, default=0.012)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_pre = _base_frame(train_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    train_feat, train_dec, _, _ = train_pre
    eval_feat, eval_dec, eval_close, eval_fill = eval_pre

    entry_x, entry_pi, entry_v, entry_meta = _rollout_entry_targets(
        train_df,
        train_dec,
        train_feat.reindex(columns=FEATURE_COLS),
        horizon=int(args.horizon),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.entry_samples),
        seed=int(args.seed),
    )
    entry_net, entry_mean, entry_std, entry_train_meta = _train_pv(
        entry_x,
        entry_pi,
        entry_v,
        n_actions=len(ENTRY_ACTIONS),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
    )
    exit_x, exit_pi, exit_v, exit_meta = _rollout_exit_targets(
        train_df,
        policy,
        entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        horizon=int(args.horizon),
        max_samples=int(args.exit_samples),
        seed=int(args.seed),
    )
    exit_net, exit_mean, exit_std, exit_train_meta = _train_pv(
        exit_x,
        exit_pi,
        exit_v,
        n_actions=len(EXIT_ACTIONS),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed) + 1,
    )

    entry_bundle = PVBundle(entry_net, entry_mean, entry_std, list(FEATURE_COLS), ENTRY_ACTIONS)
    exit_bundle_pv = PVBundle(exit_net, exit_mean, exit_std, list(MODEL_COLS), EXIT_ACTIONS)
    eval_x = eval_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    entry_probs, entry_values = _predict_pv(entry_bundle, eval_x, device)
    az_exit_model = AZExitModel(exit_bundle_pv, device)

    rows = []
    base_pre = (eval_feat, eval_dec, eval_close, eval_fill)
    rows.append(_run_bt("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, base_pre, fee=float(args.fee), slip=float(args.slip)))
    for floor in (0.0, 0.40, 0.50, 0.60, 0.70):
        dec = _entry_modified_decisions(eval_dec, entry_probs, min_keep_prob=float(floor))
        rows.append(_run_bt(f"az_entry_floor{floor:.2f}", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, (eval_feat, dec, eval_close, eval_fill), fee=float(args.fee), slip=float(args.slip)))
        for ex_th in (0.45, 0.55, 0.65, 0.75):
            rows.append(
                _run_bt(
                    f"az_entry_floor{floor:.2f}_exit{ex_th:.2f}",
                    eval_df,
                    policy,
                    az_exit_model,
                    entry_cfg,
                    risk_cfg,
                    {"exit_threshold": ex_th, "min_exit_age": exit_cfg["min_exit_age"]},
                    (eval_feat, dec, eval_close, eval_fill),
                    fee=float(args.fee),
                    slip=float(args.slip),
                )
            )
    for ex_th in (0.45, 0.55, 0.65, 0.75):
        rows.append(
            _run_bt(
                f"az_exit_only{ex_th:.2f}",
                eval_df,
                policy,
                az_exit_model,
                entry_cfg,
                risk_cfg,
                {"exit_threshold": ex_th, "min_exit_age": exit_cfg["min_exit_age"]},
                base_pre,
                fee=float(args.fee),
                slip=float(args.slip),
            )
        )

    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    best_names = [ranked[0]["name"], "baseline_hf_no_limit"]
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_rows = []
        for row in rows:
            if row["name"] not in best_names:
                continue
            # Reconstruct only the chosen variants.
            name = row["name"]
            dec, exit_model, cfg = _variant_components(name, eval_dec, entry_probs, base_exit_model, az_exit_model, exit_cfg)
            cost_rows.append(_run_bt(name, eval_df, policy, exit_model, entry_cfg, risk_cfg, cfg, (eval_feat, dec, eval_close, eval_fill), fee=float(args.fee) * mult, slip=float(args.slip) * mult))
        cost_stress[f"cost_{mult:g}x"] = cost_rows

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "alphazero_style_policy_value_governor",
            "entry": {
                "state_dict": entry_net.state_dict(),
                "mean": entry_mean.astype(np.float32),
                "std": entry_std.astype(np.float32),
                "feature_cols": list(FEATURE_COLS),
                "actions": list(ENTRY_ACTIONS),
                "meta": entry_train_meta,
            },
            "exit": {
                "state_dict": exit_net.state_dict(),
                "mean": exit_mean.astype(np.float32),
                "std": exit_std.astype(np.float32),
                "feature_cols": list(MODEL_COLS),
                "actions": list(EXIT_ACTIONS),
                "meta": exit_train_meta,
            },
        },
        args.model_out,
    )
    report = {
        "type": "alphazero_style_governor_2026",
        "note": "AlphaZero-inspired adaptation: policy/value networks trained on search-improved market-replay targets. There is no true self-play because market path is exogenous; OOS inference uses only current features.",
        "model_out": str(args.model_out),
        "policy": str(args.policy),
        "exit_bundle": str(args.exit_bundle),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "label_meta": {"entry": entry_meta, "exit": exit_meta},
        "train_meta": {"entry": entry_train_meta, "exit": exit_train_meta},
        "entry_eval_policy_distribution": {
            action: int((np.argmax(entry_probs, axis=1) == i).sum()) for i, action in enumerate(ENTRY_ACTIONS)
        },
        "entry_value_mean": float(np.mean(entry_values)) if len(entry_values) else 0.0,
        "ranked": ranked,
        "cost_stress": cost_stress,
        "decision": {
            "best_name": ranked[0]["name"],
            "best_pnl": ranked[0]["eval"]["pnl"],
            "baseline_pnl": next(r for r in rows if r["name"] == "baseline_hf_no_limit")["eval"]["pnl"],
            "delta_vs_baseline": float(ranked[0]["eval"]["pnl"] - next(r for r in rows if r["name"] == "baseline_hf_no_limit")["eval"]["pnl"]),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "model": str(args.model_out), "top": ranked[:8], "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
