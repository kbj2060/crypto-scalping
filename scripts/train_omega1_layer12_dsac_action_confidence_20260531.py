#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_omega1_layer12_action_confidence_20260531 as base


MODEL_ID = "omega1_layer12_dsac_action_confidence_20260531"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_layer12_dsac_action_confidence_20260531"


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


class Actor(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), torch.clamp(self.log_std(h), -5.0, 1.0)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        z = mu + torch.randn_like(mu) * std
        action = torch.tanh(z)
        log_prob = -0.5 * (((z - mu) / torch.clamp(std, min=1e-6)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = log_prob - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6))
        return action, log_prob

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(x)
        return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, dim: int, hidden: int, n_quantiles: int, dropout: float):
        super().__init__()
        self.n_quantiles = int(n_quantiles)

        def block() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dim + 1, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.n_quantiles),
            )

        self.q1 = block()
        self.q2 = block()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x, a], dim=1)
        return self.q1(z), self.q2(z)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return z, mean, std


def _standardize_apply(x: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return np.nan_to_num((arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _single_trade_rewards(
    frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
    exposure: float,
) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    n = len(frame)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(0, max(0, n - 2)):
        fill_i = min(i + 1, n - 1)
        for cls, side in ((1, 1), (2, -1)):
            entry = open_px[fill_i] * (1.0 + slip if side > 0 else 1.0 - slip)
            exit_px = open_px[min(fill_i + max_hold_bars, n - 1)]
            reason_px = exit_px * (1.0 - slip if side > 0 else 1.0 + slip)
            end_j = min(fill_i + int(max_hold_bars), n - 1)
            for j in range(fill_i, end_j + 1):
                if side > 0:
                    tp_hit = high[j] >= entry * (1.0 + tp_pct)
                    sl_hit = low[j] <= entry * (1.0 - sl_pct)
                    if tp_hit and sl_hit:
                        reason_px = entry * (1.0 - sl_pct) * (1.0 - slip)
                        break
                    if tp_hit:
                        reason_px = entry * (1.0 + tp_pct) * (1.0 - slip)
                        break
                    if sl_hit:
                        reason_px = entry * (1.0 - sl_pct) * (1.0 - slip)
                        break
                else:
                    tp_hit = low[j] <= entry * (1.0 - tp_pct)
                    sl_hit = high[j] >= entry * (1.0 + sl_pct)
                    if tp_hit and sl_hit:
                        reason_px = entry * (1.0 + sl_pct) * (1.0 + slip)
                        break
                    if tp_hit:
                        reason_px = entry * (1.0 - tp_pct) * (1.0 + slip)
                        break
                    if sl_hit:
                        reason_px = entry * (1.0 + sl_pct) * (1.0 + slip)
                        break
            raw = (reason_px - entry) / max(entry, 1e-12) if side > 0 else (entry - reason_px) / max(entry, 1e-12)
            cash_after_entry_fee = 1.0 - fee * exposure
            pnl = cash_after_entry_fee * (1.0 + raw * exposure) - cash_after_entry_fee * fee * exposure - 1.0
            out[i, cls] = np.float32(pnl)
    return out


def _quantile_huber(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    td = target[:, None] - pred
    huber = torch.where(td.abs() <= 1.0, 0.5 * td.pow(2), td.abs() - 0.5)
    weight = (taus[None, :] - (td.detach() < 0).float()).abs()
    return (weight * huber).mean()


def _prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    train = base._read_csv(args.split_dir / "training_features_2025.csv")
    oos = base._read_csv(args.split_dir / "training_features_2026_rebuilt.csv")
    train = base._add_layer2(train, 2025, args)
    oos = base._add_layer2(oos, 2026, args)
    train = base._add_labels(train, 2025, args.label_dir)
    oos = base._add_labels(oos, 2026, args.label_dir)
    feature_sets = base._feature_sets(train, oos)
    keep = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    if keep:
        missing = sorted(set(keep) - set(feature_sets))
        if missing:
            raise ValueError(f"unknown feature sets: {missing}; available={sorted(feature_sets)}")
        feature_sets = {name: feature_sets[name] for name in keep}
    return train, oos, feature_sets


def _train_dsac(
    x: np.ndarray,
    rewards3: np.ndarray,
    *,
    hidden: int,
    n_quantiles: int,
    updates: int,
    batch_size: int,
    lr: float,
    cvar_frac: float,
    action_l1: float,
    bc_lambda: float,
    seed: int,
    device: str,
) -> tuple[Actor, Critic, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    actor = Actor(x.shape[1], int(hidden), dropout=0.05).to(device)
    critic = Critic(x.shape[1], int(hidden), int(n_quantiles), dropout=0.05).to(device)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=float(lr), weight_decay=1e-4)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=float(lr), weight_decay=1e-4)
    taus = torch.linspace(0.5 / int(n_quantiles), 1.0 - 0.5 / int(n_quantiles), int(n_quantiles), device=device)
    class_actions = np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
    best_cls = rewards3.argmax(axis=1)
    bc_target = class_actions[best_cls]
    reward_scale = max(float(np.nanstd(rewards3[:, 1:])), 1e-4)
    scaled_rewards = np.tanh(rewards3 / max(reward_scale * 2.5, 1e-6)).astype(np.float32)
    logs: list[dict[str, float]] = []
    n = len(x)
    for step in range(1, int(updates) + 1):
        idx = rng.integers(0, n, size=int(batch_size))
        cls = rng.integers(0, 3, size=int(batch_size))
        xb = torch.from_numpy(x[idx]).to(device)
        ab = torch.from_numpy(class_actions[cls, None]).to(device)
        rb = torch.from_numpy(scaled_rewards[idx, cls]).to(device)
        q1, q2 = critic(xb, ab)
        critic_loss = _quantile_huber(q1, rb, taus) + _quantile_huber(q2, rb, taus)
        opt_c.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 3.0)
        opt_c.step()

        idx_a = rng.integers(0, n, size=int(batch_size))
        xa = torch.from_numpy(x[idx_a]).to(device)
        act, logp = actor.sample(xa)
        qa1, qa2 = critic(xa, act)
        qa = torch.minimum(qa1, qa2).sort(dim=1).values
        k = max(1, int(int(n_quantiles) * float(cvar_frac)))
        q_cvar = qa[:, :k].mean(dim=1)
        bc = torch.from_numpy(bc_target[idx_a, None].astype(np.float32)).to(device)
        actor_loss = -q_cvar.mean() + float(action_l1) * act.abs().mean() + float(bc_lambda) * F.smooth_l1_loss(act, bc)
        opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 3.0)
        opt_a.step()
        if step == 1 or step % 500 == 0 or step == int(updates):
            logs.append(
                {
                    "step": float(step),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "mean_abs_action": float(act.detach().abs().mean().cpu()),
                    "mean_cvar": float(q_cvar.detach().mean().cpu()),
                }
            )
    return actor, critic, {"logs": logs, "reward_scale": float(reward_scale)}


def _predict_raw(actor: Actor, x: np.ndarray, *, device: str, batch_size: int = 8192) -> np.ndarray:
    outs: list[np.ndarray] = []
    actor.eval()
    with torch.no_grad():
        for s in range(0, len(x), int(batch_size)):
            xb = torch.from_numpy(x[s : s + int(batch_size)]).to(device)
            outs.append(actor.deterministic(xb).squeeze(1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def _proba_from_raw(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(np.asarray(raw, dtype=np.float64), -1.0, 1.0)
    strength = np.clip(np.abs(raw), 0.0, 1.0)
    proba = np.zeros((len(raw), 3), dtype=np.float64)
    proba[:, 0] = 1.0 - strength
    proba[:, 1] = np.where(raw > 0.0, strength, 0.0)
    proba[:, 2] = np.where(raw < 0.0, strength, 0.0)
    return proba


def _metrics(y: np.ndarray, raw: np.ndarray, threshold: float) -> dict[str, Any]:
    proba = _proba_from_raw(raw)
    action = np.where(raw >= float(threshold), 1, np.where(raw <= -float(threshold), 2, 0)).astype(np.int64)
    trade = action != 0
    try:
        auc = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except Exception:
        auc = float("nan")
    return {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, action)),
        "macro_f1": float(f1_score(y, action, average="macro")),
        "ovr_auc": auc,
        "proxy_trades": int(trade.sum()),
        "proxy_long_trades": int((action == 1).sum()),
        "proxy_short_trades": int((action == 2).sum()),
        "proxy_trade_rate": float(trade.mean()),
        "proxy_wr": float((action[trade] == y[trade]).mean()) if trade.any() else None,
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(action, minlength=3))},
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "mean_confidence": float(np.mean(np.max(proba, axis=1))),
        "mean_abs_raw": float(np.mean(np.abs(raw))),
    }


def _decisions(frame: pd.DataFrame, raw: np.ndarray, threshold: float) -> pd.DataFrame:
    action = np.where(raw >= float(threshold), 1, np.where(raw <= -float(threshold), 2, 0)).astype(np.int64)
    proba = _proba_from_raw(raw)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "action": action,
            "confidence": np.max(proba, axis=1),
            "raw_action": raw,
            "p_cash": proba[:, 0],
            "p_long": proba[:, 1],
            "p_short": proba[:, 2],
        }
    )


def _select_threshold(y: np.ndarray, raw: np.ndarray, frame: pd.DataFrame, args: argparse.Namespace, grid: list[float]) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for threshold in grid:
        m = _metrics(y, raw, threshold)
        dec = _decisions(frame.reset_index(drop=True), raw, threshold)
        cost = base._cost_metrics(frame.reset_index(drop=True), dec, args)
        c3 = cost["cost3"]
        calmar = float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))
        score = calmar if int(c3["trades"]) >= int(args.min_val_trades) else -1.0e9
        rows.append({"threshold": float(threshold), "score": score, "metrics": m, "validation_backtest": cost})
    best = max(rows, key=lambda r: float(r["score"]))
    return float(best["threshold"]), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight offline DSAC signed-intent model on Omega1 Layer1+Layer2 inputs.")
    parser.add_argument("--split-dir", type=Path, default=base.DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=base.DEFAULT_LABEL_DIR)
    parser.add_argument("--ai-dir", type=Path, default=base.DEFAULT_AI_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=base.DEFAULT_CHRONOS_DIR)
    parser.add_argument("--regime3-stability-dir", type=Path, default=base.DEFAULT_REGIME3_STABILITY_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=base.DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=base.DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--dir3-patch-dir", type=Path, default=base.DEFAULT_DIR3_PATCH_DIR)
    parser.add_argument("--dir3-vsnlstm-dir", type=Path, default=base.DEFAULT_DIR3_VSNLSTM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-sets", default="l1all_safe_layer2,architect_strict_l1all_layer2")
    parser.add_argument("--val-start", default="2025-10-01")
    parser.add_argument("--confidence-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70")
    parser.add_argument("--min-val-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-quantiles", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tp-pct", type=float, default=0.018)
    parser.add_argument("--sl-pct", type=float, default=0.010)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slip", type=float, default=0.00015)
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, oos, feature_sets = _prepare(args)
    y_all = train["zigzag_action"].astype(int).to_numpy()
    y_oos = oos["zigzag_action"].astype(int).to_numpy()
    val_mask = pd.to_datetime(train["timestamp"]) >= pd.Timestamp(args.val_start)
    fit_idx = np.flatnonzero(~val_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    if len(fit_idx) < 1000 or len(val_idx) < 1000:
        raise RuntimeError(f"bad 2025 fit/validation split: fit={len(fit_idx)} val={len(val_idx)}")
    thresholds = [float(x.strip()) for x in str(args.confidence_grid).split(",") if x.strip()]

    reward_full = _single_trade_rewards(
        train,
        fee=float(args.fee) * 3.0,
        slip=float(args.slip) * 3.0,
        tp_pct=float(args.tp_pct),
        sl_pct=float(args.sl_pct),
        max_hold_bars=int(args.max_hold_bars),
        exposure=float(args.exposure),
    )
    configs = [
        {"cvar_frac": 0.35, "action_l1": 0.015, "bc_lambda": 0.02},
        {"cvar_frac": 0.55, "action_l1": 0.010, "bc_lambda": 0.01},
    ]
    runs: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    for set_name, cols in feature_sets.items():
        x_fit, mean, std = _standardize_fit(train.iloc[fit_idx][cols])
        x_val = _standardize_apply(train.iloc[val_idx][cols], mean, std)
        for ci, cfg in enumerate(configs):
            actor, critic, meta = _train_dsac(
                x_fit,
                reward_full[fit_idx],
                hidden=int(args.hidden),
                n_quantiles=int(args.n_quantiles),
                updates=int(args.updates),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                cvar_frac=float(cfg["cvar_frac"]),
                action_l1=float(cfg["action_l1"]),
                bc_lambda=float(cfg["bc_lambda"]),
                seed=int(args.seed) + ci,
                device=device,
            )
            val_raw = _predict_raw(actor, x_val, device=device)
            threshold, threshold_grid = _select_threshold(y_all[val_idx], val_raw, train.iloc[val_idx], args, thresholds)
            val_metrics = _metrics(y_all[val_idx], val_raw, threshold)
            val_dec = _decisions(train.iloc[val_idx].reset_index(drop=True), val_raw, threshold)
            val_cost = base._cost_metrics(train.iloc[val_idx].reset_index(drop=True), val_dec, args)
            c3 = val_cost["cost3"]
            calmar = float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))
            run = {
                "feature_set": set_name,
                "feature_count": int(len(cols)),
                "config": cfg,
                "threshold": float(threshold),
                "validation": val_metrics,
                "validation_backtest": val_cost,
                "validation_cost3_calmar": calmar,
                "threshold_grid": threshold_grid,
                "train_meta": meta,
                "selection_score": calmar,
            }
            runs.append(run)
            print(json.dumps({"run": run}, ensure_ascii=False, default=_json_default), flush=True)
            if best is None or float(run["selection_score"]) > float(best["selection_score"]):
                best = {**run, "feature_cols": cols}
                best_state = {
                    "actor_state": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                    "critic_state": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
                    "mean": mean,
                    "std": std,
                }
    assert best is not None and best_state is not None

    x_final, mean, std = _standardize_fit(train[best["feature_cols"]])
    reward_final = reward_full
    cfg = best["config"]
    actor, critic, meta = _train_dsac(
        x_final,
        reward_final,
        hidden=int(args.hidden),
        n_quantiles=int(args.n_quantiles),
        updates=int(args.updates),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        cvar_frac=float(cfg["cvar_frac"]),
        action_l1=float(cfg["action_l1"]),
        bc_lambda=float(cfg["bc_lambda"]),
        seed=int(args.seed) + 100,
        device=device,
    )
    x_val_final = _standardize_apply(train.iloc[val_idx][best["feature_cols"]], mean, std)
    x_oos = _standardize_apply(oos[best["feature_cols"]], mean, std)
    val_raw = _predict_raw(actor, x_val_final, device=device)
    oos_raw = _predict_raw(actor, x_oos, device=device)
    threshold = float(best["threshold"])
    val_dec = _decisions(train.iloc[val_idx].reset_index(drop=True), val_raw, threshold)
    oos_dec = _decisions(oos, oos_raw, threshold)
    val_class = _metrics(y_all[val_idx], val_raw, threshold)
    oos_class = _metrics(y_oos, oos_raw, threshold)
    val_cost = base._cost_metrics(train.iloc[val_idx].reset_index(drop=True), val_dec, args)
    oos_cost = base._cost_metrics(oos, oos_dec, args)

    val_dec.to_csv(args.out_dir / "validation_decisions.csv", index=False)
    oos_dec.to_csv(args.out_dir / "oos_2026_decisions.csv", index=False)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "feature_cols": best["feature_cols"],
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "config": cfg,
            "model_id": MODEL_ID,
        },
        args.out_dir / "model.pt",
    )
    joblib.dump({"feature_cols": best["feature_cols"], "threshold": threshold, "config": cfg, "model_id": MODEL_ID}, args.out_dir / "model_meta.joblib")
    summary = {
        "model_id": MODEL_ID,
        "design": "offline DSAC-style signed actor with distributional twin critic trained on 2025 fixed-barrier action rewards",
        "device": device,
        "selection": "feature/config/threshold selected on 2025 validation Cost3 Calmar; 2026 is final OOS only",
        "train_reward": "single-trade fixed barrier rewards for cash/long/short using cost3 fee/slip",
        "best": {
            "feature_set": best["feature_set"],
            "feature_count": int(len(best["feature_cols"])),
            "config": cfg,
            "confidence_threshold": threshold,
            "validation_selection": best["validation"],
            "validation_backtest_selection": best["validation_backtest"],
            "validation_final": val_class,
            "validation_backtest_final": val_cost,
            "oos_2026": oos_class,
            "oos_2026_backtest": oos_cost,
            "final_train_meta": meta,
        },
        "all_runs": runs,
        "row_drop_events": base.DROP_EVENTS,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "model": str(args.out_dir / "model.pt"),
            "validation_decisions": str(args.out_dir / "validation_decisions.csv"),
            "oos_2026_decisions": str(args.out_dir / "oos_2026_decisions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    (args.out_dir / "selected_features.json").write_text(json.dumps(best["feature_cols"], indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
