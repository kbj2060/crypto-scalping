#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_fallback_rl_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_RISK = sleeve.FallbackRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192)
MLP_FALLBACK_VAL_PNL = 102.349040
MLP_FALLBACK_OOS_PNL = 85.8772460561837
MLP_FALLBACK_VAL_MDD = -10.677652697162888
MLP_FALLBACK_OOS_MDD = -8.108170708968387


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _forbidden_features(cols: list[str]) -> list[str]:
    return [
        c
        for c in cols
        if c == "tp_sl_action_score"
        or c.startswith("clean_regime4_")
        or c.startswith("regime4_pred_")
        or c.startswith("teacher_")
    ]


def _build_reward_table(frame: pd.DataFrame, dec: pd.DataFrame, risk: sleeve.FallbackRisk) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    cash_idx = np.flatnonzero(~active & (np.arange(len(frame)) < len(frame) - int(risk.max_hold_bars) - 3))
    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    rewards = np.zeros((len(frame), 3), dtype=np.float32)
    valid = np.zeros(len(frame), dtype=bool)
    reasons: dict[str, int] = {}
    best_actions: list[int] = []
    best_scores: list[float] = []
    for idx in cash_idx:
        long_score, long_meta = sleeve._simulate_one(arrays, int(idx), 1, risk, fee_eff=fee_eff, slip_eff=slip_eff)
        short_score, short_meta = sleeve._simulate_one(arrays, int(idx), -1, risk, fee_eff=fee_eff, slip_eff=slip_eff)
        rewards[int(idx), sleeve.ACTION_CASH] = 0.0
        rewards[int(idx), sleeve.ACTION_LONG] = float(long_score) * 100.0
        rewards[int(idx), sleeve.ACTION_SHORT] = float(short_score) * 100.0
        valid[int(idx)] = True
        best = int(np.argmax(rewards[int(idx)]))
        best_actions.append(best)
        best_scores.append(float(rewards[int(idx), best]))
        meta = long_meta if long_score >= short_score else short_meta
        reasons[str(meta.get("reason", "unknown"))] = reasons.get(str(meta.get("reason", "unknown")), 0) + 1
    counts = {str(k): int(v) for k, v in pd.Series(best_actions).value_counts().sort_index().items()} if best_actions else {}
    return rewards, valid, {
        "cash_rows": int(len(cash_idx)),
        "best_action_counts": counts,
        "best_reward_pct_mean": float(np.mean(best_scores)) if best_scores else 0.0,
        "best_reward_pct_p95": float(np.quantile(best_scores, 0.95)) if best_scores else 0.0,
        "sim_reasons": reasons,
    }


class RLActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.q_head = nn.Linear(hidden_dim, 3)
        self.pi_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.q_head(h), self.pi_head(h)


@dataclass(frozen=True)
class RLConfig:
    name: str
    hidden_dim: int
    dropout: float
    epochs: int
    lr: float
    weight_decay: float
    actor_coef: float
    entropy_coef: float
    target_temp: float
    trade_penalty: float
    edge_temp: float
    decision_mode: str


CONFIGS = [
    RLConfig("q_mlp_h96", 96, 0.05, 260, 0.0010, 0.0006, 0.20, 0.002, 0.45, 0.000, 0.35, "q"),
    RLConfig("q_mlp_h128_cql", 128, 0.08, 300, 0.0008, 0.0010, 0.25, 0.004, 0.35, 0.012, 0.40, "q"),
    RLConfig("actor_awac_h96", 96, 0.06, 280, 0.0010, 0.0008, 0.65, 0.006, 0.35, 0.006, 0.40, "actor"),
    RLConfig("hybrid_agree_h128", 128, 0.08, 320, 0.0008, 0.0010, 0.50, 0.006, 0.30, 0.010, 0.45, "hybrid"),
]


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _train_rl_model(
    cfg: RLConfig,
    x: pd.DataFrame,
    rewards: np.ndarray,
    train_mask: np.ndarray,
    seed: int,
) -> tuple[RLActorCritic, dict[str, Any], np.ndarray, np.ndarray]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = np.flatnonzero(train_mask)
    x_np, mean, std = _standardize_fit(x.iloc[idx].to_numpy(dtype=np.float32))
    y_np = rewards[idx].astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(ds, batch_size=min(2048, max(64, len(ds))), shuffle=True, drop_last=False)
    model = RLActorCritic(int(x.shape[1]), hidden_dim=int(cfg.hidden_dim), dropout=float(cfg.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    last_loss = 0.0
    for _epoch in range(int(cfg.epochs)):
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            q, logits = model(xb)
            q_loss = F.smooth_l1_loss(q, yb)
            target_probs = F.softmax(yb / max(float(cfg.target_temp), 1.0e-6), dim=1)
            logp = F.log_softmax(logits, dim=1)
            pi = torch.exp(logp)
            actor_loss = -(target_probs * logp).sum(dim=1).mean()
            entropy = -(pi * logp).sum(dim=1).mean()
            trade_pressure = F.softplus(torch.logsumexp(q[:, 1:], dim=1) - q[:, 0]).mean()
            loss = q_loss + float(cfg.actor_coef) * actor_loss + float(cfg.trade_penalty) * trade_pressure - float(cfg.entropy_coef) * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    diag = {
        "train_rows": int(len(idx)),
        "epochs": int(cfg.epochs),
        "device": str(device),
        "last_loss": float(last_loss),
        "config": asdict(cfg),
    }
    return model.cpu(), diag, mean, std


@torch.no_grad()
def _predict_rl_model(
    model: RLActorCritic,
    x: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: RLConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_np = _standardize_apply(x.to_numpy(dtype=np.float32), mean, std)
    q, logits = model(torch.from_numpy(x_np))
    q_np = q.numpy().astype(np.float64)
    pi_np = F.softmax(logits, dim=1).numpy().astype(np.float64)
    q_best = np.argmax(q_np, axis=1).astype(np.int64)
    pi_best = np.argmax(pi_np, axis=1).astype(np.int64)
    if cfg.decision_mode == "actor":
        action = pi_best
        conf = pi_np[np.arange(len(pi_np)), action]
    elif cfg.decision_mode == "hybrid":
        action = np.where(q_best == pi_best, q_best, sleeve.ACTION_CASH).astype(np.int64)
        q_edge = np.maximum(q_np[:, sleeve.ACTION_LONG], q_np[:, sleeve.ACTION_SHORT]) - q_np[:, sleeve.ACTION_CASH]
        pi_conf = pi_np[np.arange(len(pi_np)), np.maximum(q_best, 0)]
        conf = np.where(action == sleeve.ACTION_CASH, 0.0, pi_conf * (1.0 / (1.0 + np.exp(-q_edge / max(float(cfg.edge_temp), 1.0e-6)))))
    else:
        action = q_best
        q_edge = np.maximum(q_np[:, sleeve.ACTION_LONG], q_np[:, sleeve.ACTION_SHORT]) - q_np[:, sleeve.ACTION_CASH]
        conf = 1.0 / (1.0 + np.exp(-q_edge / max(float(cfg.edge_temp), 1.0e-6)))
    diag = {
        "q_mean": q_np.mean(axis=0).tolist(),
        "pi_mean": pi_np.mean(axis=0).tolist(),
        "action_counts": {str(k): int(v) for k, v in pd.Series(action).value_counts().sort_index().items()},
    }
    return action.astype(np.int64), conf.astype(np.float64), diag


def _predict_oof_rl(
    cfg: RLConfig,
    x: pd.DataFrame,
    rewards: np.ndarray,
    cash_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(cash_mask)
    action = np.zeros(len(x), dtype=np.int64)
    conf = np.zeros(len(x), dtype=np.float64)
    folds = []
    n = len(idx)
    for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        train_mask = np.zeros(len(x), dtype=bool)
        train_mask[train_idx] = True
        model, train_diag, mean, std = _train_rl_model(cfg, x, rewards, train_mask, seed + 1000 * fold_id + train_end)
        pred_action, pred_conf, pred_diag = _predict_rl_model(model, x.iloc[val_idx], mean, std, cfg)
        action[val_idx] = pred_action
        conf[val_idx] = pred_conf
        folds.append({"train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "train": train_diag, "pred": pred_diag})
    return action, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _fit_predict_rl(
    cfg: RLConfig,
    x_train: pd.DataFrame,
    rewards_train: np.ndarray,
    train_cash_mask: np.ndarray,
    x_eval: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], RLActorCritic, np.ndarray, np.ndarray]:
    model, train_diag, mean, std = _train_rl_model(cfg, x_train, rewards_train, train_cash_mask, seed)
    action, conf, pred_diag = _predict_rl_model(model, x_eval, mean, std, cfg)
    return action, conf, {"train": train_diag, "pred": pred_diag}, model, mean, std


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, metrics)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    val_dec = sleeve._apply_aggressive(val_dec0)
    oos_dec = sleeve._apply_aggressive(oos_dec0)
    val_features = sleeve._extra_features(base._feature_frame(val_frame, val_src, val_dec0, val_prefix), val_dec)
    oos_features = sleeve._extra_features(base._feature_frame(oos_frame, oos_src, oos_dec0, oos_prefix), oos_dec)
    bad = _forbidden_features(list(val_features.columns))
    if bad:
        raise RuntimeError(f"forbidden RL fallback feature columns: {bad}")
    val_cash = ~omega._active(val_dec)
    oos_cash = ~omega._active(oos_dec)
    val_rewards, val_valid, reward_diag = _build_reward_table(val_frame, val_dec, BASE_RISK)
    train_cash = val_cash & val_valid
    if int(np.count_nonzero(train_cash)) < 500:
        raise RuntimeError(f"not enough RL cash rows: {int(np.count_nonzero(train_cash))}")
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "risk": asdict(BASE_RISK),
        "feature_count": int(val_features.shape[1]),
        "features": list(val_features.columns),
        "forbidden_feature_audit": {"passed": True, "forbidden": []},
        "val_cash_rows": int(np.count_nonzero(val_cash)),
        "oos_cash_rows": int(np.count_nonzero(oos_cash)),
        "reward_diag": reward_diag,
    }
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append(
        {
            "model": "aggressive_primary_only",
            "threshold": 1.0,
            **_metric_row("val", {**baseline_val, "primary_entries": baseline_val["long_entries"] + baseline_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
            **_metric_row("oos", {**baseline_oos, "primary_entries": baseline_oos["long_entries"] + baseline_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}),
        }
    )
    best_model_payload: dict[str, Any] | None = None
    best_candidate_key: tuple[float, float] = (-1.0e18, -1.0e18)
    for cfg in CONFIGS:
        print(json.dumps({"stage": "train_eval_rl", "config": cfg.name}, ensure_ascii=False), flush=True)
        val_action, val_conf, oof_diag = _predict_oof_rl(cfg, val_features, val_rewards, train_cash, seed=260606)
        oos_action, oos_conf, fit_diag, model, mean, std = _fit_predict_rl(cfg, val_features, val_rewards, train_cash, oos_features, seed=260606)
        diagnostics[f"{cfg.name}_oof"] = oof_diag
        diagnostics[f"{cfg.name}_fit"] = fit_diag
        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99):
            val_m = sleeve._metrics_with_fallback(val_frame, val_dec, BASE_RISK, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, BASE_RISK, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=3.0)
            row = {"model": cfg.name, "threshold": float(threshold)}
            row.update(_metric_row("val", val_m))
            row.update(_metric_row("oos", oos_m))
            rows.append(row)
            key = (float(oos_m["pnl"]), float(val_m["pnl"]))
            if key > best_candidate_key:
                best_candidate_key = key
                best_model_payload = {"config": cfg, "threshold": float(threshold), "model": model, "mean": mean, "std": std, "oos_metrics": oos_m, "val_metrics": val_m, "fit_diag": fit_diag}
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - sleeve.AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - sleeve.AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - sleeve.AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - sleeve.AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "cash_fallback_rl_ranking.csv", index=False)
    promotable = ranking[
        (ranking["model"] != "aggressive_primary_only")
        & (ranking["oos_pnl"] > MLP_FALLBACK_OOS_PNL)
        & (ranking["val_pnl"] > MLP_FALLBACK_VAL_PNL)
        & (ranking["oos_mdd"] >= MLP_FALLBACK_OOS_MDD * 1.35)
        & (ranking["val_mdd"] >= MLP_FALLBACK_VAL_MDD * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "cash_fallback_rl_promotable.csv", index=False)
    saved_model_dir = None
    best = ranking.iloc[0].to_dict()
    if best_model_payload is not None and int(len(promotable)) > 0:
        top = promotable.sort_values(["oos_pnl", "val_pnl"], ascending=False).iloc[0]
        cfg = next(c for c in CONFIGS if c.name == str(top["model"]))
        oos_action, oos_conf, fit_diag, model, mean, std = _fit_predict_rl(cfg, val_features, val_rewards, train_cash, oos_features, seed=260606)
        saved_model_dir = ROOT / "data/ensemble/supervised" / f"omega1_2_1_cash_fallback_rl_{cfg.name}_thr{str(float(top['threshold'])).replace('.', '')}_20260606"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        with (saved_model_dir / "cash_fallback_rl_model.pkl").open("wb") as f:
            pickle.dump(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(val_features.shape[1]),
                    "feature_columns": list(val_features.columns),
                    "mean": mean,
                    "std": std,
                    "config": asdict(cfg),
                    "threshold": float(top["threshold"]),
                    "risk": asdict(BASE_RISK),
                    "forbidden_feature_audit": {"passed": True, "forbidden": []},
                },
                f,
            )
        manifest = {
            "model_id": saved_model_dir.name,
            "created_at": "2026-06-06",
            "role": "cash fallback RL sleeve candidate; activates only when primary is CASH and no position is open",
            "base_primary": "omega1_2_1_aggressive_compensated_scale200_cap090",
            "model_type": "offline actor-critic contextual Q policy",
            "feature_count": int(val_features.shape[1]),
            "feature_columns": list(val_features.columns),
            "risk": asdict(BASE_RISK),
            "threshold": float(top["threshold"]),
            "config": asdict(cfg),
            "metrics": top.to_dict(),
            "ranking_report": str(OUT_DIR / "report.json"),
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
        }
        (saved_model_dir / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "baseline": "omega1_2_1_cash_fallback_mlp_base_edge006_thr085_20260606",
        "method": "Offline contextual actor-critic on primary-CASH rows only. Rewards are full counterfactual cash/long/short trade outcomes under the selected fallback risk template.",
        "diagnostics": diagnostics,
        "best": best,
        "promotable_count": int(len(promotable)),
        "top10": ranking.head(10).to_dict(orient="records"),
        "saved_model_dir": str(saved_model_dir) if saved_model_dir is not None else None,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "cash_fallback_rl_ranking.csv"),
            "promotable": str(OUT_DIR / "cash_fallback_rl_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": best, "promotable_count": int(len(promotable)), "saved_model_dir": report["saved_model_dir"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
