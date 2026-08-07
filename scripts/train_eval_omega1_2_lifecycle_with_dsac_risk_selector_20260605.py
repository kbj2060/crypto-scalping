#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_supervised_risk_selector_20260604 as sup_risk  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_lifecycle_with_dsac_risk_selector_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass
class RiskOfflineData:
    x: np.ndarray
    q_targets: np.ndarray
    best_actions: np.ndarray
    weights: np.ndarray


class DSACRiskSelector(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, n_actions))
        self.critic1 = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, n_actions))
        self.critic2 = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, n_actions))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.actor(h), self.critic1(h), self.critic2(h)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _fit_norm(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    q25 = np.nanpercentile(arr, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    out = (arr - med) / scale
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite DSAC risk training matrix")
    return np.tanh(out / 3.0).astype(np.float32), {"columns": list(x.columns), "median": med, "scale": scale}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("DSAC risk feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite DSAC risk inference matrix")
    return np.tanh(out / 3.0).astype(np.float32)


def _single_dec_row(action: int, side: int, template: dict[str, float]) -> pd.Series:
    return sup_risk._single_dec_row(action, side, template)


def _build_risk_q_dataset(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    x_risk: pd.DataFrame,
    *,
    oof: bool,
    candidate_delta: float,
    min_score: float,
    fee: float,
    slip: float,
    cost_mult: float,
    max_candidates: int,
) -> tuple[pd.DataFrame, RiskOfflineData, dict[str, Any]]:
    action = sup_risk._threshold_action(src, oof=oof, thresholds=sup_risk._candidate_thresholds(float(candidate_delta)))
    candidate_idx = np.flatnonzero(action != omega.ACTION_CASH)
    if int(max_candidates) > 0 and len(candidate_idx) > int(max_candidates):
        keep = np.linspace(0, len(candidate_idx) - 1, int(max_candidates)).round().astype(np.int64)
        candidate_idx = candidate_idx[keep]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    q_rows: list[np.ndarray] = []
    best_actions: list[int] = []
    weights: list[float] = []
    reason_counts: dict[str, int] = {}
    for idx in candidate_idx:
        act = int(action[int(idx)])
        side = 1 if act == omega.ACTION_LONG else -1
        rewards = np.zeros(len(sup_risk.RISK_TEMPLATES), dtype=np.float32)
        rewards[0] = 0.0
        for cls, template in enumerate(sup_risk.RISK_TEMPLATES):
            if cls == 0:
                continue
            score, meta = omega._simulate_trade(
                frame,
                arrays,
                int(idx),
                _single_dec_row(act, side, template),
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
            )
            rewards[int(cls)] = float(score)
            reason = str(meta.get("exit_reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        best = int(np.argmax(rewards))
        if float(rewards[best]) < float(min_score):
            best = 0
        rewards[0] = max(float(rewards[0]), 0.0)
        scale = max(float(np.std(rewards)), 1e-4)
        q_rows.append(rewards)
        best_actions.append(best)
        weights.append(float(np.exp(np.clip((float(rewards[best]) - float(np.median(rewards))) / scale, -4.0, 4.0))))
    if len(candidate_idx) < 200:
        raise RuntimeError(f"not enough DSAC risk candidates: {len(candidate_idx)}")
    x_train = x_risk.iloc[candidate_idx].reset_index(drop=True)
    x_np, norm = _fit_norm(x_train)
    q = np.asarray(q_rows, dtype=np.float32)
    best_np = np.asarray(best_actions, dtype=np.int64)
    return (
        x_train,
        RiskOfflineData(x_np, q, best_np, np.asarray(weights, dtype=np.float32)),
        {
            "candidate_delta": float(candidate_delta),
            "candidates": int(len(candidate_idx)),
            "best_action_counts": {str(i): int(v) for i, v in enumerate(np.bincount(best_np, minlength=len(sup_risk.RISK_TEMPLATES)))},
            "q_mean": float(np.mean(q)),
            "q_best_mean": float(np.mean(q.max(axis=1))),
            "counterfactual_reasons": reason_counts,
            "normalizer": norm,
        },
    )


def _train_dsac_risk(
    data: RiskOfflineData,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[DSACRiskSelector, dict[str, Any]]:
    _seed_everything(seed)
    model = DSACRiskSelector(data.x.shape[1], len(sup_risk.RISK_TEMPLATES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.q_targets), torch.from_numpy(data.best_actions), torch.from_numpy(data.weights))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            xb, q_t, best_a, w = next(it)
        except StopIteration:
            it = iter(dl)
            xb, q_t, best_a, w = next(it)
        xb = xb.to(device)
        q_t = q_t.to(device)
        best_a = best_a.to(device)
        w = w.to(device)
        logits, q1, q2 = model(xb)
        critic_loss = torch.nn.functional.smooth_l1_loss(q1, q_t) + torch.nn.functional.smooth_l1_loss(q2, q_t)
        q_min = torch.minimum(q1, q2).detach()
        probs = torch.softmax(logits, dim=1)
        policy_q = (probs * q_min).sum(dim=1).mean()
        bc = (torch.nn.functional.cross_entropy(logits, best_a, reduction="none") * w).sum() / torch.clamp(w.sum(), min=1.0)
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=1).mean()
        actor_loss = -policy_q + 0.20 * bc - 0.01 * entropy
        loss = critic_loss + actor_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 250 == 0 or step == int(steps):
            last = {
                "step": int(step),
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "policy_q": float(policy_q.detach().cpu()),
            }
            print(json.dumps({"stage": "dsac_risk_train", **last}, ensure_ascii=False), flush=True)
    return model.cpu(), last


@torch.no_grad()
def _predict_risk(model: DSACRiskSelector, x: pd.DataFrame, norm: dict[str, Any], *, device: torch.device, batch_size: int, mode: str) -> np.ndarray:
    model = model.to(device)
    model.eval()
    arr = _apply_norm(x, norm)
    outs: list[np.ndarray] = []
    for start in range(0, len(arr), int(batch_size)):
        xb = torch.tensor(arr[start : start + int(batch_size)], dtype=torch.float32, device=device)
        logits, q1, q2 = model(xb)
        score = torch.minimum(q1, q2) if mode == "critic" else logits
        outs.append(torch.argmax(score, dim=1).cpu().numpy().astype(np.int64))
    pred = np.concatenate(outs) if outs else np.zeros(0, dtype=np.int64)
    if not set(np.unique(pred)).issubset(set(range(len(sup_risk.RISK_TEMPLATES)))):
        raise RuntimeError(f"unexpected DSAC risk classes: {sorted(np.unique(pred).tolist())}")
    return pred


def _risk_decision_from_dsac(
    base_x: pd.DataFrame,
    src: pd.DataFrame,
    *,
    oof: bool,
    model: DSACRiskSelector,
    norm: dict[str, Any],
    device: torch.device,
    batch_size: int,
    mode: str,
    cash_fallback_class: int,
) -> pd.DataFrame:
    action = _final_action(src, oof=oof)
    candidate = action != omega.ACTION_CASH
    risk_class = np.zeros(len(action), dtype=np.int64)
    if bool(candidate.any()):
        x_risk = sup_risk._risk_features(base_x, src, oof=oof)
        risk_class[candidate] = _predict_risk(model, x_risk.loc[candidate].reset_index(drop=True), norm, device=device, batch_size=int(batch_size), mode=mode)
        # This experiment replaces only Parent risk fields. The selector must not
        # veto the already accepted Parent final_action.
        risk_class[candidate & (risk_class == 0)] = int(cash_fallback_class)
    return sup_risk._risk_decision(src, oof=oof, action=action, risk_class=risk_class)


def _final_action(src: pd.DataFrame, *, oof: bool) -> np.ndarray:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({omega.ACTION_CASH, omega.ACTION_LONG, omega.ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    return action


def _prepare_frames_with_dsac_risk(
    *,
    threehead_dir: Path,
    quality_threshold: float,
    device: torch.device,
    risk_batch_size: int,
    risk_candidate_delta: float,
    risk_min_score: float,
    risk_max_candidates: int,
    risk_steps: int,
    risk_lr: float,
    risk_seed: int,
    cost_mult: float,
    risk_select_mode: str,
    cash_fallback_class: int,
) -> tuple[dict[str, Any], DSACRiskSelector, dict[str, Any], dict[str, Any]]:
    base_frames = feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)
    fee, slip = omega._load_fee_slip()
    bundle = feat_coord._load_3head_payloads(threehead_dir)
    train_x, train_src = feat_coord._predict_3head_frame(base_frames["train_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    val_x, val_src = feat_coord._predict_3head_frame(base_frames["val_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=True)
    oos_x, oos_src = feat_coord._predict_3head_frame(base_frames["oos_df"], bundle, quality_threshold=float(quality_threshold), device=device, oof=False)
    train_x_risk = sup_risk._risk_features(train_x, train_src, oof=True)
    _x_train, risk_data, risk_diag = _build_risk_q_dataset(
        base_frames["train_df"],
        train_src,
        train_x_risk,
        oof=True,
        candidate_delta=float(risk_candidate_delta),
        min_score=float(risk_min_score),
        fee=fee,
        slip=slip,
        cost_mult=float(cost_mult),
        max_candidates=int(risk_max_candidates),
    )
    norm = risk_diag.pop("normalizer")
    risk_model, train_diag = _train_dsac_risk(
        risk_data,
        device=device,
        seed=int(risk_seed),
        steps=int(risk_steps),
        batch_size=int(risk_batch_size),
        lr=float(risk_lr),
    )
    train_dec = _risk_decision_from_dsac(train_x, train_src, oof=True, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(risk_select_mode), cash_fallback_class=int(cash_fallback_class))
    val_dec = _risk_decision_from_dsac(val_x, val_src, oof=True, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(risk_select_mode), cash_fallback_class=int(cash_fallback_class))
    oos_dec = _risk_decision_from_dsac(oos_x, oos_src, oof=False, model=risk_model, norm=norm, device=device, batch_size=int(risk_batch_size), mode=str(risk_select_mode), cash_fallback_class=int(cash_fallback_class))
    feature_cols = omega._numeric_feature_cols(
        pd.concat([base_frames["train_df"], base_frames["val_df"]], axis=0, ignore_index=True),
        base_frames["oos_df"],
    )
    s_train = omega._build_state_frame(base_frames["train_df"], train_src, train_dec, oof=True, feature_cols=feature_cols)
    s_val = omega._build_state_frame(base_frames["val_df"], val_src, val_dec, oof=True, feature_cols=feature_cols)
    s_oos = omega._build_state_frame(base_frames["oos_df"], oos_src, oos_dec, oof=False, feature_cols=feature_cols)
    for state, src, prefix in (
        (s_train, train_src, "omega1_regime3_expertdq_oof"),
        (s_val, val_src, "omega1_regime3_expertdq_oof"),
        (s_oos, oos_src, "omega1_regime3_expertdq"),
    ):
        state["threehead_exit_p_hold_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_hold_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_p_exit_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_p_exit_feature_only"], errors="raise").to_numpy(dtype=np.float64)
        state["threehead_exit_edge_feature_only"] = pd.to_numeric(src[f"{prefix}_exit_edge_feature_only"], errors="raise").to_numpy(dtype=np.float64)
    out = dict(base_frames)
    out.update({"train_dec": train_dec, "val_dec": val_dec, "oos_dec": oos_dec, "s_train": s_train, "s_val": s_val, "s_oos": s_oos})
    return out, risk_model, norm, {"risk_data_diag": risk_diag, "risk_train_diag": train_diag}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--risk-candidate-delta", type=float, default=-0.25)
    ap.add_argument("--risk-min-score", type=float, default=0.0010)
    ap.add_argument("--risk-max-candidates", type=int, default=6000)
    ap.add_argument("--risk-steps", type=int, default=800)
    ap.add_argument("--risk-lr", type=float, default=2e-4)
    ap.add_argument("--risk-select-mode", choices=["actor", "critic"], default="actor")
    ap.add_argument("--cash-risk-fallback-class", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=600)
    ap.add_argument("--samples-per-entry", type=int, default=6)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--min-action-edge", type=float, default=0.002)
    ap.add_argument("--disable-resize", action="store_true")
    ap.add_argument("--disable-reverse", action="store_true")
    ap.add_argument("--position-only-training", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--risk-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--class-balance-actor", action="store_true")
    ap.add_argument("--select-mode", choices=["actor", "q", "actor_q"], default="actor")
    ap.add_argument("--force-parent-entry", action="store_true")
    ap.add_argument("--force-entry-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=260623)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, risk_model, risk_norm, risk_info = _prepare_frames_with_dsac_risk(
        threehead_dir=Path(args.threehead_dir),
        quality_threshold=float(args.quality_threshold),
        device=device,
        risk_batch_size=int(args.risk_batch_size),
        risk_candidate_delta=float(args.risk_candidate_delta),
        risk_min_score=float(args.risk_min_score),
        risk_max_candidates=int(args.risk_max_candidates),
        risk_steps=int(args.risk_steps),
        risk_lr=float(args.risk_lr),
        risk_seed=int(args.seed),
        cost_mult=float(args.cost_mult),
        risk_select_mode=str(args.risk_select_mode),
        cash_fallback_class=int(args.cash_risk_fallback_class),
    )
    fee, slip = omega._load_fee_slip()
    state_cols = [c for c in lifecycle._base_state(frames["s_train"]).columns if c != "timestamp"]
    bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or c.startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    norm = lifecycle._fit_norm(lifecycle._base_state(frames["s_train"])[state_cols])
    data, data_diag = lifecycle._build_dataset(
        frames,
        seq_len=int(args.seq_len),
        max_entries=int(args.max_train_entries),
        samples_per_entry=int(args.samples_per_entry),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_sim_bars=int(args.train_max_sim_bars),
        min_action_edge=float(args.min_action_edge),
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        position_only_training=bool(args.position_only_training),
        norm=norm,
    )
    print(json.dumps({"stage": "dsac_risk_lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    model, train_diag = lifecycle._train(
        data,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        class_balance_actor=bool(args.class_balance_actor),
    )
    val = lifecycle._replay(
        frames,
        "val",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    oos = lifecycle._replay(
        frames,
        "oos",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    torch.save(
        {"model_state_dict": risk_model.state_dict(), "normalizer": risk_norm, "state_columns": list(risk_norm["columns"]), "risk_templates": sup_risk.RISK_TEMPLATES},
        out_dir / "dsac_risk_selector.pt",
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": lifecycle.ACTION_NAMES},
        out_dir / "lifecycle_controller.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Exit Head feature-only + Mamba lifecycle baseline with only Parent fixed risk fields replaced by DSAC discrete risk-template selector. Parent final_action and quality threshold are preserved.",
        "threehead_dir": str(args.threehead_dir),
        "quality_threshold": float(args.quality_threshold),
        "risk_templates": sup_risk.RISK_TEMPLATES,
        "risk_selector": {
            "type": "DSACRiskSelector",
            "select_mode": str(args.risk_select_mode),
            "cash_fallback_class": int(args.cash_risk_fallback_class),
            "candidate_delta": float(args.risk_candidate_delta),
            "min_score": float(args.risk_min_score),
            "max_candidates": int(args.risk_max_candidates),
            "steps": int(args.risk_steps),
            **risk_info,
        },
        "state_columns": state_cols,
        "training": {
            "seq_len": int(args.seq_len),
            "max_train_entries": int(args.max_train_entries),
            "samples_per_entry": int(args.samples_per_entry),
            "train_max_sim_bars": int(args.train_max_sim_bars),
            "min_action_edge": float(args.min_action_edge),
            "disable_resize": bool(args.disable_resize),
            "disable_reverse": bool(args.disable_reverse),
            "class_balance_actor": bool(args.class_balance_actor),
            "select_mode": str(args.select_mode),
            "position_only_training": bool(args.position_only_training),
            "force_parent_entry": bool(args.force_parent_entry),
            "force_entry_mult": float(args.force_entry_mult),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "data_diag": data_diag,
            "train_diag": train_diag,
        },
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult), "delta_notional_resize_fee": True, "partial_exit_fee": True},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "risk_selector": str(out_dir / "dsac_risk_selector.pt"), "model": str(out_dir / "lifecycle_controller.pt")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
