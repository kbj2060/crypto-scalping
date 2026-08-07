#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23  # noqa: E402
from scripts.runtime_retest_alpha7_1_01965_decontam_20260528 import (  # noqa: E402
    CANDIDATE_DIR,
    EVAL_CSV,
    TRAIN_CSV,
    _assert_clean_frame,
    _assert_clean_parent,
    _patch_runtime_sources,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _fill_price, _json_default  # noqa: E402
from scripts.train_eval_alpha7_iqn_fallback_20260527 import _sample_tau  # noqa: E402


MODEL_ID = "decontam_deep_scout_tcn_iqn_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
HORIZONS = (12, 24, 48)


class DeepScoutTCNIQN(nn.Module):
    def __init__(self, seq_dim: int, hidden: int = 72, n_cos: int = 64, action_dim: int = 2) -> None:
        super().__init__()
        self.n_cos = int(n_cos)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.state = nn.Sequential(nn.Linear(hidden, 96), nn.GELU(), nn.LayerNorm(96))
        self.quantile = nn.Sequential(nn.Linear(n_cos, 96), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, action_dim))

    def forward(self, seq: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, D], tau: [B, N], output: [B, N, 2] for long/short q.
        h = self.net(seq.transpose(1, 2)).squeeze(-1)
        state = self.state(h).unsqueeze(1)
        basis_idx = torch.arange(1, self.n_cos + 1, device=seq.device, dtype=seq.dtype).view(1, 1, -1)
        tau_basis = torch.cos(math.pi * tau.unsqueeze(-1) * basis_idx)
        tau_emb = self.quantile(tau_basis)
        return self.head(state * tau_emb)


def _normalizer(seqs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(seqs, axis=(0, 1)).astype(np.float32),
        "std": (np.nanstd(seqs, axis=(0, 1)) + 1e-6).astype(np.float32),
    }


def _apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _build_distribution_train_set(df: pd.DataFrame, seq_cols: list[str], *, fee: float, slip: float, stride: int) -> dict[str, np.ndarray]:
    seqs: list[np.ndarray] = []
    targets: list[list[list[float]]] = []
    for i in range(v23.SEQ_LEN, len(df) - max(HORIZONS) - 2, int(stride)):
        entry_i = min(i + 1, len(df) - 1)
        long_entry = _fill_price(df, entry_i, 1, slip, entry=True)
        short_entry = _fill_price(df, entry_i, -1, slip, entry=True)
        per_horizon: list[list[float]] = []
        for h in HORIZONS:
            exit_i = min(i + h, len(df) - 1)
            long_exit = _fill_price(df, exit_i, 1, slip, entry=False)
            short_exit = _fill_price(df, exit_i, -1, slip, entry=False)
            long_reward = (long_exit - long_entry) / max(long_entry, 1e-12) - fee * 2.0
            short_reward = (short_entry - short_exit) / max(short_entry, 1e-12) - fee * 2.0
            per_horizon.append([float(long_reward), float(short_reward)])
        seqs.append(v23._seq_at(df, i, seq_cols))
        targets.append(per_horizon)
    if not seqs:
        raise RuntimeError("no deep scout IQN train sequences")
    return {"seq": np.stack(seqs).astype(np.float32), "target": np.asarray(targets, dtype=np.float32)}


def _quantile_huber_distribution_loss(pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor, sample_weight: torch.Tensor | None = None, kappa: float = 1.0) -> torch.Tensor:
    # pred [B,N,A], target [B,K,A], tau [B,N]
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = (tau[:, :, None, None] - (td.detach() < 0.0).float()).abs()
    loss = weight * huber / kappa
    loss = loss.mean(dim=(1, 2, 3))
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def _sample_weights(y: np.ndarray) -> np.ndarray:
    best = np.max(y, axis=1)
    edge = np.max(best, axis=1) - np.min(best, axis=1)
    tail = np.maximum(0.0, -np.min(y, axis=(1, 2)))
    w = 1.0 + 2.0 * np.clip(edge / max(float(np.nanpercentile(edge, 95)), 1e-8), 0.0, 3.0)
    w *= 1.0 + 1.25 * np.clip(tail / max(float(np.nanpercentile(tail, 95)), 1e-8), 0.0, 3.0)
    return np.where(np.isfinite(w), w, 1.0).astype(np.float32)


def _redo_linear(model: nn.Module, *, tau: float = 5e-3, ratio: float = 0.05) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or module.out_features < 16:
            continue
        with torch.no_grad():
            row_norm = module.weight.detach().norm(dim=1)
            weak = torch.nonzero(row_norm < float(tau) * row_norm.mean().clamp_min(1e-12), as_tuple=False).flatten()
            if weak.numel() == 0:
                continue
            weak = weak[: max(1, int(module.out_features * float(ratio)))]
            bound = math.sqrt(6.0 / max(module.weight.shape[1] + module.out_features, 1))
            module.weight[weak].uniform_(-bound, bound)
            if module.bias is not None:
                module.bias[weak].zero_()
            count += int(weak.numel())
    return count


def _train_iqn(ds: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, epochs: int, batch_size: int, lr: float, tau_samples: int, seed: int) -> tuple[DeepScoutTCNIQN, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x = _apply_norm(ds["seq"], norm)
    y = ds["target"].astype(np.float32)
    sample_w = _sample_weights(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepScoutTCNIQN(x.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w.astype(np.float64)), num_samples=len(sample_w), replacement=True)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sample_w)),
        batch_size=int(batch_size),
        sampler=sampler,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=2, min_lr=1e-5)
    losses: list[float] = []
    redo = 0
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, sw in loader:
            xb, yb, sw = xb.to(device), yb.to(device), sw.to(device)
            tau = _sample_tau(len(xb), int(tau_samples), device, xb.dtype, tail_mix=0.55, tail_max=0.30)
            pred = model(xb, tau)
            loss = _quantile_huber_distribution_loss(pred, yb, tau, sample_weight=sw)
            mean_q = pred.mean(dim=1)
            target_mean = yb.mean(dim=1)
            # Conservative penalty: when both sides are poor, keep both q values small.
            poor = (target_mean.max(dim=1).values <= 0.0).float()
            cql = (F.softplus(torch.logsumexp(mean_q, dim=1)) * poor).mean()
            loss = loss + 0.015 * cql
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        epoch_loss = total / max(count, 1)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        if (epoch + 1) % 4 == 0:
            redo += _redo_linear(model)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(json.dumps({"stage": "train_iqn", "epoch": epoch + 1, "epochs": int(epochs), "loss": epoch_loss}), flush=True)
    return model.cpu().eval(), {"losses": losses, "device": str(device), "redo_reset_neurons": int(redo), "sample_weight_p95": float(np.percentile(sample_w, 95))}


def _predict_iqn(model: DeepScoutTCNIQN, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray], *, tau_max: float, num_tau: int = 32, batch_size: int = 512) -> np.ndarray:
    seqs = np.stack([v23._seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = _apply_norm(seqs, norm)
    taus = torch.linspace(0.01, float(tau_max), int(num_tau)).view(1, -1)
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = torch.from_numpy(x[start : start + int(batch_size)])
            tb = taus.repeat(len(xb), 1)
            pred = model(xb, tb).mean(dim=1).numpy()
            out.append(pred)
    return np.vstack(out).astype(np.float32)


def _eval_costs(df: pd.DataFrame, q: np.ndarray, dec: pd.DataFrame, stack: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        f"cost{c}": precision._eval(df=df, q=q, dec=dec, stack=stack, cfg=cfg, period="full", cost_mult=c, record=False)
        for c in (1, 2, 3)
    }


def _score(c3: dict[str, Any]) -> float:
    if int(c3.get("trades", 0)) < 20:
        return -1e9 + float(c3.get("pnl", 0.0))
    return float(c3["pnl"]) + 2.0 * float(c3["mdd"]) + 40.0 * float(c3["wr"]) - 0.03 * float(c3["trades"])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Attach IQN head to decontaminated Alpha7 deep scout TCN and test.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--tau-samples", type=int, default=32)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260528)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _assert_clean_frame(TRAIN_CSV, name="train")
    _assert_clean_frame(EVAL_CSV, name="eval")
    _assert_clean_parent(CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    _assert_clean_parent(CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    _patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = loop._load_stack()
    val_df, eval_df = loop._load_frames()
    # loop._load_frames returns validation-only train split, so use the same decontam val frame as training source
    # only for the deep scout IQN experiment's 2025 pre-Q4 data would require rebuilding loop internals.
    # Instead use the current stack deep scout seq contract and v27.DEFAULT_TRAIN merged by loop via val_df source.
    # To keep selection clean, build IQN on the full 2025 frame through loop's raw loader path.
    full_train = loop._merge_state24(loop._read(v27.DEFAULT_TRAIN), loop.alpha3_full.SIDE_CLEAN4_2025)
    a7_train = loop._rename_clean4_v2(loop._read(TRAIN_CSV))
    full_train = loop._augment_with_alpha7_features(full_train, a7_train)
    full_train["timestamp"] = pd.to_datetime(full_train["timestamp"], errors="raise")
    train_df = full_train[full_train["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)

    sources = loop._decision_sources(val_df, eval_df, stack["parent"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]
    baseline_val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    baseline_eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    baseline_val = _eval_costs(val_df, baseline_val_q, val_dec, stack, cfg)
    baseline_eval = _eval_costs(eval_df, baseline_eval_q, eval_dec, stack, cfg)
    print(json.dumps({"stage": "baseline", "val_cost3": baseline_val["cost3"]["pnl"], "oos_cost3": baseline_eval["cost3"]["pnl"]}), flush=True)

    seq_cols = list(stack["deep_payload"]["seq_cols"])
    bad = [c for c in seq_cols if c.startswith(("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_"))]
    if bad:
        raise RuntimeError(f"deep scout seq contract contains forbidden legacy features: {bad[:20]}")
    missing = [c for c in seq_cols if c not in train_df.columns or c not in val_df.columns or c not in eval_df.columns]
    if missing:
        raise RuntimeError(f"deep scout IQN missing seq cols: {missing[:20]}")

    ds = _build_distribution_train_set(train_df, seq_cols, fee=float(stack["fee"]), slip=float(stack["slip"]), stride=int(args.stride))
    norm = _normalizer(ds["seq"])
    model, train_diag = _train_iqn(ds, norm, epochs=int(args.epochs), batch_size=int(args.batch_size), lr=float(args.lr), tau_samples=int(args.tau_samples), seed=int(args.seed))

    rows: list[dict[str, Any]] = []
    val_q_cache: dict[str, np.ndarray] = {}
    eval_q_cache: dict[str, np.ndarray] = {}
    for tau_max in (0.15, 0.25, 0.40, 0.75):
        val_q_cache[f"iqn_cvar_{tau_max:.2f}"] = _predict_iqn(model, val_df, seq_cols, norm, tau_max=float(tau_max))
        eval_q_cache[f"iqn_cvar_{tau_max:.2f}"] = _predict_iqn(model, eval_df, seq_cols, norm, tau_max=float(tau_max))
    for name in list(val_q_cache):
        variants = {name: val_q_cache[name]}
        for w in (0.25, 0.50, 0.75):
            variants[f"blend_w{w:.2f}_{name}"] = (1.0 - w) * baseline_val_q + w * val_q_cache[name]
        for variant, q in variants.items():
            c3 = _eval_costs(val_df, q.astype(np.float32), val_dec, stack, cfg)["cost3"]
            rows.append(
                {
                    "variant": variant,
                    "selection_score": _score(c3),
                    "val_cost3_pnl": float(c3["pnl"]),
                    "val_cost3_mdd": float(c3["mdd"]),
                    "val_cost3_wr": float(c3["wr"]),
                    "val_cost3_trades": int(c3["trades"]),
                }
            )
    grid = pd.DataFrame(rows).sort_values(["selection_score", "val_cost3_pnl"], ascending=[False, False])
    best = grid.iloc[0].to_dict()
    best_variant = str(best["variant"])
    if best_variant.startswith("blend_"):
        parts = best_variant.split("_", 2)
        w = float(parts[1][1:])
        base_name = parts[2]
        eval_q = ((1.0 - w) * baseline_eval_q + w * eval_q_cache[base_name]).astype(np.float32)
        val_q = ((1.0 - w) * baseline_val_q + w * val_q_cache[base_name]).astype(np.float32)
    else:
        eval_q = eval_q_cache[best_variant].astype(np.float32)
        val_q = val_q_cache[best_variant].astype(np.float32)
    best_val = _eval_costs(val_df, val_q, val_dec, stack, cfg)
    best_eval = _eval_costs(eval_df, eval_q, eval_dec, stack, cfg)

    model_path = args.out_dir / "deep_scout_tcn_iqn.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "seq_cols": seq_cols, "norm": norm, "train_diag": train_diag}, model_path)
    grid_path = args.out_dir / "validation_grid.csv"
    grid.to_csv(grid_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "baseline_model": "alpha7_submodel_01965_decontam_v2_tp_20260528",
        "design": "Existing DeepAlphaTCN backbone with IQN quantile head. Runtime receives lower-tail CVaR q_long/q_short in the same deep_q contract.",
        "baseline": {"val": baseline_val, "oos": baseline_eval},
        "best_by_validation": {
            **best,
            "val_metrics": best_val,
            "oos_metrics": best_eval,
            "delta_oos_cost3_pnl": float(best_eval["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
        },
        "training": {"rows": int(len(ds["seq"])), "seq_cols": len(seq_cols), "horizons": list(HORIZONS), "train_diag": train_diag},
        "artifacts": {"model": str(model_path), "grid": str(grid_path)},
        "audit": {"selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS", "live_path_modified": False},
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best_variant": best_variant, "baseline_oos_cost3": baseline_eval["cost3"], "best_oos_cost3": best_eval["cost3"], "delta": summary["best_by_validation"]["delta_oos_cost3_pnl"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
