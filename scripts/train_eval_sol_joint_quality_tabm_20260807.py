"""Rev3 joint multi-task TabM for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev3_joint_quality_20260807.json).

Shared 8-expert BatchEnsemble trunk with two heads trained JOINTLY:
  - direction: 3-class soft race-conviction target, KL loss (identical to the parent TabM-flat)
  - quality:   2 binary per-side TP-first targets (soft_long>soft_cash, soft_short>soft_cash), BCE

loss = KL + lambda * BCE, lambda in {0.25, 1.0} (pre-registered, closed grid).
Entry rules (pre-registered): side_prob_055; side_prob_055 AND own-quality>=0.45;
side_prob_055 AND own-quality>=0.50. Seed-mean VAL selection over the 6 configs; the winner must
beat the frozen parent VAL +0.6507% to earn the single OOS read.

Usage:
  python scripts/train_eval_sol_joint_quality_tabm_20260807.py --stage val
  python scripts/train_eval_sol_joint_quality_tabm_20260807.py --stage oos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import LABEL_PATH  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import replay as gate_replay  # noqa: E402
import train_eval_sol_deepfeat_candidates_20260807 as dl  # noqa: E402

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/joint_quality_rev3"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
LAMBDAS = [0.25, 1.0]
ENTRY_THRESHOLD = 0.55
QUALITY_RULES = [None, 0.45, 0.50]
BASELINE_VAL_PNL = 0.6507
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 512
LR = 1e-3


class JointTabM(nn.Module):
    """Parent FlatTabM trunk (8-expert BatchEnsemble, 128x3, SiLU/LayerNorm/Dropout) with a
    3-class direction head and a 2-logit per-side TP-first quality head on the shared trunk."""

    def __init__(self, in_dim: int, n_experts: int = 8, hidden: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.n_experts = n_experts
        self.expert_scale = nn.Parameter(torch.ones(n_experts, in_dim))
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, in_dim))
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.direction_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d = x.shape
        xe = x.unsqueeze(1) * self.expert_scale.unsqueeze(0) + self.expert_bias.unsqueeze(0)
        h = self.trunk(xe.reshape(b * self.n_experts, d)).reshape(b, self.n_experts, -1)
        return self.direction_head(h).mean(dim=1), self.quality_head(h).mean(dim=1)


@torch.no_grad()
def predict(model, x_std, rows, device, batch=4096):
    model.eval()
    probs, quals = [], []
    for i in range(0, len(rows), batch):
        xb = torch.from_numpy(x_std[rows[i : i + batch]]).to(device)
        lg, ql = model(xb)
        probs.append(torch.softmax(lg, dim=-1).cpu().numpy())
        quals.append(torch.sigmoid(ql).cpu().numpy())
    return np.concatenate(probs), np.concatenate(quals)


def train_one(seed: int, lam: float, x_std, soft, y_quality, train_rows, val_rows, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = JointTabM(x_std.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    soft_t = torch.from_numpy(soft)
    qual_t = torch.from_numpy(y_quality)
    rng = np.random.default_rng(seed)
    val_eval_rows = val_rows[::4]
    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), BATCH):
            rows = train_rows[order[i : i + BATCH]]
            xb = torch.from_numpy(x_std[rows]).to(device)
            lg, ql = model(xb)
            loss = F.kl_div(F.log_softmax(lg, dim=-1), soft_t[rows].to(device), reduction="batchmean")
            loss = loss + lam * F.binary_cross_entropy_with_logits(ql, qual_t[rows].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        proba_v, qual_v = predict(model, x_std, val_eval_rows, device)
        kl = float(F.kl_div(torch.log(torch.from_numpy(proba_v).clamp_min(1e-9)), soft_t[val_eval_rows], reduction="batchmean"))
        bce = float(F.binary_cross_entropy(torch.from_numpy(qual_v).clamp(1e-7, 1 - 1e-7), qual_t[val_eval_rows]))
        val_loss = kl + lam * bce
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[joint lam={lam} seed={seed}] epoch {epoch} val_kl={kl:.5f} val_bce={bce:.5f}{' *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def decide(proba: np.ndarray, qual: np.ndarray, q: float | None):
    arg = proba.argmax(axis=1)
    side_prob = np.take_along_axis(proba, arg[:, None], axis=1)[:, 0]
    side = np.where(arg == 1, 1, np.where(arg == 2, -1, 0))
    side = np.where(side_prob >= ENTRY_THRESHOLD, side, 0)
    if q is not None:
        own_q = np.where(side == 1, qual[:, 0], np.where(side == -1, qual[:, 1], 0.0))
        side = np.where(own_q >= q, side, 0)
    return side


def run_replay(panel, rows, side_rows, tp_moves, sl_moves, mask):
    side_full = np.zeros(len(panel), dtype=np.int64)
    side_full[rows] = side_rows
    return gate_replay(panel, side_full, tp_moves, sl_moves, mask)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols = dl.build_data()
    labels = pd.read_parquet(LABEL_PATH)
    y_quality = np.stack([
        (labels["trade_outcome_soft_long"].to_numpy() > labels["trade_outcome_soft_cash"].to_numpy()),
        (labels["trade_outcome_soft_short"].to_numpy() > labels["trade_outcome_soft_cash"].to_numpy()),
    ], axis=1).astype(np.float32)

    train_rows = np.flatnonzero(train_mask)[::dl.TRAIN_STRIDE]
    val_rows = np.flatnonzero(val_mask)
    oos_rows = np.flatnonzero(oos_mask)

    if args.stage == "val":
        preds = {}
        for lam in LAMBDAS:
            for seed in SEEDS:
                model = train_one(seed, lam, x_std, soft, y_quality, train_rows, val_rows, device)
                torch.save(model.state_dict(), OUT_DIR / f"model_lam{lam}_seed{seed}.pt")
                preds[(lam, seed)] = predict(model, x_std, val_rows, device)
        table = []
        for lam in LAMBDAS:
            for q in QUALITY_RULES:
                per = []
                for seed in SEEDS:
                    proba, qual = preds[(lam, seed)]
                    side = decide(proba, qual, q)
                    per.append(run_replay(panel, val_rows, side, tp_moves, sl_moves, val_mask))
                table.append({
                    "lam": lam, "q": q,
                    "seed_mean_pnl_pct": float(np.mean([r.get("pnl_pct", 0.0) for r in per])),
                    "n_pos_seeds": int(sum(r.get("pnl_pct", 0.0) > 0 for r in per)),
                    "seed_mean_trades": float(np.mean([r.get("n_trades", 0) for r in per])),
                    "per_seed": per,
                })
                print(json.dumps({k: table[-1][k] for k in ("lam", "q", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")}), flush=True)
        eligible = [r for r in table if r["seed_mean_trades"] >= 15]
        best = max(eligible, key=lambda r: r["seed_mean_pnl_pct"]) if eligible else None
        earns_oos = bool(best and best["seed_mean_pnl_pct"] > BASELINE_VAL_PNL)
        out = {"stage": "val", "baseline_val_pnl": BASELINE_VAL_PNL, "table": table,
               "selected": None if best is None else {k: best[k] for k in ("lam", "q", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")},
               "earns_oos_read": earns_oos}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": earns_oos}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- joint-quality stack did not beat parent baseline on VAL"}))
            return 1
        sel = prior["selected"]
        per = []
        for seed in SEEDS:
            model = JointTabM(x_std.shape[1]).to(device)
            model.load_state_dict(torch.load(OUT_DIR / f"model_lam{sel['lam']}_seed{seed}.pt", map_location=device))
            proba, qual = predict(model, x_std, oos_rows, device)
            side = decide(proba, qual, sel["q"])
            r = run_replay(panel, oos_rows, side, tp_moves, sl_moves, oos_mask)
            per.append({"seed": seed, **r})
            print(json.dumps(per[-1]), flush=True)
        pnls = [r.get("pnl_pct", 0.0) for r in per]
        out = {"stage": "oos", "selected": sel, "seed_mean_pnl_pct": float(np.mean(pnls)),
               "n_pos_seeds": int(sum(p > 0 for p in pnls)), "per_seed": per}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("selected", "seed_mean_pnl_pct", "n_pos_seeds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
