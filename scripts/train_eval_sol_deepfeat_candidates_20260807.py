"""Stage 2 (deep learning) of docs/experiments/sol_dl_rl_architecture_survey_20260807.json.

Two DL candidates against the corrected SOL triple-barrier trade-outcome label, both trained on
the soft race-conviction target, N=5 genuinely random seeds each (contract seed list):

- tabm_flat:    TabM-style BatchEnsemble MLP (repo convention, ensemble/deep_features/
                btc_deepfeat_tabm_head_20260806.TabMEnsembleHead) on the standardized flat
                feature row -- the "does depth on tabular rows beat LGBM" test.
- transformer:  causal window-48 SupervisedTransformerEncoder (d96/l3/dropout0.25 -- the BTC G2
                hygiene-line config) from ensemble/deep_features/btc_deepfeat_encoders_20260806.

Hygiene carried over from the BTC G2 lesson: 288-bar purge + 288-bar embargo at the train end,
train_stride=4 against near-duplicate windows. Entry-rule selection (same pre-registered rule set
as the LGBM control) is VAL-only on the seed-mean; `--stage oos` replays the frozen selection
once.

Usage:
  python scripts/train_eval_sol_deepfeat_candidates_20260807.py --arch tabm_flat --stage val
  python scripts/train_eval_sol_deepfeat_candidates_20260807.py --arch transformer --stage val
  python scripts/train_eval_sol_deepfeat_candidates_20260807.py --arch <arch> --stage oos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ensemble/deep_features"))

from btc_deepfeat_encoders_20260806 import SupervisedTransformerEncoder, DeepFeatModel  # noqa: E402
from btc_deepfeat_tabm_head_20260806 import TabMEnsembleHead  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, ENTRY_RULES, replay, side_state_from_proba,
    PANEL_PATH, LABEL_PATH, HORIZON_BARS,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_BASE = ROOT / "tmp/sol_dl_rl_survey_20260807"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
WINDOW = 48
TRAIN_STRIDE = 4
EMBARGO_BARS = 288
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 512
LR = 1e-3
LGBM_CONTROL_VAL_PNL = -6.903  # frozen control result, tmp/.../lgbm_cheapgate/val_results.json


class FlatTabM(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.head = TabMEnsembleHead(n_features, n_experts=8, hidden=128, n_layers=3, dropout=0.2, n_classes=3, quality_head=False)

    def forward(self, x):
        logits, _ = self.head(x)
        return logits


def build_data():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    assert (labels["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    ts = panel["timestamp"]
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    train_mask[tr_idx[-(HORIZON_BARS + EMBARGO_BARS):]] = False  # purge + embargo
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    mean = np.nanmean(x[train_mask], axis=0)
    std = np.nanstd(x[train_mask], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = np.clip(np.nan_to_num((x - mean) / std, nan=0.0), -10.0, 10.0).astype(np.float32)

    soft = labels[["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"]].to_numpy(dtype=np.float32)
    action = labels["trade_outcome_action"].to_numpy()
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    return panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols


def window_ok_rows(mask: np.ndarray, window: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    return idx[idx >= window - 1]


def get_windows(x_std: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    out = np.empty((len(rows), window, x_std.shape[1]), dtype=np.float32)
    for i, t in enumerate(rows):
        out[i] = x_std[t - window + 1 : t + 1]
    return out


@torch.no_grad()
def predict_rows(model, arch, x_std, rows, device, batch=2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        if arch == "transformer":
            xb = torch.from_numpy(get_windows(x_std, chunk, WINDOW)).to(device)
            logits, _, _ = model(xb)
        else:
            xb = torch.from_numpy(x_std[chunk]).to(device)
            logits = model(xb)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one_seed(arch: str, seed: int, x_std, soft, train_rows, val_rows, device) -> torch.nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_features = x_std.shape[1]
    if arch == "transformer":
        encoder = SupervisedTransformerEncoder(n_features, d_model=96, n_heads=4, n_layers=3, ffn_mult=2, dropout=0.25, embed_dim=32)
        model = DeepFeatModel(encoder, embed_dim=32, n_classes=3, quality_head=False, head_type="linear").to(device)
    else:
        model = FlatTabM(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    soft_t = torch.from_numpy(soft)
    best_val = np.inf
    best_state = None
    bad = 0
    rng = np.random.default_rng(seed)
    val_eval_rows = val_rows[::4]  # early-stop metric only; final predictions stay dense

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train_rows))
        total, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            rows = train_rows[order[i : i + BATCH]]
            if arch == "transformer":
                xb = torch.from_numpy(get_windows(x_std, rows, WINDOW)).to(device)
                logits, _, _ = model(xb)
            else:
                xb = torch.from_numpy(x_std[rows]).to(device)
                logits = model(xb)
            target = soft_t[rows].to(device)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1

        proba_val = predict_rows(model, arch, x_std, val_eval_rows, device)
        val_loss = float(F.kl_div(torch.log(torch.from_numpy(proba_val).clamp_min(1e-9)), soft_t[val_eval_rows], reduction="batchmean"))
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"[{arch} seed={seed}] epoch {epoch} train_loss={total/max(nb,1):.5f} val_loss={val_loss:.5f}{' *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["tabm_flat", "transformer"], required=True)
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    out_dir = OUT_BASE / f"dl_{args.arch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols = build_data()
    window = WINDOW if args.arch == "transformer" else 1
    train_rows = window_ok_rows(train_mask, window)[::TRAIN_STRIDE]
    val_rows = window_ok_rows(val_mask, window)
    oos_rows = window_ok_rows(oos_mask, window)

    if args.stage == "val":
        per_seed_proba = {}
        val_acc = {}
        for seed in SEEDS:
            model = train_one_seed(args.arch, seed, x_std, soft, train_rows, val_rows, device)
            torch.save(model.state_dict(), out_dir / f"model_seed{seed}.pt")
            proba = predict_rows(model, args.arch, x_std, val_rows, device)
            per_seed_proba[seed] = proba
            val_acc[seed] = float((proba.argmax(axis=1) == action[val_rows]).mean())
            np.save(out_dir / f"val_proba_seed{seed}.npy", proba)

        rule_table = []
        for rule in ENTRY_RULES:
            per_seed = []
            for seed in SEEDS:
                side = np.zeros(len(panel), dtype=np.int64)
                side[val_rows] = side_state_from_proba(per_seed_proba[seed], rule["threshold"])
                r = replay(panel, side, tp_moves, sl_moves, val_mask)
                per_seed.append({"seed": seed, **r})
            pnls = [r.get("pnl_pct", 0.0) for r in per_seed]
            trades = [r.get("n_trades", 0) for r in per_seed]
            rule_table.append({
                "rule": rule["name"], "threshold": rule["threshold"],
                "seed_mean_pnl_pct": float(np.mean(pnls)), "seed_min_pnl_pct": float(np.min(pnls)),
                "n_pos_seeds": int(sum(p > 0 for p in pnls)), "seed_mean_trades": float(np.mean(trades)),
                "per_seed": per_seed,
            })
            print(json.dumps({k: rule_table[-1][k] for k in ("rule", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")}))

        eligible = [r for r in rule_table if r["seed_mean_trades"] >= 15]
        best = max(eligible, key=lambda r: r["seed_mean_pnl_pct"]) if eligible else None
        earns_oos = bool(best and best["seed_mean_pnl_pct"] > 0 and best["seed_mean_pnl_pct"] > LGBM_CONTROL_VAL_PNL)
        out = {
            "stage": "val", "arch": args.arch, "seeds": SEEDS, "val_accuracy": val_acc,
            "rules": rule_table, "selected_rule": None if best is None else {k: best[k] for k in ("rule", "threshold", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")},
            "earns_oos_read": earns_oos, "lgbm_control_val_pnl": LGBM_CONTROL_VAL_PNL,
        }
        (out_dir / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("arch", "val_accuracy", "selected_rule", "earns_oos_read")}, indent=2))
    else:
        prior = json.loads((out_dir / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"arch": args.arch, "oos": "REFUSED -- candidate did not pass VAL gate"}))
            return 1
        sel = prior["selected_rule"]
        n_features = x_std.shape[1]
        per_seed = []
        proba_sum = None
        for seed in SEEDS:
            if args.arch == "transformer":
                encoder = SupervisedTransformerEncoder(n_features, d_model=96, n_heads=4, n_layers=3, ffn_mult=2, dropout=0.25, embed_dim=32)
                model = DeepFeatModel(encoder, embed_dim=32, n_classes=3, quality_head=False, head_type="linear").to(device)
            else:
                model = FlatTabM(n_features).to(device)
            model.load_state_dict(torch.load(out_dir / f"model_seed{seed}.pt", map_location=device))
            proba = predict_rows(model, args.arch, x_std, oos_rows, device)
            proba_sum = proba if proba_sum is None else proba_sum + proba
            side = np.zeros(len(panel), dtype=np.int64)
            side[oos_rows] = side_state_from_proba(proba, sel["threshold"])
            r = replay(panel, side, tp_moves, sl_moves, oos_mask)
            per_seed.append({"seed": seed, **r})
        side = np.zeros(len(panel), dtype=np.int64)
        side[oos_rows] = side_state_from_proba(proba_sum / len(SEEDS), sel["threshold"])
        ens = replay(panel, side, tp_moves, sl_moves, oos_mask)
        pnls = [r.get("pnl_pct", 0.0) for r in per_seed]
        out = {
            "stage": "oos", "arch": args.arch, "selected_rule": sel,
            "seed_mean_pnl_pct": float(np.mean(pnls)), "n_pos_seeds": int(sum(p > 0 for p in pnls)),
            "per_seed": per_seed, "seed_ensemble": ens,
        }
        (out_dir / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
