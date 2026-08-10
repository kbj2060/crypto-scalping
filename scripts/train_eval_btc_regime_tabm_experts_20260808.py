"""ETH-architecture-analog family for the BTC regime-conditioned line
(contract revision 2026-08-08, added before any VAL was seen): per-regime TabM experts.

- bear and bull experts only (chop is force-cash per the contract), 8-expert BatchEnsemble MLP
  (same module as the SOL survey's FlatTabM), soft race-conviction KL loss, standardized
  features (train stats), early stop on the regime's own VAL KL.
- The 5 contract seeds; selection metric is seed-mean VAL PnL per entry rule; OOS adoption
  additionally requires seed-mean OOS > 0 with >=3/5 seeds positive.
- `--stage select` merges this family's table with the LGBM grid's val_results.json and reports
  the overall winner (the config the single OOS read belongs to).

Usage: --stage {val, select, oos}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_deepfeat_candidates_20260807 import FlatTabM  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import ENTRY_RULES, replay, side_state_from_proba  # noqa: E402
import train_eval_btc_regime_conditioned_entry_20260808 as base  # noqa: E402

OUT_DIR = base.OUT_DIR
SEEDS = [903174, 42517, 6688211, 15093, 771442]
TRAIN_REGIMES = [0, 2]  # bear, bull; chop forced cash
MAX_EPOCHS, PATIENCE, BATCH, LR = 40, 6, 256, 1e-3


def standardize(x, train_rows):
    mean = np.nanmean(x[train_rows], axis=0)
    std = np.nanstd(x[train_rows], axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return np.clip(np.nan_to_num((x - mean) / std, nan=0.0), -10.0, 10.0).astype(np.float32)


def train_expert(seed, x_std, soft, rows_tr, rows_val, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FlatTabM(x_std.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    soft_t = torch.from_numpy(soft)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(rows_tr))
        for i in range(0, len(order), BATCH):
            rows = rows_tr[order[i : i + BATCH]]
            logits = model(torch.from_numpy(x_std[rows]).to(device))
            loss = F.kl_div(F.log_softmax(logits, dim=-1), soft_t[rows].to(device), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_std[rows_val]).to(device))
            vl = float(F.kl_div(F.log_softmax(logits, dim=-1), soft_t[rows_val].to(device), reduction="batchmean"))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, x_std, rows, device):
    model.eval()
    out = []
    for i in range(0, len(rows), 8192):
        out.append(torch.softmax(model(torch.from_numpy(x_std[rows[i : i + 8192]]).to(device)), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "select", "oos"], required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, regime, train_mask, val_mask, oos_mask = base.load_all()
    import pandas as pd  # local, for labels soft cols
    labels = pd.read_parquet(base.LABEL_PATH)
    soft = labels[["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"]].to_numpy(dtype=np.float32)[: len(panel)]
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    x_std = standardize(x, tr_idx)
    months = ts.dt.to_period("M").astype(str).to_numpy()

    if args.stage == "val":
        per_seed_proba = {}
        for seed in SEEDS:
            proba = np.zeros((len(panel), 3))
            for r in TRAIN_REGIMES:
                rows_tr = tr_idx[regime[tr_idx] == r]
                rows_v = v_idx[regime[v_idx] == r]
                model = train_expert(seed, x_std, soft, rows_tr, rows_v, device)
                torch.save(model.state_dict(), OUT_DIR / f"tabm_expert_seed{seed}_{base.REGIME_NAMES[r]}.pt")
                if len(rows_v):
                    proba[rows_v] = predict(model, x_std, rows_v, device)
            per_seed_proba[seed] = proba
            print(f"seed {seed} trained", flush=True)
        table = []
        for rule in ENTRY_RULES:
            per = []
            for seed in SEEDS:
                side_state = np.zeros(len(panel), dtype=np.int64)
                side_state[v_idx] = side_state_from_proba(per_seed_proba[seed][v_idx], rule["threshold"])
                side_state[v_idx[regime[v_idx] == 1]] = 0
                rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
                mon = {}
                for m in sorted(set(months[v_idx])):
                    mon[m] = replay(panel, side_state, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                per.append({"seed": seed, **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "mdd_pct")},
                            "n_pos_months": int(sum(v_ > 0 for v_ in mon.values()))})
            pnls = [p["pnl_pct"] or 0.0 for p in per]
            table.append({"family": "tabm", "rule": rule["name"], "threshold": rule["threshold"],
                          "seed_mean_pnl_pct": float(np.mean(pnls)), "n_pos_seeds": int(sum(p > 0 for p in pnls)),
                          "seed_mean_trades": float(np.mean([p["n_trades"] or 0 for p in per])),
                          "seed_mean_pos_months": float(np.mean([p["n_pos_months"] for p in per])),
                          "per_seed": per})
            print(json.dumps({k: table[-1][k] for k in ("rule", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades", "seed_mean_pos_months")}), flush=True)
        (OUT_DIR / "val_results_tabm.json").write_text(json.dumps({"table": table}, indent=2))
    elif args.stage == "select":
        control_pnl = json.loads((OUT_DIR / "control.json").read_text())["best_val_pnl"]
        lgbm = json.loads((OUT_DIR / "val_results.json").read_text())
        tabm = json.loads((OUT_DIR / "val_results_tabm.json").read_text())["table"]
        lgbm_best = lgbm.get("selected")
        tabm_eligible = [r for r in tabm if r["seed_mean_trades"] >= 15 and r["seed_mean_pnl_pct"] > 0
                         and r["seed_mean_pos_months"] >= 3 and r["seed_mean_pnl_pct"] > control_pnl]
        tabm_best = max(tabm_eligible, key=lambda r: r["seed_mean_pnl_pct"]) if tabm_eligible else None
        cands = []
        if lgbm_best:
            cands.append(("lgbm", lgbm_best["pnl_pct"], lgbm_best))
        if tabm_best:
            cands.append(("tabm", tabm_best["seed_mean_pnl_pct"], {k: tabm_best[k] for k in ("rule", "threshold", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")}))
        winner = max(cands, key=lambda c: c[1]) if cands else None
        out = {"lgbm_selected": lgbm_best, "tabm_selected": None if tabm_best is None else cands[-1][2] if cands and cands[-1][0] == "tabm" else None,
               "winner_family": None if winner is None else winner[0],
               "winner": None if winner is None else winner[2],
               "earns_oos_read": winner is not None}
        (OUT_DIR / "final_selection.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    else:
        sel = json.loads((OUT_DIR / "final_selection.json").read_text())
        if not sel.get("earns_oos_read") or sel["winner_family"] != "tabm":
            print(json.dumps({"oos": "winner is not the TabM family (or no winner); use the LGBM runner's oos stage"}))
            return 1
        rule_thr = sel["winner"]["threshold"]
        o_idx = np.flatnonzero(oos_mask)
        per = []
        for seed in SEEDS:
            proba = np.zeros((len(panel), 3))
            for r in TRAIN_REGIMES:
                model = FlatTabM(x_std.shape[1]).to(device)
                model.load_state_dict(torch.load(OUT_DIR / f"tabm_expert_seed{seed}_{base.REGIME_NAMES[r]}.pt", map_location=device))
                rows_o = o_idx[regime[o_idx] == r]
                if len(rows_o):
                    proba[rows_o] = predict(model, x_std, rows_o, device)
            side_state = np.zeros(len(panel), dtype=np.int64)
            side_state[o_idx] = side_state_from_proba(proba[o_idx], rule_thr)
            side_state[o_idx[regime[o_idx] == 1]] = 0
            rres = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
            per.append({"seed": seed, **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct", "exit_reasons", "long_trades", "short_trades")}})
            print(json.dumps(per[-1]), flush=True)
        pnls = [p["pnl_pct"] or 0.0 for p in per]
        out = {"stage": "oos", "selected": sel["winner"], "seed_mean_pnl_pct": float(np.mean(pnls)),
               "n_pos_seeds": int(sum(p > 0 for p in pnls)), "per_seed": per,
               "adopted": bool(np.mean(pnls) > 0 and sum(p > 0 for p in pnls) >= 3)}
        (OUT_DIR / "oos_results_tabm.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("seed_mean_pnl_pct", "n_pos_seeds", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
