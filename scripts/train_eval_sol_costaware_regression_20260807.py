"""Rev4 cost-aware magnitude-filtered return regression for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev4_costaware_reg_20260807.json), porting the
design of arXiv:2606.00060 (2026): hourly decision cadence, return-magnitude regression, and an
execution filter that only CHANGES position when |forecast| > k x roundtrip cost -- weak signals
hold the current position rather than exiting (the paper's turnover mechanism).

Decision bars: every on-the-hour 5m bar i. Target: log(open[i+13]/open[i+1]) -- the exact
open-to-open segment this decision controls under next-bar-open execution. Replay: position-target
on the 5m grid, PnL = pos x segment return x notional, cost 5bps x |delta pos| x notional.

Families: LightGBM regression (single deterministic fit) and TabM-flat regression (5 seeds).
Closed grid per family: k in {1,2,3} x side in {long_short, long_only}. VAL-only selection;
`--stage oos` replays each passing family's frozen config once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_sol_deepfeat_candidates_20260807 as dl  # noqa: E402

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/costaware_reg_rev4"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
HORIZON = 12  # 1h on the 5m grid
COST_RT = 0.0010  # roundtrip on price move
HALF_COST = COST_RT / 2.0
MARGIN_FRACTION, LEVERAGE = 0.30, 3.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
K_GRID = [1.0, 2.0, 3.0]
SIDE_GRID = ["long_short", "long_only"]
MAX_EPOCHS = 30
PATIENCE = 5
BATCH = 512
LR = 1e-3


class TabMReg(nn.Module):
    def __init__(self, in_dim: int, n_experts: int = 8, hidden: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.n_experts = n_experts
        self.expert_scale = nn.Parameter(torch.ones(n_experts, in_dim))
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, in_dim))
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        b, d = x.shape
        xe = x.unsqueeze(1) * self.expert_scale.unsqueeze(0) + self.expert_bias.unsqueeze(0)
        h = self.trunk(xe.reshape(b * self.n_experts, d)).reshape(b, self.n_experts, -1)
        return self.head(h).mean(dim=1).squeeze(-1)


def hourly_decision_rows(panel: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    on_hour = (panel["timestamp"].dt.minute == 0).to_numpy()
    rows = np.flatnonzero(mask & on_hour)
    return rows[rows + HORIZON + 1 < len(panel)]


def build_target(panel: pd.DataFrame) -> np.ndarray:
    open_v = panel["open"].to_numpy(dtype=np.float64)
    y = np.full(len(panel), np.nan)
    valid_end = len(panel) - HORIZON - 1
    idx = np.arange(valid_end)
    y[idx] = np.log(open_v[idx + HORIZON + 1] / open_v[idx + 1])
    return y


def positions_from_forecast(yhat: np.ndarray, k: float, side: str) -> np.ndarray:
    """Position targets at each decision; weak signals (|yhat| <= k*cost) hold previous position."""
    pos = np.zeros(len(yhat))
    cur = 0.0
    thr = k * COST_RT
    for i, f in enumerate(yhat):
        if np.isfinite(f) and abs(f) > thr:
            cur = 1.0 if f > 0 else -1.0
            if side == "long_only" and cur < 0:
                cur = 0.0
        pos[i] = cur
    return pos


def replay_positions(panel: pd.DataFrame, dec_rows: np.ndarray, pos: np.ndarray) -> dict:
    """PnL over each decision's controlled segment open[i+1] -> open[i+13], compounded; cost on
    position changes. Equity marked per segment (1h granularity MDD)."""
    open_v = panel["open"].to_numpy(dtype=np.float64)
    cash = 1.0
    prev_pos = 0.0
    equity = [1.0]
    n_entries = 0
    n_changes = 0
    for j, i in enumerate(dec_rows):
        p = pos[j]
        seg_ret = open_v[i + HORIZON + 1] / open_v[i + 1] - 1.0
        turn = abs(p - prev_pos)
        if turn > 0:
            n_changes += 1
            if p != 0.0:
                n_entries += 1
        r = p * seg_ret * NOTIONAL - HALF_COST * turn * NOTIONAL
        cash *= 1.0 + r
        equity.append(cash)
        prev_pos = p
    # close any open position at the end
    if prev_pos != 0.0:
        cash *= 1.0 - HALF_COST * abs(prev_pos) * NOTIONAL
        equity.append(cash)
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    frac_long = float((pos > 0).mean()) if len(pos) else 0.0
    frac_short = float((pos < 0).mean()) if len(pos) else 0.0
    return {
        "pnl_pct": float((cash - 1.0) * 100.0),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "n_entries": int(n_entries),
        "n_position_changes": int(n_changes),
        "frac_hours_long": frac_long,
        "frac_hours_short": frac_short,
    }


def train_tabm_seed(seed, x_std, y, train_rows, val_rows, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabMReg(x_std.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    y_t = torch.from_numpy((y * 100.0).astype(np.float32))  # scale to ~unit magnitude
    rng = np.random.default_rng(seed)
    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train_rows))
        for i in range(0, len(order), BATCH):
            rows = train_rows[order[i : i + BATCH]]
            xb = torch.from_numpy(x_std[rows]).to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, y_t[rows].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(val_rows), 8192):
                preds.append(model(torch.from_numpy(x_std[val_rows[i : i + 8192]]).to(device)).cpu())
            val_loss = float(nn.functional.mse_loss(torch.cat(preds), y_t[val_rows]))
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[tabm_reg seed={seed}] epoch {epoch} val_mse={val_loss:.6f}{' *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def tabm_predict(model, x_std, rows, device):
    model.eval()
    out = []
    for i in range(0, len(rows), 8192):
        out.append(model(torch.from_numpy(x_std[rows[i : i + 8192]]).to(device)).cpu().numpy())
    return np.concatenate(out) / 100.0


def select_table(panel, dec_rows, yhat_by_seed: dict, family: str) -> list[dict]:
    table = []
    for k in K_GRID:
        for side in SIDE_GRID:
            per = []
            for seed, yhat in yhat_by_seed.items():
                pos = positions_from_forecast(yhat, k, side)
                per.append({"seed": seed, **replay_positions(panel, dec_rows, pos)})
            pnls = [r["pnl_pct"] for r in per]
            entries = [r["n_entries"] for r in per]
            table.append({
                "family": family, "k": k, "side": side,
                "seed_mean_pnl_pct": float(np.mean(pnls)),
                "n_pos_seeds": int(sum(p > 0 for p in pnls)),
                "seed_mean_entries": float(np.mean(entries)),
                "per_seed": per,
            })
            print(json.dumps({kk: table[-1][kk] for kk in ("family", "k", "side", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_entries")}), flush=True)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols = dl.build_data()
    y = build_target(panel)

    raw = pd.read_csv(ROOT / "data/splits/year_oos/sol_features_2024_2026.csv", low_memory=False)
    raw_x = raw[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    train_rows = hourly_decision_rows(panel, train_mask)
    train_rows = train_rows[np.isfinite(y[train_rows])]
    val_rows = hourly_decision_rows(panel, val_mask)
    oos_rows = hourly_decision_rows(panel, oos_mask)
    eval_rows = val_rows if args.stage == "val" else oos_rows

    lgbm_path = OUT_DIR / "lgbm_reg.txt"
    if args.stage == "val":
        reg = lgb.LGBMRegressor(objective="regression", n_estimators=600, learning_rate=0.05,
                                num_leaves=63, min_child_samples=100, feature_fraction=0.8,
                                bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                random_state=SEEDS[0], n_jobs=-1, verbosity=-1)
        reg.fit(raw_x[train_rows], y[train_rows])
        reg.booster_.save_model(str(lgbm_path))
    booster = lgb.Booster(model_file=str(lgbm_path))
    yhat_lgbm = booster.predict(raw_x[eval_rows])

    if args.stage == "val":
        yhat_tabm = {}
        for seed in SEEDS:
            model = train_tabm_seed(seed, x_std, y, train_rows, val_rows, device)
            torch.save(model.state_dict(), OUT_DIR / f"tabm_reg_seed{seed}.pt")
            yhat_tabm[seed] = tabm_predict(model, x_std, val_rows, device)
            np.save(OUT_DIR / f"val_yhat_seed{seed}.npy", yhat_tabm[seed])

        table_l = select_table(panel, val_rows, {"lgbm": yhat_lgbm}, "lgbm_reg")
        table_t = select_table(panel, val_rows, yhat_tabm, "tabm_reg")
        out = {"stage": "val", "families": {}}
        for fam, table in (("lgbm_reg", table_l), ("tabm_reg", table_t)):
            eligible = [r for r in table if r["seed_mean_entries"] >= 15]
            best = max(eligible, key=lambda r: r["seed_mean_pnl_pct"]) if eligible else None
            earns = bool(best and best["seed_mean_pnl_pct"] > 0)
            out["families"][fam] = {
                "table": table,
                "selected": None if best is None else {kk: best[kk] for kk in ("k", "side", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_entries")},
                "earns_oos_read": earns,
            }
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({f: {"selected": v["selected"], "earns_oos_read": v["earns_oos_read"]} for f, v in out["families"].items()}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        results = {}
        for fam, info in prior["families"].items():
            if not info.get("earns_oos_read"):
                results[fam] = "REFUSED -- VAL gate failed"
                continue
            sel = info["selected"]
            if fam == "lgbm_reg":
                yhats = {"lgbm": yhat_lgbm}
            else:
                yhats = {}
                for seed in SEEDS:
                    model = TabMReg(x_std.shape[1]).to(device)
                    model.load_state_dict(torch.load(OUT_DIR / f"tabm_reg_seed{seed}.pt", map_location=device))
                    yhats[seed] = tabm_predict(model, x_std, oos_rows, device)
            per = []
            for seed, yhat in yhats.items():
                pos = positions_from_forecast(yhat, sel["k"], sel["side"])
                per.append({"seed": seed, **replay_positions(panel, oos_rows, pos)})
                print(json.dumps(per[-1]), flush=True)
            pnls = [r["pnl_pct"] for r in per]
            results[fam] = {"selected": sel, "seed_mean_pnl_pct": float(np.mean(pnls)),
                            "n_pos_seeds": int(sum(p > 0 for p in pnls)), "per_seed": per}
        (OUT_DIR / "oos_results.json").write_text(json.dumps({"stage": "oos", "results": results}, indent=2))
        print(json.dumps({f: (v if isinstance(v, str) else {kk: v[kk] for kk in ("selected", "seed_mean_pnl_pct", "n_pos_seeds")}) for f, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
