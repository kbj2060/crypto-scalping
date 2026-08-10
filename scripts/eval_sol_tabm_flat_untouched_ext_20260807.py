"""Untouched-window confirmation read for the SOL survey's surviving candidate
(parent contract docs/experiments/sol_dl_rl_architecture_survey_20260807.json; TabM-flat 5-seed
stack, frozen rule side_prob_055; VAL +0.65% / OOS 2026Q1 +4.08% seed-mean).

Window: 2026-04-01 .. panel end (2026-07-21) -- never used for any training, selection, or gate
in this line. Everything is frozen; this script is inference + replay only.

PRE-REGISTERED PASS RULE (fixed before the window is read, same bar as the parent's OOS
adoption): seed-mean PnL > 0 AND >=3/5 seeds positive AND seed-mean trades >= 15. A fail closes
the candidate as a 2026Q1 artifact; a pass upgrades it to "confirmed on a second untouched
window" (still research-grade, promotion would additionally need the full sizing/router stack).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import replay, side_state_from_proba  # noqa: E402
import train_eval_sol_deepfeat_candidates_20260807 as dl  # noqa: E402

DL_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/dl_tabm_flat"
OUT_PATH = ROOT / "tmp/sol_dl_rl_survey_20260807/tabm_flat_untouched_ext.json"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
THRESHOLD = 0.55  # frozen parent rule
EXT_START = pd.Timestamp("2026-04-01")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols = dl.build_data()
    ts = panel["timestamp"]
    ext_mask = (ts >= EXT_START).to_numpy()
    ext_rows = np.flatnonzero(ext_mask)
    close = panel["close"].to_numpy()
    buyhold_pct = float((close[ext_rows[-1]] / close[ext_rows[0]] - 1.0) * 100.0)

    per_seed = []
    proba_sum = None
    for seed in SEEDS:
        model = dl.FlatTabM(x_std.shape[1]).to(device)
        model.load_state_dict(torch.load(DL_DIR / f"model_seed{seed}.pt", map_location=device))
        proba = dl.predict_rows(model, "tabm_flat", x_std, ext_rows, device)
        proba_sum = proba if proba_sum is None else proba_sum + proba
        side = np.zeros(len(panel), dtype=np.int64)
        side[ext_rows] = side_state_from_proba(proba, THRESHOLD)
        r = replay(panel, side, tp_moves, sl_moves, ext_mask)
        per_seed.append({"seed": seed, **r})
        print(json.dumps(per_seed[-1]), flush=True)

    side = np.zeros(len(panel), dtype=np.int64)
    side[ext_rows] = side_state_from_proba(proba_sum / len(SEEDS), THRESHOLD)
    ens = replay(panel, side, tp_moves, sl_moves, ext_mask)

    pnls = [r.get("pnl_pct", 0.0) for r in per_seed]
    trades = [r.get("n_trades", 0) for r in per_seed]
    passed = bool(np.mean(pnls) > 0 and sum(p > 0 for p in pnls) >= 3 and np.mean(trades) >= 15)
    out = {
        "window": [str(EXT_START.date()), str(ts.iloc[-1])],
        "buyhold_pct": buyhold_pct,
        "seed_mean_pnl_pct": float(np.mean(pnls)),
        "n_pos_seeds": int(sum(p > 0 for p in pnls)),
        "seed_mean_trades": float(np.mean(trades)),
        "per_seed": per_seed,
        "seed_ensemble": ens,
        "pre_registered_pass": passed,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("window", "buyhold_pct", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades", "pre_registered_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
