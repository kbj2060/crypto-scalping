#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble import train_rl_meta_gating as mg


def _load_actor(best_path: Path, device: torch.device) -> mg.MetaDSACActor:
    actor = mg.MetaDSACActor().to(device)
    ckpt = torch.load(best_path, map_location=device)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        actor.load_state_dict(ckpt)
    actor.eval()
    return actor


def _split_val(df: pd.DataFrame, val_split: float, n_folds: int) -> list[pd.DataFrame]:
    split = int(len(df) * (1 - val_split))
    df_val = df.iloc[split:].reset_index(drop=True)
    fold_size = max(len(df_val) // n_folds, 200)
    folds = [
        df_val.iloc[i * fold_size: (i + 1) * fold_size].reset_index(drop=True)
        for i in range(n_folds)
    ]
    folds[-1] = df_val.iloc[(n_folds - 1) * fold_size:].reset_index(drop=True)
    return folds


def _eval(actor: mg.MetaDSACActor, folds: list[pd.DataFrame], short_mode: str) -> dict:
    out = []
    for i, f in enumerate(folds):
        df_fold = f.copy()
        s = df_fold["meta_short_raw"].astype(float)
        if short_mode == "negate":
            df_fold["meta_short_raw"] = -s
        elif short_mode == "signed_2x_minus_1":
            # [0,1] 확률형 raw를 [-1,1] signed action으로 재해석
            df_fold["meta_short_raw"] = (s * 2.0) - 1.0
        elif short_mode != "baseline":
            raise ValueError(f"unknown short_mode: {short_mode}")
        m = mg._validate(actor, df_fold, next(actor.parameters()).device)
        m["fold"] = i
        out.append(m)

    agg = {
        "pnl_pct": float(sum(x["pnl_pct"] for x in out)),
        "trades": int(sum(x["trades"] for x in out)),
    }
    if agg["trades"] > 0:
        wr_num = sum(x["wr"] * x["trades"] for x in out)
        agg["wr"] = float(wr_num / agg["trades"])
    else:
        agg["wr"] = 0.0
    agg["sortino_avg"] = float(sum(x["sortino"] for x in out) / max(len(out), 1))

    return {"folds": out, "agg": agg}


def main() -> None:
    ap = argparse.ArgumentParser(description="Meta gating: short sign reinterpretation diagnostic")
    ap.add_argument("--csv", default=mg._DEFAULT_CSV)
    ap.add_argument("--best", default=mg._BEST_PATH)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=3)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    best_path = Path(args.best)
    if not csv_path.exists():
        raise SystemExit(f"csv not found: {csv_path}")
    if not best_path.exists():
        raise SystemExit(f"best checkpoint not found: {best_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    actor = _load_actor(best_path, device)
    folds = _split_val(df, args.val_split, args.n_folds)

    base = _eval(actor, folds, short_mode="baseline")
    neg = _eval(actor, folds, short_mode="negate")
    signed = _eval(actor, folds, short_mode="signed_2x_minus_1")

    report = {
        "csv": str(csv_path),
        "best": str(best_path),
        "device": str(device),
        "val_split": args.val_split,
        "n_folds": args.n_folds,
        "baseline_unsigned_short": base,
        "negated_short_signed": neg,
        "signed_2x_minus_1_short": signed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
