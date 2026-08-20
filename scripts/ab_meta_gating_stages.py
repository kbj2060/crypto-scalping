#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_stage(stage: str, csv_path: str, val_split: float, device: str) -> dict:
    env = os.environ.copy()
    env["META_USE_KELLY_ALLOC"] = "0"
    env["META_USE_HYSTERESIS"] = "0"
    env["META_USE_RESIDUAL"] = "0"

    if stage == "A":
        env["META_USE_KELLY_ALLOC"] = "1"
    elif stage == "B":
        env["META_USE_KELLY_ALLOC"] = "1"
        env["META_USE_HYSTERESIS"] = "1"
    elif stage == "C":
        env["META_USE_KELLY_ALLOC"] = "1"
        env["META_USE_HYSTERESIS"] = "1"
        env["META_USE_RESIDUAL"] = "1"
    elif stage != "BASE":
        raise ValueError(f"unknown stage: {stage}")

    cmd = [
        PY,
        "-c",
        (
            "import json,torch,pandas as pd;"
            "from ensemble import train_rl_meta_gating as m;"
            f"df=pd.read_csv(r'{csv_path}');"
            "df=df.dropna(subset=['close']).reset_index(drop=True);"
            f"sp=int(len(df)*(1-{val_split}));"
            "dv=df.iloc[sp:].reset_index(drop=True);"
            f"dev=torch.device('{device}');"
            "net=m.MetaGatingNet().to(dev);"
            "ck=torch.load(m._BEST_PATH,map_location=dev,weights_only=False);"
            "net.load_state_dict(ck['network']);"
            "met=m._run_validation(net,dv,dev);"
            "obj=m._val_objective(met);"
            "print(json.dumps({'metrics':met,'obj':obj},ensure_ascii=False))"
        ),
    ]
    out = subprocess.check_output(cmd, cwd=str(ROOT), env=env, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B/C stage evaluation for meta gating")
    ap.add_argument("--csv", default="data/splits/year_oos/rl_meta_2026.csv")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {}
    for st in ["BASE", "A", "B", "C"]:
        results[st] = run_stage(st, args.csv, args.val_split, args.device)

    out = {"csv": args.csv, "val_split": args.val_split, "results": results}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
