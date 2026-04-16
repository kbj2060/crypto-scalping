#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BT_SCRIPT = ROOT / "scripts" / "backtest_dual_specialist_dsac.py"


def _run_case(
    *,
    name: str,
    mode: str,
    csv_path: str,
    start: str,
    end: str,
    env_overrides: dict[str, str],
    out_dir: Path,
    max_rows: int,
) -> dict:
    out_json = out_dir / f"{name}.json"
    env = os.environ.copy()
    env.update(env_overrides)
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mpl")

    cmd = [
        "python",
        str(BT_SCRIPT),
        "--csv-path",
        csv_path,
        "--start",
        start,
        "--end",
        end,
        "--mode",
        mode,
        "--out-json",
        str(out_json),
    ]
    if max_rows > 0:
        cmd += ["--max-rows", str(max_rows)]

    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with out_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "name": name,
        "mode": mode,
        "env": env_overrides,
        "metrics": payload.get("metrics", {}),
        "extra": payload.get("extra", {}),
        "out_json": str(out_json),
    }


def _delta(base: dict, cur: dict) -> dict:
    b = base.get("metrics", {})
    c = cur.get("metrics", {})
    return {
        "pnl_pct": float(c.get("pnl_pct", 0.0)) - float(b.get("pnl_pct", 0.0)),
        "mdd_pct": float(c.get("mdd_pct", 0.0)) - float(b.get("mdd_pct", 0.0)),
        "trades": int(c.get("trades", 0)) - int(b.get("trades", 0)),
        "wr_pct": float(c.get("wr_pct", 0.0)) - float(b.get("wr_pct", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Blueprint A/B template runner (no trading_bot.py edits)")
    ap.add_argument("--csv-path", default="data/splits/year_oos/rl_base_2025.csv")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "ensemble" / "metrics" / f"blueprint_ab_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage sequence:
    # S0 baseline specialist-only (pure_rl)
    # S1 proposed + hard-no-go 중심 (scaling/exit 약화)
    # S2 S1 + scaling 활성
    # S3 S2 + exit 강화 (TP/SL은 제외)
    cases = [
        {
            "name": "S0_baseline_pure_rl",
            "mode": "pure_rl",
            "env": {},
        },
        {
            "name": "S1_no_go_core",
            "mode": "proposed",
            "env": {
                "P_ENABLE_TPSL": "false",
                "P_QUALITY_MULT": "0.00",
                "P_AGREE_YES_MULT": "1.00",
                "P_AGREE_NO_BASE": "1.00",
                "P_AGREE_NO_EXCESS": "0.00",
                "P_M7_OPP_EXIT": "1.10",
                "P_OPP_PRESSURE_EXIT": "9.00",
                "P_TRAIL_ARM": "9.00",
                "P_TRAIL_GAP": "9.00",
                "P_REDUCE_NET_EDGE": "-9.00",
                "P_REDUCE_MULT": "1.00",
            },
        },
        {
            "name": "S2_no_go_plus_scaling",
            "mode": "proposed",
            "env": {
                "P_ENABLE_TPSL": "false",
                "P_QUALITY_MULT": "0.20",
                "P_AGREE_YES_MULT": "0.90",
                "P_AGREE_NO_BASE": "0.30",
                "P_AGREE_NO_EXCESS": "0.20",
                "P_M7_OPP_EXIT": "1.10",
                "P_OPP_PRESSURE_EXIT": "9.00",
                "P_TRAIL_ARM": "9.00",
                "P_TRAIL_GAP": "9.00",
                "P_REDUCE_NET_EDGE": "-9.00",
                "P_REDUCE_MULT": "1.00",
            },
        },
        {
            "name": "S3_no_go_scaling_exit",
            "mode": "proposed",
            "env": {
                "P_ENABLE_TPSL": "false",
                "P_QUALITY_MULT": "0.20",
                "P_AGREE_YES_MULT": "0.90",
                "P_AGREE_NO_BASE": "0.30",
                "P_AGREE_NO_EXCESS": "0.20",
                "P_M7_OPP_EXIT": "0.60",
                "P_OPP_PRESSURE_EXIT": "1.15",
                "P_TRAIL_ARM": "0.012",
                "P_TRAIL_GAP": "0.008",
                "P_REDUCE_NET_EDGE": "0.05",
                "P_REDUCE_MULT": "0.65",
            },
        },
    ]

    results = []
    for case in cases:
        rec = _run_case(
            name=case["name"],
            mode=case["mode"],
            csv_path=args.csv_path,
            start=args.start,
            end=args.end,
            env_overrides=case["env"],
            out_dir=out_dir,
            max_rows=args.max_rows,
        )
        results.append(rec)

    baseline = results[0]
    summary = []
    for rec in results:
        row = {
            "name": rec["name"],
            "mode": rec["mode"],
            "metrics": rec["metrics"],
            "out_json": rec["out_json"],
        }
        if rec is not baseline:
            row["delta_vs_baseline"] = _delta(baseline, rec)
        summary.append(row)

    payload = {
        "created_at": ts,
        "window": {"start": args.start, "end": args.end},
        "csv_path": args.csv_path,
        "max_rows": args.max_rows,
        "cases": summary,
    }

    if args.out_json:
        out_json = ROOT / args.out_json
    else:
        out_json = out_dir / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": str(out_json), "baseline": baseline["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
