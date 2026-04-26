#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BT_SCRIPT = ROOT / "scripts" / "backtest_dual_specialist_dsac.py"


def _run_backtest(
    csv_path: str,
    start: str,
    end: str,
    env_overrides: dict[str, str],
    out_json: Path,
    max_rows: int = 0,
) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mpl")
    out_json.parent.mkdir(parents=True, exist_ok=True)
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
        "proposed",
        "--out-json",
        str(out_json),
    ]
    if max_rows > 0:
        cmd += ["--max-rows", str(int(max_rows))]
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=180,
    )
    with out_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sample_params(rng: random.Random) -> dict[str, float]:
    return {
        "P_JUMP_Z_TH": rng.uniform(2.6, 4.0),
        "P_CHOP_STD_TH": rng.uniform(1.0, 1.8),
        "P_TH_BULL_LONG": rng.uniform(0.12, 0.34),
        "P_TH_BULL_SHORT": rng.uniform(0.30, 0.72),
        "P_TH_BEAR_LONG": rng.uniform(0.30, 0.72),
        "P_TH_BEAR_SHORT": rng.uniform(0.12, 0.34),
        # Focused search: relax normal/chop entry barriers.
        "P_TH_NORMAL": rng.uniform(0.20, 0.40),
        "P_TH_CHOP": rng.uniform(0.55, 0.85),
        # Focused search: allow larger sizing than previous best.
        "P_KELLY_CAP": rng.uniform(0.20, 0.30),
        "P_KELLY_MIN": rng.uniform(0.01, 0.06),
        "P_QUALITY_MULT": rng.uniform(0.00, 0.25),
        "P_AGREE_YES_MULT": rng.uniform(0.45, 0.95),
        "P_AGREE_NO_BASE": rng.uniform(0.05, 0.30),
        "P_AGREE_NO_EXCESS": rng.uniform(0.00, 0.20),
        "P_HARD_STOP": rng.uniform(0.02, 0.04),
        "P_M7_OPP_EXIT": rng.uniform(0.52, 0.80),
        "P_OPP_PRESSURE_EXIT": rng.uniform(0.80, 1.40),
        "P_TRAIL_ARM": rng.uniform(0.006, 0.020),
        "P_TRAIL_GAP": rng.uniform(0.004, 0.015),
        "P_REDUCE_NET_EDGE": rng.uniform(0.00, 0.12),
        "P_REDUCE_MULT": rng.uniform(0.40, 0.90),
        # Focused search: loosen TP/SL floor to reduce premature exits.
        "P_MIN_TP_OFFSET": rng.uniform(0.003, 0.012),
        "P_MIN_SL_OFFSET": rng.uniform(0.004, 0.015),
    }


def _fmt_params(params: dict[str, float]) -> dict[str, str]:
    return {k: f"{v:.6f}" for k, v in params.items()}


def _score(metrics: dict) -> float:
    pnl = float(metrics.get("pnl_pct", 0.0))
    mdd = abs(float(metrics.get("mdd_pct", 0.0)))
    trades = int(metrics.get("trades", 0))
    wr = float(metrics.get("wr_pct", 0.0))
    # Prefer positive pnl with controlled drawdown and non-zero-but-not-excessive turnover.
    turnover_pen = 0.0
    if trades < 100:
        turnover_pen += (100 - trades) * 0.03
    if trades > 3500:
        turnover_pen += (trades - 3500) * 0.01
    return pnl - (1.2 * mdd) - turnover_pen + (0.10 * wr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", default="data/splits/year_oos/rl_base_2025.csv")
    p.add_argument("--q1-start", default="2025-01-01")
    p.add_argument("--q1-end", default="2025-03-31")
    p.add_argument("--h1-start", default="2025-01-01")
    p.add_argument("--h1-end", default="2025-06-30")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--q1-max-rows", type=int, default=3000)
    p.add_argument("--h1-max-rows", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    rng = random.Random(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "data" / "ensemble" / "metrics" / f"proposed_tune_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict] = []
    for i in range(args.trials):
        params = _sample_params(rng)
        env_map = _fmt_params(params)
        out_path = run_dir / f"trial_{i:03d}_q1.json"
        try:
            payload = _run_backtest(
                args.csv_path,
                args.q1_start,
                args.q1_end,
                env_map,
                out_path,
                max_rows=args.q1_max_rows,
            )
        except Exception:
            continue
        metrics = payload.get("metrics", {})
        trials.append(
            {
                "trial": i,
                "score": _score(metrics),
                "params": params,
                "q1_metrics": metrics,
                "q1_out": str(out_path),
            }
        )

    trials.sort(key=lambda x: x["score"], reverse=True)
    finalists = trials[: max(1, args.topk)]

    validated: list[dict] = []
    for item in finalists:
        env_map = _fmt_params(item["params"])
        out_path = run_dir / f"trial_{item['trial']:03d}_h1.json"
        try:
            payload = _run_backtest(
                args.csv_path,
                args.h1_start,
                args.h1_end,
                env_map,
                out_path,
                max_rows=args.h1_max_rows,
            )
        except Exception:
            continue
        h1_metrics = payload.get("metrics", {})
        rec = dict(item)
        rec["h1_metrics"] = h1_metrics
        rec["h1_score"] = _score(h1_metrics)
        rec["h1_out"] = str(out_path)
        validated.append(rec)

    validated.sort(key=lambda x: x.get("h1_score", -1e18), reverse=True)
    best = validated[0] if validated else None

    out = {
        "created_at": ts,
        "trials": args.trials,
        "seed": args.seed,
        "csv_path": args.csv_path,
        "q1_window": [args.q1_start, args.q1_end],
        "h1_window": [args.h1_start, args.h1_end],
        "topk_validated": args.topk,
        "best": best,
        "validated": validated,
    }

    if args.out_json:
        out_path = ROOT / args.out_json
    else:
        out_path = run_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": str(out_path), "best_h1": (best or {}).get("h1_metrics", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
