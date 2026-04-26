#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BT = ROOT / "scripts" / "backtest_dual_specialist_dsac.py"


@dataclass
class Row:
    name: str
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    score: float
    out_json: str


def run_one(
    profile_name: str,
    env_patch: dict[str, str],
    csv_path: str,
    max_rows: int,
    long_ckpt: str,
    out_dir: Path,
) -> Row:
    out_json = out_dir / f"limit_idea_{profile_name}.json"
    env = os.environ.copy()
    env.update(env_patch)
    cmd = [
        "python",
        str(BT),
        "--csv-path",
        csv_path,
        "--mode",
        "classic",
        "--max-rows",
        str(max_rows),
        "--long-ckpt",
        long_ckpt,
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    m = payload["metrics"]
    score = float(m["pnl_pct"]) - 0.5 * abs(float(m["mdd_pct"]))
    return Row(
        name=profile_name,
        pnl_pct=float(m["pnl_pct"]),
        mdd_pct=float(m["mdd_pct"]),
        trades=int(m["trades"]),
        wr_pct=float(m["wr_pct"]),
        score=score,
        out_json=str(out_json),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run limit-order idea suite and rank best profile.")
    ap.add_argument("--csv-path", default="data/ensemble/backup/rl_training_2025_m7_pre_prune_20260329.csv")
    ap.add_argument("--max-rows", type=int, default=12000)
    ap.add_argument("--long-ckpt", default="data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--out-json", default="data/ensemble/metrics/limit_idea_suite_result.json")
    args = ap.parse_args()

    out_path = (ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_path.parent

    profiles: dict[str, dict[str, str]] = {
        "baseline_market_like": {
            "DSAC_LIMIT_ENTRY_ENABLE": "false",
        },
        "idea1_asym_obi": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_OBI_WEIGHT": "3.2",
            "DSAC_LIMIT_REPLACE_ENABLE": "false",
        },
        "idea2_queue_collapse_replace": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_REPLACE_ENABLE": "true",
            "DSAC_LIMIT_REPLACE_MULT": "1.45",
            "DSAC_LIMIT_REPLACE_PRESSURE_TH": "0.72",
        },
        "idea3_hawkes_tail_proxy": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_ENTRY_URGENCY_JUMP_Z": "3.2",
            "DSAC_LIMIT_LIQ_WEIGHT": "1.6",
            "DSAC_LIMIT_REPLACE_ENABLE": "true",
        },
        "idea4_toxicity_regime_switch": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_ENTRY_TOXICITY_CUT": "0.45",
            "DSAC_LIMIT_ENTRY_MIN_BPS": "0.8",
            "DSAC_LIMIT_ENTRY_BASE_BPS": "2.0",
        },
        "idea5_spoof_wall_proxy": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_OBI_WEIGHT": "2.6",
            "DSAC_LIMIT_ENTRY_SIGNAL_BONUS_BPS": "3.0",
        },
        "idea6_vol_adaptive_ttl": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_TTL_ADAPT_ENABLE": "true",
            "DSAC_LIMIT_TTL_VOL_K": "0.55",
            "DSAC_LIMIT_TTL_MIN_BARS": "2",
            "DSAC_LIMIT_TTL_MAX_BARS": "10",
        },
        "idea7_vwap_pegged_proxy": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_VWAP_WEIGHT": "2.8",
            "DSAC_LIMIT_ENTRY_BASE_BPS": "2.2",
            "DSAC_LIMIT_ENTRY_MAX_BPS": "10.0",
        },
        "idea8_liq_magnet_proxy": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_LIQ_WEIGHT": "2.6",
            "DSAC_LIMIT_ENTRY_BASE_BPS": "2.0",
            "DSAC_LIMIT_ENTRY_MAX_BPS": "11.0",
        },
        "idea9_combo_safe": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_OBI_WEIGHT": "2.2",
            "DSAC_LIMIT_VWAP_WEIGHT": "1.5",
            "DSAC_LIMIT_LIQ_WEIGHT": "1.4",
            "DSAC_LIMIT_REPLACE_ENABLE": "true",
            "DSAC_LIMIT_REPLACE_MULT": "1.25",
            "DSAC_LIMIT_TTL_ADAPT_ENABLE": "true",
            "DSAC_LIMIT_TTL_MIN_BARS": "3",
            "DSAC_LIMIT_TTL_MAX_BARS": "9",
        },
        "idea10_full_stack": {
            "DSAC_LIMIT_ENTRY_ENABLE": "true",
            "DSAC_LIMIT_ENTRY_MIN_BPS": "1.0",
            "DSAC_LIMIT_ENTRY_BASE_BPS": "2.5",
            "DSAC_LIMIT_ENTRY_MAX_BPS": "12.0",
            "DSAC_LIMIT_OBI_WEIGHT": "2.8",
            "DSAC_LIMIT_VWAP_WEIGHT": "2.2",
            "DSAC_LIMIT_LIQ_WEIGHT": "2.0",
            "DSAC_LIMIT_REPLACE_ENABLE": "true",
            "DSAC_LIMIT_REPLACE_MULT": "1.35",
            "DSAC_LIMIT_REPLACE_PRESSURE_TH": "0.75",
            "DSAC_LIMIT_TTL_ADAPT_ENABLE": "true",
            "DSAC_LIMIT_TTL_MIN_BARS": "3",
            "DSAC_LIMIT_TTL_MAX_BARS": "12",
        },
    }

    rows: list[Row] = []
    for name, patch in profiles.items():
        row = run_one(
            profile_name=name,
            env_patch=patch,
            csv_path=args.csv_path,
            max_rows=args.max_rows,
            long_ckpt=args.long_ckpt,
            out_dir=out_dir,
        )
        rows.append(row)
        print(
            f"{name:28s} pnl={row.pnl_pct:+8.3f}% mdd={row.mdd_pct:+7.3f}% "
            f"trades={row.trades:4d} wr={row.wr_pct:6.2f}% score={row.score:+8.3f}"
        )

    rows_sorted = sorted(rows, key=lambda r: r.score, reverse=True)
    payload = {
        "csv_path": args.csv_path,
        "max_rows": int(args.max_rows),
        "ranking_metric": "score = pnl_pct - 0.5*abs(mdd_pct)",
        "rows": [asdict(r) for r in rows_sorted],
        "best": asdict(rows_sorted[0]) if rows_sorted else None,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nBest: {rows_sorted[0].name if rows_sorted else '-'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
