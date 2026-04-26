#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass


@dataclass
class PeriodCfg:
    name: str
    csv_path: str
    start: str
    end: str
    max_rows: int


def _run_backtest(period: PeriodCfg, env: dict[str, str], out_dir: str, tag: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    key_src = {
        "period": period.__dict__,
        "tag": tag,
        "env": {k: env[k] for k in sorted(env.keys()) if k.startswith("ULT_")},
    }
    key = hashlib.md5(json.dumps(key_src, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out_json = os.path.join(out_dir, f"{period.name}_{tag}_{key}.json")
    if not os.path.exists(out_json):
        cmd = (
            "MPLCONFIGDIR=/tmp/mpl PYTHONWARNINGS=ignore "
            "python scripts/backtest_ultimate_3model_ensemble.py "
            "--ensemble-only "
            f"--csv-path {period.csv_path} "
            f"--start '{period.start}' --end '{period.end}' "
            f"--max-rows {int(period.max_rows)} "
            f"--out-json {out_json}"
        )
        r = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"backtest failed\ncmd={cmd}\n{r.stdout[-1500:]}")
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["comparison"]["results"]


def _u(pnl_pct: float) -> float:
    return math.tanh(float(pnl_pct) / 80.0)


def _overfit_pack(is_m: dict, oos_m: dict) -> dict:
    is_p = float(is_m.get("pnl_pct", 0.0))
    oos_p = float(oos_m.get("pnl_pct", 0.0))
    retention = (oos_p / is_p) if abs(is_p) > 1e-12 else 0.0
    return {
        "is_pnl": is_p,
        "oos_pnl": oos_p,
        "retention": retention,
        "gap": is_p - oos_p,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/ensemble/metrics/dual_profile_tuning")
    ap.add_argument("--summary", default="data/ensemble/metrics/dual_profile_tuning_summary.json")
    ap.add_argument("--is-max-rows", type=int, default=7000)
    ap.add_argument("--oos-max-rows", type=int, default=7000)
    ap.add_argument("--ext-max-rows", type=int, default=7000)
    args = ap.parse_args()

    # recent split (IS/OOS) + external OOS
    is_period = PeriodCfg(
        name="recent_is",
        csv_path="data/rl_training_data_latest.csv",
        start="2026-03-04 11:25:00",
        end="2026-03-24 23:55:00",
        max_rows=int(args.is_max_rows),
    )
    oos_period = PeriodCfg(
        name="recent_oos",
        csv_path="data/rl_training_data_latest.csv",
        start="2026-03-25 00:00:00",
        end="2026-04-13 09:00:00",
        max_rows=int(args.oos_max_rows),
    )
    ext_period = PeriodCfg(
        name="ext_oos_2025q4",
        csv_path="data/splits/year_oos/rl_training_2025_m7.csv",
        start="2025-11-01 00:00:00",
        end="2025-12-31 23:55:00",
        max_rows=int(args.ext_max_rows),
    )

    static_candidates = [
        {"name": "st_a", "ULT_W_PRIMARY": "0.55", "ULT_W_LONG": "0.25", "ULT_W_SHORT": "0.20", "ULT_ENTRY_TH": "0.11", "ULT_CLOSE_TH": "0.035", "ULT_FLIP_TH": "0.18", "ULT_KELLY_SCALE": "0.80"},
        {"name": "st_b", "ULT_W_PRIMARY": "0.50", "ULT_W_LONG": "0.30", "ULT_W_SHORT": "0.20", "ULT_ENTRY_TH": "0.10", "ULT_CLOSE_TH": "0.030", "ULT_FLIP_TH": "0.16", "ULT_KELLY_SCALE": "0.85"},
        {"name": "st_c", "ULT_W_PRIMARY": "0.60", "ULT_W_LONG": "0.25", "ULT_W_SHORT": "0.15", "ULT_ENTRY_TH": "0.12", "ULT_CLOSE_TH": "0.040", "ULT_FLIP_TH": "0.20", "ULT_KELLY_SCALE": "0.75"},
    ]

    regime_candidates = [
        {
            "name": "rg_a",
            "ULT_BULL_W_PRIMARY": "0.15", "ULT_BULL_W_LONG": "0.75", "ULT_BULL_W_SHORT": "0.10",
            "ULT_BEAR_W_PRIMARY": "0.20", "ULT_BEAR_W_LONG": "0.10", "ULT_BEAR_W_SHORT": "0.70",
            "ULT_CHOP_W_PRIMARY": "0.65", "ULT_CHOP_W_LONG": "0.20", "ULT_CHOP_W_SHORT": "0.15",
            "ULT_NORMAL_W_PRIMARY": "0.45", "ULT_NORMAL_W_LONG": "0.35", "ULT_NORMAL_W_SHORT": "0.20",
            "ULT_ENTRY_TH": "0.09", "ULT_CLOSE_TH": "0.028", "ULT_FLIP_TH": "0.14", "ULT_KELLY_SCALE": "0.95",
        },
        {
            "name": "rg_b",
            "ULT_BULL_W_PRIMARY": "0.20", "ULT_BULL_W_LONG": "0.70", "ULT_BULL_W_SHORT": "0.10",
            "ULT_BEAR_W_PRIMARY": "0.25", "ULT_BEAR_W_LONG": "0.10", "ULT_BEAR_W_SHORT": "0.65",
            "ULT_CHOP_W_PRIMARY": "0.60", "ULT_CHOP_W_LONG": "0.20", "ULT_CHOP_W_SHORT": "0.20",
            "ULT_NORMAL_W_PRIMARY": "0.50", "ULT_NORMAL_W_LONG": "0.30", "ULT_NORMAL_W_SHORT": "0.20",
            "ULT_ENTRY_TH": "0.10", "ULT_CLOSE_TH": "0.030", "ULT_FLIP_TH": "0.16", "ULT_KELLY_SCALE": "0.90",
        },
        {
            "name": "rg_c",
            "ULT_BULL_W_PRIMARY": "0.10", "ULT_BULL_W_LONG": "0.80", "ULT_BULL_W_SHORT": "0.10",
            "ULT_BEAR_W_PRIMARY": "0.20", "ULT_BEAR_W_LONG": "0.05", "ULT_BEAR_W_SHORT": "0.75",
            "ULT_CHOP_W_PRIMARY": "0.70", "ULT_CHOP_W_LONG": "0.15", "ULT_CHOP_W_SHORT": "0.15",
            "ULT_NORMAL_W_PRIMARY": "0.40", "ULT_NORMAL_W_LONG": "0.40", "ULT_NORMAL_W_SHORT": "0.20",
            "ULT_ENTRY_TH": "0.08", "ULT_CLOSE_TH": "0.025", "ULT_FLIP_TH": "0.13", "ULT_KELLY_SCALE": "1.00",
        },
    ]

    base_env = os.environ.copy()

    # Tune static (conservative objective)
    static_trials = []
    for c in static_candidates:
        env = base_env.copy()
        env.update(c)
        out_is = _run_backtest(is_period, env, args.out_dir, c["name"])
        out_oos = _run_backtest(oos_period, env, args.out_dir, c["name"])
        out_ext = _run_backtest(ext_period, env, args.out_dir, c["name"])

        m_is = out_is["ultimate_ensemble_3m"]
        m_oos = out_oos["ultimate_ensemble_3m"]
        m_ext = out_ext["ultimate_ensemble_3m"]

        u_is, u_oos, u_ext = _u(m_is["pnl_pct"]), _u(m_oos["pnl_pct"]), _u(m_ext["pnl_pct"])
        gap = max(0.0, u_is - u_oos)
        # conservative: OOS/ext + MDD penalty heavier
        score = (0.15 * u_is + 0.45 * u_oos + 0.40 * u_ext) - 0.20 * gap - 0.08 * ((abs(m_oos["mdd_pct"]) + abs(m_ext["mdd_pct"])) / 10.0)

        static_trials.append(
            {
                "name": c["name"],
                "params": {k: v for k, v in c.items() if k != "name"},
                "score": float(score),
                "is": m_is,
                "oos": m_oos,
                "ext_oos": m_ext,
                "overfit": _overfit_pack(m_is, m_oos),
            }
        )

    static_trials.sort(key=lambda x: x["score"], reverse=True)
    best_static = static_trials[0]

    # Tune regime-weighted (progressive objective)
    regime_trials = []
    for c in regime_candidates:
        env = base_env.copy()
        env.update(c)
        out_is = _run_backtest(is_period, env, args.out_dir, c["name"])
        out_oos = _run_backtest(oos_period, env, args.out_dir, c["name"])
        out_ext = _run_backtest(ext_period, env, args.out_dir, c["name"])

        m_is = out_is["ultimate_ensemble_regime_weighted"]
        m_oos = out_oos["ultimate_ensemble_regime_weighted"]
        m_ext = out_ext["ultimate_ensemble_regime_weighted"]

        u_is, u_oos, u_ext = _u(m_is["pnl_pct"]), _u(m_oos["pnl_pct"]), _u(m_ext["pnl_pct"])
        gap = max(0.0, u_is - u_oos)
        # progressive: pnl priority but still penalize overfit
        score = (0.25 * u_is + 0.45 * u_oos + 0.30 * u_ext) - 0.12 * gap - 0.05 * ((abs(m_oos["mdd_pct"]) + abs(m_ext["mdd_pct"])) / 10.0)

        regime_trials.append(
            {
                "name": c["name"],
                "params": {k: v for k, v in c.items() if k != "name"},
                "score": float(score),
                "is": m_is,
                "oos": m_oos,
                "ext_oos": m_ext,
                "overfit": _overfit_pack(m_is, m_oos),
            }
        )

    regime_trials.sort(key=lambda x: x["score"], reverse=True)
    best_regime = regime_trials[0]

    payload = {
        "periods": {
            "is": is_period.__dict__,
            "oos": oos_period.__dict__,
            "ext_oos": ext_period.__dict__,
        },
        "best_static": best_static,
        "best_regime_weighted": best_regime,
        "static_trials": static_trials,
        "regime_trials": regime_trials,
    }

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(json.dumps({
        "best_static": {
            "name": best_static["name"],
            "score": best_static["score"],
            "oos_pnl": best_static["oos"]["pnl_pct"],
            "oos_mdd": best_static["oos"]["mdd_pct"],
        },
        "best_regime_weighted": {
            "name": best_regime["name"],
            "score": best_regime["score"],
            "oos_pnl": best_regime["oos"]["pnl_pct"],
            "oos_mdd": best_regime["oos"]["mdd_pct"],
        },
        "summary": args.summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
