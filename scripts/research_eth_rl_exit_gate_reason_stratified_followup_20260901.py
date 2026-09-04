#!/usr/bin/env python3
"""Follow-up to research_eth_rl_exit_gate_oracle_smoketest_20260901.py: breaks the v0/v1 policy
result down by the candidate's own terminal reason (sl/tp/timeout). No new model fitting -- reuses
the already-saved validation_v0_scored.csv / validation_v1_scored.csv verbatim.

Why: the smoke test's aggregate full-pool VALIDATION policy result (avg net +32.64bp vs a -12.52bp
no-early-exit baseline) turned out to be a bit-for-bit identical outcome between v0 and v1 (evidence
-signal columns never selected by the HGB splitter -- verified this is a real zero-importance
finding, not a data-join bug: the sig_* columns have genuine non-degenerate variance in the joined
VALIDATION data). Separately, since oracle_exit_label is TAUTOLOGICAL for sl/tp-terminal candidates
by construction (every checkpoint of an sl-terminal candidate has label=1, every checkpoint of a
tp-terminal candidate has label=0 -- see the main script's docstring), the aggregate number conflates
"the model correctly reconstructs known barrier-distance geometry" (sl/tp subpopulations, ~27.5% of
the pool) with actual exit-timing skill on the genuinely uncertain population (timeout-terminal,
~72% of the pool, checkpoint-level oracle_exit_label base rate ~46-48%, far from tautological). This
script isolates the three subpopulations to see how much of the aggregate result survives on the
hard (timeout-only) slice.

Outputs: tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901/reason_stratified_followup.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901"
ROUNDTRIP_COST = 0.0010


def checkpoint_auc(df: pd.DataFrame, prob_col: str) -> float | None:
    if df["oracle_exit_label"].nunique() < 2:
        return None  # tautological subpopulation (sl: all label=1, tp: all label=0) -- AUC undefined
    return float(roc_auc_score(df["oracle_exit_label"], df[prob_col]))


def policy_stats(df: pd.DataFrame, trigger_col: str) -> dict:
    """Candidate-level, sequential first-trigger policy vs. no-early-exit, restricted to whatever
    subset of `df` (a validation_v{0,1}_scored.csv slice) is passed in."""
    df = df.sort_values(["cand_timestamp", "checkpoint_t"])
    trig = (
        df[df[trigger_col]]
        .groupby("cand_timestamp", as_index=False)
        .first()[["cand_timestamp", "pos_unrealized"]]
        .rename(columns={"pos_unrealized": "early_move"})
    )
    cand = df.groupby("cand_timestamp", as_index=False).agg(price_move_terminal=("price_move_terminal", "first"))
    m = cand.merge(trig, on="cand_timestamp", how="left")
    realized = m["early_move"].where(m["early_move"].notna(), m["price_move_terminal"])
    net = realized.to_numpy() - ROUNDTRIP_COST
    no_exit_net = m["price_move_terminal"].to_numpy() - ROUNDTRIP_COST

    def _summ(x: np.ndarray) -> dict:
        return {
            "win_rate": float((x > 0).mean()),
            "avg_net_bp": float(x.mean() * 10000.0),
            "median_net_bp": float(np.median(x) * 10000.0),
        }

    return {
        "n_candidates": int(len(m)),
        "trigger_rate": float(m["early_move"].notna().mean()),
        "no_early_exit_baseline": _summ(no_exit_net),
        "policy": _summ(net),
    }


def main() -> None:
    out: dict = {
        "script": "scripts/research_eth_rl_exit_gate_reason_stratified_followup_20260901.py",
        "parent_script": "scripts/research_eth_rl_exit_gate_oracle_smoketest_20260901.py",
        "purpose": "isolate how much of the aggregate VALIDATION policy result is tautological "
                   "(sl/tp-terminal, oracle_exit_label constant by construction) vs. genuine "
                   "exit-timing signal on the non-tautological timeout-terminal subpopulation.",
    }
    for variant in ["v0", "v1"]:
        scored = pd.read_csv(SMOKE_DIR / f"validation_{variant}_scored.csv")
        prob_col, trig_col = f"{variant}_prob", f"{variant}_trigger"
        by_reason = {}
        for reason in ["sl", "tp", "timeout"]:
            sub = scored[scored["reason"] == reason]
            by_reason[reason] = {
                "n_checkpoints": int(len(sub)),
                "checkpoint_oracle_label_base_rate": float(sub["oracle_exit_label"].mean()),
                "checkpoint_auc": checkpoint_auc(sub, prob_col),
                **policy_stats(sub, trig_col),
            }
        out[variant] = by_reason

    (SMOKE_DIR / "reason_stratified_followup.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
