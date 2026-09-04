#!/usr/bin/env python3
"""Second follow-up to research_eth_rl_exit_gate_oracle_smoketest_20260901.py: targets the TP
give-back problem found in research_eth_rl_exit_gate_reason_stratified_followup_20260901.py (tp
-terminal candidates: no-exit +740bp avg -> policy +49bp avg, i.e. the classifier's early-exit
trigger fires on 96% of eventually-TP-bound trades and gives back most of their eventual profit).

Hypothesis: the single oracle_exit_label pools SL-avoidance ("cut this before it hits its stop")
and TP-timing ("is riding further to the target still worth it") into one binary decision, and the
classifier ends up applying an SL-avoidance-flavored trigger broadly (94.4% VALIDATION trigger
rate) because that population's payoff (-410bp -> -22bp) dominates fitting. A currently-profitable
position being cut short is exactly the TP-giveback failure mode; a currently-underwater-or-flat
position being cut short is exactly the SL-avoidance success mode. Gating the trigger on the
position's OWN current state (no new model, no refit -- pure post-hoc filter on the ALREADY-SAVED,
ALREADY-VALIDATION-scored v0_trigger/v0_prob from the main smoke test) should separate these:

  Gate A (directional): only allow a trigger when pos_unrealized <= 0 (at or below breakeven).
  Gate B (structural):  only allow a trigger when pos_dist_to_sl <= pos_dist_to_tp (structurally
                         closer to the stop than the target).

Both gates are pure AND-filters on the existing v0_trigger decision -- no re-fitting, no new
VALIDATION exposure beyond what the main smoke test already used (this reuses the SAME VALIDATION-
scored probabilities exactly once already spent; comparing gated variants of an already-frozen
decision is not a new independent VALIDATION touch of the underlying model).

Outputs: tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901/unrealized_gate_followup.json
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_exit_gate_oracle_smoketest_20260901 as exit_smoke  # noqa: E402

SMOKE_DIR = ROOT / "tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901"
ROUNDTRIP_COST = 0.0010


def policy_stats(df: pd.DataFrame, trigger_mask: pd.Series) -> dict:
    df = df.assign(_trigger=trigger_mask.to_numpy()).sort_values(["cand_timestamp", "checkpoint_t"])
    trig = (
        df[df["_trigger"]]
        .groupby("cand_timestamp", as_index=False)
        .first()[["cand_timestamp", "pos_unrealized"]]
        .rename(columns={"pos_unrealized": "early_move"})
    )
    cand = df.groupby("cand_timestamp", as_index=False).agg(price_move_terminal=("price_move_terminal", "first"))
    m = cand.merge(trig, on="cand_timestamp", how="left")
    realized = m["early_move"].where(m["early_move"].notna(), m["price_move_terminal"])
    net = realized.to_numpy() - ROUNDTRIP_COST
    return {
        "n_candidates": int(len(m)),
        "trigger_rate": float(m["early_move"].notna().mean()),
        "win_rate": float((net > 0).mean()),
        "avg_net_bp": float(net.mean() * 10000.0),
        "median_net_bp": float(np.median(net) * 10000.0),
    }


def main() -> None:
    scored = pd.read_csv(SMOKE_DIR / "validation_v0_scored.csv")
    scored["cand_timestamp"] = pd.to_datetime(scored["cand_timestamp"])

    # pos_dist_to_tp/pos_dist_to_sl weren't kept in the saved scored CSV (only pos_unrealized was) --
    # re-derive them from the original candidates' tp_move/sl_move (entry-time constants) + the
    # already-present pos_unrealized, exactly matching build_checkpoints' own formulas.
    val_cand = exit_smoke.load_saved_candidates("validation")[["timestamp", "tp_move", "sl_move"]]
    val_cand = val_cand.rename(columns={"timestamp": "cand_timestamp"})
    scored = scored.merge(val_cand, on="cand_timestamp", how="left")
    scored["pos_dist_to_tp"] = scored["tp_move"] - scored["pos_unrealized"]
    scored["pos_dist_to_sl"] = scored["pos_unrealized"] + scored["sl_move"].abs()

    base_trigger = scored["v0_trigger"].astype(bool)
    gate_a = base_trigger & (scored["pos_unrealized"] <= 0.0)
    gate_b = base_trigger & (scored["pos_dist_to_sl"] <= scored["pos_dist_to_tp"])
    gate_ab = base_trigger & (scored["pos_unrealized"] <= 0.0) & (scored["pos_dist_to_sl"] <= scored["pos_dist_to_tp"])

    variants = {
        "baseline_no_early_exit": scored["v0_trigger"] & False,  # never triggers -> pure ride-to-terminal
        "original_v0_trigger": base_trigger,
        "gate_a_unrealized_le_0": gate_a,
        "gate_b_closer_to_sl_than_tp": gate_b,
        "gate_a_and_b": gate_ab,
    }

    out: dict = {
        "script": "scripts/research_eth_rl_exit_gate_unrealized_gate_followup_20260901.py",
        "parent_script": "scripts/research_eth_rl_exit_gate_oracle_smoketest_20260901.py",
        "purpose": "test whether gating the existing v0 trigger on the position's OWN current "
                   "state (underwater/breakeven, or structurally closer to SL than TP) removes "
                   "the TP give-back (tp-terminal: +740bp no-exit -> +49bp under the ungated "
                   "policy) while preserving the SL-avoidance benefit (sl-terminal: -410bp -> -22bp).",
    }
    for name, trig in variants.items():
        by_reason = {}
        for reason in ["sl", "tp", "timeout"]:
            mask = scored["reason"] == reason
            by_reason[reason] = policy_stats(scored[mask], trig[mask])
        by_reason["ALL"] = policy_stats(scored, trig)
        out[name] = by_reason
        print(f"=== {name} ===")
        for reason in ["sl", "tp", "timeout", "ALL"]:
            s = by_reason[reason]
            print(f"  {reason:8s} n={s['n_candidates']:6d} trigger_rate={s['trigger_rate']:.3f} "
                  f"win={s['win_rate']:.3f} avg_bp={s['avg_net_bp']:8.2f} median_bp={s['median_net_bp']:8.2f}")

    (SMOKE_DIR / "unrealized_gate_followup.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {SMOKE_DIR / 'unrealized_gate_followup.json'}")


if __name__ == "__main__":
    main()
