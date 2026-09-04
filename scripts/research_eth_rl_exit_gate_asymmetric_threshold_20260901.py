#!/usr/bin/env python3
"""Third follow-up to research_eth_rl_exit_gate_oracle_smoketest_20260901.py.

research_eth_rl_exit_gate_unrealized_gate_followup_20260901.py tested a hard post-hoc gate
("only allow the v0 trigger when pos_unrealized<=0") to fix the TP give-back problem (tp-terminal:
+740bp no-exit -> +49bp under the ungated v0 policy). Result: NET NEGATIVE in aggregate (ALL-pool
avg_net_bp 32.64 -> 26.75bp, median 10.04 -> -27.00bp) -- the hard gate also blocked genuinely
useful EARLY (still-positive-unrealized) SL-avoidance triggers, hurting the sl-terminal bucket far
more (avg -22.38 -> -68.63bp, win_rate 0.302 -> 0.000) than the tp-bucket gain justified. A "closer
to SL than TP" structural gate was tested too and was also net negative, just less so.

This script tries a softer, principled alternative instead of a hard AND-gate: an ASYMMETRIC
PROBABILITY THRESHOLD -- one threshold for checkpoints where pos_unrealized<=0 (require LESS
conviction to cut a currently-losing/flat position -- this is where the valuable SL-avoidance
signal lives), a separate, INDEPENDENT threshold for pos_unrealized>0 (require MORE conviction to
cut a currently-winning position -- this is where TP give-back happens). Unlike the hard gate, this
does not categorically block early positive-unrealized SL warnings; it just raises the bar for them.

Split discipline: both thresholds are swept and selected ENTIRELY on the TRAIN-internal embargoed
holdout (the same holdout the main smoke test already used for its single threshold calibration --
no new model, just a different post-hoc threshold-selection criterion: maximize the ALL-pool
sequential-first-trigger policy's aggregate net bp instead of matching the oracle label's base
rate). VALIDATION is scored exactly once with the resulting fixed (threshold_neg, threshold_pos)
pair, reusing the ALREADY-SAVED v0_prob/pos_unrealized from the main smoke test's single VALIDATION
pass -- this is not a new independent VALIDATION exposure of the underlying model, just a different
decision rule applied to probabilities already computed and already spent once.

Outputs: tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901/asymmetric_threshold_followup.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_entry_gate_oracle_smoketest_20260831 as entry_smoke  # noqa: E402
import research_eth_rl_exit_gate_oracle_smoketest_20260901 as exit_smoke  # noqa: E402

SMOKE_DIR = exit_smoke.OUT_DIR
ROUNDTRIP_COST = exit_smoke.ROUNDTRIP_COST
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 19), 4)  # 19 x 19 = 361 combos


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def policy_stats(df: pd.DataFrame, trigger_mask: np.ndarray) -> dict:
    d = df.assign(_trigger=trigger_mask).sort_values(["cand_timestamp", "checkpoint_t"])
    trig = (
        d[d["_trigger"]]
        .groupby("cand_timestamp", as_index=False)
        .first()[["cand_timestamp", "pos_unrealized"]]
        .rename(columns={"pos_unrealized": "early_move"})
    )
    cand = d.groupby("cand_timestamp", as_index=False).agg(price_move_terminal=("price_move_terminal", "first"))
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
    out: dict = {
        "script": "scripts/research_eth_rl_exit_gate_asymmetric_threshold_20260901.py",
        "parent_script": "scripts/research_eth_rl_exit_gate_oracle_smoketest_20260901.py",
        "prior_followup_hard_gate_result": "NET NEGATIVE (see research_eth_rl_exit_gate_"
                                            "unrealized_gate_followup_20260901.py output) -- "
                                            "motivates this softer asymmetric-threshold attempt.",
    }

    log("Step 1: rebuild TRAIN v0 checkpoint+context table (reusing exit_smoke functions)")
    train_cand = exit_smoke.load_saved_candidates("train")
    eth_klines = entry_smoke.load_klines(entry_smoke.ETH_KLINES_PATH)
    btc_klines = entry_smoke.load_klines(entry_smoke.BTC_KLINES_PATH)
    eth_klines["atr_pct_192"] = entry_smoke._atr_pct(eth_klines, window=entry_smoke.ATR_WINDOW)
    pos_lookup = exit_smoke.build_kline_lookup(eth_klines)
    train_ckpt, ckpt_report = exit_smoke.build_checkpoints(train_cand, eth_klines, pos_lookup)
    log(f"  TRAIN checkpoints: {ckpt_report}")

    training_features_full = entry_smoke.load_training_features_full()
    regime_df, _ = entry_smoke.score_regime_gbm3(training_features_full)
    market_context = entry_smoke.build_market_context(training_features_full)
    train_v0, join_report = exit_smoke.join_context(train_ckpt, market_context, regime_df, None)
    log(f"  TRAIN v0 modeling pool: {len(train_v0)} ({join_report})")

    log("Step 2: internal TRAIN-holdout split + fit model_a on fit_df (SAME split as main smoke test)")
    fit_df, holdout_df, split_info = entry_smoke.internal_train_holdout_split(
        train_v0, exit_smoke.INTERNAL_HOLDOUT_FRAC, exit_smoke.EMBARGO_BARS
    )
    log(f"  {split_info}")
    x_fit = entry_smoke.prep_x(fit_df, exit_smoke.V0_FEATURE_COLS)
    y_fit = fit_df["oracle_exit_label"].to_numpy()
    model_a = entry_smoke.fit_hgb(x_fit, y_fit)
    x_hold = entry_smoke.prep_x(holdout_df, exit_smoke.V0_FEATURE_COLS)
    holdout_df = holdout_df.assign(prob=model_a.predict_proba(x_hold)[:, 1])
    log(f"  holdout_df: {len(holdout_df)} rows, prob range [{holdout_df['prob'].min():.3f}, {holdout_df['prob'].max():.3f}]")

    log("Step 3: sweep (threshold_neg, threshold_pos) on TRAIN-holdout, maximize ALL-pool avg_net_bp")
    neg_mask = (holdout_df["pos_unrealized"] <= 0.0).to_numpy()
    pos_mask = ~neg_mask
    prob = holdout_df["prob"].to_numpy()

    best = None
    grid_results = []
    t0 = time.time()
    for t_neg in THRESHOLD_GRID:
        for t_pos in THRESHOLD_GRID:
            trigger = (neg_mask & (prob >= t_neg)) | (pos_mask & (prob >= t_pos))
            stats = policy_stats(holdout_df, trigger)
            grid_results.append({"t_neg": float(t_neg), "t_pos": float(t_pos), "avg_net_bp": stats["avg_net_bp"],
                                  "median_net_bp": stats["median_net_bp"], "win_rate": stats["win_rate"],
                                  "trigger_rate": stats["trigger_rate"]})
            if best is None or stats["avg_net_bp"] > best["avg_net_bp"]:
                best = {"t_neg": float(t_neg), "t_pos": float(t_pos), **stats}
    log(f"  swept {len(grid_results)} combos in {time.time() - t0:.1f}s")
    log(f"  BEST on TRAIN-holdout: t_neg={best['t_neg']:.3f} t_pos={best['t_pos']:.3f} -> {best}")

    # Also report the single-threshold baseline (t_neg==t_pos, i.e. what the main smoke test did,
    # modulo its base-rate-matching vs this script's bp-maximizing criterion) for direct comparison.
    symmetric = [g for g in grid_results if abs(g["t_neg"] - g["t_pos"]) < 1e-9]
    best_symmetric = max(symmetric, key=lambda g: g["avg_net_bp"])
    log(f"  best SYMMETRIC (t_neg==t_pos) on TRAIN-holdout for reference: {best_symmetric}")

    out["train_holdout_split_info"] = split_info
    out["train_holdout_grid_search"] = {
        "n_combos": len(grid_results),
        "best_asymmetric": best,
        "best_symmetric_reference": best_symmetric,
        "top_10_by_avg_net_bp": sorted(grid_results, key=lambda g: -g["avg_net_bp"])[:10],
    }

    log("Step 4: apply chosen (t_neg, t_pos) to the ALREADY-SAVED VALIDATION v0 probabilities (single application)")
    scored = pd.read_csv(SMOKE_DIR / "validation_v0_scored.csv")
    scored["cand_timestamp"] = pd.to_datetime(scored["cand_timestamp"])
    val_neg_mask = (scored["pos_unrealized"] <= 0.0).to_numpy()
    val_pos_mask = ~val_neg_mask
    val_prob = scored["v0_prob"].to_numpy()

    variants = {
        "original_v0_trigger_(base_rate_matched_symmetric)": scored["v0_trigger"].to_numpy(),
        "bp_optimal_symmetric_(train_holdout_selected)": val_prob >= best_symmetric["t_neg"],
        "bp_optimal_asymmetric_(train_holdout_selected)": (
            (val_neg_mask & (val_prob >= best["t_neg"])) | (val_pos_mask & (val_prob >= best["t_pos"]))
        ),
    }
    out["validation_comparison"] = {}
    for name, trig in variants.items():
        by_reason = {r: policy_stats(scored[scored["reason"] == r], trig[(scored["reason"] == r).to_numpy()])
                     for r in ["sl", "tp", "timeout"]}
        by_reason["ALL"] = policy_stats(scored, trig)
        out["validation_comparison"][name] = by_reason
        log(f"=== VALIDATION: {name} ===")
        for r in ["sl", "tp", "timeout", "ALL"]:
            s = by_reason[r]
            log(f"  {r:8s} n={s['n_candidates']:6d} trigger_rate={s['trigger_rate']:.3f} "
                f"win={s['win_rate']:.3f} avg_bp={s['avg_net_bp']:8.2f} median_bp={s['median_net_bp']:8.2f}")

    (SMOKE_DIR / "asymmetric_threshold_followup.json").write_text(json.dumps(out, indent=2))
    log(f"Done. Written to {SMOKE_DIR / 'asymmetric_threshold_followup.json'}")


if __name__ == "__main__":
    main()
