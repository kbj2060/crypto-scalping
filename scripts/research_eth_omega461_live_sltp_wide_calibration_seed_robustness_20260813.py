#!/usr/bin/env python3
"""RESEARCH ONLY -- seed-robustness check for a side-finding shared by two prior experiments:

  - docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md: symmetric MFE-width mechanism,
    base102 feature panel, tp_scale=9.0 (sl_ratio = cfg sl_mult/tp_mult = 0.5, same formula as the
    rest of that grid) -- portfolio (h48qual>zig075 priority) VAL no_gate +130.25%/-18.45% MDD,
    with_gate +143.53%/-15.47% MDD, vs baseline +36.82%/-24.34% (no_gate) / +54.88%/-31.11%
    (with_gate).
  - docs/experiments/eth_omega461_live_sltp_asymmetric_tpsl_20260813.md: asymmetric mechanism (TP
    from MFE prediction, SL decoupled and scaled off the ORIGINAL live ATR-floor SL), tp_scale=9.0,
    sl_scale=1.5 -- portfolio VAL no_gate +123.68%/-15.51% MDD, with_gate +123.53%/-22.22% MDD.

Both are the only two cells (out of the full symmetric scale grid x 2 feature sets, and the full
21-cell asymmetric tp_scale x sl_scale grid) where PnL AND MDD both improve over baseline
SIMULTANEOUSLY on BOTH no_gate and with_gate -- independently re-verified against report.json for
both experiments before writing this script (see the check embedded in the orchestrator exchange
this script's docstring accompanies).

IMPORTANT SCOPE NOTE: the underlying problem this was investigating (SLTP width -> trade count) is
CLOSED per orchestrator instruction (2026-08-13) -- both tested mechanisms (constant floor retuning,
symmetric/asymmetric MFE-width learning) failed the (a)+(b) success bar and that investigation does
NOT reopen here. This script does ONLY ONE thing: check whether the wide-side "beats baseline on
both PnL and MDD" result is a genuine, seed-robust finding or a single-seed noise artifact -- exactly
the discipline this session has repeatedly needed (memory: tabm_hp_low_signal_pattern -- single-seed
HP "winners" are often just noise; also the final-boss v2/v3 tracks' VAL-improves/OOS-flips pattern).
Per CLAUDE.md's Seed-Diversity Ensemble Promotion Gate spirit (N>=5 TRUE random seeds, not a fixed
arithmetic increment), the MFE/MAE regressor's random_state is the only thing varied across trials;
everything else (TP_SCALE=9.0, SL_SCALE=1.5 for the asymmetric config, base102 feature panel, TRAIN
window, live bundles/sidecars, VAL window) is held fixed.

Reuses, does NOT reimplement:
  - research_eth_omega461_live_sltp_mfe_width_20260813.py (mfe_width): base_sweep, base102_panel,
    train_mfe_models, predicted_width, apply_mfe_width_sltp, _ledger_stats, _duration_gated,
    _as_router_component, _load_tb_labels.
  - research_eth_omega461_live_sltp_asymmetric_tpsl_20260813.py (asym): apply_asymmetric_tpsl.
  - replay_omega4_6_1_greedy_router_20260706.py (router): greedy_replay (h48qual>zig075 single
    shared position slot, the live PRIORITY mechanism).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
VAL window = 2025-10-01..2025-12-31 (identical to both prior experiments). OOS NOT run -- promotion
decision is the orchestrator's, not this script's, even if the result reproduces.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import research_eth_omega461_live_sltp_asymmetric_tpsl_20260813 as asym  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

base_sweep = mfe_width.base_sweep

OUT_DIR = ROOT / "tmp/research_20260813/omega461_live_sltp_wide_calibration_seed_robustness"

# Genuinely random (os-entropy-seeded random.sample, NOT a fixed arithmetic increment -- verified
# non-arithmetic: diffs are -331874253/+379648622/-362189691/-125768392) -- generated once and
# hardcoded here so the run is reproducible/auditable after the fact.
SEEDS = [453827194, 121952941, 501601563, 139411872, 13643480]

CONFIG_SYMMETRIC = {"tp_scale": 9.0}          # sl_ratio taken from each component's own cfg (0.5)
CONFIG_ASYMMETRIC = {"tp_scale": 9.0, "sl_scale": 1.5}


def log(msg: str) -> None:
    print(msg, flush=True)


def run_one_seed(seed: int, base_cols: list[str], panel_train: pd.DataFrame, panel_val: pd.DataFrame,
                  feat_cols: list[str], train_labels: pd.DataFrame, val_labels: pd.DataFrame,
                  prepped: dict[str, dict[str, Any]], router_base: dict[str, dict[str, Any]],
                  val_frame: pd.DataFrame, fee0: float, slip0: float) -> dict[str, Any]:
    models, train_diag = mfe_width.train_mfe_models(panel_train, feat_cols, train_labels, seed=seed)
    val_diag = mfe_width.val_sanity_gate(models, panel_val, feat_cols, val_labels)
    x_val_scoring = panel_val[feat_cols]

    widths = {}
    for name, p in prepped.items():
        side = pd.to_numeric(p["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
        widths[name] = mfe_width.predicted_width(models, x_val_scoring, side)

    results: dict[str, Any] = {}

    # --- symmetric scale=9.0 ---
    comps_sym = {}
    for name, p in prepped.items():
        cfg = base_sweep.COMPONENTS[name]
        sl_ratio = float(cfg["sl_mult"]) / float(cfg["tp_mult"])
        dec_sym, _ = mfe_width.apply_mfe_width_sltp(p["dec"], widths[name], tp_scale=CONFIG_SYMMETRIC["tp_scale"], sl_ratio=sl_ratio,
                                                      min_tp=mfe_width.FLOOR_TP, min_sl=mfe_width.FLOOR_SL, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
        comps_sym[name] = {**router_base[name], "dec": dec_sym}
    _, ledger_sym = router.greedy_replay(val_frame, comps_sym, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    results["symmetric_scale9"] = {"no_gate": mfe_width._ledger_stats(ledger_sym, val_frame),
                                    "with_gate": mfe_width._duration_gated(ledger_sym, val_frame, router.DURATION_THRESHOLD)}

    # --- asymmetric tp_scale=9.0, sl_scale=1.5 ---
    comps_asym = {}
    for name, p in prepped.items():
        cfg = base_sweep.COMPONENTS[name]
        dec_asym, _ = asym.apply_asymmetric_tpsl(p["dec"], widths[name], tp_scale=CONFIG_ASYMMETRIC["tp_scale"], sl_scale=CONFIG_ASYMMETRIC["sl_scale"],
                                                   min_tp=mfe_width.FLOOR_TP, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
        comps_asym[name] = {**router_base[name], "dec": dec_asym}
    _, ledger_asym = router.greedy_replay(val_frame, comps_asym, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    results["asymmetric_tp9_sl1.5"] = {"no_gate": mfe_width._ledger_stats(ledger_asym, val_frame),
                                        "with_gate": mfe_width._duration_gated(ledger_asym, val_frame, router.DURATION_THRESHOLD)}

    return {"seed": seed, "train_diag": train_diag, "val_sanity_gate": val_diag, "results": results,
            "ledgers": {"symmetric_scale9": ledger_sym, "asymmetric_tp9_sl1.5": ledger_asym}}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"seeds (N=5, true random, not fixed-increment): {SEEDS}")

    log("stage=load_frames")
    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    train_frame = base_sweep.load_frame(mfe_width.TRAIN_START, mfe_width.TRAIN_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)

    bundle_h48 = torch.load(base_sweep.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle_h48["base_cols"])
    train_labels = mfe_width._load_tb_labels("train")
    val_labels = mfe_width._load_tb_labels("validation")

    log("stage=prep_components (baseline ATR-floor dec/margin/leverage, computed ONCE, seed-independent)")
    prepped: dict[str, dict[str, Any]] = {}
    baseline_component_rows = []
    for name, cfg in base_sweep.COMPONENTS.items():
        pred_csv = base_sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        p = base_sweep.prep_component(name, cfg, val_frame, pred_csv, oof=True)
        prepped[name] = p
        m_base, _ = base_sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
        )
        baseline_component_rows.append({"component": name, **{k: v for k, v in m_base.items() if k != "exit_reasons"}})

    router_base = {name: mfe_width._as_router_component(p, exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD) for name, p in prepped.items()}
    fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
    _, ledger_base_combined = router.greedy_replay(val_frame, router_base, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    baseline_combined = {"no_gate": mfe_width._ledger_stats(ledger_base_combined, val_frame),
                          "with_gate": mfe_width._duration_gated(ledger_base_combined, val_frame, router.DURATION_THRESHOLD)}
    log(f"priority_combined baseline: {json.dumps(baseline_combined)}")

    log("stage=build_panel_base102 (feature panel only, shared across all 5 seeds -- only the MFE regressor's random_state varies)")
    panel_train, feat_cols = mfe_width.base102_panel(base_cols, train_frame)
    panel_val, _ = mfe_width.base102_panel(base_cols, val_frame)

    log(f"stage=seed_sweep n_seeds={len(SEEDS)}")
    seed_results = []
    for seed in SEEDS:
        log(f"  seed={seed}")
        r = run_one_seed(seed, base_cols, panel_train, panel_val, feat_cols, train_labels, val_labels, prepped, router_base, val_frame, fee0, slip0)
        seed_results.append(r)
        for cfg_name, res in r["results"].items():
            ng, wg = res["no_gate"], res["with_gate"]
            log(f"    {cfg_name}: no_gate pnl={ng['pnl']:.2f} mdd={ng['mdd']:.2f} trades={ng['trades']} | with_gate pnl={wg['pnl']:.2f} mdd={wg['mdd']:.2f}")
        r["ledgers"]["symmetric_scale9"].to_csv(OUT_DIR / f"ledger_symmetric_scale9_seed{seed}_VAL.csv", index=False)
        r["ledgers"]["asymmetric_tp9_sl1.5"].to_csv(OUT_DIR / f"ledger_asymmetric_tp9_sl1.5_seed{seed}_VAL.csv", index=False)

    log("stage=aggregate")
    summary: dict[str, Any] = {"baseline": baseline_combined, "seeds": SEEDS, "per_config": {}}
    rows = []
    for cfg_name in ["symmetric_scale9", "asymmetric_tp9_sl1.5"]:
        ng_pnls = np.array([r["results"][cfg_name]["no_gate"]["pnl"] for r in seed_results])
        ng_mdds = np.array([r["results"][cfg_name]["no_gate"]["mdd"] for r in seed_results])
        wg_pnls = np.array([r["results"][cfg_name]["with_gate"]["pnl"] for r in seed_results])
        wg_mdds = np.array([r["results"][cfg_name]["with_gate"]["mdd"] for r in seed_results])
        ng_pnl_wins = int((ng_pnls > baseline_combined["no_gate"]["pnl"]).sum())
        ng_mdd_wins = int((ng_mdds > baseline_combined["no_gate"]["mdd"]).sum())
        wg_pnl_wins = int((wg_pnls > baseline_combined["with_gate"]["pnl"]).sum())
        wg_mdd_wins = int((wg_mdds > baseline_combined["with_gate"]["mdd"]).sum())
        reproduced = bool(ng_pnl_wins >= 4 and ng_mdd_wins >= 4 and wg_pnl_wins >= 4 and wg_mdd_wins >= 4)
        cfg_summary = {
            "no_gate_pnl_mean": float(ng_pnls.mean()), "no_gate_pnl_std": float(ng_pnls.std()),
            "no_gate_mdd_mean": float(ng_mdds.mean()), "no_gate_mdd_std": float(ng_mdds.std()),
            "with_gate_pnl_mean": float(wg_pnls.mean()), "with_gate_pnl_std": float(wg_pnls.std()),
            "with_gate_mdd_mean": float(wg_mdds.mean()), "with_gate_mdd_std": float(wg_mdds.std()),
            "no_gate_pnl_wins_vs_baseline": f"{ng_pnl_wins}/5", "no_gate_mdd_wins_vs_baseline": f"{ng_mdd_wins}/5",
            "with_gate_pnl_wins_vs_baseline": f"{wg_pnl_wins}/5", "with_gate_mdd_wins_vs_baseline": f"{wg_mdd_wins}/5",
            "reproduced": reproduced,
            "per_seed_no_gate_pnl": ng_pnls.tolist(), "per_seed_no_gate_mdd": ng_mdds.tolist(),
            "per_seed_with_gate_pnl": wg_pnls.tolist(), "per_seed_with_gate_mdd": wg_mdds.tolist(),
        }
        summary["per_config"][cfg_name] = cfg_summary
        rows.append({"config": cfg_name, **cfg_summary})
        log(f"  {cfg_name}: no_gate pnl {ng_pnl_wins}/5 wins (mean {ng_pnls.mean():.2f}+-{ng_pnls.std():.2f}), "
            f"mdd {ng_mdd_wins}/5 wins (mean {ng_mdds.mean():.2f}+-{ng_mdds.std():.2f}); "
            f"with_gate pnl {wg_pnl_wins}/5, mdd {wg_mdd_wins}/5 -- REPRODUCED={reproduced}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "seed_robustness_summary_VAL.csv", index=False)
    report = {
        "model_id": "omega461_live_sltp_wide_calibration_seed_robustness_20260813",
        "parent_experiments": [
            "docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md",
            "docs/experiments/eth_omega461_live_sltp_asymmetric_tpsl_20260813.md",
        ],
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "val_window": [base_sweep.VAL_START, base_sweep.VAL_END], "oos_run": False,
        "seeds": SEEDS, "seed_generation": "random.sample over os-entropy-seeded random module, not fixed increment",
        "configs": {"symmetric_scale9": CONFIG_SYMMETRIC, "asymmetric_tp9_sl1.5": CONFIG_ASYMMETRIC},
        "baseline_component": baseline_component_rows, "baseline_combined": baseline_combined,
        "per_seed_train_val_diag": [{"seed": r["seed"], "train_diag": r["train_diag"], "val_sanity_gate": r["val_sanity_gate"]} for r in seed_results],
        "per_seed_results": [{"seed": r["seed"], "results": r["results"]} for r in seed_results],
        "summary": summary,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=base_sweep.omega._json_default), encoding="utf-8"
    )
    log(f"stage=done report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
