#!/usr/bin/env python3
"""RESEARCH ONLY -- fresh-forward VAL/OOS evaluation of a RETRAINED exit-head bundle,
holding entry (direction/quality/side/TP/SL) decisions, risk sidecar, and margin/leverage
sizing fixed to the FROZEN live-baseline artifacts. Only the exit-head weights used inside
the causal replay loop change (loaded from a new --bundle path). This isolates the effect of
the exit-head retrain variant (e.g. a different exit_edge_min label threshold) from any other
change, since direction/quality/side/TP/SL come from the frozen prediction CSVs used to
originally certify Omega4.6.1 (same CSVs research_eth_omega461_exit_sweep_20260721.py uses),
not from the new bundle.

fresh_forward_bar_by_bar=true (single forward causal pass per replay_exit_variant call).
trade_ledgers_used_as_input=false (ledgers are only ever written as OUTPUT here).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint. Research artifact only -- no promotion-gate claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402

def _bundle(suffix: str) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{suffix}/true_3head_tabm_bundle.pt"


# --- Variants under test: (label, component -> new bundle path) -----------------------------
# giveback_min sweep (label rule actually in effect for the live checkpoints is
# entry_label_terminal_giveback, which is governed by giveback_min/adverse_unreal/
# min_mfe_for_giveback/terminal_window -- NOT exit_edge_min, which that mode ignores).
VARIANTS: dict[str, dict[str, Path]] = {
    "h48qual_gb065_control": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_control_edge0020")},
    "h48qual_gb045": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_gb045")},
    "h48qual_gb055": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_gb055")},
    "h48qual_gb075": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_gb075")},
    "h48qual_gb085": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_gb085")},
    "zig075_gb065_control": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_gb065_control")},
    "zig075_gb045": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_gb045")},
    "zig075_gb055": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_gb055")},
    "zig075_gb075": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_gb075")},
    "zig075_gb085": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_gb085")},
    "h48qual_tw08": {"h48qual": _bundle("research_exit_head_retrain_20260721_h48qual_tw08")},
    "zig075_tw08": {"zig075": _bundle("research_exit_head_retrain_20260721_zig075_tw08")},
}


def evaluate_variant(name: str, bundle_overrides: dict[str, Path]) -> pd.DataFrame:
    components = {}
    for cname, cfg in base.COMPONENTS.items():
        cfg2 = dict(cfg)
        if cname in bundle_overrides:
            cfg2["bundle"] = bundle_overrides[cname]
        components[cname] = cfg2

    val_frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    oos_frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)

    rows = []
    for cname, cfg in components.items():
        if cname not in bundle_overrides:
            continue  # only score components we actually retrained for this variant
        val_pred = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        oos_pred = base.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"

        val_prepped = base.prep_component(cname, cfg, val_frame, val_pred, oof=True)
        m_val, ledger_val = base.replay_exit_variant(
            val_prepped["frame"], val_prepped["x"], val_prepped["dec"], val_prepped["loaded"],
            risk_margin_fraction=val_prepped["margin"], risk_leverage=val_prepped["leverage"],
            exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=val_prepped["fee"], slip=val_prepped["slip"],
            cost_mult=base.COST_MULT, notional_scaled_sltp=val_prepped["notional_scaled_sltp"], device=base.DEVICE,
        )
        rows.append({"variant": name, "component": cname, "split": "VAL", **m_val, "exit_reasons": json.dumps(m_val["exit_reasons"])})

        oos_prepped = base.prep_component(cname, cfg, oos_frame, oos_pred, oof=False)
        m_oos, ledger_oos = base.replay_exit_variant(
            oos_prepped["frame"], oos_prepped["x"], oos_prepped["dec"], oos_prepped["loaded"],
            risk_margin_fraction=oos_prepped["margin"], risk_leverage=oos_prepped["leverage"],
            exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=oos_prepped["fee"], slip=oos_prepped["slip"],
            cost_mult=base.COST_MULT, notional_scaled_sltp=oos_prepped["notional_scaled_sltp"], device=base.DEVICE,
        )
        rows.append({"variant": name, "component": cname, "split": "OOS", **m_oos, "exit_reasons": json.dumps(m_oos["exit_reasons"])})
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = ROOT / "tmp/research_20260721_exit_head_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, overrides in VARIANTS.items():
        print(f"stage=evaluate variant={name}", flush=True)
        df = evaluate_variant(name, overrides)
        print(df[["variant", "component", "split", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)
        all_rows.append(df)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_dir / "exit_head_retrain_variants.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
