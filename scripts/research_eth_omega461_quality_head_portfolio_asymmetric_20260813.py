#!/usr/bin/env python3
"""RESEARCH ONLY -- portfolio-level VAL check of the h48qual quality_head live-ATR relabel
(scripts/research_eth_omega461_quality_head_liveatr_relabel_20260813.py), gated on that component's
own VAL result already looking like a genuine, non-trivial improvement (PnL +5.45%->+20.20%, WR
41.4%->48.0%, admission rate 1.79%->9.60% of all VAL bars, admission bias 87% short->68% long --
see tmp/causal_regen_20260516/eth_omega461_quality_head_liveatr_relabel_20260813/report.json and
this session's own CSV analysis). zig075 and regime3 routing are completely untouched.

Follows the exact same portfolio-validation pattern the exit-head-liveatr-relabel fix used tonight
(scripts/research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py, see
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md "후속 2"): the real live
adapter (trading_bot_modules/omega4_6_1_live.py, read for reference only, never imported/touched)
shares ONE account-level position slot with h48qual>zig075 priority -- a component looking better in
isolation does not automatically mean the combined single-account system improves. This script
reuses replay_omega4_6_1_greedy_router_20260706.greedy_replay unchanged (byte-identical
PRIORITY/SCALE_MAP/LEVERAGE_CAP/NOTIONAL_CAP to the live module) and reuses
research_eth_omega461_exit_head_portfolio_asymmetric_20260813's `_component_cfg`/
`_prepare_component_val`/`_ledger_metrics` unchanged (imported as a module, not copy-pasted) --
that script already solved "greedy.prepare_component hardcodes oof=False but VAL predictions are
oof=True" via a local `_prepare_component_val` copy, no need to re-solve it.

KEY DIFFERENCE from the exit-head predecessor's own portfolio script: that experiment only changed
exit_head weights, so both compared variants could share the SAME (static) h48qual/zig075
prediction CSVs -- entry-side (direction/quality-gated) decisions were identical between baseline
and candidate by construction. This experiment changes h48qual's QUALITY head, which changes
entry-side decisions directly. `greedy.prepare_component`/`_prepare_component_val` both read `dec`
(the side/TP/SL/quality-gated action) from a static prediction CSV file, not from a fresh forward
pass of `cfg["bundle"]` (`cfg["bundle"]`'s loaded model is only ever consulted for the EXIT head
during replay) -- the same trap already found and fixed for the component-level eval
(_evaluate_val_quality in the quality-head-relabel script). So this script uses TWO DIFFERENT
h48qual prediction sources depending on variant: the long-established static CSV
(sweep.EXT_PRED_DIR/h48qual/validation_predictions_q050.csv) for `baseline_both_original`, and this
session's own fresh-recomputed new-quality-head CSV
(tmp/.../eth_omega461_quality_head_liveatr_relabel_20260813/h48qual/validation_predictions_new_q050.csv,
already self-verified: the SAME pipeline applied to the unchanged original bundle reproduced the
static CSV's PnL/MDD/trades/wr to 0.0 absolute difference) for `asymmetric_h48qual_quality_liveatr_zig075_original`.
zig075 uses the SAME static CSV in both variants -- completely unchanged, per the coordinator's
explicit instruction.

SCALE_MAP note: greedy_replay (reused unchanged) DOES apply the live SCALE_MAP multiplier
(h48qual_L=0.38, h48qual_S=2.499, zig075_L=2.446, zig075_S=2.478) when sizing a position -- unlike
the component-level replay_exit_variant/prep_component harness used for the earlier per-component
VAL check, which omits it. So THIS script's absolute PnL/MDD figures are at true live scale; the
component-level numbers reported earlier this session are not directly comparable to these on an
absolute basis (relative baseline-vs-candidate comparisons were valid at both levels regardless).

VAL only: 2025-10-01..2025-12-31 (research_eth_omega461_exit_sweep_20260721.VAL_START/VAL_END).
OOS (2026-01-01..2026-03-31) is never loaded or scored here -- this script only runs if the
coordinator's decision gate says so, and even then a single OOS confirmation is a SEPARATE
deliberate step, not automatic.

fresh_forward_bar_by_bar=true (greedy_replay is a single causal forward pass). trade_ledgers_used_as_input=false
(ledgers are output only). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
direction_head/exit_head unchanged for h48qual (only quality_head weights differ from baseline);
zig075 and regime3 routing completely unchanged. No duration-gate post-filter (matches current live
running with the gate off, and matches the exit-head predecessor's own portfolio script -- isolates
the one variable in question).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or any other experiment's output directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as exit_portfolio  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_quality_head_portfolio_asymmetric_20260813"
NEW_H48QUAL_BUNDLE = ROOT / "tmp/causal_regen_20260516/eth_omega461_quality_head_liveatr_relabel_20260813/h48qual/true_3head_tabm_bundle.pt"
H48QUAL_NEW_PRED_CSV = ROOT / "tmp/causal_regen_20260516/eth_omega461_quality_head_liveatr_relabel_20260813/h48qual/validation_predictions_new_q050.csv"
DEVICE = torch.device("cpu")

# Reused unchanged from the exit-head predecessor's portfolio script -- these functions are generic
# over which bundle/pred_csv is passed in, not specific to the exit-head experiment.
_component_cfg = exit_portfolio._component_cfg
_prepare_component_val = exit_portfolio._prepare_component_val
_ledger_metrics = exit_portfolio._ledger_metrics


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _align_frame_and_predictions(val_frame: pd.DataFrame, pred_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Generalized version of exit_portfolio._align_frame_and_predictions: takes explicit
    {label: pred_csv_path} instead of {component: q_tag} (that predecessor assumed a single static
    CSV per component works for every variant; here h48qual's path differs by variant, so the
    caller passes the exact per-variant path set). Intersects all pred timestamp sets with the
    frame's, writes aligned copies under THIS script's own OUT_DIR (never touches the exit-head
    predecessor's output directory or any other experiment's files)."""
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(val_frame["timestamp"])
    for label, path in pred_paths.items():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[label] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = val_frame[val_frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    for label, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{label}: alignment failed after timestamp intersection")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = OUT_DIR / f"_aligned_{label}_validation_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        aligned_paths[label] = out_path
    return aligned_frame, aligned_paths


def run_variant(name: str, comp_cfgs: dict[str, dict[str, Any]], val_frame: pd.DataFrame,
                 pred_paths: dict[str, Path], *, fee: float, slip: float) -> dict[str, Any]:
    components = {}
    for cname, cfg in comp_cfgs.items():
        components[cname] = _prepare_component_val(val_frame, pred_paths[cname], cfg, DEVICE)
        print(f"  {cname}: bundle={Path(cfg['bundle']).parent.name} nonzero_side={(components[cname]['dec']['side'] != 0).mean():.3f}", flush=True)
    _diag, ledger = greedy.greedy_replay(val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT_DIR / f"portfolio_ledger_{name}.csv", index=False)
    metrics = _ledger_metrics(ledger)
    print(f"  {name}: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("stage=load_val_frame", flush=True)
    val_frame_raw = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"  VAL frame rows={len(val_frame_raw)} range=[{val_frame_raw['timestamp'].min()}, {val_frame_raw['timestamp'].max()}]", flush=True)
    fee, slip = omega._load_fee_slip()

    h48qual_baseline_static = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{sweep.COMPONENTS['h48qual']['q_tag']}.csv"
    zig075_static = sweep.EXT_PRED_DIR / "zig075" / f"validation_predictions_{sweep.COMPONENTS['zig075']['q_tag']}.csv"

    print("stage=align_frame_and_predictions", flush=True)
    val_frame, aligned = _align_frame_and_predictions(val_frame_raw, {
        "h48qual_baseline": h48qual_baseline_static,
        "h48qual_new": H48QUAL_NEW_PRED_CSV,
        "zig075": zig075_static,
    })
    print(f"  aligned rows={len(val_frame)} (from raw {len(val_frame_raw)})", flush=True)

    variants = {
        "baseline_both_original": {
            "comp_cfgs": {"h48qual": _component_cfg("h48qual"), "zig075": _component_cfg("zig075")},
            "pred_paths": {"h48qual": aligned["h48qual_baseline"], "zig075": aligned["zig075"]},
        },
        "asymmetric_h48qual_quality_liveatr_zig075_original": {
            "comp_cfgs": {"h48qual": _component_cfg("h48qual", bundle_override=NEW_H48QUAL_BUNDLE), "zig075": _component_cfg("zig075")},
            "pred_paths": {"h48qual": aligned["h48qual_new"], "zig075": aligned["zig075"]},
        },
    }

    results: dict[str, Any] = {}
    for name, v in variants.items():
        print(f"stage=run_variant name={name}", flush=True)
        results[name] = run_variant(name, v["comp_cfgs"], val_frame, v["pred_paths"], fee=fee, slip=slip)

    report = {
        "design": (
            "Portfolio-level (single shared position slot, h48qual>zig075 priority) VAL replay via "
            "the existing replay_omega4_6_1_greedy_router_20260706.greedy_replay, reused unchanged "
            "(includes live SCALE_MAP). baseline_both_original = both components on original frozen "
            "bundles + original static prediction CSVs. asymmetric_... = h48qual on the new "
            "quality-head-liveatr-relabeled bundle + this session's fresh-recomputed prediction CSV "
            "(self-verified against the static baseline CSV, 0.0 pnl diff on the unchanged bundle); "
            "zig075 fully unchanged (same bundle, same static CSV, in both variants)."
        ),
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "duration_gate_applied": False,
        "scale_map_applied": True,
        "new_h48qual_bundle": str(NEW_H48QUAL_BUNDLE),
        "h48qual_new_pred_csv": str(H48QUAL_NEW_PRED_CSV),
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
