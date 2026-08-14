#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority #4, VAL-side evaluation. Compares the GBDT h48qual exit_head
trained by scripts/train_eval_omega461_gbdt_exit_head_liveatr_20260813.py against the CURRENT
Odyssey2 baseline -- h48qual's TabM live-ATR-relabel exit head
(tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/h48qual/
true_3head_tabm_bundle.pt, NOT the original live h48qual bundle) -- at both the component level
(h48qual standalone ledger, research_eth_omega461_exit_sweep_20260721.replay_exit_variant) and the
portfolio level (h48qual+zig075 single-account greedy router,
replay_omega4_6_1_greedy_router_20260706.greedy_replay via
research_eth_omega461_exit_head_portfolio_asymmetric_20260813's already-certified prep/align
helpers, imported and reused unchanged). zig075 is not touched.

=== G0 self-check (runs first, unconditionally) ===
Re-derives the two ALREADY-PUBLISHED reference numbers -- component-level via
h48cons._evaluate_val (100% pre-existing code, zero new logic in the call itself) and
portfolio-level via research_eth_omega461_exit_head_portfolio_asymmetric_20260813.run_variant
(also imported unchanged) -- through this exact script/import chain, and asserts they match
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md's published VAL numbers
within a small tolerance. Because G0 calls the SAME already-certified functions the GBDT variant
below also calls (no reimplementation of the replay loop anywhere in this file), a G0 pass mostly
validates "did the GBDT-runtime-injection wrapper get wired in without disturbing the untouched
TabM code paths" -- it is not a strong test of the GBDT numbers themselves, just of this harness's
plumbing (bundle paths, alignment, imports). Per the project's methodology discipline, if G0 fails
this script aborts BEFORE computing or trusting any GBDT number.

=== GBDT injection ===
The GBDT model is injected at the exact per-bar call site both harnesses share
(train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one, which does
`torch.softmax(model(x)["exit"], dim=-1).mean(dim=1)`) via a duck-typed wrapper
(GBDTExitHeadWrapper) whose __call__ returns log(predict_proba) reshaped to (batch, k=1, 2).
softmax(log(p)) == p exactly (p already sums to 1), so the TabM-shaped softmax/ensemble-pooling
machinery it was not designed for reproduces predict_proba unchanged. mean/std are identity zeros/
ones (GBDT needs no standardization; _predict_exit_prob_one computes (row-mean)/std before calling
the model, so identity scaling hands the wrapper the raw feature row it was trained on).
_predict_exit_prob_one / _prepare_exit_runtime / replay_exit_variant / greedy_replay /
prepare_component are all imported and called UNMODIFIED -- this file adds zero lines to any of
them.

=== Promotion gate ===
GBDT must be non-worse than the TabM live-ATR baseline on BOTH PnL and MDD at BOTH the component
and portfolio level on VAL before OOS is allowed to run (see
scripts/research_eth_omega461_gbdt_exit_head_oos_20260813.py, which reads this script's
report.json and refuses to proceed if gate_pass is False).

fresh_forward_bar_by_bar=true (replay_exit_variant and greedy_replay are both single causal forward
passes, i increasing, only bar i and already-closed history used at bar i).
trade_ledgers_used_as_input=false (ledgers are written-only outputs).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. direction_head/
quality_head/encoder are frozen and unchanged (bit-identical across the original/TabM-liveATR/GBDT
h48qual variants -- only exit_head differs). VAL window 2025-10-01..2025-12-31
(research_eth_omega461_exit_sweep_20260721.VAL_START/VAL_END). OOS is never loaded here.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT touch zig075.
"""
from __future__ import annotations

import json
import pickle
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega461_gbdt_exit_head_liveatr_20260813 as gbdt_train  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_val_20260813"
GBDT_BUNDLE = gbdt_train.OUT_DIR / "h48qual" / "gbdt_exit_bundle.pkl"

# Published in docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md ("VAL 평가
# 결과" and "후속 2" tables). Used only for the G0 self-check below, not for the GBDT comparison.
G0_REFERENCE = {
    "component_baseline_original": {"pnl": 5.45, "mdd": -11.62, "trades": 29},
    "component_tabm_liveatr": {"pnl": 9.23, "mdd": -7.59, "trades": 63},
    "portfolio_baseline_both_original": {"pnl": 36.82, "mdd": -24.34, "trades": 29},
    "portfolio_asymmetric_tabm_liveatr": {"pnl": 46.59, "mdd": -21.70, "trades": 35},
}
G0_TOLERANCE_PP = 0.05  # percentage points on pnl/mdd; trades must match exactly (deterministic replay)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _close_to_reference(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= G0_TOLERANCE_PP
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= G0_TOLERANCE_PP
        and int(actual["trades"]) == int(expected["trades"])
    )


class GBDTExitHeadWrapper:
    """Duck-types just enough of train_eval_omega1_2_tabm_3head_20260603.ThreeHeadTabM's __call__
    contract for train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one to use it as a
    drop-in replacement for `model` in `runtime[expert] = (model, mean, std)`. That function does:
        probs = torch.softmax(model(x)["exit"], dim=-1).mean(dim=1)[..., 1]
    dim=-1 is the 2-class softmax; dim=1 is the k-ensemble-member pooling dim. This wrapper fakes a
    k=1 ensemble by returning logits = log(predict_proba) reshaped to (batch, 1, 2); since
    softmax(log(p)) == p (p already sums to 1 across the 2 classes), the pooled probability the
    surrounding TabM-shaped machinery produces is exactly `predict_proba`, unchanged."""

    def __init__(self, model: Any, device: torch.device, columns: list[str]) -> None:
        classes = [int(c) for c in np.asarray(model.classes_)]
        if classes != [0, 1]:
            raise RuntimeError(f"GBDT exit model classes_ must be [0, 1], got {classes}")
        self.model = model
        self.device = device
        self.columns = list(columns)

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_np = x.detach().cpu().numpy().astype(np.float64)
        # Predict with a column-named DataFrame (not a raw ndarray) so it matches the DataFrame
        # LightGBM was fit on -- purely cosmetic (avoids an sklearn "X does not have valid feature
        # names" warning on every single-row call) since column ORDER, not names, is what the
        # underlying values depend on, but this keeps the fit/predict contract explicit.
        proba = self.model.predict_proba(pd.DataFrame(x_np, columns=self.columns))
        logits = np.log(np.clip(proba, 1.0e-12, 1.0)).astype(np.float32)
        return {"exit": torch.from_numpy(logits).to(self.device).unsqueeze(1)}  # (batch, k=1, 2)


def _load_gbdt_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _gbdt_loaded_models(base_cols: list[str], gbdt_models: dict[str, Any], device: torch.device) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Shape-compatible with parent._load_payloads' return value, for harnesses (replay_exit_variant)
    that accept `loaded_models` and build their own exit runtime internally via
    train_eval_omega4_2_risk_sidecar_20260622._prepare_exit_runtime."""
    cols = list(base_cols) + list(parent.POS_COLS)
    scaler = {"columns": cols, "mean": np.zeros(len(cols), dtype=np.float32), "std": np.ones(len(cols), dtype=np.float32)}
    return {expert: (GBDTExitHeadWrapper(gbdt_models[expert], device, cols), scaler) for expert in hard.EXPERT_NAMES}


def _inject_gbdt_exit_runtime(prepped: dict[str, Any], gbdt_models: dict[str, Any], device: torch.device, base_cols: list[str]) -> dict[str, Any]:
    """Shape-compatible override for harnesses (greedy_replay, via prepare_component /
    _prepare_component_val) that already built `exit_runtime` -- replaces only that dict key,
    everything else (dec/atr/margin/leverage/route/exit_threshold, all exit-head-independent)
    untouched. Returns a new dict (does not mutate `prepped`)."""
    cols = list(base_cols) + list(parent.POS_COLS)
    n = int(prepped["base_np"].shape[1])
    if n != len(cols):
        raise RuntimeError(f"GBDT injection column count mismatch: base_np width={n} vs base_cols+POS_COLS={len(cols)}")
    zeros, ones = np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
    out = dict(prepped)
    out["exit_runtime"] = {expert: (GBDTExitHeadWrapper(gbdt_models[expert], device, cols), zeros, ones) for expert in hard.EXPERT_NAMES}
    return out


def _evaluate_component_val(gbdt_models: dict[str, Any]) -> dict[str, Any]:
    """h48qual-standalone VAL ledger, TabM live-ATR baseline vs GBDT. dec/x/frame/margin/leverage
    are exit-head-independent (direction/quality/encoder frozen, sidecar risk model unaffected by
    exit_head) so this prepares the component ONCE via sweep.prep_component (loads the TabM
    live-ATR bundle normally) and reuses that single prep for both replay_exit_variant calls,
    swapping only `loaded_models` for the GBDT leg -- simpler and strictly more consistent than
    re-running prep_component per variant (h48cons._evaluate_val's pattern), since it removes any
    chance of the two legs seeing subtly different frame/dec/margin/leverage."""
    cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_pred = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{cfg['q_tag']}.csv"

    prepped = sweep.prep_component("h48qual", cfg, val_frame, val_pred, oof=True)
    m_tabm, _ledger_tabm = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )

    base_cols = list(torch.load(cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    gbdt_loaded = _gbdt_loaded_models(base_cols, gbdt_models, sweep.DEVICE)
    m_gbdt, _ledger_gbdt = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], gbdt_loaded,
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    return {"tabm_liveatr": m_tabm, "gbdt": m_gbdt}


def _run_portfolio_variant_gbdt(
    val_frame: pd.DataFrame, aligned_pred_paths: dict[str, Path], gbdt_models: dict[str, Any], *, fee: float, slip: float,
) -> dict[str, Any]:
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    h48qual_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["h48qual"], h48qual_cfg, portfolio.DEVICE)
    h48qual_gbdt = _inject_gbdt_exit_runtime(h48qual_prepped, gbdt_models, portfolio.DEVICE, base_cols)
    zig075_prepped = portfolio._prepare_component_val(val_frame, aligned_pred_paths["zig075"], zig075_cfg, portfolio.DEVICE)
    components = {"h48qual": h48qual_gbdt, "zig075": zig075_prepped}
    _diag, ledger = greedy.greedy_replay(val_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE)
    ledger.to_csv(OUT_DIR / "portfolio_ledger_asymmetric_h48qual_gbdt_zig075_original.csv", index=False)
    metrics = portfolio._ledger_metrics(ledger)
    print(f"  asymmetric_h48qual_gbdt_zig075_original: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== stage=G0_self_check ===", flush=True)
    g0_component = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    print(f"  component baseline_original: {g0_component['baseline']}", flush=True)
    print(f"  component tabm_liveatr: {g0_component['h48cons_relabel']}", flush=True)
    g0_ok_component_baseline = _close_to_reference(g0_component["baseline"], G0_REFERENCE["component_baseline_original"])
    g0_ok_component_tabm = _close_to_reference(g0_component["h48cons_relabel"], G0_REFERENCE["component_tabm_liveatr"])

    val_frame_raw = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    fee, slip = omega._load_fee_slip()
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    val_frame, aligned_pred_paths = portfolio._align_frame_and_predictions(val_frame_raw, q_tags)
    print(f"  VAL aligned rows={len(val_frame)} (from raw {len(val_frame_raw)})", flush=True)

    portfolio_baseline = portfolio.run_variant(
        "baseline_both_original",
        {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    portfolio_tabm_liveatr = portfolio.run_variant(
        "asymmetric_h48qual_liveatr_zig075_original",
        {"h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE), "zig075": portfolio._component_cfg("zig075")},
        val_frame, aligned_pred_paths, fee=fee, slip=slip,
    )
    g0_ok_portfolio_baseline = _close_to_reference(portfolio_baseline, G0_REFERENCE["portfolio_baseline_both_original"])
    g0_ok_portfolio_tabm = _close_to_reference(portfolio_tabm_liveatr, G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"])

    g0_pass = bool(g0_ok_component_baseline and g0_ok_component_tabm and g0_ok_portfolio_baseline and g0_ok_portfolio_tabm)
    g0_report = {
        "component_baseline_original": {"actual": g0_component["baseline"], "reference": G0_REFERENCE["component_baseline_original"], "match": g0_ok_component_baseline},
        "component_tabm_liveatr": {"actual": g0_component["h48cons_relabel"], "reference": G0_REFERENCE["component_tabm_liveatr"], "match": g0_ok_component_tabm},
        "portfolio_baseline_both_original": {"actual": portfolio_baseline, "reference": G0_REFERENCE["portfolio_baseline_both_original"], "match": g0_ok_portfolio_baseline},
        "portfolio_asymmetric_tabm_liveatr": {"actual": portfolio_tabm_liveatr, "reference": G0_REFERENCE["portfolio_asymmetric_tabm_liveatr"], "match": g0_ok_portfolio_tabm},
        "tolerance_pp": G0_TOLERANCE_PP,
        "pass": g0_pass,
    }
    print(f"stage=G0_result pass={g0_pass}", flush=True)

    if not g0_pass:
        report = {
            "stage_reached": "G0_self_check",
            "g0": g0_report,
            "gate_pass": False,
            "note": "G0 failed -- this harness does not reproduce the published TabM live-ATR reference numbers. Aborting before evaluating GBDT (per methodology discipline, GBDT numbers from an unverified harness are not trustworthy).",
        }
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(f"report={OUT_DIR / 'report.json'}", flush=True)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    print("=== stage=gbdt_evaluation ===", flush=True)
    if not GBDT_BUNDLE.exists():
        raise FileNotFoundError(f"GBDT bundle not found, run train_eval_omega461_gbdt_exit_head_liveatr_20260813.py first: {GBDT_BUNDLE}")
    gbdt_bundle = _load_gbdt_bundle(GBDT_BUNDLE)
    gbdt_models = gbdt_bundle["models"]

    component_gbdt = _evaluate_component_val(gbdt_models)
    print(f"  component tabm_liveatr (rechecked): {component_gbdt['tabm_liveatr']}", flush=True)
    print(f"  component gbdt: {component_gbdt['gbdt']}", flush=True)
    portfolio_gbdt = _run_portfolio_variant_gbdt(val_frame, aligned_pred_paths, gbdt_models, fee=fee, slip=slip)

    gate_component_pnl = float(component_gbdt["gbdt"]["pnl"]) >= float(component_gbdt["tabm_liveatr"]["pnl"])
    gate_component_mdd = float(component_gbdt["gbdt"]["mdd"]) >= float(component_gbdt["tabm_liveatr"]["mdd"])
    gate_portfolio_pnl = float(portfolio_gbdt["pnl"]) >= float(portfolio_tabm_liveatr["pnl"])
    gate_portfolio_mdd = float(portfolio_gbdt["mdd"]) >= float(portfolio_tabm_liveatr["mdd"])
    gate_pass = bool(gate_component_pnl and gate_component_mdd and gate_portfolio_pnl and gate_portfolio_mdd)
    print(
        f"stage=gate_result component_pnl={gate_component_pnl} component_mdd={gate_component_mdd} "
        f"portfolio_pnl={gate_portfolio_pnl} portfolio_mdd={gate_portfolio_mdd} gate_pass={gate_pass}",
        flush=True,
    )

    report = {
        "stage_reached": "gbdt_evaluation",
        "g0": g0_report,
        "gbdt_bundle": str(GBDT_BUNDLE),
        "gbdt_library": gbdt_bundle.get("gbdt_library"),
        "component_level": {
            "tabm_liveatr": component_gbdt["tabm_liveatr"],
            "gbdt": component_gbdt["gbdt"],
            "gate_pnl_nonworse": gate_component_pnl,
            "gate_mdd_nonworse": gate_component_mdd,
        },
        "portfolio_level": {
            "baseline_both_original": portfolio_baseline,
            "asymmetric_h48qual_liveatr_zig075_original": portfolio_tabm_liveatr,
            "asymmetric_h48qual_gbdt_zig075_original": portfolio_gbdt,
            "gate_pnl_nonworse": gate_portfolio_pnl,
            "gate_mdd_nonworse": gate_portfolio_mdd,
        },
        "gate_pass": gate_pass,
        "gate_rule": "GBDT non-worse than TabM live-ATR baseline on PnL AND MDD, at BOTH component and portfolio level",
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "oos_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
