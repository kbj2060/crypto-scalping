#!/usr/bin/env python3
"""RESEARCH ONLY -- random-direction risk-management ablation.

=== User question (2026-08-17 session) ===
Odyssey4 cannot predict direction (Odyssey1 settled fact, N>=5 seeds: ungated direction_head loses
to always-short 0/5 -- docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md)
but the surrounding risk-management stack (quality-threshold gate, zig075 sustained-uptrend SHORT
entry veto -- CONFIRMED -- h48qual regime-aware exit guard, ATR TP/SL, risk-sidecar sizing) is the
one part of this system with validated value. Before designing a full human-in-the-loop direction
system, test the extreme case directly: if the direction pick itself is REPLACED with a uniform
coin-flip (LONG/SHORT, 50/50, every bar) while every other layer of the live-matching Odyssey4
pipeline is left byte-for-byte unmodified, does the quality gate + veto + guard + sizing alone
produce something survivable? This is the mirror image of the already-settled "ungated direction
head loses to always-short" result: THAT question removed the gate and kept the direction head;
THIS question removes the direction head and keeps the gate + full risk stack.

=== Design ===
Reuses (never edits) the exact modules that produced the live Odyssey4 G0 reference numbers:
  - eth_omega461_multiwindow_confirmation_gate_20260814 (gate): window defs, aligned prediction CSVs,
    COMP_CFGS_ASYMMETRIC_TABM_LIVEATR (the deployed bundle configs).
  - research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 (guard): sustained-uptrend
    detector (build_detector, zero new free parameters, calibrated on 2025-Q1+Q2 only) and the
    h48qual regime-aware exit-guard side-channel (guard_base_np/guard_exit_runtime/guard_pos_idx).
  - research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 (veto_mod):
    greedy_replay_entry_veto, the exact G0 portfolio replay (priority routing, TP/SL, exit-head,
    zig075 SHORT veto) -- called completely unmodified.
  - replay_omega4_6_1_greedy_router_20260706.prepare_component /
    research_eth_omega461_exit_head_portfolio_asymmetric_20260813._prepare_component_val: the
    component-preparation functions that turn a raw prediction CSV into (dec, margin, leverage,
    exit-head runtime). THIS script's only new code is `prepare_component_direction_override` below,
    a documented copy of those two (which are identical except the oof flag) with ONE inserted step:
    right after the raw prediction CSV is read, {prefix}_final_action and {prefix}_quality_for_action
    are overwritten using a caller-supplied `side_selector(n_rows) -> np.ndarray[+1/-1]` instead of
    argmax(direction_proba). quality_for_action is recomputed from that CSV's OWN
    {prefix}_quality_p_long / {prefix}_quality_p_short columns at the SELECTED side (quality_for_
    action is direction-dependent by construction -- reusing the model's original quality_for_action,
    which reflects the model's OWN pick, would silently keep information this ablation is supposed to
    remove). {prefix}_dir_p_*/_dir_action/_dir_confidence/_dir_side_edge/_dir_trade_prob are left
    UNTOUCHED -- they still enter the risk sidecar as raw features (renamed "parent_dir_*" by
    rs._risk_feature_frame), exactly as they would for a real human-in-the-loop design where the
    model's own (now direction-overruled) opinion remains available to the risk/sizing layer as
    context. Nothing else changes: ATR TP/SL (direction-independent), the risk-sidecar GBM weights,
    the zig075 veto detector/threshold, the h48qual regime-exit guard, SCALE_MAP/LEVERAGE_CAP/
    NOTIONAL_CAP, priority routing.

Three side_selector arms, all using the identical mechanism above (only the selector differs):
  - random:       per-bar np.random.default_rng(seed).choice([-1, +1]), N=5 independently-drawn
                   seeds (SeedSequence-spawned, not a fixed base+increment -- see
                   feedback memory tabm_hp_low_signal_pattern on why increment seeds are invalid).
  - always_long:   constant +1 every bar, still gated by quality_threshold at that side.
  - always_short:  constant -1 every bar, still gated by quality_threshold at that side.
This isolates the causal contribution of direction *quality* by holding the gate + full risk stack
fixed across all three arms and only varying the direction-selection rule -- a cleaner comparison
than the repo's usual ungated max(always_long, always_short) benchmark (reported separately below,
context only, since it uses a different -- ungated -- mechanism).

Windows: val, oos_q1, oos_q2 (the three judged-tier windows with a locked G0 reference in
docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md). 2025 Q1-Q3 context
windows are not included in this ablation (can be added later; the judged tier is the direct
comparison point for the G0 table).

fresh_forward_bar_by_bar=true (single causal forward pass; row i's random draw and quality gate
depend only on that row's own quality_p_long/quality_p_short, never on future rows).
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
Does NOT modify any imported module (greedy/guard/gate/veto_mod/parent/omega/rs/atr_eval/hard are
read-only imports). No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_random_direction_risk_management_ablation_20260817"
DEVICE = portfolio.DEVICE
JUDGED_WINDOWS = ("val", "oos_q1", "oos_q2")
N_SEEDS = 5

# G0 reference (Odyssey4 with_gate, verbatim from docs/model_contracts/
# odyssey4_eth_entry_veto_baseline_contract_20260814.md -- not recomputed here, only reused for
# the comparison table).
G0_ODYSSEY4_WITH_GATE = {
    "val": {"pnl": 77.31, "mdd": -21.76, "trades": 26},
    "oos_q1": {"pnl": 67.25, "mdd": -15.48, "trades": 19},
    "oos_q2": {"pnl": -12.69, "mdd": -20.76, "trades": 10},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _side_selector_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.integers(0, 2, size=n) == 0, -1, 1).astype(np.int64)


def _side_selector_constant(n: int, side: int) -> np.ndarray:
    return np.full(n, side, dtype=np.int64)


def prepare_component_direction_override(
    frame: pd.DataFrame, pred_csv: Path, cfg: dict, device: torch.device, *, oof: bool,
    side_selector: Callable[[int], np.ndarray],
) -> dict[str, Any]:
    """Documented copy of replay_omega4_6_1_greedy_router_20260706.prepare_component /
    research_eth_omega461_exit_head_portfolio_asymmetric_20260813._prepare_component_val (those two
    are identical except the oof flag on parent._to_decisions -- copied here as ONE function
    parameterized by oof, exactly as portfolio._prepare_component_val's own docstring already
    explains why the duplication exists: the shared script's oof flag isn't exposed as a parameter).
    The only added logic vs both originals is the block marked below; every other line is unchanged.
    """
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("timestamp mismatch")

    # --- direction override: only new logic vs prepare_component / _prepare_component_val ---
    prefix = omega._tabm_prefix(oof)
    n = len(pred)
    random_side = side_selector(n)  # +1 = LONG (ACTION_LONG=1), -1 = SHORT (ACTION_SHORT=2)
    quality_p_long = pd.to_numeric(pred[f"{prefix}quality_p_long"], errors="raise").to_numpy(dtype=np.float64)
    quality_p_short = pd.to_numeric(pred[f"{prefix}quality_p_short"], errors="raise").to_numpy(dtype=np.float64)
    quality_for_random = np.where(random_side > 0, quality_p_long, quality_p_short)
    threshold = pd.to_numeric(pred[f"{prefix}quality_threshold"], errors="raise").to_numpy(dtype=np.float64)
    random_action = np.where(random_side > 0, omega.ACTION_LONG, omega.ACTION_SHORT)
    final_action_random = np.where(quality_for_random >= threshold, random_action, omega.ACTION_CASH)
    pred[f"{prefix}final_action"] = final_action_random
    pred[f"{prefix}quality_for_action"] = quality_for_random
    # dir_p_cash/long/short, dir_action, dir_confidence, dir_side_edge, dir_trade_prob deliberately
    # left untouched -- still the model's own real direction-head assessment, available to the risk
    # sidecar as context features via rs._risk_feature_frame's "parent_dir_*" columns.
    # --- end direction override block ---

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=oof)
    dec, _ = atr_eval._apply_atr_safety_sltp(dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"],
                                              sl_mult=cfg["sl_mult"], min_tp=cfg["min_tp"], min_sl=cfg["min_sl"],
                                              max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
    atr = atr_eval._atr_pct(frame, cfg["atr_window"])
    loaded = parent._load_payloads(models, device=device)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = rs._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in rs.MARGIN_CFG_KEYS})
    lev = rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in rs.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(x, loaded)
    route = hard._route_id(frame)
    return {
        "dec": dec, "atr": atr, "margin": margin, "leverage": lev, "base_np": base_np,
        "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": route, "exit_threshold": cfg["exit_threshold"],
        "n_random_long": int((random_side[quality_for_random >= threshold] > 0).sum()),
        "n_random_short": int((random_side[quality_for_random >= threshold] < 0).sum()),
        "n_gated_cash": int((quality_for_random < threshold).sum()),
    }


def build_ablation_components(
    window_name: str, windows: dict[str, Any], score_by_base: dict, threshold: float, out_dir: Path,
    device: torch.device, *, side_selector: Callable[[int], np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Direction-overridden mirror of guard.prepare_regime_aware_components: same detector mask, same
    h48qual regime-exit-guard side-channel (reused UNMODIFIED from the real components -- the exit
    guard only ever evaluates an already-open position's exit-head probability using realized
    trade-path values (move/mfe/mae/hold), it does not depend on how the entry direction was chosen),
    but dec/margin/leverage/base_np/exit_runtime/pos_idx are rebuilt via
    prepare_component_direction_override for BOTH h48qual and zig075.
    """
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, out_dir)

    h48qual_ablated = prepare_component_direction_override(
        aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], device,
        oof=w["oof"], side_selector=side_selector,
    )
    # h48qual's ORIGINAL (pre-liveATR-relabel) exit-head side-channel: reused unmodified from the
    # real (non-ablated) pipeline -- the regime-exit guard evaluates an open position's exit
    # probability, which does not depend on entry-direction selection.
    h48qual_original_real = (
        portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    )(aligned_frame, aligned_paths["h48qual"], gate.COMP_CFGS_BASELINE_BOTH_ORIGINAL["h48qual"], device)

    zig075_ablated = prepare_component_direction_override(
        aligned_frame, aligned_paths["zig075"], gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device,
        oof=w["oof"], side_selector=side_selector,
    )

    mask, n_nan = guard._detector_mask_for_frame(aligned_frame, window_name, score_by_base, threshold)
    h48qual_guarded = dict(h48qual_ablated)
    h48qual_guarded["guard_base_np"] = h48qual_original_real["base_np"]
    h48qual_guarded["guard_exit_runtime"] = h48qual_original_real["exit_runtime"]
    h48qual_guarded["guard_pos_idx"] = h48qual_original_real["pos_idx"]
    h48qual_guarded["guard_exit_threshold"] = h48qual_original_real["exit_threshold"]
    h48qual_guarded["sustained_uptrend_mask"] = mask

    zig075_vetoed = dict(zig075_ablated)
    zig075_vetoed["short_entry_veto_mask"] = mask

    components = {"h48qual": h48qual_guarded, "zig075": zig075_vetoed}
    return aligned_frame, components


def run_arm(
    arm_label: str, window_name: str, windows: dict, score_by_base: dict, threshold: float, out_dir: Path,
    device: torch.device, fee: float, slip: float, *, side_selector: Callable[[int], np.ndarray],
) -> dict[str, Any]:
    aligned_frame, components = build_ablation_components(
        window_name, windows, score_by_base, threshold, out_dir, device, side_selector=side_selector,
    )
    diag, ledger = veto_mod.greedy_replay_entry_veto(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
    )
    metrics = portfolio._ledger_metrics(ledger)
    with_gate = None
    try:
        import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    except Exception as exc:  # pragma: no cover -- diagnostic only, no_gate is the primary number
        with_gate = {"error": str(exc)}
    return {
        "arm": arm_label, "window": window_name,
        "no_gate": metrics, "with_gate": with_gate,
        "n_random_long_h48qual": components["h48qual"].get("n_random_long"),
        "n_random_short_h48qual": components["h48qual"].get("n_random_short"),
        "n_gated_cash_h48qual": components["h48qual"].get("n_gated_cash"),
        "n_random_long_zig075": components["zig075"].get("n_random_long"),
        "n_random_short_zig075": components["zig075"].get("n_random_short"),
        "n_gated_cash_zig075": components["zig075"].get("n_gated_cash"),
        "veto_bars": diag.get("veto_bars"),
        "trades": int(len(ledger)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=detector_build (reused from guard module, zero new free parameters) ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    log(f"  primary(p90)={threshold:.10f}")

    seed_sequence = np.random.SeedSequence(20260817)
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"  N_SEEDS={N_SEEDS} independently-spawned seeds: {seeds}")

    results: list[dict[str, Any]] = []
    for window_name in JUDGED_WINDOWS:
        log(f"=== window={window_name} ===")

        log(f"  arm=always_long")
        results.append(run_arm(
            "always_long", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
            side_selector=lambda n: _side_selector_constant(n, 1),
        ))
        log(f"  arm=always_short")
        results.append(run_arm(
            "always_short", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
            side_selector=lambda n: _side_selector_constant(n, -1),
        ))
        for seed in seeds:
            log(f"  arm=random seed={seed}")
            results.append(run_arm(
                f"random_seed{seed}", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                side_selector=lambda n, _seed=seed: _side_selector_random(n, _seed),
            ))

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    log(f"\nwrote {OUT_DIR / 'ablation_results.csv'}")

    log("\n=== summary ===")
    for window_name in JUDGED_WINDOWS:
        g0 = G0_ODYSSEY4_WITH_GATE[window_name]
        wdf = df[df["window"] == window_name]
        random_rows = wdf[wdf["arm"].str.startswith("random_seed")]
        random_pnl = random_rows["with_gate"].apply(lambda d: d.get("pnl") if isinstance(d, dict) else None)
        random_mdd = random_rows["with_gate"].apply(lambda d: d.get("mdd") if isinstance(d, dict) else None)
        always_long = wdf[wdf["arm"] == "always_long"].iloc[0]
        always_short = wdf[wdf["arm"] == "always_short"].iloc[0]
        log(f"[{window_name}] G0 real model with_gate: pnl={g0['pnl']:+.2f}% mdd={g0['mdd']:.2f}% trades={g0['trades']}")
        log(f"  always_long (gated):  pnl={always_long['with_gate'].get('pnl'):+.2f}% mdd={always_long['with_gate'].get('mdd'):.2f}% trades={always_long['trades']}")
        log(f"  always_short (gated): pnl={always_short['with_gate'].get('pnl'):+.2f}% mdd={always_short['with_gate'].get('mdd'):.2f}% trades={always_short['trades']}")
        log(f"  random (N={N_SEEDS} seeds): pnl mean={random_pnl.mean():+.2f}% std={random_pnl.std():.2f}%  mdd mean={random_mdd.mean():.2f}% std={random_mdd.std():.2f}%")
        log(f"  random per-seed pnl: {random_pnl.tolist()}")

    report = {
        "design": __doc__,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "seeds": seeds,
        "g0_reference": G0_ODYSSEY4_WITH_GATE,
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
