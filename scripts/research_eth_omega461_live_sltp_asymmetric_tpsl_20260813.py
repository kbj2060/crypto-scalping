#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to research_eth_omega461_live_sltp_mfe_width_20260813.py (see
docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md for the parent experiment this
extends). Same TP/SL-width-only variable isolation, same reused harnesses
(research_eth_omega461_exit_sweep_20260721.py / replay_omega4_6_1_greedy_router_20260706.py), same
VAL-only / OOS-forbidden / fresh-forward constraints. Does NOT touch
trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.

Orchestrator hypothesis (2026-08-13, after reviewing the parent experiment's result): the parent
experiment's win-rate collapse (baseline 41-48% -> 27-37% across tp_scale 1.0-6.0, which is what
drove PnL/MDD sharply negative even though hold-time/trade-count improved exactly as diagnosed) came
from SL narrowing IN LOCKSTEP with TP -- both driven by the same `predicted_mfe_width * tp_scale *
fixed_sl_ratio` formula -- cutting into stop distances that used to be wide enough to ride out
ordinary 5m noise. If only TP had narrowed, winning trades would simply close faster without any
change to losing-trade risk.

This script decouples the two axes:
  - TP: UNCHANGED mechanism from the parent experiment -- per-row predicted MFE (base102 feature
    panel, same trained long/short HistGradientBoostingRegressor recipe) * tp_scale, same
    TP_SCALE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0].
  - SL: taken directly from the ORIGINAL live ATR-floor stop_loss that prep_component's
    _apply_atr_safety_sltp already computed (i.e. NOT derived from the MFE prediction at all), times
    an independent sl_scale in {1.0 (baseline SL, completely unchanged), 1.5, 2.0}. sl_scale=1.0
    means the parent experiment's "just don't touch SL" fast check the orchestrator asked for first.

Feature set: base102 ONLY (the parent experiment found it decisively better than the
FINAL10+autoencoder-latent16 control on every VAL metric and almost every downstream replay cell --
not repeated here, per explicit orchestrator instruction to save time).

2D grid: 7 tp_scale x 3 sl_scale = 21 cells, each checked at BOTH the per-component level
(research_eth_omega461_exit_sweep_20260721.replay_exit_variant) and the priority-combined portfolio
level (replay_omega4_6_1_greedy_router_20260706.greedy_replay, h48qual>zig075 single shared position
slot -- the actual live combination mechanism). Looking for any cell where avg_hold_bars is
meaningfully shorter than baseline AND trades meaningfully higher AND pnl/mdd are not worse than
baseline (ideally better) -- simultaneously on both no_gate and with_gate (duration-gate) portfolio
numbers.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.
VAL window = 2025-10-01..2025-12-31 (identical to the parent experiment). OOS NOT run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402

base_sweep = mfe_width.base_sweep

OUT_DIR = ROOT / "tmp/research_20260813/omega461_live_sltp_asymmetric_tpsl"
TP_SCALE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]
SL_SCALE_GRID = [1.0, 1.5, 2.0]


def log(msg: str) -> None:
    print(msg, flush=True)


def apply_asymmetric_tpsl(dec: pd.DataFrame, width: np.ndarray, *, tp_scale: float, sl_scale: float,
                           min_tp: float, max_tp: float, max_sl: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """TP: same driver as the parent experiment (predicted MFE * tp_scale, floored/capped).
    SL: NOT derived from width at all -- scaled directly off dec's OWN pre-existing stop_loss (the
    live ATR-floor value prep_component already computed), so sl_scale=1.0 reproduces the exact
    baseline SL untouched."""
    out = dec.copy().reset_index(drop=True)
    active = base_sweep.omega._active(out)
    baseline_sl = pd.to_numeric(out["stop_loss"], errors="raise").to_numpy(dtype=np.float64)  # read BEFORE any overwrite below
    tp_raw = np.asarray(width, dtype=np.float64) * float(tp_scale)
    tp = np.clip(np.maximum(float(min_tp), tp_raw), 0.0, float(max_tp))
    sl = np.clip(baseline_sl * float(sl_scale), 0.0, float(max_sl))
    out.loc[active, "take_profit"] = tp[active]
    out.loc[active, "stop_loss"] = sl[active]
    out.loc[~active, ["take_profit", "stop_loss"]] = 0.0
    active_tp, active_sl = tp[active], sl[active]
    diag = {
        "tp_scale": float(tp_scale), "sl_scale": float(sl_scale),
        "tp_p50": float(np.quantile(active_tp, 0.5)) if len(active_tp) else 0.0,
        "sl_p50": float(np.quantile(active_sl, 0.5)) if len(active_sl) else 0.0,
        "tp_floor_bind_rate": float((active_tp <= min_tp + 1.0e-12).mean()) if len(active_tp) else 0.0,
        "sl_cap_bind_rate": float((active_sl >= max_sl - 1.0e-12).mean()) if len(active_sl) else 0.0,
    }
    return out, diag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("stage=load_frames")
    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    train_frame = base_sweep.load_frame(mfe_width.TRAIN_START, mfe_width.TRAIN_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)

    import torch
    bundle_h48 = torch.load(base_sweep.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle_h48["base_cols"])

    train_labels = mfe_width._load_tb_labels("train")

    log("stage=prep_components (baseline ATR-floor dec/margin/leverage, computed ONCE)")
    prepped: dict[str, dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] = []
    for name, cfg in base_sweep.COMPONENTS.items():
        pred_csv = base_sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        p = base_sweep.prep_component(name, cfg, val_frame, pred_csv, oof=True)
        prepped[name] = p
        m_base, _ = base_sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
        )
        baseline_rows.append({"component": name, "variant": "baseline_atr_floor", "tp_scale": None, "sl_scale": None,
                               **{k: v for k, v in m_base.items() if k != "exit_reasons"}})
        log(f"  component={name} baseline pnl={m_base['pnl']:.2f} trades={m_base['trades']} avg_hold_bars={m_base['avg_hold_bars']:.1f}")

    router_base = {name: mfe_width._as_router_component(p, exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD) for name, p in prepped.items()}
    fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
    _, ledger_base_combined = router.greedy_replay(val_frame, router_base, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
    baseline_combined = {"no_gate": mfe_width._ledger_stats(ledger_base_combined, val_frame),
                          "with_gate": mfe_width._duration_gated(ledger_base_combined, val_frame, router.DURATION_THRESHOLD)}
    log(f"priority_combined baseline: {json.dumps(baseline_combined)}")

    log("stage=build_panel_base102 + train_mfe_models (identical recipe/seed to the parent experiment)")
    panel_train, feat_cols = mfe_width.base102_panel(base_cols, train_frame)
    panel_val, _ = mfe_width.base102_panel(base_cols, val_frame)
    models, train_diag = mfe_width.train_mfe_models(panel_train, feat_cols, train_labels, seed=mfe_width.MFE_MODEL_SEED)
    val_labels = mfe_width._load_tb_labels("validation")
    val_diag = mfe_width.val_sanity_gate(models, panel_val, feat_cols, val_labels)
    log(f"train_diag: {json.dumps(train_diag)}")
    log(f"val_sanity_gate: {json.dumps(val_diag)} (should match the parent experiment's base102 numbers exactly)")
    x_val_scoring = panel_val[feat_cols]

    log(f"stage=2d_grid tp_scale x sl_scale = {len(TP_SCALE_GRID)} x {len(SL_SCALE_GRID)} = {len(TP_SCALE_GRID) * len(SL_SCALE_GRID)} cells")
    all_rows: list[dict[str, Any]] = list(baseline_rows)
    combined_results: dict[str, Any] = {"baseline": baseline_combined}
    grid_summary: list[dict[str, Any]] = []

    widths = {}
    for name, p in prepped.items():
        side = pd.to_numeric(p["dec"]["side"], errors="raise").to_numpy(dtype=np.int64)
        widths[name] = mfe_width.predicted_width(models, x_val_scoring, side)

    for sl_scale in SL_SCALE_GRID:
        for tp_scale in TP_SCALE_GRID:
            cell_tag = f"tp{tp_scale:g}_sl{sl_scale:g}"
            for name, p in prepped.items():
                cfg = base_sweep.COMPONENTS[name]
                dec_new, wdiag = apply_asymmetric_tpsl(p["dec"], widths[name], tp_scale=tp_scale, sl_scale=sl_scale,
                                                        min_tp=mfe_width.FLOOR_TP, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
                m, _ = base_sweep.replay_exit_variant(
                    p["frame"], p["x"], dec_new, p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                    exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
                    notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
                )
                all_rows.append({"component": name, "variant": cell_tag, "tp_scale": tp_scale, "sl_scale": sl_scale,
                                  **{k: v for k, v in m.items() if k != "exit_reasons"}, **{f"width_{k}": v for k, v in wdiag.items()}})

            comps = {}
            for name, p in prepped.items():
                cfg = base_sweep.COMPONENTS[name]
                dec_new, _ = apply_asymmetric_tpsl(p["dec"], widths[name], tp_scale=tp_scale, sl_scale=sl_scale,
                                                    min_tp=mfe_width.FLOOR_TP, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"])
                comps[name] = {**router_base[name], "dec": dec_new}
            _, ledger = router.greedy_replay(val_frame, comps, fee=fee0, slip=slip0, cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
            no_gate = mfe_width._ledger_stats(ledger, val_frame)
            with_gate = mfe_width._duration_gated(ledger, val_frame, router.DURATION_THRESHOLD)
            combined_results[cell_tag] = {"no_gate": no_gate, "with_gate": with_gate,
                                           "source_component_counts": ledger["source_component"].value_counts().to_dict() if len(ledger) else {}}
            ledger.to_csv(OUT_DIR / f"priority_combined_ledger_{cell_tag}_VAL.csv", index=False)

            hold_ok = no_gate["avg_hold_bars"] < baseline_combined["no_gate"]["avg_hold_bars"] * 0.7 and no_gate["trades"] > baseline_combined["no_gate"]["trades"] * 1.2
            pnl_mdd_ok = (no_gate["pnl"] >= baseline_combined["no_gate"]["pnl"] and no_gate["mdd"] >= baseline_combined["no_gate"]["mdd"]
                          and with_gate["pnl"] >= baseline_combined["with_gate"]["pnl"] and with_gate["mdd"] >= baseline_combined["with_gate"]["mdd"])
            grid_summary.append({"tp_scale": tp_scale, "sl_scale": sl_scale, "no_gate_pnl": no_gate["pnl"], "no_gate_mdd": no_gate["mdd"],
                                  "no_gate_trades": no_gate["trades"], "no_gate_avg_hold_bars": no_gate["avg_hold_bars"],
                                  "with_gate_pnl": with_gate["pnl"], "with_gate_mdd": with_gate["mdd"],
                                  "hold_and_trades_meaningfully_better": bool(hold_ok), "pnl_and_mdd_not_worse": bool(pnl_mdd_ok),
                                  "BOTH_GOALS_MET": bool(hold_ok and pnl_mdd_ok)})
            log(f"  {cell_tag}: no_gate pnl={no_gate['pnl']:.2f} mdd={no_gate['mdd']:.2f} trades={no_gate['trades']} "
                f"avg_hold={no_gate['avg_hold_bars']:.1f} | with_gate pnl={with_gate['pnl']:.2f} mdd={with_gate['mdd']:.2f} "
                f"| hold_ok={hold_ok} pnl_mdd_ok={pnl_mdd_ok}")

    component_df = pd.DataFrame(all_rows)
    component_df.to_csv(OUT_DIR / "component_variants_VAL.csv", index=False)
    grid_df = pd.DataFrame(grid_summary)
    grid_df.to_csv(OUT_DIR / "grid_summary_VAL.csv", index=False)
    log("\n=== 2D grid summary (portfolio, no_gate) ===")
    log(grid_df.to_string(index=False))

    winners = grid_df[grid_df["BOTH_GOALS_MET"]]
    log(f"\ncells meeting BOTH goals simultaneously: {len(winners)} / {len(grid_df)}")
    if len(winners):
        log(winners.to_string(index=False))

    report = {
        "model_id": "omega461_live_sltp_asymmetric_tpsl_20260813",
        "parent_experiment": "docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "val_window": [base_sweep.VAL_START, base_sweep.VAL_END], "oos_run": False,
        "feature_set": "base102_only", "tp_scale_grid": TP_SCALE_GRID, "sl_scale_grid": SL_SCALE_GRID,
        "train_diag": train_diag, "val_sanity_gate": val_diag,
        "component_variants_val": all_rows, "priority_combined_val": combined_results,
        "grid_summary": grid_summary, "n_cells_meeting_both_goals": int(len(winners)),
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=base_sweep.omega._json_default), encoding="utf-8"
    )
    log(f"\nstage=done report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
