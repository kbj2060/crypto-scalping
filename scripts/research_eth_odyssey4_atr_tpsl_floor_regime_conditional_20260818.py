#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to the CLOSED axis in
docs/experiments/eth_omega461_atr_tpsl_floor_independent_percomponent_20260815.md (see also
eth_omega461_atr_tpsl_recalibration_pilot_20260813.md and
eth_omega4_6_1_atr_tpsl_floor_binding_investigation_20260812.md). Prior finding: the live
"ATR-adaptive" TP/SL floor (min_tp=0.075/min_sl=0.040) binds 95-98.5% of the time on ETH 5m --
effectively a FIXED-width barrier, not ATR-adaptive. Three independent single-global-floor
recalibrations (widen via mult, narrow via absolute value, per-component isolation) were all
REJECTED: widening was monotonically worse; zig075 narrowing was decisively worse on VAL (3/3,
portfolio PnL -34% to -43%); h48qual narrowing PASSED VAL (best candidate portfolio PnL
+36.82%->+43.91%) but REVERSED on the pre-registered single-touch OOS (PnL +49.32%->+38.16%, MDD
nearly doubled -16.20%->-28.64%). Closing memo's own explicit suggestion for a "qualitatively
different hypothesis": regime-conditional floor -- this script is that follow-up.

Literature grounding (2026-08-18, via paper-lookup skill): Kaminski & Lo (2014), "When Do
Stop-Loss Rules Stop Losses?", Journal of Financial Markets (83 citations, Semantic Scholar).
Their finding: under the Random Walk Hypothesis, stop-loss rules always DECREASE expected
return; but in the PRESENCE OF MOMENTUM, stop-loss rules CAN ADD VALUE -- shown via an explicit
regime-switching model. This directly motivates conditioning barrier width on a momentum/trend
regime rather than applying one global width, which is exactly the mechanism the 3 prior CLOSED
attempts could not express (a single scalar cannot be "narrow when it helps, wide when it hurts").

Regime check before designing candidates (2026-08-18, this session): computed atr_pct(window=192)
percentiles split by bull/bear/chop route (hard._route_id) on the full 2025 TRAIN population --
found the three regimes' ATR distributions are actually SIMILAR (p50 atr_pct 0.273%/0.294%/0.254%
for bull/bear/chop respectively; floor still binds 96-99% in ALL THREE regimes even at p90). This
means the regime-conditional hypothesis is NOT grounded in "regime X has structurally wider ATR
that would naturally clear the floor" -- it is grounded purely in the Kaminski-Lo mechanism
(momentum/trend regimes benefit from responsive stops; range-bound/chop regimes do not, because
losses in chop are closer to noise than persistent drawdown). Also checked whether VAL vs OOS-Q1
regime MIX differs enough to explain the h48qual narrowing's VAL-pass/OOS-reversal on its own:
it does NOT (VAL chop=49.4% vs OOS-Q1 chop=50.2%, bull/bear within ~2pp) -- so this experiment
tests the regime-conditional mechanism on its own merits, not as a "regime mix shifted" story.

Design: keep the origin scope, narrow ONLY h48qual (zig075 narrowing was decisively rejected on
VAL regardless of regime nuance -- not retested here to keep the grid small per the closing
memo's own multiple-testing caution after 2 consecutive rejections). Within h48qual, apply the
narrowed floor ONLY on bars routed to bull or bear (trending); chop bars keep the exact live
floor (0.075/0.040) unchanged. Reuses the ALREADY-CHARACTERIZED p50-cross/p25-cross floor values
from the 08-15 experiment (0.0324/0.0162 and 0.0252/0.0126) for continuity -- no new untested
floor magnitudes, only a new (regime-conditional) APPLICATION of already-known values. Skips
h48qual_p75 (already the weakest of the 3 in 08-15, with_gate PnL below baseline).

zig075 stays at the exact live floor in every candidate row (component_to_change is always
"h48qual" here). tp_mult/sl_mult/max_tp/max_sl unchanged everywhere (caps never bind, out of
scope). exit_threshold unchanged (0.95, isolates the floor axis only).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. VAL window =
2025-10-01..2025-12-31. OOS window = 2026-01-01..2026-03-31, single touch, opened only if a VAL
candidate clears the SAME bar as 08-15 (beat baseline on pnl AND mdd, no_gate AND with_gate).
Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
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

import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as helpers  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_atr_tpsl_floor_regime_conditional_20260818"

# (label, trending_min_tp, trending_min_sl). chop bars always stay at live floor (0.075/0.040).
# component_to_change is always h48qual. Values reused verbatim from the 08-15 experiment's
# p50-cross / p25-cross candidates (see module docstring).
CANDIDATE_GRID: list[tuple[str, float, float]] = [
    ("baseline", 0.075, 0.040),
    ("h48qual_trending_p50", 0.0324, 0.0162),
    ("h48qual_trending_p25", 0.0252, 0.0126),
]
LIVE_MIN_TP, LIVE_MIN_SL = 0.075, 0.040
BASELINE_EXIT_THRESHOLD = 0.95


def log(msg: str) -> None:
    print(f"[atr_floor_regime] {msg}", flush=True)


def prep_component_regime(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path, *, oof: bool,
                           trending_min_tp: float, trending_min_sl: float) -> dict[str, Any]:
    """Byte-for-byte copy of research_eth_omega461_exit_sweep_20260721.prep_component, except the
    single _apply_atr_safety_sltp call is replaced by two calls (live floor + trending floor)
    whose take_profit/stop_loss columns are then spliced per-row by regime -- everything else
    (direction/quality decisions, risk-sidecar margin/leverage, which are computed AFTER and FROM
    the spliced dec) is identical to the shared harness. Not modifying prep_component itself since
    other sibling experiments still rely on its single-floor behavior."""
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    src_raw = pd.read_csv(pred_csv)
    for c in src_raw.columns:
        if str(src_raw[c].dtype).lower().startswith("str"):
            src_raw[c] = src_raw[c].astype(object)
    src_raw["timestamp"] = pd.to_datetime(src_raw["timestamp"])
    keep_ts = set(src_raw["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src = src_raw[src_raw["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if len(src) != len(frame) or not src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"{name}: prediction/frame timestamp mismatch ({len(src)} vs {len(frame)})")

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(src, oof=oof)

    dec_live, _ = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=LIVE_MIN_TP, min_sl=LIVE_MIN_SL, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    dec_trending, _ = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=trending_min_tp, min_sl=trending_min_sl, max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    route = hard._route_id(frame)  # 0=bull, 1=bear, 2=chop_expert (hard.EXPERT_NAMES order)
    is_trending = route != 2
    dec = dec_live.copy()
    dec.loc[is_trending, "take_profit"] = dec_trending.loc[is_trending, "take_profit"]
    dec.loc[is_trending, "stop_loss"] = dec_trending.loc[is_trending, "stop_loss"]

    atr_pct = atr_eval._atr_pct(frame, cfg["atr_window"])
    fee, slip = base_sweep.omega._load_fee_slip()
    loaded = parent._load_payloads(models, device=base_sweep.DEVICE)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)

    features = rs._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)

    mapping = pkl["selected_mapping"]
    margin_kwargs = {k: mapping[k] for k in rs.MARGIN_CFG_KEYS}
    margin = rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **margin_kwargs)
    leverage = None
    if pkl["dynamic_leverage"]:
        lev_kwargs = {k: mapping[k] for k in rs.LEVERAGE_CFG_KEYS}
        leverage = rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **lev_kwargs)

    return dict(
        frame=frame, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
        fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"],
        regime_trending_share=float(is_trending.mean()),
    )


def run_split(split_name: str, frame: pd.DataFrame, *, oof: bool,
              grid: list[tuple[str, float, float]]) -> dict[str, Any]:
    log(f"=== split={split_name} rows={len(frame)} range=[{frame['timestamp'].min()}, {frame['timestamp'].max()}] ===")
    component_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for label, tmin_tp, tmin_sl in grid:
        prepped: dict[str, dict[str, Any]] = {}
        for name, base_cfg in base_sweep.COMPONENTS.items():
            cfg = dict(base_cfg)
            pred_csv = base_sweep.EXT_PRED_DIR / name / (f"validation_predictions_{cfg['q_tag']}.csv" if oof
                                                           else f"oos_predictions_{cfg['q_tag']}.csv")
            if name == "h48qual":
                p = prep_component_regime(name, cfg, frame, pred_csv, oof=oof,
                                           trending_min_tp=tmin_tp, trending_min_sl=tmin_sl)
            else:
                p = base_sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)
            prepped[name] = p

            m, _ledger = base_sweep.replay_exit_variant(
                p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
                notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
            )
            component_rows.append({
                "split": split_name, "candidate": label, "component": name,
                "trending_min_tp": tmin_tp, "trending_min_sl": tmin_sl,
                **{k: v for k, v in m.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m["exit_reasons"]),
            })
            log(f"  [{label}] component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} "
                f"avg_hold={m['avg_hold_bars']:.1f}"
                + (f" trending_share={p['regime_trending_share']*100:.1f}%" if "regime_trending_share" in p else ""))

        router_components = {name: helpers._as_router_component(p, exit_threshold=BASELINE_EXIT_THRESHOLD)
                              for name, p in prepped.items()}
        fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
        _, ledger_combined = router.greedy_replay(frame, router_components, fee=fee0, slip=slip0,
                                                   cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
        no_gate = helpers._ledger_stats(ledger_combined, frame)
        with_gate = helpers._duration_gated(ledger_combined, frame, router.DURATION_THRESHOLD)
        portfolio_rows.append({"split": split_name, "candidate": label,
                                "trending_min_tp": tmin_tp, "trending_min_sl": tmin_sl,
                                "no_gate": no_gate, "with_gate": with_gate})
        log(f"  [{label}] PORTFOLIO no_gate pnl={no_gate['pnl']:.2f}% mdd={no_gate['mdd']:.2f}% trades={no_gate['trades']} | "
            f"with_gate pnl={with_gate['pnl']:.2f}% mdd={with_gate['mdd']:.2f}% trades={with_gate['trades']}")

    return {"component_rows": component_rows, "portfolio_rows": portfolio_rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END,
                                       base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    val_result = run_split("VAL", val_frame, oof=True, grid=CANDIDATE_GRID)

    val_df = pd.DataFrame(val_result["component_rows"])
    val_df.to_csv(OUT_DIR / "component_val.csv", index=False)
    val_portfolio = pd.DataFrame([
        {"split": r["split"], "candidate": r["candidate"],
         "trending_min_tp": r["trending_min_tp"], "trending_min_sl": r["trending_min_sl"],
         "no_gate_pnl": r["no_gate"]["pnl"], "no_gate_mdd": r["no_gate"]["mdd"], "no_gate_trades": r["no_gate"]["trades"],
         "with_gate_pnl": r["with_gate"]["pnl"], "with_gate_mdd": r["with_gate"]["mdd"], "with_gate_trades": r["with_gate"]["trades"]}
        for r in val_result["portfolio_rows"]
    ])
    val_portfolio.to_csv(OUT_DIR / "portfolio_val.csv", index=False)

    baseline_row = val_result["portfolio_rows"][0]
    assert baseline_row["candidate"] == "baseline"
    g0 = baseline_row["no_gate"]
    g0_ok = (abs(g0["pnl"] - 36.82) < 0.5) and (abs(g0["mdd"] - (-24.34)) < 0.5) and (g0["trades"] == 29)
    log(f"G0 self-consistency check: pnl={g0['pnl']:.2f} (expect 36.82) mdd={g0['mdd']:.2f} (expect -24.34) "
        f"trades={g0['trades']} (expect 29) -> {'PASS' if g0_ok else 'FAIL'}")

    baseline_no_gate, baseline_with_gate = val_result["portfolio_rows"][0]["no_gate"], val_result["portfolio_rows"][0]["with_gate"]
    qualifiers = []
    for r in val_result["portfolio_rows"][1:]:
        ng, wg = r["no_gate"], r["with_gate"]
        beats = (ng["pnl"] >= baseline_no_gate["pnl"] and ng["mdd"] >= baseline_no_gate["mdd"] and
                 wg["pnl"] >= baseline_with_gate["pnl"] and wg["mdd"] >= baseline_with_gate["mdd"])
        if beats:
            qualifiers.append(r)
    log(f"VAL qualifiers (beat baseline on pnl AND mdd, no_gate AND with_gate): "
        f"{[q['candidate'] for q in qualifiers] if qualifiers else 'NONE'}")

    result: dict[str, Any] = {
        "g0_self_consistency": {"ok": bool(g0_ok), "measured": g0, "expected": {"pnl": 36.82, "mdd": -24.34, "trades": 29}},
        "val_candidate_grid": CANDIDATE_GRID,
        "val_qualifiers": [q["candidate"] for q in qualifiers],
        "oos_run": False,
    }

    if not g0_ok:
        log("ABORTING before OOS: G0 self-consistency check failed.")
        (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
        return 1

    if not qualifiers:
        log("No candidate cleared the VAL bar. OOS NOT opened -- negative pilot result.")
        (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
        return 0

    best = max(qualifiers, key=lambda r: r["no_gate"]["pnl"])
    best_idx = [r["candidate"] for r in val_result["portfolio_rows"]].index(best["candidate"])
    best_label, best_min_tp, best_min_sl = CANDIDATE_GRID[best_idx]
    log(f"Best VAL qualifier: {best_label} (trending_min_tp={best_min_tp}, trending_min_sl={best_min_sl}) "
        f"-- opening SINGLE-TOUCH OOS now.")

    oos_frame = base_sweep.load_frame(base_sweep.OOS_START, base_sweep.OOS_END,
                                       base_csv=base_sweep.BASE_2026, wide24_csv=base_sweep.WIDE24_2026)
    route_finite = np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    n_route_bad = int((~route_finite).sum())
    if n_route_bad:
        oos_frame = oos_frame[route_finite].reset_index(drop=True)
        log(f"dropped {n_route_bad} OOS bars with non-finite Regime3 route probabilities (known WIDE24_2026 gap)")
    oos_grid = [("baseline", 0.075, 0.040), (best_label, best_min_tp, best_min_sl)]
    oos_result = run_split("OOS", oos_frame, oof=False, grid=oos_grid)
    oos_df = pd.DataFrame(oos_result["component_rows"])
    oos_df.to_csv(OUT_DIR / "component_oos.csv", index=False)
    oos_portfolio = pd.DataFrame([
        {"split": r["split"], "candidate": r["candidate"],
         "trending_min_tp": r["trending_min_tp"], "trending_min_sl": r["trending_min_sl"],
         "no_gate_pnl": r["no_gate"]["pnl"], "no_gate_mdd": r["no_gate"]["mdd"], "no_gate_trades": r["no_gate"]["trades"],
         "with_gate_pnl": r["with_gate"]["pnl"], "with_gate_mdd": r["with_gate"]["mdd"], "with_gate_trades": r["with_gate"]["trades"]}
        for r in oos_result["portfolio_rows"]
    ])
    oos_portfolio.to_csv(OUT_DIR / "portfolio_oos.csv", index=False)

    oos_baseline, oos_candidate = oos_result["portfolio_rows"][0], oos_result["portfolio_rows"][1]
    oos_survives = (oos_candidate["no_gate"]["pnl"] >= oos_baseline["no_gate"]["pnl"] and
                    oos_candidate["no_gate"]["mdd"] >= oos_baseline["no_gate"]["mdd"])
    result.update({
        "oos_run": True,
        "oos_window": [base_sweep.OOS_START, base_sweep.OOS_END],
        "chosen_candidate": best_label,
        "chosen_min_tp": best_min_tp, "chosen_min_sl": best_min_sl,
        "oos_baseline_no_gate": oos_baseline["no_gate"], "oos_candidate_no_gate": oos_candidate["no_gate"],
        "oos_baseline_with_gate": oos_baseline["with_gate"], "oos_candidate_with_gate": oos_candidate["with_gate"],
        "oos_survives_no_gate_pnl_and_mdd": bool(oos_survives),
    })
    log(f"OOS result: baseline pnl={oos_baseline['no_gate']['pnl']:.2f}% mdd={oos_baseline['no_gate']['mdd']:.2f}% | "
        f"candidate pnl={oos_candidate['no_gate']['pnl']:.2f}% mdd={oos_candidate['no_gate']['mdd']:.2f}% -> "
        f"{'SURVIVES' if oos_survives else 'REVERSES'}")

    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
