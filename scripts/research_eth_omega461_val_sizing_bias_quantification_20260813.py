#!/usr/bin/env python3
"""RESEARCH ONLY -- quantify how much of h48qual/zig075's VAL-vs-OOS PnL gap is attributable to the
risk sidecar's VAL-fit SIZING curve, as opposed to genuine entry/exit/direction signal.

Follow-up to docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md section 5-1,
which qualitatively flagged (by unpickling the live risk_sidecar.pkl files for both components)
that selection_scope="validation_only" -- i.e. the margin_fraction/leverage mapping
(train_eval_omega4_2_risk_sidecar_20260622.py:1417-1445, "selected_mapping") is grid-searched to
directly maximize VAL log_risk_utility (selection_objective="log_risk" for both live sidecars,
confirmed by unpickling). That document stated this qualitatively but did not quantify it. This
script produces the quantitative table: docs/experiments/eth_val_only_sizing_bias_quantification_20260813.md.

METHOD (freezes entry/exit completely, only re-prices the SAME trades under a different sizing
rule -- see CLAUDE.md Futures Risk Sizing Contract, and the caller's explicit warning about the
"double leverage" trap):

1. Build ONE ledger per component x split using research_eth_omega461_exit_sweep_20260721's own
   prep_component()/replay_exit_variant() (the exact harness at least 4 other scripts tonight
   already reused, itself structurally identical to the certified
   train_eval_omega4_2_risk_sidecar_20260622._replay_with_risk loop), fed the REAL live sidecar's
   VAL-selected margin/leverage mapping (prep_component() already reproduces
   _Component.entry_decision()'s sizing math exactly: _risk_margins/_risk_leverage against
   pkl["selected_mapping"]). This is a single causal bar-by-bar forward pass -- entry timing, exit
   timing/reason, side, and raw price-move-based returns are fixed the instant this ledger exists.
2. Re-price that SAME frozen ledger under a flat sizing rule using
   train_eval_omega4_2_risk_sidecar_20260622._ledger_metrics_with_margins() -- the SAME function
   the sidecar's own promotion pipeline (lines ~1350-1500) already uses internally to compare
   margin/leverage candidates on a fixed ledger without re-simulating entries/exits. It only
   rescales `net_per_notional` (a pure price-move-and-fee-derived per-unit-of-notional return,
   already computed once) by a new notional; it never touches take_profit/stop_loss (both live
   sidecars have notional_scaled_sltp=False anyway, confirmed by unpickling, so TP/SL were never
   coupled to notional even in the original replay). Because step 2 reuses the IDENTICAL ledger
   object for both the "actual" and "flat" scenarios, entry_signal_i/exit_i/side/reason are
   byte-for-byte identical between them by construction -- trade count and win/loss direction
   literally cannot differ. An integrity check asserts the ledger-relayered "actual" PnL matches
   the true bar-by-bar replay's own PnL (they should, since price is only realized at trade
   close in both methods).
3. Flat sizing = BASE_TEMPLATE (train_eval_omega1_2_tabm_diffusion_risk_20260603.BASE_TEMPLATE:
   notional=0.45, leverage=2.0 -> margin_fraction=0.225), the SAME fixed sizing every
   always-short/always-long baseline in this sub-project already uses, applied identically to
   every active trade regardless of side/score/regime (the flattest reading of "flat sizing" --
   no EXPERT_SCALES, no side asymmetry). This constant was never part of the sidecar's own
   VAL grid search (unlike any point on the selected_mapping sigmoid curve, e.g. its own
   score-median, which would still encode VAL-fit floor/cap/leverage bounds and therefore would
   NOT be a valid VAL-blind counterfactor -- deliberately not used here).

Judgment rule (caller-specified): if (actual_VAL_pnl - flat_VAL_pnl) >> (actual_OOS_pnl -
flat_OOS_pnl), the sizing curve is fit specifically to VAL and does not generalize -- direct
evidence for the bias. If the two margins are comparable, sizing is not the main driver of the
VAL-to-OOS flip and the qualitative claim in the referenced document should be softened.

No seeds: everything here replays FROZEN, already-trained live artifacts (parent bundles + risk
sidecar HistGradientBoostingRegressor). There is no retraining and therefore no seed-diversity
axis to report (the CLAUDE.md Seed-Diversity Ensemble Promotion Gate targets genuinely-retrained
seed ensembles, not deterministic replay of frozen artifacts).

Windows: VAL=2025-10-01..2025-12-31, OOS=2026-01-01..2026-03-31 (research_eth_omega461_exit_sweep_
20260721 defaults == CLAUDE.md canonical OOS window; VAL starts a month later than the canonical
2025-09-01 because no frozen OOF prediction exists before 2025-10-01 -- same documented deviation
as every other sibling script tonight).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false (the ledger built in step 1 is
generated fresh by this run's own single forward pass, not read from a prior saved ledger; it is
only reused WITHIN this run for the diagnostic re-pricing in step 2 -- purely diagnostic per the
caller's brief, not a promotion/model-selection claim). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT retrain anything.
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

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260813/omega461_val_sizing_bias_quantification"

# Deliberately NOT derived from any risk_sidecar.pkl / selected_mapping -- BASE_TEMPLATE predates
# and is independent of the VAL-only sizing grid search, and is already the fixed sizing every
# always-short/always-long baseline in this sub-project uses. Applied with NO EXPERT_SCALES and NO
# side asymmetry -- the flattest possible reading of "flat sizing" per the caller's brief.
FLAT_NOTIONAL = float(omega.BASE_TEMPLATE["notional"])
FLAT_LEVERAGE = float(omega.BASE_TEMPLATE["leverage"])
FLAT_MARGIN = FLAT_NOTIONAL / FLAT_LEVERAGE

PNL_MATCH_TOL_PP = 0.05  # percentage points; integrity check tolerance for the ledger-relayer


def _flat_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, FLAT_MARGIN, dtype=np.float64), np.full(n, FLAT_LEVERAGE, dtype=np.float64)


def _drop_route_nan_gaps(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """research_eth_omega461_exit_sweep_20260721.load_frame() left-merges the regime3_current
    wide24 overlay onto the base feature CSV. The overlay CSV is missing one contiguous 95-bar
    block (2026-02-28 16:05..23:55, ~7.9h, verified via direct diff against the base CSV -- a
    pre-existing gap in the shared overlay data file, not introduced by this script; likely a
    side effect of the overlay file recovery documented in
    docs/model_contracts/eth_omega4_6_1_live_risk_assessment_20260812.md issue 5). Left join turns
    that gap into NaN route-probability rows, which train_omega1_..._volpca_20260602._route_id()
    correctly refuses to route. Dropping this single small contiguous block locally (NOT patching
    the shared overlay CSV or load_frame -- other sessions/scripts depend on both) costs 95/25633
    OOS bars (0.37%) and is a one-evening skip, not scattered holes, so it does not distort the
    causal walk beyond simulating one missing-data evening.
    """
    route_vals = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    finite = np.isfinite(route_vals).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        bad_ts = frame.loc[~finite, "timestamp"]
        print(
            f"WARNING {label}: dropping {n_bad} bars with non-finite regime3 route probs "
            f"(pre-existing overlay CSV gap, not introduced by this script) "
            f"range=[{bad_ts.min()}, {bad_ts.max()}]",
            flush=True,
        )
    return frame.loc[finite].reset_index(drop=True)


def run_component_split(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path, *, split: str, oof: bool) -> dict[str, Any]:
    print(f"stage=prep component={name} split={split}", flush=True)
    prepped = sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)

    print(f"stage=replay_actual component={name} split={split}", flush=True)
    m_actual_full, ledger = sweep.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"],
        device=sweep.DEVICE,
    )
    if ledger.empty:
        raise RuntimeError(f"{name}/{split}: empty ledger, cannot quantify sizing bias")

    # Re-price the SAME frozen ledger two ways -- entry_signal_i/exit_i/side/reason never change.
    m_actual_relayered, ledger_actual = rs._ledger_metrics_with_margins(prepped["frame"], ledger, None)
    flat_margins, flat_leverage = _flat_arrays(len(prepped["frame"]))
    m_flat, ledger_flat = rs._ledger_metrics_with_margins(prepped["frame"], ledger, flat_margins, flat_leverage)

    pnl_gap = abs(float(m_actual_relayered["pnl"]) - float(m_actual_full["pnl"]))
    if pnl_gap > PNL_MATCH_TOL_PP:
        raise RuntimeError(
            f"{name}/{split}: ledger-relayered actual pnl {m_actual_relayered['pnl']:.4f} vs "
            f"bar-by-bar replay pnl {m_actual_full['pnl']:.4f} (gap {pnl_gap:.4f}pp) -- rescaling "
            f"method is unsound, refusing to report a sizing-bias number built on it"
        )
    if int(len(ledger_actual)) != int(len(ledger_flat)) or int(len(ledger_actual)) != int(len(ledger)):
        raise RuntimeError(f"{name}/{split}: trade count changed across sizing scenarios -- entry/exit not frozen")
    if not ledger_actual["entry_signal_i"].equals(ledger_flat["entry_signal_i"]) or not ledger_actual["exit_i"].equals(ledger_flat["exit_i"]):
        raise RuntimeError(f"{name}/{split}: entry/exit indices diverged across sizing scenarios")
    if not ledger_actual["reason"].equals(ledger_flat["reason"]) or not ledger_actual["win"].equals(ledger_flat["win"]):
        raise RuntimeError(f"{name}/{split}: exit reason or win/loss direction diverged across sizing scenarios")

    long_mask = ledger["side"] > 0
    short_mask = ledger["side"] < 0
    return {
        "component": name, "split": split, "trades": int(len(ledger)),
        "long_trades": int(long_mask.sum()), "short_trades": int(short_mask.sum()),
        "actual_pnl": float(m_actual_relayered["pnl"]), "actual_mdd": float(m_actual_relayered["mdd"]),
        "actual_avg_margin_fraction": float(ledger["margin_fraction"].mean()),
        "actual_avg_leverage": float(ledger["leverage"].mean()),
        "actual_avg_notional": float(ledger["notional"].mean()),
        "actual_avg_notional_long": float(ledger.loc[long_mask, "notional"].mean()) if long_mask.any() else float("nan"),
        "actual_avg_notional_short": float(ledger.loc[short_mask, "notional"].mean()) if short_mask.any() else float("nan"),
        "flat_pnl": float(m_flat["pnl"]), "flat_mdd": float(m_flat["mdd"]),
        "flat_margin_fraction": FLAT_MARGIN, "flat_leverage": FLAT_LEVERAGE, "flat_notional": FLAT_NOTIONAL,
        "bar_by_bar_pnl_check": float(m_actual_full["pnl"]), "bar_by_bar_mdd_check": float(m_actual_full["mdd"]),
        "pnl_integrity_gap_pp": float(pnl_gap),
        "exit_reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().to_dict().items()},
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    val_frame = _drop_route_nan_gaps(val_frame, label="VAL")
    oos_frame = _drop_route_nan_gaps(oos_frame, label="OOS")
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    print(f"FLAT sizing: notional={FLAT_NOTIONAL} leverage={FLAT_LEVERAGE} margin_fraction={FLAT_MARGIN:.6f} (BASE_TEMPLATE, no EXPERT_SCALES, no side asymmetry)", flush=True)

    rows: list[dict[str, Any]] = []
    for name, cfg in sweep.COMPONENTS.items():
        val_pred = sweep.EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        oos_pred = sweep.EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        rows.append(run_component_split(name, cfg, val_frame, val_pred, split="VAL", oof=True))
        rows.append(run_component_split(name, cfg, oos_frame, oos_pred, split="OOS", oof=False))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sizing_bias_raw.csv", index=False)

    summary_rows = []
    for name in sweep.COMPONENTS:
        val_row = df[(df["component"] == name) & (df["split"] == "VAL")].iloc[0]
        oos_row = df[(df["component"] == name) & (df["split"] == "OOS")].iloc[0]
        val_pnl_margin = float(val_row["actual_pnl"] - val_row["flat_pnl"])
        oos_pnl_margin = float(oos_row["actual_pnl"] - oos_row["flat_pnl"])
        val_mdd_margin = float(val_row["actual_mdd"] - val_row["flat_mdd"])  # positive = actual has shallower (better) MDD than flat
        oos_mdd_margin = float(oos_row["actual_mdd"] - oos_row["flat_mdd"])
        ratio = (val_pnl_margin / oos_pnl_margin) if abs(oos_pnl_margin) > 1e-9 else float("inf") * (1 if val_pnl_margin >= 0 else -1)
        summary_rows.append({
            "component": name,
            "val_actual_pnl": float(val_row["actual_pnl"]), "val_flat_pnl": float(val_row["flat_pnl"]), "val_pnl_dominance_margin": val_pnl_margin,
            "oos_actual_pnl": float(oos_row["actual_pnl"]), "oos_flat_pnl": float(oos_row["flat_pnl"]), "oos_pnl_dominance_margin": oos_pnl_margin,
            "val_actual_mdd": float(val_row["actual_mdd"]), "val_flat_mdd": float(val_row["flat_mdd"]), "val_mdd_dominance_margin": val_mdd_margin,
            "oos_actual_mdd": float(oos_row["actual_mdd"]), "oos_flat_mdd": float(oos_row["flat_mdd"]), "oos_mdd_dominance_margin": oos_mdd_margin,
            "val_over_oos_pnl_margin_ratio": ratio,
            "val_trades": int(val_row["trades"]), "oos_trades": int(oos_row["trades"]),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "sizing_bias_summary.csv", index=False)

    with open(OUT_DIR / "sizing_bias_full.json", "w") as f:
        json.dump({"rows": rows, "summary": summary_rows, "flat_sizing": {"notional": FLAT_NOTIONAL, "leverage": FLAT_LEVERAGE, "margin_fraction": FLAT_MARGIN}}, f, indent=2, default=str)

    pd.set_option("display.width", 200)
    print("\n=== per component/split ===", flush=True)
    print(df[["component", "split", "trades", "actual_pnl", "actual_mdd", "flat_pnl", "flat_mdd", "bar_by_bar_pnl_check", "pnl_integrity_gap_pp"]].to_string(index=False), flush=True)
    print("\n=== dominance margin summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nstage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
