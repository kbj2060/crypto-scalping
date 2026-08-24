#!/usr/bin/env python3
"""DIAGNOSTIC ONLY -- Odyssey2 #22: intervention-surface + hindsight-ceiling audit for the whole
"evidence signal -> live Omega4.6.1 post-entry injection" candidate class.

=== Why this exists ===
Four independent injections of the cross-window-validated ETH reversal evidence signals into the
live Omega4.6.1 model have now failed (contract #18 hard exit veto: VAL -29.92pp; #19 exit_head
feature: stage-0 diagnostic reversed on VAL, retraining withheld; #20 sizing-sidecar GBM feature:
both components negative; #21 soft exit-threshold relaxation: VAL winners were byte-identical
no-ops, the one real intervention lost VAL by -18.18pp, single-touch OOS rejected). A standalone
rule built from the same signals also lost every one of 36 window x K cells to always_long/
always_short.

Rather than attempt a 5th variant, this script asks the structural question those four experiments
never measured: **how much of the live portfolio can this signal class even touch, and what is the
absolute best any exit-timing overlay driven by it could achieve?** It computes, per window:

  (a) INTERVENTION SURFACE -- how many baseline trades have >=1 counter-evidence fire while open
      (counter-evidence = bottom-side orthogonal_combo for a SHORT, top-side for a LONG), broken
      down by component/side, plus the share of portfolio PnL those trades carry. #18/#21 only ever
      scoped h48qual SHORT; this audit deliberately widens to BOTH components and BOTH sides, so
      "would a differently-scoped variant have more surface?" is answered with numbers rather than
      assumed.

  (b) HINDSIGHT CEILING -- with the trade population frozen (same entries), replace each touched
      trade's exit with the best of {its actual exit} U {every fire bar while it was open}, chosen
      with perfect hindsight. That is an upper bound no learnable overlay can exceed, because a real
      overlay must decide at the fire bar without knowing the outcome. If the ceiling is small, the
      entire exit-overlay axis is dead on arithmetic, independent of how cleverly the trigger is
      filtered. Also reported: "always act on the first fire" (the trade-frozen analogue of #18) and
      the help/hurt split of individual fires (how selective an overlay would have to be).

The frozen-population caveat is explicit and load-bearing: exiting early frees a portfolio slot and
would change SUBSEQUENT entries (#18 saw trades 26->28 in VAL). This audit deliberately does NOT
model that channel -- it isolates and bounds the "change the exit of an already-open trade" channel,
which is exactly the channel #18/#21 targeted. The re-entry channel is already measured empirically
by #18/#21's own full replays and is reported there.

=== What this is NOT ===
Not a candidate, not a promotion basis, not a new intervention. It produces no tradeable variant and
proposes no threshold. It is an accounting/power audit of a class of interventions, in the same
"stage-0 diagnostic before committing effort" tradition as
docs/experiments/eth_omega461_evidence_signal_exit_head_feature_rank_correlation_20260814.md.

=== Compliance ===
The baseline ledger analyzed here is produced INSIDE this run by an unmodified causal bar-by-bar
forward replay (gate.run_portfolio_variant -> greedy.greedy_replay); no pre-existing saved ledger,
no saved parent exit timestamp, and no future row is used to make any decision -- the hindsight
counterfactual is explicitly labelled a CEILING (an upper bound computed with future information ON
PURPOSE), never a strategy result, exactly as the repo's Fresh-Forward rule requires oracle
constructions to be treated. fresh_forward_bar_by_bar=true (for the baseline replay whose ledger is
audited), trade_ledgers_used_as_input=false (the ledger is this run's own output, consumed only for
accounting), saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
No retraining, no GPU beyond what the existing replay uses, no new hyperparameter, no threshold
search. Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py /
.env. Imports and reads (never edits) eth_omega461_multiwindow_confirmation_gate_20260814.py,
research_eth_omega461_evidence_veto_exit_overlay_20260814.py (for its already-validated signal
plumbing), replay_omega4_6_1_greedy_router_20260706.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
research_eth_omega461_live_sltp_mfe_width_20260813.py, research_eth_omega461_exit_sweep_20260721.py.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_evidence_veto_exit_overlay_20260814 as veto  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_evidence_intervention_surface_ceiling_20260815"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
RETURN_RECON_TOL = 1.0e-9
RANDOM_CONTROL_REPS = 20
RANDOM_CONTROL_SEED = 20260815

# Signal constants: reused verbatim from veto (#18), which in turn reused them verbatim from the
# evidence-study lineage. The top side uses the same definition mirrored, exactly as
# analyze_eth_creative_reversal_evidence_signals_20260814.build_signals() writes it:
#   bottom: (p_fast<=0.10)&(p_slow<=0.10)&(delta_z<=-2.0)
#   top:    (p_fast>=0.90)&(p_slow>=0.90)&(delta_z>=+2.0)
OSC_LOW, OSC_HIGH = veto.OSCILLATOR_DECILE, 1.0 - veto.OSCILLATOR_DECILE
DZ_LOW, DZ_HIGH = veto.DELTA_Z_THRESHOLD, -veto.DELTA_Z_THRESHOLD


def log(msg: str) -> None:
    print(f"[surface_ceiling] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _evidence_scores(base_csv: Path) -> pd.DataFrame:
    """Both sides of orthogonal_combo on a full year CSV (never window-by-window, so a window
    starting mid-year is not artificially NaN-truncated at its own start). Same loading/indicator
    path as veto._evidence_veto_score, extended with the top side."""
    df = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ind = add_creative_indicators(compute_indicators(df))
    out = ind[["timestamp"]].copy()
    out["evi_bottom"] = ((ind["p_fast"] <= OSC_LOW) & (ind["p_slow"] <= OSC_LOW) & (ind["delta_z"] <= DZ_LOW)).fillna(False).to_numpy(dtype=bool)
    out["evi_top"] = ((ind["p_fast"] >= OSC_HIGH) & (ind["p_slow"] >= OSC_HIGH) & (ind["delta_z"] >= DZ_HIGH)).fillna(False).to_numpy(dtype=bool)
    return out


def _masks_for_frame(aligned_frame: pd.DataFrame, window_name: str, score_by_base: dict[Path, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    score = score_by_base[gate.WINDOW_DEFS[window_name]["base_csv"]]
    merged = aligned_frame[["timestamp"]].merge(score, on="timestamp", how="left")
    if len(merged) != len(aligned_frame) or not merged["timestamp"].equals(aligned_frame["timestamp"]):
        raise RuntimeError(f"{window_name}: evidence score merge failed (row count/order mismatch)")
    return (merged["evi_bottom"].fillna(False).to_numpy(dtype=bool),
            merged["evi_top"].fillna(False).to_numpy(dtype=bool))


def _trade_return_at(open_arr: np.ndarray, close_arr: np.ndarray, *, entry_i: int, exit_i: int, side: int,
                     notional: float, fee_eff: float, slip_eff: float) -> float:
    """Exact algebraic reconstruction of greedy.greedy_replay's own per-trade return accounting:
        entry_price = open[entry_i] * (1 +- slip_eff)          (+ for long, - for short)
        exit_price  = close[exit_i] * (1 -+ slip_eff)
        raw         = (exit-entry)/entry (long) or (entry-exit)/entry (short)
        cash path   = E -> E(1-fee*N) -> E(1-fee*N)(1+raw*N-fee*N)
    Verified against the replay's own ledger for every baseline trade in stage G0c before any
    counterfactual below is trusted."""
    entry_price = float(open_arr[entry_i]) * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    exit_px = float(close_arr[exit_i]) * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    raw = (exit_px - entry_price) / entry_price if side > 0 else (entry_price - exit_px) / entry_price
    return float((1.0 - fee_eff * notional) * (1.0 + raw * notional - fee_eff * notional) - 1.0)


def _portfolio_metrics(ledger: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    return {"no_gate": portfolio._ledger_metrics(ledger),
            "with_gate": mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)}


def _audit_window(wname: str, windows: dict[str, Any], score_by_base: dict[Path, pd.DataFrame],
                  fee: float, slip: float) -> dict[str, Any]:
    fee_eff, slip_eff = float(fee) * float(sweep.COST_MULT), float(slip) * float(sweep.COST_MULT)
    result = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR,
                                        fee=fee, slip=slip, device=DEVICE, out_dir=OUT_DIR,
                                        variant_label="asymmetric_tabm_liveatr")
    frame = result["aligned_frame"]
    ledger = pd.read_csv(result["ledger_path"])
    open_arr = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    m_bottom, m_top = _masks_for_frame(frame, wname, score_by_base)

    # ---- G0c: per-trade return reconstruction must be exact before any counterfactual is trusted ----
    recon = np.array([
        _trade_return_at(open_arr, close_arr, entry_i=int(r.entry_i), exit_i=int(r.exit_i), side=int(r.side),
                         notional=float(r.notional), fee_eff=fee_eff, slip_eff=slip_eff)
        for r in ledger.itertuples()
    ]) if len(ledger) else np.array([])
    recon_max_abs_err = float(np.abs(recon - ledger["trade_return"].to_numpy(dtype=np.float64)).max()) if len(ledger) else 0.0

    # ---- per-trade surface + counterfactual exits ----
    rows: list[dict[str, Any]] = []
    for r in ledger.itertuples():
        entry_i, exit_i, side = int(r.entry_i), int(r.exit_i), int(r.side)
        mask = m_bottom if side < 0 else m_top          # counter-evidence for the position's direction
        fires = np.flatnonzero(mask[entry_i:exit_i]) + entry_i if exit_i > entry_i else np.array([], dtype=int)
        base_ret = float(r.trade_return)
        cand = {int(f): _trade_return_at(open_arr, close_arr, entry_i=entry_i, exit_i=int(f), side=side,
                                         notional=float(r.notional), fee_eff=fee_eff, slip_eff=slip_eff)
                for f in fires}
        first_fire = int(fires[0]) if len(fires) else None
        best_fire = max(cand, key=cand.get) if cand else None
        rows.append({
            "entry_timestamp": r.entry_timestamp, "exit_timestamp": r.exit_timestamp,
            "component": r.source_component, "side": side, "reason": r.reason,
            "entry_i": entry_i, "exit_i": exit_i, "hold_bars": exit_i - entry_i,
            "baseline_return": base_ret, "n_fires": int(len(fires)), "touched": bool(len(fires) > 0),
            "first_fire_i": first_fire,
            "first_fire_return": float(cand[first_fire]) if first_fire is not None else None,
            "best_fire_i": best_fire,
            "best_fire_return": float(cand[best_fire]) if best_fire is not None else None,
            "n_fires_that_help": int(sum(1 for v in cand.values() if v > base_ret)),
            "n_fires_that_hurt": int(sum(1 for v in cand.values() if v <= base_ret)),
        })
    trades = pd.DataFrame(rows)

    # ---- counterfactual ledgers (trade population frozen; only exits move) ----
    def _rebuild(kind: str) -> pd.DataFrame:
        led = ledger.copy()
        if not len(trades):
            return led
        for idx, t in trades.iterrows():
            if not t["touched"]:
                continue
            if kind == "always_first_fire":
                new_i, new_ret = t["first_fire_i"], t["first_fire_return"]
            else:  # hindsight_best -- upper bound: keep baseline unless a fire bar beats it
                if t["best_fire_return"] is None or t["best_fire_return"] <= t["baseline_return"]:
                    continue
                new_i, new_ret = t["best_fire_i"], t["best_fire_return"]
            led.at[idx, "exit_i"] = int(new_i)
            led.at[idx, "exit_timestamp"] = str(frame["timestamp"].iloc[int(new_i)])
            led.at[idx, "trade_return"] = float(new_ret)
            led.at[idx, "reason"] = "counterfactual_evidence_exit"
            led.at[idx, "win"] = int(new_ret > 0)
        return led

    variants = {"baseline": ledger, "always_first_fire": _rebuild("always_first_fire"),
                "hindsight_best_fire": _rebuild("hindsight_best")}
    metrics = {k: _portfolio_metrics(v, frame) for k, v in variants.items()}
    for k, v in variants.items():
        if k != "baseline":
            v.to_csv(OUT_DIR / f"counterfactual_ledger_{wname}_{k}.csv", index=False)
    trades.to_csv(OUT_DIR / f"trade_surface_{wname}.csv", index=False)

    # ---- MATCHED RANDOM CONTROL (mandatory: a hindsight max over many candidate bars is large for
    # ANY candidate set, so the fire-bar ceiling is uninterpretable on its own). Per replicate, each
    # touched trade gets the SAME NUMBER of candidate bars as it had fires, drawn uniformly without
    # replacement from its own hold window [entry_i, exit_i) -- identical count, identical support,
    # only the bar SELECTION differs. Any excess of the fire ceiling over this control is the only
    # part attributable to the evidence signal itself. Seed is fixed and reported. ----
    ctrl_ceiling_ng, ctrl_ceiling_wg, ctrl_first_ng, ctrl_first_wg, ctrl_help, ctrl_total = [], [], [], [], [], []
    ctrl_cond_gain: list[float] = []  # mean per-trade gain of best candidate over baseline, conditional on >0
    for rep in range(RANDOM_CONTROL_REPS):
        rng = np.random.default_rng(RANDOM_CONTROL_SEED + rep)
        led_best, led_first = ledger.copy(), ledger.copy()
        help_n = tot_n = 0
        rep_gains: list[float] = []
        for idx, t in trades.iterrows():
            if not t["touched"]:
                continue
            lo, hi, k = int(t["entry_i"]), int(t["exit_i"]), int(t["n_fires"])
            pool = np.arange(lo, hi)
            pick = pool if k >= len(pool) else rng.choice(pool, size=k, replace=False)
            rets = {int(b): _trade_return_at(open_arr, close_arr, entry_i=lo, exit_i=int(b), side=int(t["side"]),
                                             notional=float(ledger.at[idx, "notional"]), fee_eff=fee_eff, slip_eff=slip_eff)
                    for b in pick}
            base_ret = float(t["baseline_return"])
            help_n += sum(1 for v in rets.values() if v > base_ret)
            tot_n += len(rets)
            b_first = int(min(rets))
            led_first.at[idx, "exit_i"], led_first.at[idx, "trade_return"] = b_first, float(rets[b_first])
            b_best = max(rets, key=rets.get)
            if rets[b_best] > base_ret:
                led_best.at[idx, "exit_i"], led_best.at[idx, "trade_return"] = b_best, float(rets[b_best])
                rep_gains.append(float(rets[b_best] - base_ret))
        ctrl_cond_gain.append(float(np.mean(rep_gains)) if rep_gains else 0.0)
        m_best, m_first = _portfolio_metrics(led_best, frame), _portfolio_metrics(led_first, frame)
        ctrl_ceiling_ng.append(m_best["no_gate"]["pnl"]); ctrl_ceiling_wg.append(m_best["with_gate"]["pnl"])
        ctrl_first_ng.append(m_first["no_gate"]["pnl"]); ctrl_first_wg.append(m_first["with_gate"]["pnl"])
        ctrl_help.append(help_n); ctrl_total.append(tot_n)
    random_control = {
        "reps": RANDOM_CONTROL_REPS, "seed": RANDOM_CONTROL_SEED,
        "ceiling_with_gate_mean": float(np.mean(ctrl_ceiling_wg)), "ceiling_with_gate_std": float(np.std(ctrl_ceiling_wg)),
        "ceiling_no_gate_mean": float(np.mean(ctrl_ceiling_ng)), "ceiling_no_gate_std": float(np.std(ctrl_ceiling_ng)),
        "always_first_with_gate_mean": float(np.mean(ctrl_first_wg)), "always_first_no_gate_mean": float(np.mean(ctrl_first_ng)),
        "help_rate_mean": float(np.mean(ctrl_help) / np.mean(ctrl_total)) if np.mean(ctrl_total) > 0 else 0.0,
        "candidate_bars_mean": float(np.mean(ctrl_total)),
        "conditional_gain_mean": float(np.mean(ctrl_cond_gain)),
    }
    # Direct decomposition of the ceiling into P(an improving exit exists) x E[size of that
    # improvement] -- measured, not inferred from the ceiling gap.
    _improving = trades[trades["touched"] & trades.apply(lambda t: t["best_fire_return"] is not None and t["best_fire_return"] > t["baseline_return"], axis=1)] if len(trades) else trades
    fire_cond_gain = float((_improving["best_fire_return"] - _improving["baseline_return"]).mean()) if len(_improving) else 0.0

    touched = trades[trades["touched"]] if len(trades) else trades
    tot_abs = float(trades["baseline_return"].abs().sum()) if len(trades) else 0.0
    by_group: dict[str, Any] = {}
    if len(trades):
        for (comp, sd), g in trades.groupby(["component", "side"]):
            by_group[f"{comp}_{'L' if sd > 0 else 'S'}"] = {
                "trades": int(len(g)), "touched": int(g["touched"].sum()),
                "fires": int(g["n_fires"].sum()),
                "fires_help": int(g["n_fires_that_help"].sum()), "fires_hurt": int(g["n_fires_that_hurt"].sum()),
            }
    return {
        "tier": gate.WINDOW_DEFS[wname]["tier"],
        "n_bars": int(len(frame)),
        "signal_activation": {"bottom_bars": int(m_bottom.sum()), "top_bars": int(m_top.sum()),
                              "bottom_rate": float(m_bottom.mean()), "top_rate": float(m_top.mean())},
        "return_reconstruction_max_abs_err": recon_max_abs_err,
        "return_reconstruction_exact": bool(recon_max_abs_err < RETURN_RECON_TOL),
        "surface": {
            "trades": int(len(trades)),
            "trades_touched": int(touched["touched"].sum()) if len(trades) else 0,
            "touched_frac": float(touched["touched"].sum() / len(trades)) if len(trades) else 0.0,
            "total_fires_in_position": int(trades["n_fires"].sum()) if len(trades) else 0,
            "fires_help": int(trades["n_fires_that_help"].sum()) if len(trades) else 0,
            "fires_hurt": int(trades["n_fires_that_hurt"].sum()) if len(trades) else 0,
            "touched_abs_return_share": float(touched["baseline_return"].abs().sum() / tot_abs) if tot_abs > 0 else 0.0,
            "by_component_side": by_group,
        },
        "metrics": metrics,
        "random_control": random_control,
        "signal_excess_over_random": {
            "ceiling_with_gate_pp": float(metrics["hindsight_best_fire"]["with_gate"]["pnl"] - random_control["ceiling_with_gate_mean"]),
            "ceiling_no_gate_pp": float(metrics["hindsight_best_fire"]["no_gate"]["pnl"] - random_control["ceiling_no_gate_mean"]),
            "help_rate_fire": float(trades["n_fires_that_help"].sum() / max(trades["n_fires"].sum(), 1)) if len(trades) else 0.0,
            "help_rate_random": random_control["help_rate_mean"],
            "trades_with_improving_exit_fire": int(len(_improving)),
            "conditional_gain_fire": fire_cond_gain,
            "conditional_gain_random": random_control["conditional_gain_mean"],
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 #22 -- intervention-surface + hindsight-ceiling audit of the evidence-signal "
            "post-entry injection class. Measures (a) how many live-baseline trades a counter-evidence "
            "signal can touch at all (both components, both sides -- wider than #18/#21's h48qual-SHORT "
            "scope) and (b) the frozen-population upper bound on any exit-timing overlay driven by it "
            "(per touched trade, best of actual exit U fire bars, chosen with perfect hindsight). "
            "Diagnostic only -- no candidate, no promotion basis."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "hindsight_ceiling_is_an_upper_bound_not_a_strategy": True,
        "frozen_population_caveat": (
            "Counterfactual exits do not free portfolio slots, so subsequent entries are held fixed. "
            "This isolates the 'change the exit of an already-open trade' channel (what #18/#21 "
            "targeted); the re-entry channel is measured empirically by #18/#21's own full replays."
        ),
        "signal": {
            "bottom": f"(p_fast<={OSC_LOW})&(p_slow<={OSC_LOW})&(delta_z<={DZ_LOW})",
            "top": f"(p_fast>={OSC_HIGH})&(p_slow>={OSC_HIGH})&(delta_z>={DZ_HIGH})",
            "provenance": "reused verbatim from research_eth_omega461_evidence_veto_exit_overlay_20260814 / analyze_eth_creative_reversal_evidence_signals_20260814 -- no new threshold, no search",
        },
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=G0a_reference_reproduction ===")
    g0a: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        res = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR,
                                         fee=fee, slip=slip, device=DEVICE, out_dir=OUT_DIR,
                                         variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = veto.G0_REQUIRED[wname]
        ok = _close(res["no_gate"], ref_ng) and _close(res["with_gate"], ref_wg)
        g0a[wname] = {"no_gate": res["no_gate"], "with_gate": res["with_gate"],
                      "reference_no_gate": ref_ng, "reference_with_gate": ref_wg, "match": ok}
        log(f"  {wname}: no_gate={res['no_gate']['pnl']:.2f}%/{res['no_gate']['mdd']:.2f}%/{res['no_gate']['trades']} "
            f"with_gate={res['with_gate']['pnl']:.2f}%/{res['with_gate']['mdd']:.2f}%/{res['with_gate']['trades']} match={ok}")
    g0a_pass = all(v["match"] for v in g0a.values())
    report["g0a_reference_reproduction"] = {"windows": g0a, "pass": g0a_pass}
    log(f"stage=G0a_result pass={g0a_pass}")
    if not g0a_pass:
        report["gate_pass"] = False
        report["note"] = "G0a failed -- baseline reference numbers not reproduced. Aborting before trusting any surface/ceiling number."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
        return 1

    log("=== stage=signal_build ===")
    score_by_base = {sweep.BASE_2025: _evidence_scores(sweep.BASE_2025), sweep.BASE_2026: _evidence_scores(sweep.BASE_2026)}

    log("=== stage=surface_and_ceiling (6 windows) ===")
    by_window: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        by_window[wname] = _audit_window(wname, windows, score_by_base, fee, slip)
        w = by_window[wname]
        s, m = w["surface"], w["metrics"]
        log(f"  {wname:8s} tier={w['tier']:11s} bars={w['n_bars']:6d} evi_bottom={w['signal_activation']['bottom_bars']:4d} evi_top={w['signal_activation']['top_bars']:4d} "
            f"recon_exact={w['return_reconstruction_exact']}")
        log(f"           trades={s['trades']:3d} touched={s['trades_touched']:3d} ({s['touched_frac'] * 100:5.1f}%) fires_in_pos={s['total_fires_in_position']:4d} "
            f"help={s['fires_help']:3d} hurt={s['fires_hurt']:3d} touched_abs_ret_share={s['touched_abs_return_share'] * 100:5.1f}%")
        log(f"           with_gate PnL: baseline={m['baseline']['with_gate']['pnl']:8.2f}%  always_first_fire={m['always_first_fire']['with_gate']['pnl']:8.2f}%  "
            f"HINDSIGHT_CEILING={m['hindsight_best_fire']['with_gate']['pnl']:8.2f}%")
        log(f"           no_gate   PnL: baseline={m['baseline']['no_gate']['pnl']:8.2f}%  always_first_fire={m['always_first_fire']['no_gate']['pnl']:8.2f}%  "
            f"HINDSIGHT_CEILING={m['hindsight_best_fire']['no_gate']['pnl']:8.2f}%")
        rc, ex = w["random_control"], w["signal_excess_over_random"]
        log(f"           RANDOM CONTROL ({rc['reps']} reps, matched bar counts): ceiling with_gate={rc['ceiling_with_gate_mean']:8.2f}% (sd {rc['ceiling_with_gate_std']:.2f}) "
            f"always_first={rc['always_first_with_gate_mean']:8.2f}%  help_rate={rc['help_rate_mean'] * 100:5.1f}%")
        log(f"           SIGNAL EXCESS over random: ceiling_with_gate={ex['ceiling_with_gate_pp']:+8.2f}pp  help_rate fire={ex['help_rate_fire'] * 100:5.1f}% vs random={ex['help_rate_random'] * 100:5.1f}%  "
            f"cond_gain fire={ex['conditional_gain_fire'] * 100:5.2f}% vs random={ex['conditional_gain_random'] * 100:5.2f}% (n_improving={ex['trades_with_improving_exit_fire']})")
        log(f"           by component/side: {json.dumps(s['by_component_side'])}")
    report["by_window"] = by_window

    recon_all_exact = all(w["return_reconstruction_exact"] for w in by_window.values())
    report["g0c_return_reconstruction_all_windows_exact"] = recon_all_exact
    oos = ("oos_q1", "oos_q2")
    report["headline"] = {
        "oos_confirm_touched_trades": {w: by_window[w]["surface"]["trades_touched"] for w in oos},
        "oos_confirm_trades": {w: by_window[w]["surface"]["trades"] for w in oos},
        "oos_confirm_ceiling_gain_pp_with_gate": {
            w: float(by_window[w]["metrics"]["hindsight_best_fire"]["with_gate"]["pnl"] - by_window[w]["metrics"]["baseline"]["with_gate"]["pnl"])
            for w in oos},
        "all_windows_ceiling_gain_pp_with_gate": {
            w: float(by_window[w]["metrics"]["hindsight_best_fire"]["with_gate"]["pnl"] - by_window[w]["metrics"]["baseline"]["with_gate"]["pnl"])
            for w in gate.ALL_WINDOWS},
        "all_windows_always_first_fire_delta_pp_with_gate": {
            w: float(by_window[w]["metrics"]["always_first_fire"]["with_gate"]["pnl"] - by_window[w]["metrics"]["baseline"]["with_gate"]["pnl"])
            for w in gate.ALL_WINDOWS},
        "signal_excess_over_random_ceiling_pp_with_gate": {
            w: by_window[w]["signal_excess_over_random"]["ceiling_with_gate_pp"] for w in gate.ALL_WINDOWS},
        "help_rate_fire_vs_random": {
            w: {"fire": by_window[w]["signal_excess_over_random"]["help_rate_fire"],
                "random": by_window[w]["signal_excess_over_random"]["help_rate_random"]} for w in gate.ALL_WINDOWS},
        "fires_help_hurt_total": {
            "help": int(sum(by_window[w]["surface"]["fires_help"] for w in gate.ALL_WINDOWS)),
            "hurt": int(sum(by_window[w]["surface"]["fires_hurt"] for w in gate.ALL_WINDOWS))},
    }
    report["gate_pass"] = bool(recon_all_exact)
    report["stage_reached"] = "surface_and_ceiling"
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done recon_all_exact={recon_all_exact}")
    log(f"HEADLINE {json.dumps(report['headline'], default=omega._json_default)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
