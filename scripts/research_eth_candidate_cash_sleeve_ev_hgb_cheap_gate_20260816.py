#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate (ORACLE / HINDSIGHT measurement, NOT a causal backtest and NOT a
model) for a candidate "cash-sleeve EV-HGB" mechanism on ETH's primary (h48qual/zig075 ThreeHeadTabM
system + Odyssey4's locked entry-veto/regime-guard layer).

=== What this is porting ===
A prior investigation found BTC has a live-tested (but currently unwired/dead-code) mechanism: two
independent sklearn HistGradientBoostingRegressor models (long_model/short_model) that only fire
when the PRIMARY model is flat (CASH, no open position), predicting simulated fallback-trade EV.
This script does NOT train anything -- it only asks the question a cheap_gate must answer BEFORE any
training is justified: on ETH's real VAL/OOS CASH bars, is there enough cost-inclusive, hindsight
edge for an EV-HGB model to even have something to learn?

=== ORACLE / HINDSIGHT framing (read before trusting any number below) ===
Every "long_net"/"short_net" figure here is computed with PERFECT knowledge of each bar's own
future price path (up to primary_takeover/max_hold) -- no model, causal or otherwise, produces
these numbers. They exist ONLY to upper-bound how much edge would even be there for a real EV-HGB
model to try to capture. Nothing in this script is a promotion claim, a strategy result, or a
substitute for the Fresh-Forward Validation/OOS/Test Rule's causal walk-forward requirement -- an
actual EV-HGB candidate would still need real bar-by-bar causal training/inference and its own
walk-forward test before any promotion claim.

=== Primary definition (reused, not reimplemented) ===
"Primary is in CASH at bar i" is read directly off the actual Odyssey4-baseline greedy account-level
replay ledger (entry_i/exit_i integer bar indices) -- NOT re-derived from raw model outputs, so it
automatically reflects every real constraint (single shared position slot, h48qual>zig075 priority,
margin/leverage caps, the locked regime-aware exit guard on h48qual, and the locked zig075 SHORT
entry veto during detected sustained uptrends -- Odyssey4's own G0 baseline, see
docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md). This script imports and
calls, unmodified:
  - eth_omega461_multiwindow_confirmation_gate_20260814 (window definitions/loading)
  - research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 (h48qual regime-aware exit
    guard, component prep, detector)
  - research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814
    (greedy_replay_entry_veto -- the exact locked Odyssey4 baseline replay function)
None of those modules are edited. A per-bar `held` boolean array is reconstructed from the returned
ledger's entry_i/exit_i columns (both inclusive -- see docstring of `_held_mask` for why), which is
exactly the same as re-running the account-level simulation bar-by-bar, just cheaper.

=== Fallback-trade oracle simulation (this script's only new logic) ===
For every bar i where the primary holds no position ("CASH"), simulate BOTH a long and a short
fallback trade: entry at bar i+1's open (with slippage, matching every other replay engine in this
lineage), exit at the first of:
  - take_profit / stop_loss barrier, checked bar-by-bar against each bar's CLOSE price (NOT
    intrabar high/low) -- this mirrors replay_omega4_6_1_greedy_router_20260706.greedy_replay's own
    barrier-check convention exactly, so the fallback trade is judged by the same rule the live
    system's own TP/SL barriers use, not a more favorable intrabar-touch rule.
  - max_hold_bars (192)
  - "primary_takeover": the bar before the primary's `held` array goes True again (the account only
    has one position slot; the fallback must have vacated it by the time the primary needs it).
Costs are stressed at 3x normal fee/slip (COST_STRESS_MULT), matching the BTC mechanism's own
design. Fee/slip constants (FEE_RATE=0.0005, SLIP_RATE=0.0002) are this repo's real ETH backtest
constants, read via train_eval_omega1_2_tabm_diffusion_risk_20260603._load_fee_slip() (same source
every other ETH Omega4.6.1/Odyssey script in this lineage uses -- not redefined here).

Fixed risk profile (given, NOT modified per the earlier BTC survey's own recommendation):
  take_profit=0.026 (price move), stop_loss=0.014 (price move), notional=0.405, leverage=2.0
  (=> margin_fraction = notional/leverage = 0.2025, Futures Risk Sizing Contract identity check),
  max_hold_bars=192, ev_min=0.002.

=== Window boundaries -- deviation from CLAUDE.md's generic default, logged explicitly per its own
escape clause ("날짜 경계가 바뀌면 리포트에 명시해야 한다") ===
CLAUDE.md's generic Fresh-Forward default is VAL=2025-09-01..12-31 / OOS=2026-01-01..03-31. This
script uses VAL=2025-10-01..12-31 / OOS(=oos_q1)=2026-01-01..03-31 instead -- OOS matches the
generic default exactly, but VAL's start is one month later. Reason: the h48qual/zig075 ThreeHeadTabM
parent artifact's own train/validation split boundary is SPLIT_TS=2025-10-01 (see
train_eval_omega1_2_tabm_3head_20260603.py:33) -- September 2025 is INSIDE that model's TRAIN split,
so its "validation_predictions_*.csv" simply does not exist for Sep 2025 (using Sep data would mean
silently falling back to in-sample/train-split predictions, which this repo's own convention
already segregates into the context-only "2025q3" tier, never a val/oos gating tier). Using the
already-established, already-confirmed val/oos_q1 windows this specific sub-project relies on
throughout (eth_omega461_multiwindow_confirmation_gate_20260814.WINDOW_DEFS) avoids injecting
in-sample bars under the "validation" label, which would be a worse violation of the Fresh-Forward
rule's own intent than a one-month boundary shift.

fresh_forward_bar_by_bar=true (primary ledger comes from an unmodified single causal bar-by-bar
replay; the fallback-trade oracle walk is itself bar-by-bar forward from each CASH bar, though it
uses each bar's own REALIZED future path by design -- an oracle, stated as such throughout, not a
causal signal). trade_ledgers_used_as_input=false (the primary ledger informs ONLY which bars are
CASH -- a structural fact about the account, not a per-trade signal fed into any decision).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false for the primary; the
oracle fallback simulation explicitly and admittedly DOES use each bar's own future price path (that
is the entire point of an oracle upper bound) -- never claimed otherwise.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
Does NOT modify any imported module. No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
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
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816"
DEVICE = guard.DEVICE

# --- fixed risk profile (given, unmodified) ---
TAKE_PROFIT = 0.026
STOP_LOSS = 0.014
NOTIONAL = 0.405
LEVERAGE = 2.0
MARGIN_FRACTION = NOTIONAL / LEVERAGE
MAX_HOLD_BARS = 192
EV_MIN = 0.002
COST_STRESS_MULT = 3.0

WINDOWS = ("val", "oos_q1")
G0_TOLERANCE_PP = 0.05

# Odyssey4-baseline reference numbers (no_gate) for the entry-veto-applied replay, copied verbatim
# from docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md's G0 table --
# sanity check only, proves this script wired the exact locked baseline engine, not re-derived here.
G0_ODYSSEY4_NO_GATE = {
    "val": {"pnl": 41.13, "mdd": -21.70, "trades": 35},
    "oos_q1": {"pnl": 93.27, "mdd": -15.48, "trades": 24},
}


def log(msg: str) -> None:
    print(f"[cash_sleeve_cheap_gate] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def build_primary_ledger(wname: str, windows: dict[str, Any], score_by_base, threshold: float, fee: float, slip: float):
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, DEVICE)
    mask, n_nan = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
    veto_components = veto_mod._attach_veto_mask(components, mask)
    diag, ledger = veto_mod.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=1.0, device=DEVICE)
    return aligned_frame, ledger, diag


def _held_mask(n: int, ledger: pd.DataFrame) -> np.ndarray:
    """True at bar i iff the primary held an open position DURING bar i's price action.
    replay_omega4_6_1_greedy_router_20260706.greedy_replay sets `pos` nonzero starting the iteration
    AFTER the signal bar (entry executes at entry_i = signal_i + 1's open) and keeps `pos` nonzero
    through the exit bar inclusive (the exit itself is decided using that bar's close, then `pos`
    resets) -- so [entry_i, exit_i] inclusive is exactly the held range; entry_signal_i itself is
    correctly a CASH bar (no position was open yet during it)."""
    held = np.zeros(n, dtype=bool)
    for row in ledger.itertuples():
        held[int(row.entry_i):int(row.exit_i) + 1] = True
    return held


def _next_held_bar(held: np.ndarray) -> np.ndarray:
    n = len(held)
    nxt = np.full(n, n, dtype=np.int64)
    running = n
    for i in range(n - 1, -1, -1):
        if held[i]:
            running = i
        nxt[i] = running
    return nxt


def _simulate_side(arrays: dict[str, np.ndarray], entry_i: int, side: int, fee_eff: float, slip_eff: float,
                    hold_end: int, capped_reason: str) -> dict[str, Any]:
    open_, close = arrays["open"], arrays["close"]
    entry_price = open_[entry_i] * (1 + slip_eff if side > 0 else 1 - slip_eff)
    cash_after_entry = 1.0 - fee_eff * NOTIONAL
    raw_exit = 0.0
    reason = capped_reason
    exit_j = hold_end
    for j in range(entry_i, hold_end + 1):
        c = close[j]
        move = (c * (1 - slip_eff) - entry_price) / entry_price if side > 0 else (entry_price - c * (1 + slip_eff)) / entry_price
        if move >= TAKE_PROFIT:
            reason, raw_exit, exit_j = "take_profit", move, j
            break
        if move <= -abs(STOP_LOSS):
            reason, raw_exit, exit_j = "stop_loss", move, j
            break
    else:
        c = close[hold_end]
        raw_exit = (c * (1 - slip_eff) - entry_price) / entry_price if side > 0 else (entry_price - c * (1 + slip_eff)) / entry_price
    cash_final = cash_after_entry * (1.0 + raw_exit * NOTIONAL)
    cash_final -= cash_after_entry * fee_eff * NOTIONAL
    net_return = cash_final - 1.0
    return {"net_return": float(net_return), "reason": reason, "exit_j": int(exit_j), "hold_bars": int(exit_j - entry_i + 1)}


def run_cash_sleeve_oracle(aligned_frame: pd.DataFrame, held: np.ndarray, fee: float, slip: float) -> tuple[pd.DataFrame, int]:
    arrays = {c: pd.to_numeric(aligned_frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(aligned_frame)
    valid_n = n - 2  # same bar range replay_omega4_6_1_greedy_router_20260706.greedy_replay itself evaluates (range(0, n-2))
    fee_eff, slip_eff = fee * COST_STRESS_MULT, slip * COST_STRESS_MULT
    nxt_held = _next_held_bar(held)

    records: list[dict[str, Any]] = []
    for i in range(0, valid_n):
        if held[i]:
            continue
        entry_i = i + 1
        if entry_i >= n:
            continue
        k = int(nxt_held[entry_i])
        time_cap = min(entry_i + MAX_HOLD_BARS - 1, n - 1)
        takeover_cap = k - 1
        if takeover_cap < entry_i:
            hold_end, capped_reason = entry_i, "primary_takeover"
        elif takeover_cap < time_cap:
            hold_end, capped_reason = takeover_cap, "primary_takeover"
        else:
            hold_end, capped_reason = time_cap, "max_hold_bars"

        long_res = _simulate_side(arrays, entry_i, +1, fee_eff, slip_eff, hold_end, capped_reason)
        short_res = _simulate_side(arrays, entry_i, -1, fee_eff, slip_eff, hold_end, capped_reason)
        records.append({
            "i": i, "timestamp": str(aligned_frame["timestamp"].iloc[i]),
            "long_net": long_res["net_return"], "long_reason": long_res["reason"],
            "short_net": short_res["net_return"], "short_reason": short_res["reason"],
            "hold_bars": long_res["hold_bars"],
        })
    return pd.DataFrame(records), valid_n


def summarize_window(wname: str, cash_df: pd.DataFrame, valid_n: int) -> dict[str, Any]:
    n_cash = len(cash_df)
    cash_frac = n_cash / valid_n if valid_n else 0.0
    if n_cash == 0:
        return {"window": wname, "valid_bars": valid_n, "cash_bars": 0, "cash_frac": 0.0, "note": "no CASH bars in this window"}

    best = cash_df[["long_net", "short_net"]].max(axis=1)
    qualifies = best > EV_MIN
    oracle_frac_of_cash = float(qualifies.mean())
    oracle_frac_of_all_bars = float(qualifies.sum() / valid_n)
    oracle_avg_edge_qualifying = float(best[qualifies].mean()) if qualifies.any() else 0.0
    oracle_cum_pnl_pct = float(best[qualifies].sum() * 100.0)

    always_short_sum_pct = float(cash_df["short_net"].sum() * 100.0)
    always_long_sum_pct = float(cash_df["long_net"].sum() * 100.0)
    always_short_mean_pct = float(cash_df["short_net"].mean() * 100.0)
    always_long_mean_pct = float(cash_df["long_net"].mean() * 100.0)
    naive_baseline_sum_pct = max(always_short_sum_pct, always_long_sum_pct)

    return {
        "window": wname,
        "valid_bars": int(valid_n),
        "cash_bars": int(n_cash),
        "cash_frac": float(cash_frac),
        "oracle_ev_min_qualifying_frac_of_cash_bars": oracle_frac_of_cash,
        "oracle_ev_min_qualifying_frac_of_all_valid_bars": oracle_frac_of_all_bars,
        "oracle_avg_net_edge_pct_on_qualifying_bars": oracle_avg_edge_qualifying * 100.0,
        "oracle_upper_bound_cum_pnl_pct_NOT_ACHIEVABLE": oracle_cum_pnl_pct,
        "always_short_all_cash_bars_sum_pct": always_short_sum_pct,
        "always_short_all_cash_bars_mean_pct": always_short_mean_pct,
        "always_long_all_cash_bars_sum_pct": always_long_sum_pct,
        "always_long_all_cash_bars_mean_pct": always_long_mean_pct,
        "naive_directional_baseline_sum_pct": naive_baseline_sum_pct,
        "oracle_minus_naive_baseline_pct": oracle_cum_pnl_pct - naive_baseline_sum_pct,
        "long_reason_counts": cash_df["long_reason"].value_counts().to_dict(),
        "short_reason_counts": cash_df["short_reason"].value_counts().to_dict(),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    log(f"fee={fee} slip={slip} (normal, ETH repo constants) -> stressed fee_eff={fee * COST_STRESS_MULT} slip_eff={slip * COST_STRESS_MULT} (3x)")
    log(f"risk profile: take_profit={TAKE_PROFIT} stop_loss={STOP_LOSS} notional={NOTIONAL} leverage={LEVERAGE} "
        f"margin_fraction={MARGIN_FRACTION} max_hold_bars={MAX_HOLD_BARS} ev_min={EV_MIN}")
    assert abs(NOTIONAL - MARGIN_FRACTION * LEVERAGE) < 1e-9, "Futures Risk Sizing Contract: notional must equal margin_fraction*leverage"

    report: dict[str, Any] = {
        "type": "cheap_gate_oracle_hindsight_measurement_NOT_a_backtest_NOT_a_model",
        "candidate": "eth_candidate_cash_sleeve_ev_hgb",
        "ported_from": "BTC cash-sleeve mechanism (dead-code, live-tested) survey -- spec given verbatim by the orchestrating session",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "oracle_fallback_simulation_uses_future_price_path_by_design": True,
        "windows_used": {"val": "2025-10-01..2025-12-31 (sub-project's own established VAL, see module docstring for why not CLAUDE.md's generic 2025-09-01)", "oos_q1": "2026-01-01..2026-03-31 (matches CLAUDE.md generic default exactly)"},
        "risk_profile": {"take_profit": TAKE_PROFIT, "stop_loss": STOP_LOSS, "notional": NOTIONAL, "leverage": LEVERAGE,
                          "margin_fraction": MARGIN_FRACTION, "max_hold_bars": MAX_HOLD_BARS, "ev_min": EV_MIN,
                          "cost_stress_mult": COST_STRESS_MULT, "fee_normal": fee, "slip_normal": slip},
    }

    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    log("=== stage=detector_build (h48qual regime-aware guard + zig075 SHORT veto, both reused unmodified) ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    log(f"  detector threshold(p90)={threshold:.10f}")

    per_window: dict[str, Any] = {}
    g0_check: dict[str, Any] = {}
    for wname in WINDOWS:
        log(f"=== stage=primary_ledger window={wname} ===")
        aligned_frame, ledger, diag = build_primary_ledger(wname, windows, score_by_base, threshold, fee, slip)
        no_gate = portfolio._ledger_metrics(ledger)
        ref = G0_ODYSSEY4_NO_GATE[wname]
        ok = _close(no_gate, ref)
        g0_check[wname] = {"actual": no_gate, "reference": ref, "match": ok}
        log(f"  G0 sanity (no_gate vs locked Odyssey4 baseline): actual={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} "
            f"reference={ref['pnl']:.2f}%/{ref['mdd']:.2f}%/{ref['trades']} match={ok}")

        n = len(aligned_frame)
        held = _held_mask(n, ledger)
        valid_n = n - 2
        cash_bars_direct = int((~held[:valid_n]).sum())
        log(f"  n_bars={n} valid_n={valid_n} primary_held_bars={int(held[:valid_n].sum())} cash_bars={cash_bars_direct} "
            f"cash_frac={cash_bars_direct / valid_n:.4f}")

        log(f"=== stage=cash_sleeve_oracle_simulation window={wname} ===")
        cash_df, valid_n2 = run_cash_sleeve_oracle(aligned_frame, held, fee, slip)
        assert valid_n2 == valid_n
        assert len(cash_df) == cash_bars_direct, f"cash bar count mismatch: {len(cash_df)} vs {cash_bars_direct}"
        cash_df.to_csv(OUT_DIR / f"cash_sleeve_oracle_bars_{wname}.csv", index=False)

        summary = summarize_window(wname, cash_df, valid_n)
        per_window[wname] = summary
        log(f"  cash_frac={summary['cash_frac']:.4f}  oracle_qualifying_frac_of_cash={summary['oracle_ev_min_qualifying_frac_of_cash_bars']:.4f}  "
            f"oracle_avg_edge_pct={summary['oracle_avg_net_edge_pct_on_qualifying_bars']:.4f}  "
            f"oracle_upper_bound_cum_pnl_pct={summary['oracle_upper_bound_cum_pnl_pct_NOT_ACHIEVABLE']:.2f}  "
            f"always_short_sum_pct={summary['always_short_all_cash_bars_sum_pct']:.2f}  "
            f"always_long_sum_pct={summary['always_long_all_cash_bars_sum_pct']:.2f}")

    # combined VAL+OOS-Q1 totals (simple concatenation -- both are already disjoint fixed windows)
    combined_cash_bars = sum(per_window[w]["cash_bars"] for w in WINDOWS)
    combined_valid_bars = sum(per_window[w]["valid_bars"] for w in WINDOWS)
    combined_oracle_cum = sum(per_window[w]["oracle_upper_bound_cum_pnl_pct_NOT_ACHIEVABLE"] for w in WINDOWS)
    combined_naive = sum(per_window[w]["naive_directional_baseline_sum_pct"] for w in WINDOWS)
    combined = {
        "cash_bars": combined_cash_bars,
        "valid_bars": combined_valid_bars,
        "cash_frac": combined_cash_bars / combined_valid_bars if combined_valid_bars else 0.0,
        "oracle_upper_bound_cum_pnl_pct_NOT_ACHIEVABLE_sum_of_windows": combined_oracle_cum,
        "naive_directional_baseline_sum_pct_sum_of_windows": combined_naive,
    }
    log(f"=== combined VAL+OOS-Q1 === cash_frac={combined['cash_frac']:.4f}  "
        f"oracle_cum_pnl_sum={combined_oracle_cum:.2f}%  naive_baseline_sum={combined_naive:.2f}%")

    report["g0_sanity_check_vs_locked_odyssey4_baseline"] = g0_check
    report["g0_pass"] = bool(all(v["match"] for v in g0_check.values()))
    report["per_window"] = per_window
    report["combined_val_oos_q1"] = combined

    # Verdict (descriptive only -- the orchestrating session/user makes the actual proceed/close call)
    oracle_frac_avg = np.mean([per_window[w]["oracle_ev_min_qualifying_frac_of_cash_bars"] for w in WINDOWS])
    verdict = "SIGNAL_PRESENT_NEEDS_HUMAN_DECISION" if oracle_frac_avg >= 0.05 and combined_oracle_cum > combined_naive else "LIKELY_CLOSE_NEGATIVE_AT_CHEAP_GATE"
    report["verdict_heuristic"] = {
        "rule": "oracle_ev_min_qualifying_frac_of_cash_bars >= 5% of CASH bars AND oracle_upper_bound_cum_pnl > naive_directional_baseline, averaged/summed across VAL+OOS-Q1",
        "oracle_qualifying_frac_avg": float(oracle_frac_avg),
        "verdict": verdict,
    }
    log(f"=== verdict_heuristic === {verdict} (oracle_qualifying_frac_avg={oracle_frac_avg:.4f})")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0 if report["g0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
