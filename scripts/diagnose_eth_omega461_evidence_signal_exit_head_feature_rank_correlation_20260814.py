#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap, no-retrain, no-GPU pre-gate for the "add the evidence signal as an
exit_head training feature" direction (Odyssey2 #18 follow-up, user chose exit_head retrain over
sizing-sidecar/entry-side options -- docs/experiments/eth_omega461_evidence_veto_exit_overlay_
20260814.md's rejected hard-veto overlay is a DIFFERENT mechanism from this).

Reuses the ALREADY-COMPUTED h48qual asymmetric_tabm_liveatr trade ledgers written by
research_eth_omega461_evidence_veto_exit_overlay_20260814.py's own main_run stage (6 windows,
gate.run_portfolio_variant, variant_label="asymmetric_tabm_liveatr") -- no new simulation, just
reads the existing ledger CSVs. Also reuses build_signal (unmodified import, orthogonal_combo)
from that same script for the evidence signal.

=== What this checks (the same "0단계 진단" pattern this sub-project always runs before a retrain,
matching e.g. eth_h48qual_quality_for_action_rank_correlation_20260811.md) ===
For every BAR where h48qual holds a SHORT position (not per-trade -- see confound note in main()):
does the bottom-evidence signal firing ON THAT BAR correlate with a favorable forward-K-bar price
move for the short? If evidence-firing reliably predicts a worse forward move, exit_head has
something to learn from this feature; if not (or sign-unstable across windows), a retrain is
unlikely to find a useful gradient regardless of how the feature is added.

fresh_forward_bar_by_bar=true (the underlying ledgers were produced by a causal bar-by-bar replay;
this script only reads them and computes a fixed-window overlap check, no new simulation, no
forward-looking join). trade_ledgers_used_as_input=true -- unlike every OTHER script in this
lineage, this one explicitly IS a diagnostic-only, promotion-irrelevant read of ledgers (per this
repo's Fresh-Forward rule: "저장된 trade ledger ... replay는 diagnostic ... 전용. 모델 선택,
승격 ... 근거로 쓰지 않는다" -- this script's role is exactly that diagnostic use, never promotion
evidence on its own). saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env.
Does NOT modify any imported module or the ledger CSVs it reads (read-only). No retraining, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from research_eth_omega461_evidence_veto_exit_overlay_20260814 import build_signal  # noqa: E402

LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_evidence_veto_exit_overlay_20260814"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_evidence_signal_exit_head_feature_rank_correlation_20260814"

FORWARD_K_BARS = 12  # 1h -- the horizon an exit_head reading this feature per-bar would actually care about


def log(msg: str) -> None:
    print(f"[evidence_exit_head_feature_diag] {msg}", flush=True)


def _window_evidence_exposure_bar_level(window_name: str, score_by_base: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """BAR-LEVEL (not trade-level) diagnostic: for every bar where h48qual holds a SHORT position,
    does evidence_veto firing ON THAT BAR predict a favorable forward K-bar price move (down, good
    for a short) in the following FORWARD_K_BARS? This is the level exit_head would actually
    consume the feature at (a per-bar 'cur_evidence_veto' column), unlike a trade-level any-fire
    aggregate, which confounds with hold duration (longer/winning trades simply have more bars for
    a rare event to appear somewhere, inflating a spurious positive correlation -- caught and
    discarded in this session before being reported as a finding)."""
    ledger_path = LEDGER_DIR / f"portfolio_ledger_{window_name}_asymmetric_tabm_liveatr.csv"
    ledger = pd.read_csv(ledger_path)
    ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
    ledger["exit_timestamp"] = pd.to_datetime(ledger["exit_timestamp"])
    short_trades = ledger[(ledger["source_component"] == "h48qual") & (ledger["side"] < 0)].reset_index(drop=True)
    if short_trades.empty:
        return pd.DataFrame(columns=["timestamp", "evidence_fired", "forward_k_return_for_short"])

    base_csv = gate.WINDOW_DEFS[window_name]["base_csv"]
    # build_signal's output only carries timestamp/evidence_veto/delta_z_nan -- reload close
    # separately (same read/sort/dedup discipline as _evidence_veto_score) for the forward-return calc.
    raw = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "close"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    score = score_by_base[base_csv].merge(raw, on="timestamp", how="left")
    score["fwd_close"] = score["close"].shift(-FORWARD_K_BARS)
    score["forward_k_return_for_short"] = (score["close"] - score["fwd_close"]) / score["close"]  # positive = price fell = good for a short

    rows: list[dict[str, Any]] = []
    ts_index = score.set_index("timestamp")
    for _, tr in short_trades.iterrows():
        mask = (ts_index.index >= tr["entry_timestamp"]) & (ts_index.index < tr["exit_timestamp"])
        sub = ts_index.loc[mask]
        for ts, r in sub.iterrows():
            if pd.isna(r["forward_k_return_for_short"]):
                continue
            rows.append({"timestamp": ts, "evidence_fired": bool(r["evidence_veto"]), "forward_k_return_for_short": float(r["forward_k_return_for_short"])})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=build_signal ===")
    score_by_base = build_signal()

    report: dict[str, Any] = {
        "design": (
            "Cheap pre-retrain diagnostic, BAR-LEVEL (not trade-level -- see note below): for "
            "every bar where h48qual holds a SHORT position (ledgers reused from "
            "research_eth_omega461_evidence_veto_exit_overlay_20260814.py's own "
            "asymmetric_tabm_liveatr run, no new simulation), does the orthogonal_combo evidence "
            "signal firing ON THAT BAR correlate with a favorable forward-12-bar (1h) price move "
            "for the short? This is the level exit_head would actually read the feature at "
            "(per-bar), unlike a trade-level any-fire-during-hold aggregate. Gates whether an "
            "exit_head retrain with this feature is worth the server-GPU cost."
        ),
        "confound_caught_and_discarded": (
            "An earlier trade-level version of this diagnostic (evidence fired ANYWHERE during the "
            "whole hold vs final trade_return) showed a consistent POSITIVE correlation in every "
            "window (VAL point-biserial rho=+0.675, p=0.016; 2025q3 spearman rho=+0.598, p=0.009) "
            "-- the WRONG sign for the intended use (would mean evidence firing predicts a BETTER "
            "outcome, not worse). Root cause: winning/longer-held trades simply have more bars "
            "during which a rare (~0.5-0.7%/bar) event can appear somewhere, inflating a spurious "
            "positive correlation via a hold-duration confound. This bar-level version fixes that "
            "by testing forward return FROM the specific bar where the signal fires, not the "
            "trade's eventual outcome."
        ),
        "forward_k_bars": FORWARD_K_BARS,
        "trade_ledgers_used_as_input": True,
        "trade_ledgers_role": "diagnostic_only_per_repo_fresh_forward_rule_never_promotion_evidence",
        "windows": {},
    }

    pooled_fired: list[float] = []
    pooled_not_fired: list[float] = []
    for wname in gate.ALL_WINDOWS:
        df = _window_evidence_exposure_bar_level(wname, score_by_base)
        n = len(df)
        n_fired = int(df["evidence_fired"].sum()) if n else 0
        if n < 20 or n_fired < 2 or n_fired == n:
            corr = {"n": n, "n_fired": n_fired, "skipped": "too_few_bars_or_no_variation"}
        else:
            rho, p = stats.pointbiserialr(df["evidence_fired"].astype(int), df["forward_k_return_for_short"])
            corr = {"n": n, "n_fired": n_fired, "point_biserial_rho": float(rho), "p_value": float(p)}
            pooled_fired.extend(df.loc[df["evidence_fired"], "forward_k_return_for_short"].tolist())
            pooled_not_fired.extend(df.loc[~df["evidence_fired"], "forward_k_return_for_short"].tolist())
        mean_fired = float(df.loc[df["evidence_fired"], "forward_k_return_for_short"].mean()) if n_fired > 0 else float("nan")
        mean_not_fired = float(df.loc[~df["evidence_fired"], "forward_k_return_for_short"].mean()) if (n - n_fired) > 0 else float("nan")
        report["windows"][wname] = {
            "n_short_hold_bars": n, "n_evidence_fired_bars": n_fired,
            "bar_level_fired_vs_forward_k_return": corr,
            "mean_forward_k_return_when_fired": mean_fired,
            "mean_forward_k_return_when_not_fired": mean_not_fired,
        }
        log(f"  {wname:8s} n_bars={n:6d} fired={n_fired:4d} "
            f"pointbiserial_rho={corr.get('point_biserial_rho', float('nan')):+.4f} p={corr.get('p_value', float('nan')):.4f}  "
            f"mean_fwd_ret fired={mean_fired:+.4f} not_fired={mean_not_fired:+.4f}")
        if n:
            df.to_csv(OUT_DIR / f"short_hold_bar_evidence_exposure_{wname}.csv", index=False)

    signed_rhos = [w["bar_level_fired_vs_forward_k_return"].get("point_biserial_rho") for w in report["windows"].values() if isinstance(w["bar_level_fired_vs_forward_k_return"], dict) and "point_biserial_rho" in w["bar_level_fired_vs_forward_k_return"]]
    n_negative = sum(1 for r in signed_rhos if r < 0)
    n_positive = sum(1 for r in signed_rhos if r > 0)
    pooled_rho, pooled_p = (stats.pointbiserialr([1] * len(pooled_fired) + [0] * len(pooled_not_fired), pooled_fired + pooled_not_fired) if pooled_fired and pooled_not_fired else (float("nan"), float("nan")))
    report["summary"] = {
        "windows_with_correlation_computed": len(signed_rhos),
        "windows_negative_rho": n_negative,  # negative = evidence firing on a bar predicts a WORSE forward move for the short = the useful direction for an exit feature
        "windows_positive_rho": n_positive,
        "pooled_all_windows_point_biserial_rho": float(pooled_rho), "pooled_p_value": float(pooled_p),
        "pooled_n_fired": len(pooled_fired), "pooled_n_not_fired": len(pooled_not_fired),
        "note": "negative point-biserial rho is the useful direction here (evidence firing on a bar during a SHORT hold predicts a WORSE forward price move for the short over the next 12 bars, i.e. price rises against the short -- exactly the counter-evidence signal exit_head would need to learn to weight toward exiting). Sign consistency across windows, and the pooled statistic (much larger n than any single window), matter more than any single window's per-window significance.",
    }
    log(f"  SUMMARY: {n_negative} negative / {n_positive} positive / {len(signed_rhos)} windows computed  pooled_rho={pooled_rho:+.4f} pooled_p={pooled_p:.4f} (n_fired={len(pooled_fired)}, n_not_fired={len(pooled_not_fired)})")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
